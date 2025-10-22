import http.server
import socketserver
import json
import os
import base64
import cgi
import urllib.parse
from datetime import datetime
import logging
import webbrowser
import threading
import time
import mimetypes
import subprocess
import queue
import uuid
import sys
import random
import numpy as np
from PIL import Image
import torch
import traceback
from framepack_start_end import process_video as framepack_process_video

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 確保基礎輸出目錄存在
OUTPUT_DIR = "storyboard_outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"創建基礎目錄: {OUTPUT_DIR}")

# 輔助函數：創建專案資料夾結構
def create_project_structure(project_folder):
    """創建專案資料夾及其子資料夾"""
    image_folder = os.path.join(project_folder, "images")
    os.makedirs(project_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    return image_folder

# 任務隊列和處理狀態
task_queue = queue.Queue()
task_status = {}  # 存儲任務狀態的字典
stop_processing = False
preview_jobs = {}
preview_cancel_flags = {}

## 已移除 Qwen 預載（warm-up），改為每次推理時才載入

# 配置參數
default_params = {
    "prompt": "Character movements based on storyboard sequence",
    "n_prompt": "",
    "seed": 31337,
    "total_second_length": 5,
    "latent_window_size": 9,
    "steps": 25,
    "cfg": 1.0,
    "gs": 10.0,
    "rs": 0.0,
    "gpu_memory_preservation": 6,
    "use_teacache": True,
    "mp4_crf": 16
}

def create_tasks_from_nodes_and_transitions_direct(nodes, transitions, project_folder):
    """
    直接從節點和轉場數據創建視頻處理任務（用於重新生成）
    
    Args:
        nodes: 節點數據列表
        transitions: 轉場數據列表
        project_folder: 現有的專案資料夾路徑
    
    Returns:
        添加的任務ID列表，或錯誤信息
    """
    try:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_image_folder = os.path.join(project_folder, "images")
        
        if len(nodes) < 2:
            error_msg = f"節點少於2個，無法創建任務"
            logger.warning(error_msg)
            return {"success": False, "message": "需要至少2個節點才能生成影片。"}
        
        # 檢查所有節點是否都有圖片
        logger.info(f"檢查重生任務的圖片，專案圖片資料夾: {project_image_folder}")
        
        # 列出專案圖片資料夾中的所有文件
        if os.path.exists(project_image_folder):
            available_images = os.listdir(project_image_folder)
            logger.info(f"專案圖片資料夾中的文件: {available_images}")
        else:
            logger.error(f"專案圖片資料夾不存在: {project_image_folder}")
            return {"success": False, "message": f"專案圖片資料夾不存在: {project_image_folder}"}
        
        missing_images = []
        node_image_paths = {}  # 儲存每個節點的實際圖片路徑
        
        for i, node in enumerate(nodes):
            has_image = node.get("hasImage", False)
            image_path = node.get("imagePath")
            
            logger.info(f"檢查節點 {i}: hasImage={has_image}, imagePath={image_path}")
            
            if not has_image:
                missing_images.append(i)
                logger.warning(f"節點 {i} 標記為沒有圖片")
                continue
            
            # 嘗試多種方式找到圖片文件
            found_image = None
            
            # 方法1: 如果提供了 imagePath，嘗試使用它
            if image_path:
                # 嘗試直接使用 imagePath（可能是文件名）
                full_path = os.path.join(project_image_folder, os.path.basename(image_path))
                if os.path.exists(full_path):
                    found_image = full_path
                    logger.info(f"節點 {i} 找到圖片（方法1-basename）: {full_path}")
                else:
                    # 嘗試直接使用 imagePath
                    alt_path = os.path.join(project_image_folder, image_path)
                    if os.path.exists(alt_path):
                        found_image = alt_path
                        logger.info(f"節點 {i} 找到圖片（方法1-direct）: {alt_path}")
            
            # 方法2: 如果還沒找到，嘗試常見的命名模式
            if not found_image:
                common_patterns = [
                    f"node_{i}.png",
                    f"node_{i}.jpg",
                    f"node_{i}.jpeg",
                    f"{i}.png",
                    f"{i}.jpg",
                    f"{i}.jpeg"
                ]
                
                for pattern in common_patterns:
                    test_path = os.path.join(project_image_folder, pattern)
                    if os.path.exists(test_path):
                        found_image = test_path
                        logger.info(f"節點 {i} 找到圖片（方法2-pattern）: {found_image}")
                        break
            
            # 方法3: 如果還沒找到，按索引順序查找圖片文件
            if not found_image:
                image_files = [f for f in available_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_files.sort()  # 確保順序一致
                
                if i < len(image_files):
                    found_image = os.path.join(project_image_folder, image_files[i])
                    logger.info(f"節點 {i} 找到圖片（方法3-index）: {found_image}")
            
            if found_image:
                node_image_paths[i] = found_image
                logger.info(f"節點 {i} 最終使用圖片: {found_image}")
            else:
                missing_images.append(i)
                logger.warning(f"節點 {i} 找不到任何圖片文件")
        
        if missing_images:
            error_msg = f"節點 {missing_images} 缺少圖片文件"
            logger.error(error_msg)
            return {"success": False, "message": f"節點 {missing_images} 缺少圖片文件"}
        
        task_ids = []
        
        # 為每相鄰的節點對創建任務
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            # 使用我們之前找到的實際圖片路徑
            start_image_path = node_image_paths.get(i)
            end_image_path = node_image_paths.get(i + 1)
            
            if not start_image_path or not end_image_path:
                logger.error(f"無法找到節點 {i} 或 {i+1} 的圖片路徑")
                continue
            
            logger.info(f"重生任務 {i}->{i+1}: 起始圖片={start_image_path}, 結束圖片={end_image_path}")
            
            # 查找對應的轉場描述
            transition_text = ""
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    transition_text = transition.get("text", "")
                    break
            
            # 重生時重用現有的 task ID，而不是創建新的
            # 查找現有的 task ID
            existing_task_id = None
            if os.path.exists(project_folder):
                for filename in os.listdir(project_folder):
                    if filename.startswith(f"video_{i}_{i+1}_") and filename.endswith('.mp4'):
                        existing_task_id = filename[:-4]  # 移除 .mp4 擴展名
                        logger.info(f"找到現有任務 ID: {existing_task_id}")
                        break
            
            # 如果找不到現有任務，創建新的
            if existing_task_id:
                task_id = existing_task_id
                logger.info(f"重用現有任務 ID: {task_id}")
            else:
                timestamp = datetime.now().strftime("%H%M%S")
                task_id = f"video_{i}_{i+1}_{timestamp}"
                logger.info(f"創建新任務 ID: {task_id}")
            
            # 設置任務時長（基於節點時間差）
            time_range = None
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    time_range = transition.get("time_range")
                    break
            
            second_length = 5  # 默認5秒
            if time_range and len(time_range) >= 2:
                second_length = max(0.1, float(time_range[1]) - float(time_range[0]))
            
            # 獲取 seed（從結束節點）
            seed = end_node.get("seed", default_params["seed"])
            if seed is None:
                seed = random.randint(0, 2147483647)
            
            logger.info(f"創建重新生成任務 {task_id}: {start_image_path} -> {end_image_path}")
            logger.info(f"  轉場描述: '{transition_text}'")
            logger.info(f"  時長: {second_length} 秒")
            logger.info(f"  Seed: {seed}")
            
            # 創建任務字典（與正常處理流程保持一致）
            # 重生任務會直接覆蓋現有文件，不需要額外的刪除操作
            task = {
                "id": task_id,
                "start_image": start_image_path,
                "end_image": end_image_path,
                "project_folder": project_folder,
                "is_regenerate": True,  # 標記為重生任務
                "params": {
                    "prompt": transition_text,
                    "seed": seed,
                    "second_length": second_length
                }
            }
            
            # 將任務添加到隊列
            task_queue.put(task)
            
            task_ids.append(task_id)
            
            # 初始化或更新任務狀態
            if task_id in task_status:
                # 如果任務已存在（重生情況），更新狀態但保留輸出文件信息
                existing_output = task_status[task_id].get("output_file")
                task_status[task_id] = {
                    "status": "pending",
                    "message": f"重生任務已創建，等待處理",
                    "progress": 0,
                    "output_file": existing_output  # 保留現有的輸出文件
                }
                logger.info(f"更新現有任務狀態: {task_id}")
            else:
                # 新任務
                task_status[task_id] = {
                    "status": "pending", 
                    "message": f"任務已創建，等待處理",
                    "progress": 0
                }
                logger.info(f"創建新任務狀態: {task_id}")
        
        logger.info(f"成功創建 {len(task_ids)} 個重新生成任務")
        return {"success": True, "task_ids": task_ids}
        
    except Exception as e:
        logger.error(f"創建重新生成任務時發生錯誤: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error creating regeneration tasks: {str(e)}"}


# 模擬 FramePack 處理函數
def process_video_task(task_id, start_image_path, end_image_path=None, params=None, project_folder=None):
    """
    處理視頻生成任務
    
    Args:
        task_id: 任務ID
        start_image_path: 起始幀圖片路徑
        end_image_path: 結束幀圖片路徑 (可選)
        params: 其他參數字典
        project_folder: 專案資料夾路徑
    """
    global task_status
    
    try:
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 0
        task_status[task_id]["message"] = "啟動處理任務..."
        
        # 合併默認參數和提供的參數
        if params is None:
            params = {}
        actual_params = default_params.copy()
        actual_params.update(params)
        
        logger.info(f"開始處理任務 {task_id}，起始幀：{start_image_path}, 結束幀：{end_image_path}")
        
        # 檢查是否應該停止處理
        if stop_processing:
            task_status[task_id]["status"] = "cancelled"
            task_status[task_id]["message"] = "任務被取消"
            return None
        
        # 準備輸出文件名（使用專案資料夾）
        if project_folder:
            output_filename = os.path.join(project_folder, f"{task_id}.mp4")
        else:
            # 如果沒有專案資料夾，使用臨時目錄
            import tempfile
            output_filename = os.path.join(tempfile.gettempdir(), f"{task_id}.mp4")
        
        # 使用 framepack（已在啟動時匯入函式；模型載入應由外部模組在呼叫時處理）
        if framepack_process_video:
            # 使用自定義進度回調來更新任務狀態
            def progress_callback(percentage, message):
                task_status[task_id]["progress"] = percentage
                task_status[task_id]["message"] = message
                logger.info(f"任務 {task_id} 進度: {percentage}% - {message}")
            
            # 調用framepack_start_end.py中的處理函數
            logger.info(f"使用framepack_process_video處理任務 {task_id}")
            result = framepack_process_video(
                start_image_path, 
                end_image_path, 
                progress_callback=progress_callback,
                prompt=actual_params["prompt"],
                n_prompt=actual_params["n_prompt"],
                seed=actual_params["seed"],
                total_second_length=actual_params["total_second_length"],
                latent_window_size=actual_params["latent_window_size"],
                steps=actual_params["steps"],
                cfg=actual_params["cfg"],
                gs=actual_params["gs"],
                rs=actual_params["rs"],
                gpu_memory_preservation=actual_params["gpu_memory_preservation"],
                use_teacache=actual_params["use_teacache"],
                mp4_crf=actual_params["mp4_crf"],
                output=output_filename
            )
            
            if result:
                task_status[task_id]["status"] = "completed"
                task_status[task_id]["progress"] = 100
                task_status[task_id]["message"] = "處理完成！"
                task_status[task_id]["output_file"] = result
                logger.info(f"任務 {task_id} 處理完成，輸出文件: {result}")
                return result
            else:
                task_status[task_id]["status"] = "error"
                task_status[task_id]["message"] = "處理失敗"
                logger.error(f"任務 {task_id} 處理失敗")
                return None
        else:
            # 如果無法導入 framepack，則標記任務為錯誤
            logger.error(f"任務 {task_id} 無法處理: framepack 模組未成功導入。")
            task_status[task_id]["status"] = "error"
            task_status[task_id]["message"] = "處理模組不可用 (framepack missing)"
            task_status[task_id]["progress"] = 0 # 確保進度為0
            return None
        
    except Exception as e:
        logger.error(f"處理任務 {task_id} 出錯: {e}")
        traceback.print_exc()
        task_status[task_id]["status"] = "error"
        task_status[task_id]["message"] = f"錯誤: {str(e)}"
        return None

# 任務處理線程
def task_processor():
    global task_status, stop_processing
    
    logger.info("任務處理線程已啟動")
    
    while not stop_processing:
        try:
            # 從隊列獲取任務，非阻塞
            try:
                task = task_queue.get(block=False)
                task_id = task.get("id")
                logger.info(f"開始處理任務: {task_id}")
                
                # 處理任務
                start_image = task.get("start_image")
                end_image = task.get("end_image")
                params = task.get("params", {})
                project_folder = task.get("project_folder")
                is_regenerate = task.get("is_regenerate", False)
                
                # 如果是重生任務，重置任務狀態為等待中
                if is_regenerate:
                    task_status[task_id] = {
                        "status": "queued",
                        "progress": 0,
                        "message": "重生任務已加入隊列",
                        "output_file": task_status.get(task_id, {}).get("output_file")  # 保留原有的輸出文件信息
                    }
                    logger.info(f"重生任務 {task_id} 狀態重置為等待中")
                
                result = process_video_task(task_id, start_image, end_image, params, project_folder)
                
                # 標記任務完成
                task_queue.task_done()
                
            except queue.Empty:
                # 隊列為空，等待
                time.sleep(1)
                continue
                
        except Exception as e:
            logger.error(f"任務處理線程發生錯誤: {e}")
            traceback.print_exc()
            time.sleep(5)  # 出錯後稍微等待
    
    logger.info("任務處理線程已停止")

# 從故事板JSON生成任務
def create_tasks_from_storyboard(storyboard_file):
    """
    從故事板JSON文件生成視頻處理任務
    
    Args:
        storyboard_file: JSON文件路徑（已經在專案資料夾內）
    
    Returns:
        添加的任務ID列表，或錯誤信息
    """
    try:
        with open(storyboard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        job_id = data.get("job_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        nodes = data.get("nodes", [])
        transitions = data.get("transitions", [])
        
        # 獲取專案資料夾路徑（JSON 文件已經在專案資料夾內）
        project_folder = os.path.dirname(storyboard_file)
        project_image_folder = os.path.join(project_folder, "images")
        
        if len(nodes) < 2:
            error_msg = f"故事板 {storyboard_file} 節點少於2個，無法創建任務"
            logger.warning(error_msg)
            return {"success": False, "message": "Storyboard needs at least 2 nodes to generate videos."}
        
        # 獲取節點圖像路徑
        task_ids = []
        
        # 為每一對連續節點創建任務
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            # 獲取圖像路徑 - 直接在專案的 images 資料夾中查找
            start_image_path = start_node.get("imagePath")
            if start_image_path:
                start_image_path = os.path.join(project_image_folder, os.path.basename(start_image_path))
                if not os.path.exists(start_image_path):
                    logger.error(f"無法找到節點 {i} 的圖片: {start_image_path}")
                    continue
            
            # 處理結束節點圖像
            end_image_path = end_node.get("imagePath")
            if end_image_path:
                end_image_path = os.path.join(project_image_folder, os.path.basename(end_image_path))
                if not os.path.exists(end_image_path):
                    logger.warning(f"無法找到節點 {i+1} 的圖片: {end_image_path}")
                    end_image_path = None
            
            # 查找對應的轉場描述
            transition_text = ""
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    transition_text = transition.get("text", "")
                    break
            
            # 設置任務參數 - 簡化命名方式，添加時間戳確保唯一性
            timestamp = datetime.now().strftime("%H%M%S")
            task_id = f"video_{i}_{i+1}_{timestamp}"
            
            # 設置任務時長（基於節點時間差）
            time_range = None
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    time_range = transition.get("time_range")
                    break
            
            second_length = 5  # 默認5秒
            if time_range and len(time_range) >= 2:
                second_length = max(1, time_range[1] - time_range[0])
            
            # 獲取節點的 seed（使用結束節點的 seed，因為那是要生成的目標）
            node_seed = end_node.get("seed")
            if node_seed is None:
                node_seed = default_params["seed"]  # 使用預設 seed 如果未指定
            
            # 創建任務
            task = {
                "id": task_id,
                "start_image": start_image_path,
                "end_image": end_image_path,
                "project_folder": project_folder,  # 加入專案資料夾路徑
                "params": {
                    "prompt": f"Character movement: {transition_text}" if transition_text else default_params["prompt"],
                    "total_second_length": second_length,
                    "seed": node_seed  # 包含節點的 seed
                }
            }
            
            # 初始化任務狀態
            task_status[task_id] = {
                "status": "queued",
                "progress": 0,
                "message": "In queue",
                "created_at": datetime.now().isoformat()
            }
            
            # 添加到任務隊列
            task_queue.put(task)
            task_ids.append(task_id)
            
            logger.info(f"創建任務 {task_id}，從節點 {i} 到節點 {i+1}")
        
        return {"success": True, "task_ids": task_ids, "project_folder": project_folder}
        
    except Exception as e:
        logger.error(f"從故事板創建任務時出錯: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error creating tasks: {str(e)}"}

# 自定義 HTTP 請求處理器
class StoryboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # 處理根路徑請求，提供 storyboard.html
        if path == '/' or path == '':
            self.path = '/storyboard.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
        # 健康檢查
        elif path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({
                'status': 'ok',
                'message': '伺服器運行中'
            })
            self.wfile.write(response.encode('utf-8'))
            return
            
        # 獲取任務狀態
        elif path.startswith('/task_status'):
            query = urllib.parse.parse_qs(parsed_path.query)
            task_id = query.get('id', [''])[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if task_id and task_id in task_status:
                response = json.dumps({
                    'status': 'success',
                    'task': task_status[task_id]
                })
            elif not task_id:
                # 返回所有任務狀態
                response = json.dumps({
                    'status': 'success',
                    'tasks': task_status
                })
            else:
                response = json.dumps({
                    'status': 'error',
                    'message': f'找不到任務 {task_id}'
                })
            
            self.wfile.write(response.encode('utf-8'))
            return

        # 新增：查詢預覽生成任務狀態
        elif path.startswith('/preview_status'):
            query = urllib.parse.parse_qs(parsed_path.query)
            job_id = query.get('id', [''])[0]

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            task = task_status.get(job_id)
            if task:
                response = json.dumps({'status': 'success', 'task': task})
            else:
                response = json.dumps({'status': 'error', 'message': '找不到任務'})
            self.wfile.write(response.encode('utf-8'))
            return

        # 取消預覽任務
        elif path.startswith('/cancel_preview'):
            query = urllib.parse.parse_qs(parsed_path.query)
            job_id = query.get('id', [''])[0]

            if not job_id:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': '缺少任務ID'}).encode('utf-8'))
                return

            if job_id in task_status and task_status[job_id]['status'] == 'processing':
                preview_cancel_flags[job_id] = True
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'message': '已請求取消'}).encode('utf-8'))
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success', 'message': '任務不在處理中或已完成'}).encode('utf-8'))
            return
            
        # 列出所有故事板JSON文件（只從專案資料夾中搜尋）
        elif path == '/list_storyboards':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                files = []
                
                # 只搜尋專案資料夾中的 JSON 文件（新格式：Xnodes_YYYYMMDD_HHMMSS 或舊格式：storyboard_YYYYMMDD_HHMMSS）
                for item in os.listdir(OUTPUT_DIR):
                    if ('nodes_' in item or item.startswith('storyboard_')) and os.path.isdir(os.path.join(OUTPUT_DIR, item)):
                        project_path = os.path.join(OUTPUT_DIR, item)
                        # 在每個專案資料夾中搜尋 JSON 文件
                        for filename in os.listdir(project_path):
                            if filename.endswith('.json'):
                                file_path = os.path.join(project_path, filename)
                                file_stats = os.stat(file_path)
                                files.append({
                                    'filename': filename,
                                    'path': file_path,
                                    'project_folder': item,
                                    'size': file_stats.st_size,
                                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                                })
                
                # 按修改時間排序（最新的在前）
                files.sort(key=lambda x: x['modified'], reverse=True)
                
                response = json.dumps({
                    'status': 'success',
                    'files': files
                })
                
            except Exception as e:
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                
            self.wfile.write(response.encode('utf-8'))
            return

        # 新增：處理讀取特定故事板JSON文件的請求
        elif path == '/load_storyboard':
            query_components = urllib.parse.parse_qs(parsed_path.query)
            filename = query_components.get('file', [None])[0]

            if not filename:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": "Filename parameter is missing"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return

            # 安全性檢查，防止路徑遍歷
            if '..' in filename or filename.startswith('/') or filename.startswith('\\\\'):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": "Invalid filename"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return

            # 只在專案資料夾中搜尋文件
            file_path = None
            project_folder = None
            
            # 在專案資料夾中搜尋文件（新格式：Xnodes_YYYYMMDD_HHMMSS 或舊格式：storyboard_YYYYMMDD_HHMMSS）
            for item in os.listdir(OUTPUT_DIR):
                if ('nodes_' in item or item.startswith('storyboard_')) and os.path.isdir(os.path.join(OUTPUT_DIR, item)):
                    candidate_path = os.path.join(OUTPUT_DIR, item, filename)
                    if os.path.exists(candidate_path):
                        file_path = candidate_path
                        project_folder = item
                        break

            if not file_path:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": f"Storyboard file '{filename}' not found in any project folder"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 處理圖片路徑 - 從專案資料夾的 images 載入
                project_image_folder = os.path.join(OUTPUT_DIR, project_folder, "images")
                logger.info(f"載入專案圖片資料夾: {project_image_folder}")
                
                # 列出資料夾中的所有文件用於調試
                if os.path.exists(project_image_folder):
                    available_images = os.listdir(project_image_folder)
                    logger.info(f"專案圖片資料夾中的文件: {available_images}")
                else:
                    logger.warning(f"專案圖片資料夾不存在: {project_image_folder}")
                
                for node in data.get('nodes', []):
                    if node.get('imagePath'):
                        # 嘗試直接使用 imagePath
                        image_path = os.path.join(project_image_folder, node['imagePath'])
                        
                        # 如果直接路徑不存在，嘗試只使用文件名
                        if not os.path.exists(image_path):
                            basename = os.path.basename(node['imagePath'])
                            image_path = os.path.join(project_image_folder, basename)
                            logger.info(f"嘗試使用基本文件名: {basename}")
                        
                        if os.path.exists(image_path):
                            try:
                                with open(image_path, 'rb') as img_file:
                                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                                    node['imageData'] = image_data
                                    node['imageType'] = 'image/png'
                                    # 保留 imagePath 以便識別，但前端會使用 imageData
                                    logger.info(f"成功載入節點 {node.get('index')} 的圖片: {node['imagePath']} -> {image_path}")
                            except Exception as e:
                                logger.error(f"讀取圖片失敗 {image_path}: {e}")
                        else:
                            logger.warning(f"專案圖片文件不存在: {image_path} (原始路徑: {node['imagePath']})")
                    else:
                        # 如果沒有 imagePath，嘗試根據節點索引尋找圖片
                        node_index = node.get('index', 0)
                        # 嘗試多種可能的檔案名稱模式
                        # 從專案資料夾名稱提取時間戳
                        project_timestamp = project_folder.split('_', 1)[1] if '_' in project_folder else datetime.now().strftime("%Y%m%d_%H%M%S")
                        possible_names = [
                            f"node_{node_index}_{project_timestamp}.png",
                            f"node_{node_index}.png",
                            f"{node_index}.png"
                        ]
                        
                        for possible_name in possible_names:
                            image_path = os.path.join(project_image_folder, possible_name)
                            if os.path.exists(image_path):
                                try:
                                    with open(image_path, 'rb') as img_file:
                                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                                        node['imageData'] = image_data
                                        node['imageType'] = 'image/png'
                                        node['imagePath'] = possible_name  # 設定 imagePath 供前端使用
                                        logger.info(f"成功載入節點 {node_index} 的圖片（按索引尋找）: {possible_name}")
                                        break
                                except Exception as e:
                                    logger.error(f"讀取圖片失敗 {image_path}: {e}")
                        else:
                            logger.warning(f"節點 {node_index} 沒有找到對應的圖片檔案")
                
                # 添加專案資料夾信息到回應中
                data['project_folder'] = os.path.join(OUTPUT_DIR, project_folder) if project_folder else None
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
            
            except json.JSONDecodeError as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": f"Error decoding JSON from file '{filename}': {str(e)}"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_data = {"status": "error", "message": f"Error loading storyboard file: {str(e)}"}
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return
        
        # 視頻文件提供（支援專案資料夾結構）
        elif path.startswith('/video/'):
            video_name = path.replace('/video/', '')
            video_path = None
            
            # 搜索專案資料夾中的影片文件（新格式：Xnodes_YYYYMMDD_HHMMSS 或舊格式：storyboard_YYYYMMDD_HHMMSS）
            for item in os.listdir(OUTPUT_DIR):
                if 'nodes_' in item or item.startswith('storyboard_'):
                    project_path = os.path.join(OUTPUT_DIR, item)
                    if os.path.isdir(project_path):
                        candidate_path = os.path.join(project_path, video_name)
                        if os.path.exists(candidate_path):
                            video_path = candidate_path
                            break
            
            if video_path and os.path.exists(video_path):
                logger.info(f"提供影片檔案: {video_path}")
                self.send_response(200)
                # 明確設定 MP4 的 Content-Type
                self.send_header('Content-type', 'video/mp4')
                self.send_header('Content-length', str(os.path.getsize(video_path)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                with open(video_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                logger.error(f"找不到影片檔案: {video_name}")
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(f'Video not found: {video_name}'.encode('utf-8'))
            return
            
        # 其他請求正常處理
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_OPTIONS(self):
        # 處理 CORS 預檢請求
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        # 處理儲存故事板的請求 - 新邏輯：只做驗證，不保存文件
        if self.path == '/save_storyboard':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                # 解析 JSON 資料（只做驗證）
                data = json.loads(post_data.decode('utf-8'))
                logger.info(f"收到故事板資料: {len(data['nodes'])} 個節點")
                
                # 檢查是否至少有兩個節點
                if len(data['nodes']) < 2:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    response = json.dumps({
                        'status': 'error',
                        'message': "Storyboard needs at least 2 nodes to create transitions. Please add more nodes."
                    })
                    self.wfile.write(response.encode('utf-8'))
                    return
                
                # 檢查所有節點是否都有圖片
                missing_images = []
                for i, node in enumerate(data['nodes']):
                    has_image = node.get('hasImage', False)
                    has_image_data = 'imageData' in node
                    has_image_path = 'imagePath' in node
                    
                    if not has_image or (not has_image_data and not has_image_path):
                        missing_images.append(i)
                
                if missing_images:
                    nodes_str = ", ".join([str(i) for i in missing_images])
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    response = json.dumps({
                        'status': 'error',
                        'message': f"Nodes {nodes_str} are missing images. Please add images to all nodes before saving."
                    })
                    self.wfile.write(response.encode('utf-8'))
                    return
                
                # 創建專案資料夾並保存故事板檔案
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                node_count = len(data['nodes'])
                
                # 創建專案資料夾：Xnodes_YYYYMMDD_HHMMSS
                project_folder_name = f"{node_count}nodes_{timestamp}"
                project_folder_path = os.path.join(OUTPUT_DIR, project_folder_name)
                
                # 創建專案資料夾結構
                project_image_folder = create_project_structure(project_folder_path)
                
                # 生成檔案名稱
                filename = f"storyboard_{timestamp}.json"
                file_path = os.path.join(project_folder_path, filename)
                
                # 保存故事板檔案到專案資料夾
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"專案已創建: {project_folder_path}")
                logger.info(f"故事板檔案已保存: {file_path}")
                
                # 返回成功響應，包含專案和檔案信息
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'success',
                    'message': 'Project created and storyboard saved successfully',
                    'project_folder': project_folder_name,
                    'project_path': project_folder_path,
                    'file_path': file_path,
                    'filename': filename
                })
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"處理請求時發生錯誤: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                self.wfile.write(response.encode('utf-8'))
        
        # 處理從故事板創建視頻任務的請求 - 新邏輯：接收完整數據並立即創建專案
        elif self.path == '/process_storyboard':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # 檢查是否有 storyboard_data（新格式）、project_folder（專案資料夾）或 storyboard_file（舊格式）
                storyboard_data = data.get('storyboard_data')
                project_folder = data.get('project_folder')
                storyboard_file = data.get('storyboard_file')
                
                if storyboard_data:
                    # 新邏輯：接收完整的 storyboard 數據
                    logger.info("使用新邏輯：接收完整的 storyboard 數據")
                elif project_folder:
                    # 專案邏輯：從專案資料夾讀取數據
                    logger.info(f"使用專案邏輯：從專案資料夾讀取數據 {project_folder}")
                    # 構建專案資料夾路徑
                    project_folder_path = os.path.join(OUTPUT_DIR, project_folder)
                    if not os.path.exists(project_folder_path):
                        raise ValueError(f"Project folder not found: {project_folder_path}")
                    
                    # 在專案資料夾中尋找 JSON 檔案
                    json_files = [f for f in os.listdir(project_folder_path) if f.endswith('.json')]
                    if not json_files:
                        raise ValueError(f"No JSON file found in project folder: {project_folder_path}")
                    
                    # 讀取第一個 JSON 檔案（通常只有一個）
                    storyboard_file_path = os.path.join(project_folder_path, json_files[0])
                    with open(storyboard_file_path, 'r', encoding='utf-8') as f:
                        storyboard_data = json.load(f)
                    
                    logger.info(f"從專案資料夾讀取檔案: {storyboard_file_path}")
                elif storyboard_file:
                    # 舊邏輯：從檔案路徑讀取數據
                    logger.info(f"使用舊邏輯：從檔案讀取數據 {storyboard_file}")
                    # 構建完整的檔案路徑
                    storyboard_file_path = os.path.join(OUTPUT_DIR, storyboard_file)
                    if not os.path.exists(storyboard_file_path):
                        raise ValueError(f"Storyboard file not found: {storyboard_file_path}")
                    
                    # 讀取檔案內容
                    with open(storyboard_file_path, 'r', encoding='utf-8') as f:
                        storyboard_data = json.load(f)
                else:
                    raise ValueError("No storyboard data, project folder, or file provided")
                
                # 確定專案資料夾路徑
                if project_folder:
                    # 如果已經有專案資料夾，直接使用
                    project_folder_path = os.path.join(OUTPUT_DIR, project_folder)
                    project_image_folder = os.path.join(project_folder_path, "images")
                    # 從專案資料夾名稱提取時間戳
                    if '_' in project_folder:
                        project_timestamp = project_folder.split('_', 1)[1]
                    else:
                        project_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                else:
                    # 如果沒有專案資料夾，創建新的
                    project_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    node_count = len(storyboard_data.get('nodes', []))
                    project_folder = f"{node_count}nodes_{project_timestamp}"
                    project_folder_path = os.path.join(OUTPUT_DIR, project_folder)
                    project_image_folder = create_project_structure(project_folder_path)
                
                # 生成 JSON 文件名（如果專案資料夾中沒有 JSON 檔案）
                json_filename = f"storyboard_{project_timestamp}.json"
                project_json_path = os.path.join(project_folder_path, json_filename)
                
                # 處理並保存圖片到專案資料夾
                for node in storyboard_data.get('nodes', []):
                    if 'imageData' in node and 'imageType' in node:
                        # 生成圖片文件名
                        image_filename = f"node_{node['index']}_{project_timestamp}.png"
                        image_path = os.path.join(project_image_folder, image_filename)
                        
                        # 保存圖片
                        try:
                            with open(image_path, 'wb') as img_file:
                                img_file.write(base64.b64decode(node['imageData']))
                            
                            # 更新節點數據
                            node['imagePath'] = image_filename
                            del node['imageData']
                            del node['imageType']
                            logger.info(f"保存圖片到專案: {image_filename}")
                        except Exception as e:
                            logger.error(f"保存圖片失敗: {e}")
                
                # 保存 JSON 到專案資料夾
                with open(project_json_path, 'w', encoding='utf-8') as f:
                    json.dump(storyboard_data, f, ensure_ascii=False, indent=2)
                logger.info(f"保存 JSON 到專案: {project_json_path}")
                
                # 現在使用專案中的 JSON 創建任務
                result = create_tasks_from_storyboard(project_json_path)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                if result.get("success", False):
                    response = json.dumps({
                        'status': 'success',
                        'message': f'Created {len(result["task_ids"])} video tasks',
                        'task_ids': result["task_ids"],
                        'project_folder': project_folder
                    })
                else:
                    response = json.dumps({
                        'status': 'error',
                        'message': result.get("message", "Unknown error occurred")
                    })
                    
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"處理故事板任務創建請求時發生錯誤: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                self.wfile.write(response.encode('utf-8'))
        
        # 重新生成影片（在現有專案資料夾內）
        elif self.path == '/regenerate_video':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                project_folder = data.get('project_folder')
                nodes = data.get('nodes', [])
                transitions = data.get('transitions', [])
                
                logger.info(f"重生請求: project_folder={project_folder}")
                logger.info(f"重生請求: nodes數量={len(nodes)}, transitions數量={len(transitions)}")
                
                if not project_folder:
                    raise ValueError("No project folder provided")
                
                if not os.path.exists(project_folder):
                    raise ValueError(f"Project folder does not exist: {project_folder}")
                
                logger.info(f"專案資料夾存在: {project_folder}")
                
                # 直接從提供的節點和轉場數據創建任務
                result = create_tasks_from_nodes_and_transitions_direct(nodes, transitions, project_folder)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                if result.get("success", False):
                    response = json.dumps({
                        'status': 'success',
                        'message': f'Created {len(result["task_ids"])} regeneration tasks',
                        'task_ids': result["task_ids"]
                    })
                else:
                    response = json.dumps({
                        'status': 'error',
                        'message': result.get("message", "Unknown error occurred")
                    })
                    
                self.wfile.write(response.encode('utf-8'))
                
            except Exception as e:
                logger.error(f"重新生成影片請求時發生錯誤: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({
                    'status': 'error',
                    'message': str(e)
                })
                self.wfile.write(response.encode('utf-8'))
        
        # 新增：處理生成節點圖片的請求（僅打印訊息）
        elif self.path == '/generate_node_image':
            try:
                content_length = int(self.headers.get('Content-Length', '0'))
                post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
                payload = json.loads(post_data.decode('utf-8') or '{}')
                node_index = payload.get('node_index')
                prev_node_index = payload.get('prev_node_index')
                has_prev_image = payload.get('has_prev_image')
                transition_text = payload.get('transition_text', '')
                duration_seconds = float(payload.get('duration_seconds', 1.0))
                # Debug: 印出時間差到 console
                logger.info(f"[Generate] duration_seconds={duration_seconds}")

                # 驗證：需要有前一張圖片
                if not has_prev_image:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': '前一節點沒有圖片，無法生成。'})
                    self.wfile.write(resp.encode('utf-8'))
                    return

                # 解析前一張圖片：使用 base64 數據
                prev_image_file = None
                try:
                    if payload.get('prev_image_data') and payload.get('prev_image_type'):
                        # base64 儲存成暫存檔
                        import tempfile
                        fname = f"prev_tmp_{int(time.time())}.png"
                        candidate = os.path.join(tempfile.gettempdir(), fname)
                        with open(candidate, 'wb') as f:
                            f.write(base64.b64decode(payload['prev_image_data']))
                        prev_image_file = candidate
                except Exception as e:
                    logger.error(f"解析前一張圖片失敗: {e}")

                if not prev_image_file or not os.path.exists(prev_image_file):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': '找不到或無法保存前一張圖片。'})
                    self.wfile.write(resp.encode('utf-8'))
                    return

                # 建立預覽任務 ID 與初始狀態，先回應前端以顯示進度條
                job_id = f"preview_{uuid.uuid4().hex}"
                task_status[job_id] = {
                    'status': 'processing',
                    'progress': 1,
                    'message': '準備中（等待分類）...',
                    'created_at': datetime.now().isoformat()
                }
                preview_cancel_flags[job_id] = False

                def progress_cb(pct, msg):
                    task_status[job_id]['progress'] = int(pct)
                    task_status[job_id]['message'] = str(msg)

                def cancel_cb():
                    return preview_cancel_flags.get(job_id, False)

                # 先回應，避免前端等待 Qwen 推理時無法顯示進度
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = json.dumps({'status': 'queued', 'job_id': job_id, 'message': '已開始處理（分類→生成）'})
                self.wfile.write(resp.encode('utf-8'))

                # 背景執行：依序進行 Qwen 分類與單幀生成
                def run_job():
                    try:
                        # 1) Qwen 分類階段：載入 → 推理（延後卸載，若 label==2 還需後續框選）
                        task_status[job_id]['message'] = '分類中（Qwen）...'
                        task_status[job_id]['progress'] = 5
                        qwen_label = None
                        try:
                            try:
                                from Qwen_inference import get_qwen_model_and_processor, classify_image_edit_task, extract_remove_bounding_boxes, unload_qwen
                            except Exception:
                                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                                from Qwen_inference import get_qwen_model_and_processor, classify_image_edit_task, extract_remove_bounding_boxes, unload_qwen

                            get_qwen_model_and_processor()
                            qwen_label = classify_image_edit_task(prev_image_file, transition_text)
                            task_status[job_id]['qwen_label'] = int(qwen_label)
                            logger.info(f"QwenClassification label: {qwen_label} (image={prev_image_file}, prompt='{transition_text}')")
                        except Exception as e:
                            logger.error(f"呼叫 QwenClassification 失敗: {e}")

                        # 依 Qwen 分類結果選擇路線
                        try:
                            label_int = int(qwen_label) if qwen_label is not None else 0
                        except Exception:
                            label_int = 0

                        task_status[job_id]['message'] = '開始生成圖片...'
                        task_status[job_id]['progress'] = 10

                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_image_path = tmp_file.name

                        prompt_text = transition_text or ''

                        # 分類後若非 ObjectClear (label!=2)，此時即可卸載 Qwen；
                        if label_int != 2 and 'unload_qwen' in locals():
                            try:
                                unload_qwen()
                            except Exception as _ue:
                                logger.warning(f"Qwen 卸載失敗: {_ue}")
                        # 根據 label 選擇實作；FLUX/ObjectClear 先輸出 placeholder 圖片，保持後處理一致
                        if label_int == 1:
                            logger.info("選擇 FLUX 生成")
                            try:
                                from reference_flux_kontext import generate_flux_frame
                            except Exception as _ie:
                                logger.error(f"載入 FLUX 產生器失敗: {_ie}")
                                task_status[job_id]['status'] = 'error'
                                task_status[job_id]['message'] = 'FLUX 模組未安裝或不可用'
                                return
                            try:
                                generate_flux_frame(
                                    image=prev_image_file,
                                    prompt=prompt_text,
                                    output=tmp_image_path,
                                    guidance_scale=2.5,
                                    precision="bf16",
                                    local_files_only=False,
                                )
                            except Exception as _e:
                                raise RuntimeError(f"FLUX 生成失敗: {_e}")
                        elif label_int == 2:
                            logger.info("選擇 ObjectClear：先解析移除框選，並在當前圖片上疊加框選圖")
                            boxes_result = {"boxes": []}
                            try:
                                # 呼叫 Qwen 的框選抽取
                                boxes_result = extract_remove_bounding_boxes(prev_image_file, prompt_text) or {"boxes": []}
                                task_status[job_id]['remove_boxes'] = boxes_result
                            except Exception as _e:
                                logger.warning(f"ObjectClear 框選抽取失敗: {_e}")
                            finally:
                                # 完成框選後再卸載 Qwen
                                if 'unload_qwen' in locals():
                                    try:
                                        unload_qwen()
                                    except Exception as _ue:
                                        logger.warning(f"Qwen 卸載失敗: {_ue}")

                            # 產生框選可視化圖，輸出為預覽
                            try:
                                from PIL import Image, ImageDraw, ImageFont
                                base_img = Image.open(prev_image_file).convert('RGBA')
                                overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
                                draw = ImageDraw.Draw(overlay)
                                W, H = base_img.size
                                boxes = boxes_result.get('boxes', []) if isinstance(boxes_result, dict) else []
                                # 使用預設字型
                                try:
                                    font = ImageFont.load_default()
                                except Exception:
                                    font = None
                                for item in boxes:
                                    box = item.get('box_xyxy') if isinstance(item, dict) else None
                                    label_text = str(item.get('label')) if isinstance(item, dict) and item.get('label') is not None else 'object'
                                    if isinstance(box, (list, tuple)) and len(box) == 4:
                                        x0, y0, x1, y1 = [int(v) for v in box]
                                        # 邊界裁切
                                        x0 = max(0, min(x0, W))
                                        x1 = max(0, min(x1, W))
                                        y0 = max(0, min(y0, H))
                                        y1 = max(0, min(y1, H))
                                        if x1 <= x0 or y1 <= y0:
                                            continue
                                        # 只畫邊框（不填色；避免重疊時遮蔽其他框）
                                        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

                                        # 繪製標籤背景與文字
                                        if font is not None and label_text:
                                            pad = 2
                                            # 計算文字框大小
                                            try:
                                                tb = draw.textbbox((0, 0), label_text, font=font)
                                                text_w = tb[2] - tb[0]
                                                text_h = tb[3] - tb[1]
                                            except Exception:
                                                text_w, text_h = draw.textsize(label_text, font=font)

                                            bg_w = text_w + 2 * pad
                                            bg_h = text_h + 2 * pad
                                            # 盡量畫在框上方，放不下就畫在框內側
                                            bg_x0 = x0
                                            bg_y0 = y0 - bg_h if y0 - bg_h >= 0 else y0
                                            if bg_x0 + bg_w > W:
                                                bg_x0 = max(0, W - bg_w)
                                            bg_y1 = min(H, bg_y0 + bg_h)
                                            # 背景（半透明黑）
                                            draw.rectangle([bg_x0, bg_y0, bg_x0 + bg_w, bg_y1], fill=(0, 0, 0, 160))
                                            # 文字（白）
                                            draw.text((bg_x0 + pad, bg_y0 + pad), label_text, font=font, fill=(255, 255, 255, 255))

                                composed = Image.alpha_composite(base_img, overlay).convert('RGB')
                                composed.save(tmp_image_path)
                            except Exception as _e:
                                raise RuntimeError(f"建立 ObjectClear 框選圖失敗: {_e}")
                        else:
                            # FramePack 
                            try:
                                from generate_preview import generate_one_frame
                            except Exception as e:
                                logger.error(f"載入 generate_preview 失敗: {e}")
                                task_status[job_id]['status'] = 'error'
                                task_status[job_id]['message'] = '後端未安裝生成模組'
                                return
                            generate_one_frame(prev_image_file, prompt_text, tmp_image_path, progress_cb=progress_cb, cancel_cb=cancel_cb, duration_seconds=duration_seconds)

                        with open(tmp_image_path, 'rb') as img_file:
                            import base64
                            image_data = base64.b64encode(img_file.read()).decode('utf-8')
                        try:
                            os.unlink(tmp_image_path)
                        except:
                            pass

                        if preview_cancel_flags.get(job_id, False):
                            task_status[job_id]['status'] = 'cancelled'
                            task_status[job_id]['message'] = '已取消生成'
                            return

                        task_status[job_id]['status'] = 'completed'
                        task_status[job_id]['progress'] = 100
                        task_status[job_id]['message'] = '生成完成。'
                        task_status[job_id]['image_data'] = image_data
                        task_status[job_id]['image_type'] = 'image/png'
                    except Exception as e:
                        if preview_cancel_flags.get(job_id, False):
                            task_status[job_id]['status'] = 'cancelled'
                            task_status[job_id]['message'] = '已取消生成'
                        else:
                            task_status[job_id]['status'] = 'error'
                            task_status[job_id]['message'] = f'生成失敗: {str(e)}'

                threading.Thread(target=run_job, daemon=True).start()
            except Exception as e:
                logger.error(f"處理生成圖片請求時發生錯誤: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = json.dumps({'status': 'error', 'message': str(e)})
                self.wfile.write(resp.encode('utf-8'))
        
        else:
            # 其他 POST 請求返回 404
            self.send_response(404)
            self.end_headers()
        

# 自動開啟瀏覽器
def open_browser(PORT):
    """在啟動伺服器後自動開啟瀏覽器"""
    time.sleep(1)  # 等待伺服器啟動
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    # 設置伺服器端口
    PORT = 7860
    
    # 啟動任務處理線程
    processor_thread = threading.Thread(target=task_processor, daemon=True)
    processor_thread.start()
    
    # 創建伺服器
    with socketserver.TCPServer(("", PORT), StoryboardHandler) as httpd:
        logger.info("啟動故事板系統...")
        logger.info(f"輸出目錄: {os.path.abspath(OUTPUT_DIR)}")
        logger.info(f"伺服器啟動在 http://localhost:{PORT}")
        
        # 在新執行緒中開啟瀏覽器，以免阻塞伺服器啟動
        threading.Thread(target=open_browser, args=(PORT,)).start()

        # 已移除啟動時 Qwen 暖機
        
        # 啟動伺服器
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("伺服器關閉")
            stop_processing = True  # 停止任務處理線程
            processor_thread.join(timeout=5)  # 等待任務處理線程結束
            httpd.server_close()