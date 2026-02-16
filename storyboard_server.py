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
import tempfile
import shutil
import numpy as np
from PIL import Image
import torch
import traceback
import torchvision.io as tvio

# FramePack 相關
from framepack_start_end import process_video as framepack_process_video, process_storyboard_continuous, unload_framepack

# Qwen 相關（分類、框選、縮圖比例分析）
from Qwen_inference import (
    get_qwen_model_and_processor,
    classify_image_edit_task,
    extract_remove_bounding_boxes,
    analyze_zoom_out_ratio,
    unload_qwen
)

# Qwen Image Edit 2509
from Qwen_image_edit2509 import (
    generate_qwen_image_edit_2509,
    unload_qwen_image_edit_2509
)

# DIS-SAM（物件分割）
from dis_sam_inference import (
    generate_masks_with_dis_sam,
    unload_dis_sam_model
)

# ObjectClear（物件移除）
from inference_objectclear import infer_on_two_images

# Diffusers Image Outpaint（擴圖）
from diffusers_image_outpaint_inference import (
    outpaint_center_shrink,
    unload_outpaint
)

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

def create_tasks_from_nodes_and_transitions_direct(nodes, transitions, project_folder, target_pairs=None):
    """
    直接從節點和轉場數據創建視頻處理任務（用於重新生成）
    
    Args:
        nodes: 節點數據列表
        transitions: 轉場數據列表
        project_folder: 現有的專案資料夾路徑
        target_pairs: 指定要處理的節點對列表 [(from_idx, to_idx), ...]，若為 None 則處理全部
    
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
        
        if target_pairs is not None:
            target_pair_set = {tuple(pair) for pair in target_pairs}
            relevant_indices = {idx for pair in target_pair_set for idx in pair}
        else:
            target_pair_set = None
            relevant_indices = set(range(len(nodes)))
        
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
            if i not in relevant_indices:
                continue
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
        
        # 為指定的節點對創建任務（未指定則處理全部）
        for i in range(len(nodes) - 1):
            pair = (i, i + 1)
            if target_pair_set is not None and pair not in target_pair_set:
                continue
            
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


"""
def process_video_task(task_id, start_image_path, end_image_path=None, params=None, project_folder=None):
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
        
        # 倒序生成：本段 i->i+1 使用「時間上下一段」(i+1)->(i+2) 的影片，取其最前 19 幀當歷史幀
        prev_video_path = None
        try:
            if project_folder:
                parts = os.path.basename(task_id).split("_")
                if len(parts) >= 4 and parts[0] == "video":
                    from_idx = int(parts[1])
                    to_idx = int(parts[2])
                    next_from = to_idx
                    next_to = to_idx + 1
                    prefix = f"video_{next_from}_{next_to}_"
                    latest_file = None
                    latest_mtime = None
                    for fname in os.listdir(project_folder):
                        if fname.startswith(prefix) and fname.endswith(".mp4"):
                            fpath = os.path.join(project_folder, fname)
                            mtime = os.path.getmtime(fpath)
                            if latest_mtime is None or mtime > latest_mtime:
                                latest_mtime = mtime
                                latest_file = fpath
                    if latest_file:
                        prev_video_path = latest_file
                        logger.info(f"找到時間上下一段影片作為歷史幀來源: {prev_video_path}")
        except Exception as e:
            logger.warning(f"尋找時間上下一段影片時發生錯誤: {e}")
        
        # 使用 framepack（已在啟動時匯入函式；模型載入應由外部模組在呼叫時處理）
        if framepack_process_video:
            # 使用自定義進度回調來更新任務狀態
            def progress_callback(percentage, message):
                task_status[task_id]["progress"] = percentage
                task_status[task_id]["message"] = message
                logger.info(f"任務 {task_id} 進度: {percentage}% - {message}")
            
            # 調用 framepack_start_end 處理函數
            logger.info(f"使用 framepack_start_end 處理任務 {task_id}")
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
                output=output_filename,
                prev_video_path=prev_video_path
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
"""

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
def start_continuous_generation(storyboard_file):
    """
    啟動連續生成流程（在單獨線程中運行）
    
    Args:
        storyboard_file: JSON文件路徑
    
    Returns:
        任務ID和專案資料夾
    """
    try:
        with open(storyboard_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        transitions = data.get("transitions", [])
        
        project_folder = os.path.dirname(storyboard_file)
        project_image_folder = os.path.join(project_folder, "images")
        
        if len(nodes) < 2:
            return {"success": False, "message": "Storyboard needs at least 2 nodes to generate videos."}
        
        # 構建所有片段的信息
        num_segments = len(nodes) - 1
        segments = []
        
        for i in range(num_segments):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            start_image_path = start_node.get("imagePath")
            if start_image_path:
                start_image_path = os.path.join(project_image_folder, os.path.basename(start_image_path))
                if not os.path.exists(start_image_path):
                    logger.error(f"無法找到節點 {i} 的圖片: {start_image_path}")
                    continue
            
            end_image_path = end_node.get("imagePath")
            if end_image_path:
                end_image_path = os.path.join(project_image_folder, os.path.basename(end_image_path))
                if not os.path.exists(end_image_path):
                    logger.warning(f"無法找到節點 {i+1} 的圖片: {end_image_path}")
                    end_image_path = None
            
            transition_text = ""
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    transition_text = transition.get("text", "")
                    break
            
            time_range = None
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    time_range = transition.get("time_range")
                    break
            
            second_length = 5
            if time_range and len(time_range) >= 2:
                second_length = max(1, time_range[1] - time_range[0])
            
            node_seed = end_node.get("seed")
            if node_seed is None:
                node_seed = default_params["seed"]
            
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = os.path.join(project_folder, f"video_{i}_{i+1}_{timestamp}.mp4")
            
            segments.append({
                "start_image_path": start_image_path,
                "end_image_path": end_image_path,
                "prompt": f"Character movement: {transition_text}" if transition_text else default_params["prompt"],
                "seed": node_seed,
                "total_second_length": second_length,
                "output_path": output_path
            })
        
        # 創建一個特殊的任務ID來追踪整體進度
        task_id = f"continuous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task_status[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "開始連續生成...",
            "created_at": datetime.now().isoformat(),
            "total_segments": num_segments,
            "current_segment": -1
        }
        
        # 啟動線程執行連續生成
        def run_continuous_generation():
            try:
                def progress_callback(segment_idx, percentage, message):
                    task_status[task_id].update({
                        "status": "processing",
                        "progress": percentage,
                        "message": message,
                        "current_segment": segment_idx
                    })
                
                final_path = process_storyboard_continuous(segments, project_folder, progress_callback)
                
                if final_path:
                    task_status[task_id].update({
                        "status": "completed",
                        "progress": 100,
                        "message": "全部完成",
                        "output_file": os.path.basename(final_path)
                    })
                    logger.info(f"連續生成完成: {final_path}")
                else:
                    task_status[task_id].update({
                        "status": "error",
                        "message": "生成失敗"
                    })
                    logger.error("連續生成失敗")
                    
            except Exception as e:
                task_status[task_id].update({
                    "status": "error",
                    "message": str(e)
                })
                logger.error(f"連續生成過程中發生錯誤: {e}")
                traceback.print_exc()
        
        thread = threading.Thread(target=run_continuous_generation, daemon=True)
        thread.start()
        
        return {
            "success": True,
            "task_id": task_id,
            "project_folder": os.path.basename(project_folder),
            "num_segments": num_segments
        }
        
    except Exception as e:
        logger.error(f"啟動連續生成時出錯: {e}")
        traceback.print_exc()
        return {"success": False, "message": f"Error starting continuous generation: {str(e)}"}


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
        # 為每一對連續節點創建任務（倒序入隊：最後一段先做，取時間上下一段最前 19 幀當歷史）
        num_segments = len(nodes) - 1
        tasks_to_queue = []
        task_ids_ordered = [None] * num_segments
        
        for i in range(num_segments):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            
            start_image_path = start_node.get("imagePath")
            if start_image_path:
                start_image_path = os.path.join(project_image_folder, os.path.basename(start_image_path))
                if not os.path.exists(start_image_path):
                    logger.error(f"無法找到節點 {i} 的圖片: {start_image_path}")
                    continue
            
            end_image_path = end_node.get("imagePath")
            if end_image_path:
                end_image_path = os.path.join(project_image_folder, os.path.basename(end_image_path))
                if not os.path.exists(end_image_path):
                    logger.warning(f"無法找到節點 {i+1} 的圖片: {end_image_path}")
                    end_image_path = None
            
            transition_text = ""
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    transition_text = transition.get("text", "")
                    break
            
            timestamp = datetime.now().strftime("%H%M%S")
            task_id = f"video_{i}_{i+1}_{timestamp}"
            
            time_range = None
            for transition in transitions:
                if transition.get("from_node") == i and transition.get("to_node") == i + 1:
                    time_range = transition.get("time_range")
                    break
            second_length = 5
            if time_range and len(time_range) >= 2:
                second_length = max(1, time_range[1] - time_range[0])
            
            node_seed = end_node.get("seed")
            if node_seed is None:
                node_seed = default_params["seed"]
            
            task = {
                "id": task_id,
                "start_image": start_image_path,
                "end_image": end_image_path,
                "project_folder": project_folder,
                "params": {
                    "prompt": f"Character movement: {transition_text}" if transition_text else default_params["prompt"],
                    "total_second_length": second_length,
                    "seed": node_seed
                }
            }
            task_status[task_id] = {
                "status": "queued",
                "progress": 0,
                "message": "In queue",
                "created_at": datetime.now().isoformat()
            }
            tasks_to_queue.append((i, task_id, task))
            task_ids_ordered[i] = task_id
        
        for i in reversed(range(num_segments)):
            _, task_id, task = tasks_to_queue[i]
            task_queue.put(task)
            logger.info(f"創建任務 {task_id}，從節點 {i} 到節點 {i+1}（倒序入隊）")
        
        task_ids = [tid for tid in task_ids_ordered if tid is not None]
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
                
                # 添加專案資料夾與段落影片資訊
                project_folder_path = os.path.join(OUTPUT_DIR, project_folder) if project_folder else None
                data['project_folder'] = project_folder_path
                data['project_folder_name'] = project_folder

                segment_videos = []
                if project_folder_path and os.path.exists(project_folder_path):
                    node_count = len(data.get('nodes', []))
                    for i in range(max(0, node_count - 1)):
                        prefix = f"video_{i}_{i+1}_"
                        latest_file = None
                        latest_mtime = None

                        for fname in os.listdir(project_folder_path):
                            if fname.startswith(prefix) and fname.endswith('.mp4'):
                                fpath = os.path.join(project_folder_path, fname)
                                mtime = os.path.getmtime(fpath)
                                if latest_mtime is None or mtime > latest_mtime:
                                    latest_file = fname
                                    latest_mtime = mtime

                        if latest_file:
                            task_id = latest_file[:-4]
                            output_file = os.path.join(project_folder_path, latest_file)
                            segment_videos.append({
                                "from": i,
                                "to": i + 1,
                                "file": latest_file,
                                "task_id": task_id
                            })

                            existing_status = task_status.get(task_id, {})
                            if existing_status.get("status") not in {"queued", "processing"}:
                                task_status[task_id] = {
                                    "status": "completed",
                                    "progress": 100,
                                    "message": "已完成",
                                    "output_file": output_file,
                                }

                data['segment_videos'] = segment_videos
                # 偵測 final.mp4 是否存在，供前端在載入時顯示於最下方
                final_mp4_path = os.path.join(project_folder_path, "final.mp4")
                data['has_final_mp4'] = os.path.isfile(final_mp4_path)
                
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
        
        # 視頻文件提供（支援專案資料夾結構；路徑可為 /video/檔名 或 /video/專案資料夾/檔名）
        elif path.startswith('/video/'):
            rest = path.replace('/video/', '').lstrip('/')
            parts = rest.split('/')
            video_path = None
            if len(parts) == 2:
                project_folder_name, video_name = parts
                candidate = os.path.join(OUTPUT_DIR, project_folder_name, video_name)
                if os.path.exists(candidate):
                    video_path = candidate
            if video_path is None:
                video_name = rest if len(parts) <= 1 else parts[-1]
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
                
                # 使用新的連續生成流程
                result = start_continuous_generation(project_json_path)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                if result.get("success", False):
                    response = json.dumps({
                        'status': 'success',
                        'message': f'Started continuous generation for {result["num_segments"]} segments',
                        'task_id': result["task_id"],
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
                project_folder_value = data.get('project_folder')
                from_node_value = data.get('from_node')
                to_node_value = data.get('to_node')
                new_seed = data.get('new_seed')
                
                logger.info(f"重生請求: project_folder={project_folder_value}, from_node={from_node_value}, to_node={to_node_value}, new_seed={new_seed}")
                
                if project_folder_value is None:
                    raise ValueError("No project folder provided")
                if from_node_value is None or to_node_value is None:
                    raise ValueError("from_node and to_node are required")
                
                from_node = int(from_node_value)
                to_node = int(to_node_value)
                
                # 解析專案資料夾路徑（允許傳入相對或絕對路徑）
                if os.path.isabs(project_folder_value):
                    project_folder_path = project_folder_value
                    project_folder_name = os.path.relpath(project_folder_value, OUTPUT_DIR)
                else:
                    project_folder_name = project_folder_value
                    project_folder_path = os.path.join(OUTPUT_DIR, project_folder_value)
                
                if not os.path.exists(project_folder_path):
                    raise ValueError(f"Project folder does not exist: {project_folder_path}")
                
                # 找到專案中的 storyboard JSON 檔案（取最新修改時間）
                json_files = [
                    os.path.join(project_folder_path, f)
                    for f in os.listdir(project_folder_path)
                    if f.endswith('.json')
                ]
                if not json_files:
                    raise ValueError(f"No storyboard JSON file found in project folder: {project_folder_path}")
                storyboard_file = max(json_files, key=os.path.getmtime)
                
                with open(storyboard_file, 'r', encoding='utf-8') as f:
                    storyboard_data = json.load(f)
                
                nodes = storyboard_data.get('nodes', [])
                transitions = storyboard_data.get('transitions', [])
                
                # 更新節點 seed（若有提供）
                if new_seed is not None and 0 <= to_node < len(nodes):
                    nodes[to_node]['seed'] = new_seed
                    storyboard_data['nodes'] = nodes
                    with open(storyboard_file, 'w', encoding='utf-8') as f:
                        json.dump(storyboard_data, f, ensure_ascii=False, indent=2)
                        logger.info(f"已更新節點 {to_node} 的 seed 並寫回 {storyboard_file}")
                
                # 僅為指定段落創建任務
                target_pair = [(from_node, to_node)]
                result = create_tasks_from_nodes_and_transitions_direct(
                    nodes,
                    transitions,
                    project_folder_path,
                    target_pairs=target_pair,
                )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                if result.get("success", False):
                    response = json.dumps({
                        'status': 'success',
                        'message': f'Created {len(result["task_ids"])} regeneration tasks',
                        'task_ids': result["task_ids"],
                        'project_folder': project_folder_name
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
        
        # 串接所有片段影片為一支最終影片
        elif self.path == '/concat_videos':
            content_length = int(self.headers.get('Content-Length', '0'))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            
            try:
                data = json.loads(post_data.decode('utf-8') or '{}')
                project_folder_value = data.get('project_folder')
                
                if not project_folder_value:
                    raise ValueError("Missing 'project_folder' in request body")
                
                # 解析專案資料夾路徑（允許相對或絕對）
                if os.path.isabs(project_folder_value):
                    project_folder_path = project_folder_value
                else:
                    project_folder_path = os.path.join(OUTPUT_DIR, project_folder_value)
                
                if not os.path.isdir(project_folder_path):
                    raise ValueError(f"Project folder does not exist: {project_folder_path}")
                
                # 找到專案中的 storyboard JSON 檔案（取最新修改時間）
                json_files = [
                    os.path.join(project_folder_path, f)
                    for f in os.listdir(project_folder_path)
                    if f.endswith('.json')
                ]
                if not json_files:
                    raise ValueError(f"No storyboard JSON file found in project folder: {project_folder_path}")
                storyboard_file = max(json_files, key=os.path.getmtime)
                
                with open(storyboard_file, 'r', encoding='utf-8') as f:
                    storyboard_data = json.load(f)
                
                nodes = storyboard_data.get('nodes', [])
                if len(nodes) < 2:
                    raise ValueError("Storyboard needs at least 2 nodes to concat videos.")
                
                num_segments = len(nodes) - 1
                
                # 依序為每一段尋找最新的 segment 視頻
                segment_paths = []
                for i in range(num_segments):
                    prefix = f"video_{i}_{i+1}_"
                    latest_file = None
                    latest_mtime = None
                    for fname in os.listdir(project_folder_path):
                        if fname.startswith(prefix) and fname.endswith('.mp4'):
                            fpath = os.path.join(project_folder_path, fname)
                            mtime = os.path.getmtime(fpath)
                            if latest_mtime is None or mtime > latest_mtime:
                                latest_mtime = mtime
                                latest_file = fpath
                    if latest_file is None:
                        raise ValueError(f"Missing segment video for transition {i}->{i+1} in project '{project_folder_path}'")
                    segment_paths.append(latest_file)
                
                if len(segment_paths) != num_segments:
                    raise ValueError("Segment video count mismatch; cannot concat.")
                
                # 將單一 mp4 轉為 BxCxTxHxW 並對應到 [-1,1]
                def read_video_to_bcthw(path):
                    video, _, _ = tvio.read_video(path, pts_unit="sec")
                    if video.numel() == 0:
                        raise ValueError(f"Video file is empty or unreadable: {path}")
                    video = video.float() / 127.5 - 1.0        # T,H,W,C
                    video = video.permute(3, 0, 1, 2)          # C,T,H,W
                    video = video.unsqueeze(0)                 # 1,C,T,H,W
                    return video
                
                from diffusers_helper.utils import save_bcthw_as_mp4
                
                # 直接銜接：依序讀入各段並在時間維度上 cat，不做 overlap
                parts = [read_video_to_bcthw(p) for p in segment_paths]
                history = torch.cat(parts, dim=2)
                
                # 最終輸出檔案名稱：放在當前專案資料夾下，統一命名為 final.mp4
                final_path = os.path.join(project_folder_path, "final.mp4")
                save_bcthw_as_mp4(history, final_path, fps=30, crf=16)
                
                logger.info(f"Concat videos completed for project '{project_folder_path}', output: {final_path}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = {
                    "status": "success",
                    "message": "Concatenation completed",
                    "file": os.path.basename(final_path)
                }
                self.wfile.write(json.dumps(resp).encode('utf-8'))
            
            except Exception as e:
                logger.error(f"Error while concatenating videos: {e}")
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = {
                    "status": "error",
                    "message": str(e)
                }
                self.wfile.write(json.dumps(resp).encode('utf-8'))
        
        # 新增：處理生成節點圖片的請求
        elif self.path == '/generate_node_image':
            try:
                content_length = int(self.headers.get('Content-Length', '0'))
                post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
                payload = json.loads(post_data.decode('utf-8') or '{}')
                node_index = payload.get('node_index')
                prev_node_index = payload.get('prev_node_index')
                user_prompt = payload.get('user_prompt', '')  # 用戶輸入的 prompt
                negative_prompt = payload.get('negative_prompt', '')  # 負面提示詞
                reference_image_data = payload.get('reference_image_data')  # 參考圖片 base64 數據
                reference_image_type = payload.get('reference_image_type', 'image/png')  # 參考圖片類型
                duration_seconds = float(payload.get('duration_seconds', 1.0))
                
                logger.info(f"[Generate] 節點 {node_index}, prompt: {user_prompt[:50]}..., negative_prompt: {negative_prompt[:50] if negative_prompt else 'N/A'}, duration: {duration_seconds}s")

                # 驗證：需要有參考圖片
                if not reference_image_data:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': '請提供參考圖片（前一個節點的圖片或上傳的圖片）'})
                    self.wfile.write(resp.encode('utf-8'))
                    return

                # 驗證：需要有 prompt
                if not user_prompt or not user_prompt.strip():
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': '請輸入 Prompt 描述'})
                    self.wfile.write(resp.encode('utf-8'))
                    return

                # 解析參考圖片：使用 base64 數據
                reference_image_file = None
                try:
                    # base64 儲存成暫存檔
                    import tempfile
                    fname = f"ref_tmp_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
                    candidate = os.path.join(tempfile.gettempdir(), fname)
                    with open(candidate, 'wb') as f:
                        f.write(base64.b64decode(reference_image_data))
                    reference_image_file = candidate
                    logger.info(f"[Generate] 參考圖片已保存到臨時文件: {reference_image_file}")
                except Exception as e:
                    logger.error(f"解析參考圖片失敗: {e}")
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': f'無法保存參考圖片: {str(e)}'})
                    self.wfile.write(resp.encode('utf-8'))
                    return

                if not reference_image_file or not os.path.exists(reference_image_file):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    resp = json.dumps({'status': 'error', 'message': '找不到或無法保存參考圖片。'})
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
                        # Qwen 分類階段（使用用戶輸入的 prompt）
                        task_status[job_id]['message'] = '分類中（Qwen）...'
                        task_status[job_id]['progress'] = 5
                        
                        get_qwen_model_and_processor()
                        qwen_label = classify_image_edit_task(reference_image_file, user_prompt)
                        task_status[job_id]['qwen_label'] = int(qwen_label)
                        logger.info(f"QwenClassification label: {qwen_label}, prompt: {user_prompt[:50]}...")

                        # 準備生成
                        task_status[job_id]['message'] = '開始生成圖片...'
                        task_status[job_id]['progress'] = 10

                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            tmp_image_path = tmp_file.name

                        # 根據 label 選擇實作
                        if qwen_label == 1:
                            logger.info("選擇 Qwen Image Edit")
                            # 先卸載 Qwen 分類模型，避免顯存衝突
                            try:
                                unload_qwen()
                            except Exception as e:
                                logger.warning(f"Qwen 卸載失敗: {e}")
                            
                            try:
                                logger.info(f"Qwen Image Edit - user_prompt: {user_prompt[:50]}...")
                                logger.info(f"Qwen Image Edit - negative_prompt: {negative_prompt[:50] if negative_prompt else 'N/A'}...")

                                result_path = generate_qwen_image_edit_2509(
                                    image_path=reference_image_file,
                                    prompt=user_prompt,
                                    output_path=tmp_image_path,
                                    negative_prompt=(negative_prompt or "blurry, low quality, distorted"),
                                    steps=25,
                                    seed=torch.randint(0, 2**32, (1,)).item()
                                )

                                if result_path is None:
                                    raise RuntimeError("Qwen Image Edit 生成失敗")

                                logger.info("Qwen Image Edit 生成完成")

                                # 卸載 Qwen Image Edit 模型（2509）
                                try:
                                    unload_qwen_image_edit_2509()
                                except Exception as e:
                                    logger.warning(f"Qwen Image Edit 卸載失敗: {e}")

                            except Exception as e:
                                logger.error(f"Qwen Image Edit 失敗: {e}")
                                raise RuntimeError(f"Qwen Image Edit 生成失敗: {e}")
                                
                        elif qwen_label == 2:
                            logger.info("選擇 ObjectClear")
                            try:
                                # 構建給 Qwen 的 prompt：如果有 native_prompt，則在最後加上
                                qwen_prompt = user_prompt
                                if negative_prompt:
                                    qwen_prompt = user_prompt + " do not remove the object list below: " + negative_prompt
                                    logger.info(f"ObjectClear 使用擴展 prompt: {qwen_prompt[:100]}...")
                                
                                # 呼叫 Qwen 的框選任務
                                boxes_result = extract_remove_bounding_boxes(reference_image_file, qwen_prompt) or {"boxes": []}
                                task_status[job_id]['remove_boxes'] = boxes_result
                                
                                # 完成框選後立即卸載 Qwen
                                try:
                                    unload_qwen()
                                except Exception as e:
                                    logger.warning(f"Qwen 卸載失敗: {e}")
                                
                                # 使用 DIS-SAM 產生精確 mask（兩階段：SAM + IS-Net 精煉）
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
                                    tmp_mask_path = tmp_mask.name
                                
                                # 呼叫 DIS-SAM 推理（直接傳入 boxes_result dict，DIS-SAM 會自動處理）
                                logger.info("開始 DIS-SAM 推理（SAM + IS-Net 精煉，使用用戶 prompt）...")
                                mask_path = generate_masks_with_dis_sam(
                                    image_path=reference_image_file,
                                    boxes_json=boxes_result,  # 直接傳入 dict，DIS-SAM 接受這種格式
                                    output_mask_path=tmp_mask_path,
                                    sam_model_type="vit_h",  # 使用 vit_l（更快）或 vit_h（更準確）
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                    use_refinement=True,  # 使用 IS-Net 精煉（完整 DIS-SAM）
                                    auto_download=True
                                )
                                
                                # 完成 DIS-SAM 推理後卸載模型以釋放顯存
                                try:
                                    unload_dis_sam_model()
                                    logger.info("DIS-SAM 模型已卸載")
                                except Exception as e:
                                    logger.warning(f"DIS-SAM 模型卸載失敗: {e}")
                                
                                # 使用 ObjectClear 模型完成物件移除
                                # 呼叫 ObjectClear 推理
                                logger.info("開始 ObjectClear 推理...")
                                final_output_path = infer_on_two_images(
                                    sample_image_path=reference_image_file,
                                    mask_image_path=mask_path,
                                    output_path=None,
                                    use_fp16=False,
                                    steps=20,
                                    guidance_scale=2.5,
                                    seed=42,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                )
                                
                                # 將 ObjectClear 的結果複製到最終輸出
                                shutil.copy2(final_output_path, tmp_image_path)
                                
                                # 清理檔案
                                try:
                                    os.unlink(final_output_path)
                                    os.unlink(tmp_mask_path)
                                except:
                                    pass
                                    
                                logger.info("ObjectClear 推理完成")
                                
                            except Exception as e:
                                # 任何推理失敗都直接輸出黑色圖片
                                logger.error(f"ObjectClear 推理失敗: {e}")
                                print(f"ObjectClear 推理失敗: {e}")
                                
                                # 確保在錯誤時也卸載 DIS-SAM 模型
                                try:
                                    unload_dis_sam_model()
                                    logger.info("DIS-SAM 模型已卸載（錯誤處理）")
                                except Exception as unload_error:
                                    logger.warning(f"DIS-SAM 模型卸載失敗: {unload_error}")
                                
                                # 創建黑色圖片（使用參考圖片的尺寸）
                                with Image.open(reference_image_file) as img:
                                    width, height = img.size
                                black_img = Image.new('RGB', (width, height), (0, 0, 0))
                                black_img.save(tmp_image_path)
                                logger.info("已輸出黑色圖片")
                                
                        else:
                            # Diffusers Image Outpaint
                            logger.info("選擇 Diffusers Image Outpaint")
                            try:
                                # 更新進度：正在分析縮圖比例
                                task_status[job_id]['message'] = '分析最佳縮圖比例（Qwen）...'
                                task_status[job_id]['progress'] = 12
                                
                                logger.info("開始使用 Qwen 分析最佳縮圖比例...")
                                shrink_percent = analyze_zoom_out_ratio(reference_image_file, user_prompt)
                                logger.info(f"Qwen 建議的縮圖比例: {shrink_percent}%")
                                
                                # 將分析結果記錄到任務狀態中
                                task_status[job_id]['shrink_percent'] = shrink_percent
                                
                            except Exception as e:
                                logger.error(f"Qwen 分析縮圖比例失敗: {e}")
                                # 如果分析失敗，使用基於時長的預設值
                                shrink_percent = min(duration_seconds * 10, 30.0)
                                logger.info(f"使用預設縮圖比例: {shrink_percent}% (based on duration: {duration_seconds}s)")
                            
                            # 完成分析後立即卸載 Qwen（照 label 2 的簡潔寫法）
                            try:
                                unload_qwen()
                            except Exception as e:
                                logger.warning(f"Qwen 卸載失敗: {e}")
                            
                            try:
                                logger.info(f"Diffusers Outpaint - user_prompt: {user_prompt[:50]}...")
                                logger.info(f"Diffusers Outpaint - negative_prompt: {negative_prompt[:50] if negative_prompt else 'N/A'}...")
                                logger.info(f"Diffusers Outpaint - shrink_percent: {shrink_percent}% (analyzed by Qwen)")
                                
                                # 更新進度
                                task_status[job_id]['message'] = 'Outpaint 生成中...'
                                task_status[job_id]['progress'] = 15
                                
                                # 呼叫 outpaint 推理
                                result_path = outpaint_center_shrink(
                                    image=reference_image_file,
                                    prompt="keep the original art style and expand the image, " + user_prompt,
                                    shrink_percent=shrink_percent,
                                    output_path=tmp_image_path,
                                    negative_prompt=negative_prompt if negative_prompt else None,
                                    num_inference_steps=8,  
                                    seed=torch.randint(0, 2**32, (1,)).item(),
                                    overlap_percentage=2,  # 2% 重疊
                                    return_dict=False
                                )
                                
                                if result_path is None or not os.path.exists(result_path):
                                    raise RuntimeError("Diffusers Outpaint 生成失敗")
                                
                                logger.info("Diffusers Outpaint 生成完成")
                                
                                # 卸載 Outpaint 模型
                                try:
                                    unload_outpaint()
                                    logger.info("Outpaint 模型已卸載")
                                except Exception as e:
                                    logger.warning(f"Outpaint 模型卸載失敗: {e}")
             
                            except Exception as e:
                                logger.error(f"Diffusers Outpaint 生成失敗: {e}")
                                traceback.print_exc()
                                raise RuntimeError(f"Diffusers Outpaint 生成失敗: {e}")

                        # 讀取並編碼圖片
                        with open(tmp_image_path, 'rb') as img_file:
                            image_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # 清理臨時檔案（包括參考圖片）
                        try:
                            os.unlink(tmp_image_path)
                            if reference_image_file and os.path.exists(reference_image_file):
                                os.unlink(reference_image_file)
                                logger.info(f"[Generate] 已清理參考圖片臨時文件: {reference_image_file}")
                        except Exception as cleanup_error:
                            logger.warning(f"[Generate] 清理臨時文件失敗: {cleanup_error}")

                        # 檢查取消狀態
                        if preview_cancel_flags.get(job_id, False):
                            task_status[job_id]['status'] = 'cancelled'
                            task_status[job_id]['message'] = '已取消生成'
                            return

                        # 完成任務
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
        
        # 啟動伺服器
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("伺服器關閉")
            stop_processing = True  # 停止任務處理線程
            processor_thread.join(timeout=5)  # 等待任務處理線程結束
            httpd.server_close()