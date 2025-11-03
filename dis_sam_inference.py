"""
DIS-SAM 推理函數（兩階段方法：SAM + IS-Net）
- 第一階段：使用 SAM 根據 bbox 生成粗略 mask
- 第二階段：使用 IS-Net 精煉 mask 的邊界細節
- 輸入：圖片路徑與 Qwen 輸出的 bbox JSON（XYXY 像素座標）
- 輸出：精確 mask（PNG），以及可選的每個物件獨立 mask
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import urllib.request

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import cv2
import argparse

# 導入 SAM（嘗試多種方式）
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    # 嘗試從 DIS-SAM 項目導入
    try:
        import sys
        dis_sam_path = Path(__file__).resolve().parent.parent / "DIS-SAM"
        if dis_sam_path.exists():
            sys.path.insert(0, str(dis_sam_path))
            from SAM.segment_anything import sam_model_registry, SamPredictor
            SAM_AVAILABLE = True
        else:
            SAM_AVAILABLE = False
    except ImportError:
        SAM_AVAILABLE = False

if not SAM_AVAILABLE:
    raise RuntimeError("請安裝 segment-anything: pip install segment-anything")

# 導入 IS-Net 模型
ISNET_AVAILABLE = False
try:
    # 嘗試從當前目錄直接導入 isnet.py（與 dis_sam_inference.py 同級）
    from isnet import ISNetDIS
    ISNET_AVAILABLE = True
    print("[DIS-SAM] 成功導入 IS-Net 模型")
except ImportError:
    # 嘗試添加當前目錄到 sys.path 後導入
    try:
        current_dir = Path(__file__).resolve().parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from isnet import ISNetDIS
        ISNET_AVAILABLE = True
        print("[DIS-SAM] 成功導入 IS-Net 模型（通過路徑添加）")
    except ImportError as e:
        # 最後嘗試：從舊的包結構導入（向後兼容）
        try:
            from IS_Net.models.isnet import ISNetDIS
            ISNET_AVAILABLE = True
            print("[DIS-SAM] 成功導入 IS-Net 模型（從舊的包結構）")
        except ImportError:
            ISNET_AVAILABLE = False
            print(f"[DIS-SAM] 警告：無法導入 isnet 模組，DIS-SAM 精煉階段將不可用")
            print(f"[DIS-SAM] 導入錯誤: {e}")
            print(f"[DIS-SAM] 請確保 isnet.py 與 dis_sam_inference.py 在同一個目錄")
            print(f"[DIS-SAM] 當前工作目錄: {os.getcwd()}")
            print(f"[DIS-SAM] 腳本目錄: {Path(__file__).resolve().parent}")
            print(f"[DIS-SAM] Python 路徑: {sys.path[:3]}...")

# SAM 權重配置
SAM_CHECKPOINTS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size": "2.6GB"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth",
        "size": "1.2GB"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth",
        "size": "375MB"
    }
}

# DIS-SAM (IS-Net) 權重配置
DIS_SAM_CHECKPOINT = {
    "file_id": "1O0He8iJT6YkmccgHOTFNEMPIqfPPR5Np",  # Google Drive 文件 ID
    "filename": "dis_sam.pth",
    "size": "169MB",
    "note": "也可從 Google Drive 下載：https://drive.google.com/drive/folders/1fE_DCGKU3WA-HmZnqRzaazHe44Lx9RP2?usp=sharing"
}


def _download_sam_checkpoint(model_type: str, checkpoint_dir: Path) -> Path:
    """自動下載 SAM 權重到指定目錄"""
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"不支援的模型類型: {model_type}，支援: {list(SAM_CHECKPOINTS.keys())}")
    
    config = SAM_CHECKPOINTS[model_type]
    checkpoint_path = checkpoint_dir / config["filename"]
    
    if checkpoint_path.exists():
        print(f"[DIS-SAM] SAM 權重已存在: {checkpoint_path}")
        return checkpoint_path
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DIS-SAM] 開始下載 SAM {model_type} 權重 ({config['size']})...")
    print(f"[DIS-SAM] 下載來源: {config['url']}")
    print(f"[DIS-SAM] 儲存位置: {checkpoint_path}")
    
    try:
        urllib.request.urlretrieve(config["url"], checkpoint_path)
        print(f"[DIS-SAM] SAM 權重下載完成: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"\n[DIS-SAM] SAM 權重下載失敗: {e}")
        print(f"[DIS-SAM] 請手動下載: {config['url']}")
        print(f"[DIS-SAM] 存放到: {checkpoint_path}")
        raise RuntimeError(f"下載 SAM 權重失敗: {e}")


def _download_dis_sam_checkpoint(checkpoint_dir: Path) -> Path:
    """自動下載 DIS-SAM (IS-Net) 權重到指定目錄（從 Google Drive）"""
    config = DIS_SAM_CHECKPOINT
    checkpoint_path = checkpoint_dir / config["filename"]
    file_id = config["file_id"]
    
    # 檢查文件是否存在且有效
    if checkpoint_path.exists():
        # 驗證文件是否為有效的 PyTorch 權重文件（不是 HTML）
        try:
            with open(checkpoint_path, 'rb') as f:
                first_bytes = f.read(10)
                # PyTorch 權重文件應該以 pickle 魔數開頭（通常是 0x80 0x02 或 0x80 0x03）
                # HTML 文件以 <!DOCTYPE 或 <html 開頭
                if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                    print(f"[DIS-SAM] 檢測到無效的權重文件（可能是 HTML），將重新下載...")
                    checkpoint_path.unlink()
                else:
                    print(f"[DIS-SAM] IS-Net 權重已存在: {checkpoint_path}")
                    return checkpoint_path
        except Exception:
            # 如果讀取失敗，假設文件損壞，重新下載
            print(f"[DIS-SAM] 檢測到損壞的權重文件，將重新下載...")
            checkpoint_path.unlink()
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DIS-SAM] 開始下載 DIS-SAM (IS-Net) 權重 ({config['size']})...")
    print(f"[DIS-SAM] 下載來源: Google Drive")
    print(f"[DIS-SAM] 原始連結: {config['note']}")
    print(f"[DIS-SAM] 儲存位置: {checkpoint_path}")
    
    try:
        # 優先使用 gdown（最可靠）
        try:
            import gdown
            print("[DIS-SAM] 使用 gdown 下載（推薦方式）...")
            # 構建正確的 Google Drive 下載 URL
            gdrive_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(gdrive_url, str(checkpoint_path), quiet=False)
            
            # 驗證下載的文件
            if not _validate_checkpoint_file(checkpoint_path):
                checkpoint_path.unlink()
                raise RuntimeError("下載的文件無效（不是 PyTorch 權重文件）")
            
            print(f"[DIS-SAM] IS-Net 權重下載完成: {checkpoint_path}")
            return checkpoint_path
        except ImportError:
            print("[DIS-SAM] gdown 未安裝，使用 requests 下載（可能需要處理 Google Drive 確認頁面）...")
            print("[DIS-SAM] 提示: 安裝 gdown 可獲得更可靠的下載體驗: pip install gdown")
            # 如果沒有 gdown，使用 requests 手動處理 Google Drive 下載
            try:
                import requests
                print("[DIS-SAM] 使用 requests 下載...")
                session = requests.Session()
                URL = 'https://docs.google.com/uc?export=download'
                
                # 嘗試直接使用 confirm=t 下載（處理大文件病毒掃描警告）
                params = {'id': file_id, 'export': 'download', 'confirm': 't'}
                response = session.get(URL, params=params, stream=True, allow_redirects=True)
                
                # 如果還是 HTML，嘗試獲取 token
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    # 先嘗試獲取確認 token
                    params2 = {'id': file_id}
                    response2 = session.get(URL, params=params2, stream=True)
                    token = None
                    for key, value in response2.cookies.items():
                        if key.startswith('download_warning'):
                            token = value
                            break
                    
                    if token:
                        params2['confirm'] = token
                        response = session.get(URL, params=params2, stream=True, allow_redirects=True)
                    else:
                        response = response2
                
                # 檢查響應是否是 HTML（Google Drive 病毒掃描警告頁面）
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    # 解析 HTML 獲取實際下載 URL
                    import re
                    html_content = response.text
                    
                    # 方法1: 查找 form action 和參數（處理病毒掃描警告）
                    form_action_match = re.search(r'<form[^>]*action="([^"]+)"', html_content)
                    if form_action_match:
                        form_action = form_action_match.group(1)
                        # 提取所有 hidden input 的值
                        id_match = re.search(r'<input[^>]*name="id"[^>]*value="([^"]+)"', html_content)
                        export_match = re.search(r'<input[^>]*name="export"[^>]*value="([^"]+)"', html_content)
                        confirm_match = re.search(r'<input[^>]*name="confirm"[^>]*value="([^"]+)"', html_content)
                        
                        if id_match:
                            download_params = {'id': id_match.group(1)}
                            if export_match:
                                download_params['export'] = export_match.group(1)
                            if confirm_match:
                                download_params['confirm'] = confirm_match.group(1)
                            
                            print("[DIS-SAM] 檢測到 Google Drive 病毒掃描警告頁面，解析下載參數...")
                            if form_action.startswith('http'):
                                response = session.get(form_action, params=download_params, stream=True, allow_redirects=True)
                            else:
                                response = session.get('https://drive.google.com' + form_action, params=download_params, stream=True, allow_redirects=True)
                        else:
                            # 方法2: 嘗試直接構建下載 URL（使用 confirm=t）
                            print("[DIS-SAM] 嘗試使用 confirm=t 參數下載...")
                            params = {'id': file_id, 'export': 'download', 'confirm': 't'}
                            response = session.get(URL, params=params, stream=True, allow_redirects=True)
                    else:
                        # 方法3: 查找 href 中的下載鏈接
                        download_match = re.search(r'href="(/uc\?[^"]+)"', html_content)
                        if download_match:
                            download_url = 'https://docs.google.com' + download_match.group(1)
                            response = session.get(download_url, stream=True, allow_redirects=True)
                        else:
                            # 最後嘗試：使用 confirm=t 強制下載
                            print("[DIS-SAM] 使用 confirm=t 強制下載...")
                            params = {'id': file_id, 'export': 'download', 'confirm': 't'}
                            response = session.get(URL, params=params, stream=True, allow_redirects=True)
                    
                    # 再次檢查是否還是 HTML
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        raise RuntimeError("無法從 Google Drive 獲取下載鏈接，建議安裝 gdown: pip install gdown")
                
                total_size = int(response.headers.get('content-length', 0))
                if total_size > 0:
                    print(f"[DIS-SAM] 檔案大小: {total_size / 1024 / 1024:.2f} MB")
                
                with open(checkpoint_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = downloaded / total_size * 100
                                print(f"\r[DIS-SAM] 下載進度: {percent:.1f}%", end='', flush=True)
                
                print(f"\n[DIS-SAM] 下載完成，驗證文件中...")
                
                # 驗證下載的文件
                if not _validate_checkpoint_file(checkpoint_path):
                    checkpoint_path.unlink()
                    raise RuntimeError("下載的文件無效（不是 PyTorch 權重文件）")
                
                print(f"[DIS-SAM] IS-Net 權重下載完成: {checkpoint_path}")
                return checkpoint_path
            except ImportError:
                raise RuntimeError("需要安裝 gdown 或 requests 庫來下載權重文件")
    except Exception as e:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"\n[DIS-SAM] IS-Net 權重自動下載失敗: {e}")
        print(f"[DIS-SAM] 解決方案:")
        print(f"[DIS-SAM] 方案 1 (推薦): 安裝 gdown 庫後重新運行:")
        print(f"[DIS-SAM]    pip install gdown")
        print(f"[DIS-SAM] 方案 2: 手動下載權重檔案:")
        print(f"[DIS-SAM]    1. 前往: {config['note']}")
        print(f"[DIS-SAM]    2. 找到並下載 DIS-SAM-checkpoint.pth (約 169MB)")
        print(f"[DIS-SAM]    3. 放到: {checkpoint_path}")
        print(f"[DIS-SAM] 方案 3: 設定環境變數指向已有權重:")
        print(f"[DIS-SAM]    export DIS_SAM_CHECKPOINT=/path/to/your/checkpoint.pth")
        raise RuntimeError(f"下載 IS-Net 權重失敗: {e}")


def _validate_checkpoint_file(checkpoint_path: Path) -> bool:
    """驗證檢查點文件是否為有效的 PyTorch 權重文件"""
    try:
        # 檢查文件大小（應該大於 1MB，169MB 左右）
        file_size = checkpoint_path.stat().st_size
        if file_size < 1024 * 1024:  # 小於 1MB，可能是 HTML 或錯誤文件
            print(f"[DIS-SAM] 警告：下載的文件過小 ({file_size / 1024 / 1024:.2f} MB)，可能是錯誤的")
            return False
        
        # 檢查文件開頭是否是 HTML
        with open(checkpoint_path, 'rb') as f:
            first_bytes = f.read(100)
            if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                print(f"[DIS-SAM] 錯誤：下載的文件是 HTML 而不是權重文件")
                return False
        
        # 嘗試加載文件（不完整加載，只檢查格式）
        try:
            # 只讀取前幾個字節來檢查是否是 pickle 格式
            with open(checkpoint_path, 'rb') as f:
                import pickle
                # PyTorch 文件是 pickle 格式，通常以特定字節開頭
                first_byte = f.read(1)
                if first_byte != b'\x80':  # pickle 協議標記
                    # 但也可能是較新的格式，所以檢查更多
                    f.seek(0)
                    first_bytes = f.read(10)
                    # 如果看起來像文本文件，則無效
                    try:
                        first_bytes.decode('utf-8')
                        if first_bytes.startswith(b'<') or first_bytes.startswith(b'{'):
                            return False
                    except UnicodeDecodeError:
                        pass  # 二進制文件是正常的
        except Exception:
            pass  # 如果檢查失敗，假設文件可能是有效的
        
        return True
    except Exception as e:
        print(f"[DIS-SAM] 驗證文件時出錯: {e}")
        return False


# 全局變量存儲模型
_global_sam_predictor = None
_global_sam_device = None
_global_isnet_model = None
_global_isnet_device = None


def _load_sam_model(
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    auto_download: bool = True,
):
    """載入 SAM 模型（第一階段）"""
    global _global_sam_predictor, _global_sam_device
    
    if checkpoint is None:
        checkpoint = os.getenv("SAM_CHECKPOINT")
    
    if not checkpoint or not Path(checkpoint).exists():
        if auto_download:
            checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
            checkpoint = _download_sam_checkpoint(model_type, checkpoint_dir)
        else:
            raise FileNotFoundError(
                "請提供 SAM 權重檔路徑（SAM_CHECKPOINT 環境變數或函數參數 checkpoint）"
            )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 如果已經載入了相同設備的模型，直接返回
    if _global_sam_predictor is not None and _global_sam_device == device:
        return _global_sam_predictor, device
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    _global_sam_predictor = predictor
    _global_sam_device = device
    
    return predictor, device


def _load_isnet_model(
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    auto_download: bool = True,
):
    """載入 IS-Net 模型（第二階段）"""
    global _global_isnet_model, _global_isnet_device
    
    if not ISNET_AVAILABLE:
        raise RuntimeError("IS_Net.models.isnet 不可用，無法載入 IS-Net 模型")
    
    if checkpoint is None:
        checkpoint = os.getenv("DIS_SAM_CHECKPOINT")
    
    # 如果沒有指定權重，嘗試從 checkpoints 目錄查找
    if not checkpoint or not Path(checkpoint).exists():
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        # 先檢查目錄中是否有 .pth 文件
        if checkpoint_dir.exists():
            pth_files = list(checkpoint_dir.glob("*.pth"))
            # 過濾掉 SAM 權重文件
            dis_sam_files = [f for f in pth_files if "sam_vit" not in f.name and "dis_sam" in f.name]
            if dis_sam_files:
                checkpoint = str(dis_sam_files[0])
                print(f"[DIS-SAM] 自動找到 IS-Net 權重: {checkpoint}")
        
        # 如果還沒找到，嘗試自動下載
        if not checkpoint or not Path(checkpoint).exists():
            if auto_download:
                checkpoint = _download_dis_sam_checkpoint(checkpoint_dir)
            else:
                raise FileNotFoundError(
                    "請提供 DIS-SAM (IS-Net) 權重檔路徑（DIS_SAM_CHECKPOINT 環境變數或函數參數 checkpoint）\n"
                    f"下載連結：{DIS_SAM_CHECKPOINT['note']}"
                )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 如果已經載入了相同設備的模型，直接返回
    if _global_isnet_model is not None and _global_isnet_device == device:
        return _global_isnet_model, device
    
    # 創建 IS-Net 模型（輸入 5 通道：RGB + mask + box）
    net = ISNetDIS(in_ch=5)
    # PyTorch 2.6+ 需要設置 weights_only=False 來加載舊的權重文件
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=False)
    net.to(device=device)
    net.eval()
    
    _global_isnet_model = net
    _global_isnet_device = device
    
    print(f"[DIS-SAM] IS-Net 模型已載入: {checkpoint}")
    return net, device


def unload_dis_sam_model():
    """卸載 DIS-SAM 模型以釋放顯存"""
    global _global_sam_predictor, _global_sam_device
    global _global_isnet_model, _global_isnet_device
    
    # 卸載 SAM
    if _global_sam_predictor is not None:
        if hasattr(_global_sam_predictor, 'model'):
            del _global_sam_predictor.model
        del _global_sam_predictor
        _global_sam_predictor = None
        _global_sam_device = None
    
    # 卸載 IS-Net
    if _global_isnet_model is not None:
        del _global_isnet_model
        _global_isnet_model = None
        _global_isnet_device = None
    
    # 清理顯存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("[DIS-SAM] 模型已卸載")


def _ensure_xyxy(box: List[int]) -> List[int]:
    """確保 bbox 為 xyxy 格式且為 int"""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError("box_xyxy 需為 [x0, y0, x1, y1]")
    return [int(round(float(v))) for v in box]


def _save_mask_png(mask: np.ndarray, out_path: Union[str, Path]) -> None:
    """將二值 mask 儲存為 PNG（白=前景，黑=背景）"""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    out = Image.fromarray(mask)
    out.save(str(out_path))


def get_box_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    從 mask 生成 bounding box（用於 IS-Net 輸入）
    
    Args:
        mask: numpy array mask（可以是 0/1 或 0/255）
    
    Returns:
        box: numpy array，與 mask 同尺寸，box 區域為 255，其他為 0
    """
    mask_tensor = torch.from_numpy(np.array(mask)).float()
    if mask_tensor.max() <= 1:
        mask_tensor = mask_tensor * 255.0
    
    box = torch.zeros_like(mask_tensor)
    rows, cols = torch.where(mask_tensor > 0)
    
    if len(rows) == 0:
        return box.numpy().astype(np.uint8)
    
    left = int(torch.min(cols).item())
    top = int(torch.min(rows).item())
    right = int(torch.max(cols).item())
    bottom = int(torch.max(rows).item())
    
    box[top:bottom+1, left:right+1] = 255
    return box.numpy().astype(np.uint8)


class GOSNormalize(object):
    """Normalize 類（用於 IS-Net 預處理）"""
    def __init__(self, mean=[0.5,0.5,0.5,0,0], std=[1.0,1.0,1.0,1.0,1.0]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        return normalize(image, self.mean, self.std)


def _refine_mask_with_isnet(
    net,
    image: np.ndarray,
    rough_mask: np.ndarray,
    box: np.ndarray,
    device: str,
    input_size: List[int] = [1024, 1024],
    model_digit: str = "full"
) -> np.ndarray:
    """
    使用 IS-Net 精煉 mask（第二階段）
    
    Args:
        net: IS-Net 模型
        image: RGB 圖片 (H, W, 3)，numpy array，值域 0-255
        rough_mask: SAM 生成的粗略 mask (H, W)，numpy array，值域 0-255 或 0-1
        box: bounding box mask (H, W)，numpy array，值域 0-255
        device: 設備 ('cuda' 或 'cpu')
        input_size: 模型輸入尺寸 [H, W]
        model_digit: "full" (float32) 或 "half" (float16)
    
    Returns:
        refined_mask: 精煉後的 mask (H, W)，numpy array，值域 0-255
    """
    with torch.no_grad():
        # 轉換為 tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        mask_tensor = torch.from_numpy(np.array(rough_mask)).float()
        box_tensor = torch.from_numpy(np.array(box)).float()
        
        # 確保 mask 值域在 0-255
        if mask_tensor.max() <= 1:
            mask_tensor = mask_tensor * 255.0
        
        # 拼接 5 通道輸入：[RGB(3) + mask(1) + box(1)]
        inputs = torch.cat([image_tensor, mask_tensor[...,None], box_tensor[...,None]], dim=2)
        inputs = inputs.permute(2, 0, 1)[None, ...]  # [1, 5, H, W]
        # 保存原始尺寸（轉為 tuple 以確保格式正確）
        shapes_val = tuple(inputs.shape[-2:])
        
        # 調整到模型輸入尺寸
        inputs = F.interpolate(inputs, size=input_size, mode='bilinear', align_corners=False)
        
        # 處理 box 通道（二值化）
        box_channel = inputs[0][-1]
        box_channel[box_channel > 127] = 255
        box_channel[box_channel <= 127] = 0
        inputs[0][-1] = box_channel
        
        # 歸一化到 0-1
        inputs = inputs / 255.0
        
        # 轉換數據類型
        if model_digit == "full":
            inputs = inputs.type(torch.FloatTensor)
        else:
            inputs = inputs.type(torch.HalfTensor)
        
        # 移到設備並應用 normalize
        inputs = inputs.to(device)
        transform = GOSNormalize([0.5,0.5,0.5,0,0], [1.0,1.0,1.0,1.0,1.0])
        inputs = transform(inputs)
        
        # 前向傳播
        net.eval()
        model_output = net(inputs)
        # IS-Net 模型返回 tuple: ([d1, d2, d3, d4, d5, d6], [features])
        # 每個 d 的形狀是 [1, 1, H, W]，我們取第一個 side output
        if isinstance(model_output, (list, tuple)):
            predictions_list = model_output[0]  # [d1, d2, d3, d4, d5, d6]
            output = predictions_list[0]  # d1: [1, 1, H, W]
        else:
            output = model_output  # 如果只有一個輸出
        
        # 確保 output 是 4D [1, 1, H, W] 或 3D [1, H, W]
        if output.dim() == 4:
            # 已經是 [1, 1, H, W]，直接使用
            output_4d = output
        elif output.dim() == 3:
            # [1, H, W]，需要添加 channel 維度
            output_4d = output[:, None, ...]  # [1, 1, H, W]
        elif output.dim() == 2:
            # [H, W]，需要添加 batch 和 channel 維度
            output_4d = output[None, None, ...]  # [1, 1, H, W]
        else:
            raise ValueError(f"模型輸出維度錯誤: 期望 2D/3D/4D，得到 {output.shape}")
        
        # 恢復到原始尺寸
        # output_4d 是 [1, 1, H_model, W_model] (通常是 1024x1024)
        # 需要插值回原始圖片尺寸 shapes_val (例如 826x1200)
        # shapes_val 是 tuple (原始H, 原始W)
        pred = F.interpolate(
            output_4d, 
            size=shapes_val,  # tuple (H, W)
            mode='bilinear', 
            align_corners=False
        )[0, 0]  # 恢復為 [H, W]
        
        # 歸一化到 0-1
        ma = torch.max(pred)
        mi = torch.min(pred)
        pred = (pred - mi) / (ma - mi + 1e-8)
        
        # 轉換為 numpy 並二值化
        refined_mask = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
        _, binary = cv2.threshold(refined_mask, 0, 255, cv2.THRESH_OTSU)
        
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return binary


def generate_masks_with_dis_sam(
    image_path: Union[str, Path],
    boxes_json: Union[str, Path, Dict],
    output_mask_path: Union[str, Path],
    per_object_dir: Optional[Union[str, Path]] = None,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_l",
    isnet_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    use_refinement: bool = True,
    auto_download: bool = True,
) -> str:
    """
    使用 DIS-SAM（兩階段方法）依據 Qwen 輸出的 bbox 產生精確 mask
    
    Args:
        image_path: 輸入圖片路徑
        boxes_json: Qwen 的 JSON（dict 或檔案路徑），格式：{"boxes": [{"label": str, "box_xyxy": [x0,y0,x1,y1]}, ...]}
        output_mask_path: 輸出合併 mask 的 PNG 路徑
        per_object_dir:（可選）若提供，會將每個物件的 mask 另存一份
        sam_checkpoint: SAM 權重路徑（可選，會自動下載）
        sam_model_type: SAM 模型類型（"vit_h", "vit_l", "vit_b"），默認 "vit_l"
        isnet_checkpoint: IS-Net 權重路徑（可選，會自動下載或從 checkpoints 目錄查找）
        device: 'cuda' 或 'cpu'
        use_refinement: 是否使用 IS-Net 精煉（True 使用完整 DIS-SAM，False 僅使用 SAM）
        auto_download: 是否自動下載缺失的權重
    
    Returns:
        output_mask_path（字串）
    """
    image_path = Path(image_path)
    if isinstance(boxes_json, (str, Path)):
        with open(boxes_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = boxes_json
    
    boxes = meta.get("boxes", [])
    if not boxes:
        # 沒有 box，輸出全黑 mask
        dummy = np.zeros((100, 100), dtype=np.uint8)
        _save_mask_png(dummy, output_mask_path)
        return str(output_mask_path)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== 階段 1：使用 SAM 生成粗略 mask ==========
    print("[DIS-SAM] 階段 1: 使用 SAM 生成粗略 mask...")
    sam_predictor, device = _load_sam_model(
        checkpoint=sam_checkpoint,
        model_type=sam_model_type,
        device=device,
        auto_download=auto_download
    )
    
    # 載入圖片
    with Image.open(str(image_path)) as im:
        im_rgb = im.convert("RGB")
    image_np = np.array(im_rgb)
    H, W = image_np.shape[:2]
    
    # 設定圖片到 predictor
    sam_predictor.set_image(image_np)
    
    merged_rough_mask = np.zeros((H, W), dtype=np.uint8)
    per_dir = Path(per_object_dir) if per_object_dir else None
    if per_dir:
        per_dir.mkdir(parents=True, exist_ok=True)
    
    # 逐個 box 產生粗略 mask
    rough_masks = []
    for idx, item in enumerate(boxes):
        box_xyxy = _ensure_xyxy(item.get("box_xyxy", []))
        x0, y0, x1, y1 = box_xyxy
        x0 = max(0, min(x0, W))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H))
        y1 = max(0, min(y1, H))
        if x1 <= x0 or y1 <= y0:
            continue
        
        box_arr = np.array([x0, y0, x1, y1])
        
        # 使用 SAM 預測
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=False,
        )
        
        best_mask = masks[0].astype(bool)
        rough_mask = (best_mask * 255).astype(np.uint8)
        rough_masks.append(rough_mask)
        
        merged_rough_mask |= best_mask.astype(np.uint8) * 255
        
        if per_dir is not None:
            _save_mask_png(rough_mask, per_dir / f"rough_mask_{idx:02d}.png")
    
    # 如果不需要精煉，直接返回 SAM 的結果
    if not use_refinement or not ISNET_AVAILABLE:
        if not use_refinement:
            print("[DIS-SAM] 跳過精煉階段（use_refinement=False）")
        else:
            print("[DIS-SAM] 跳過精煉階段（IS-Net 不可用）")
        _save_mask_png(merged_rough_mask, output_mask_path)
        return str(output_mask_path)
    
    # ========== 階段 2：使用 IS-Net 精煉 mask ==========
    print("[DIS-SAM] 階段 2: 使用 IS-Net 精煉 mask...")
    try:
        isnet_model, device = _load_isnet_model(
            checkpoint=isnet_checkpoint,
            device=device,
            auto_download=auto_download
        )
        
        # 從粗略 mask 生成 box
        merged_box = get_box_from_mask(merged_rough_mask)
        
        # 使用 IS-Net 精煉整個合併的 mask
        refined_mask = _refine_mask_with_isnet(
            net=isnet_model,
            image=image_np,
            rough_mask=merged_rough_mask,
            box=merged_box,
            device=device,
            input_size=[1024, 1024],
            model_digit="full"
        )
        
        # 保存最終的精煉 mask
        _save_mask_png(refined_mask, output_mask_path)
        
        # 如果提供了 per_object_dir，也精煉每個物件的 mask
        if per_dir is not None and len(rough_masks) > 1:
            for idx, rough_mask in enumerate(rough_masks):
                box = get_box_from_mask(rough_mask)
                refined_single = _refine_mask_with_isnet(
                    net=isnet_model,
                    image=image_np,
                    rough_mask=rough_mask,
                    box=box,
                    device=device,
                    input_size=[1024, 1024],
                    model_digit="full"
                )
                _save_mask_png(refined_single, per_dir / f"refined_mask_{idx:02d}.png")
        
        print(f"[DIS-SAM] 精煉完成，已保存: {output_mask_path}")
        
    except Exception as e:
        print(f"[DIS-SAM] 警告：IS-Net 精煉失敗，使用 SAM 的粗略 mask: {e}")
        import traceback
        traceback.print_exc()
        _save_mask_png(merged_rough_mask, output_mask_path)
    
    return str(output_mask_path)


if __name__ == "__main__":
    """主程式入口"""
    parser = argparse.ArgumentParser(description="DIS-SAM inference: bbox.json -> mask.png")
    parser.add_argument("--image", type=str, default="image.jpg", help="輸入影像路徑")
    parser.add_argument("--boxes", type=str, default="bbox.json", help="輸入 bbox JSON 路徑")
    parser.add_argument("--output", type=str, default="mask.png", help="輸出合併 mask 路徑")
    parser.add_argument("--per_object_dir", type=str, default=None, help="每物件 mask 輸出資料夾")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="SAM 權重 .pth（或設定 SAM_CHECKPOINT 環境變數）")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM 模型類型")
    parser.add_argument("--isnet_checkpoint", type=str, default=None, help="IS-Net 權重 .pth（或設定 DIS_SAM_CHECKPOINT 環境變數）")
    parser.add_argument("--device", type=str, default=None, help="裝置：cuda/cpu")
    parser.add_argument("--no_refinement", action="store_true", help="停用 IS-Net 精煉，僅使用 SAM")
    parser.add_argument("--no_auto_download", action="store_true", help="停用自動下載權重功能")
    args = parser.parse_args()
    
    # 執行推理
    out_path = generate_masks_with_dis_sam(
        image_path=args.image,
        boxes_json=args.boxes,
        output_mask_path=args.output,
        per_object_dir=args.per_object_dir,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        isnet_checkpoint=args.isnet_checkpoint,
        device=args.device,
        use_refinement=not args.no_refinement,
        auto_download=not args.no_auto_download,
    )
    
    print(f"完成：{out_path}")
