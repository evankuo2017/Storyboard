"""
SAM 推理函數（原版 Segment Anything Model）
- 輸入：圖片路徑與 Qwen 輸出的 bbox JSON（XYXY 像素座標）
- 輸出：精確 mask（PNG），以及可選的每個物件獨立 mask
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import urllib.request

import numpy as np
from PIL import Image
import torch
import argparse

# 載入原版 SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise RuntimeError("請安裝 segment-anything: pip install segment-anything")

# SAM 權重下載配置（使用官方公開權重）
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

def _download_checkpoint(model_type: str, checkpoint_dir: Path) -> Path:
    """自動下載 SAM 權重到指定目錄"""
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"不支援的模型類型: {model_type}，支援: {list(SAM_CHECKPOINTS.keys())}")
    
    config = SAM_CHECKPOINTS[model_type]
    checkpoint_path = checkpoint_dir / config["filename"]
    
    # 如果檔案已存在，直接回傳
    if checkpoint_path.exists():
        print(f"[SAM] 權重已存在: {checkpoint_path}")
        return checkpoint_path
    
    # 建立目錄
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SAM] 開始下載 {model_type} 權重 ({config['size']})...")
    print(f"[SAM] 下載來源: {config['url']}")
    print(f"[SAM] 儲存位置: {checkpoint_path}")
    
    try:
        # 下載檔案
        urllib.request.urlretrieve(config["url"], checkpoint_path)
        print(f"[SAM] 下載完成: {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        # 清理失敗的下載
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f"\n[SAM] 自動下載失敗: {e}")
        print(f"[SAM] 請手動下載權重檔案:")
        print(f"[SAM] 1. 前往: {config['url']}")
        print(f"[SAM] 2. 下載檔案到: {checkpoint_path}")
        print(f"[SAM] 3. 或設定環境變數: export SAM_CHECKPOINT=/path/to/your/checkpoint.pth")
        print(f"[SAM] 4. 或使用參數: --checkpoint /path/to/your/checkpoint.pth")
        
        raise RuntimeError(f"下載權重失敗: {e}")


# 全局變量存儲 SAM 模型
_global_sam_predictor = None
_global_sam_device = None

def _load_sam_model(
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    auto_download: bool = True,
):
    """載入 SAM 模型"""
    global _global_sam_predictor, _global_sam_device
    
    if checkpoint is None:
        checkpoint = os.getenv("SAM_CHECKPOINT")
    
    # 如果沒有指定權重或檔案不存在，嘗試自動下載
    if not checkpoint or not Path(checkpoint).exists():
        if auto_download:
            # 建立 checkpoints 目錄
            checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
            checkpoint = _download_checkpoint(model_type, checkpoint_dir)
        else:
            raise FileNotFoundError(
                "請提供 SAM 權重檔路徑（SAM_CHECKPOINT 環境變數或函數參數 checkpoint）\n"
                "下載連結：https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 如果已經載入了相同設備的模型，直接返回
    if _global_sam_predictor is not None and _global_sam_device == device:
        return _global_sam_predictor, device

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # 存儲到全局變量
    _global_sam_predictor = predictor
    _global_sam_device = device
    
    return predictor, device


def unload_sam_model():
    """卸載 SAM 模型以釋放顯存"""
    global _global_sam_predictor, _global_sam_device
    
    if _global_sam_predictor is not None:
        # 清理模型
        if hasattr(_global_sam_predictor, 'model'):
            del _global_sam_predictor.model
        del _global_sam_predictor
        _global_sam_predictor = None
        _global_sam_device = None
        
        # 清理顯存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("SAM 模型已卸載")


def _ensure_xyxy(box: List[int]) -> List[int]:
    """確保 bbox 為 xyxy 格式且為 int"""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError("box_xyxy 需為 [x0, y0, x1, y1]")
    return [int(round(float(v))) for v in box]


def _save_mask_png(mask: np.ndarray, out_path: Union[str, Path]) -> None:
    """將二值 mask 儲存為 PNG（白=前景，黑=背景）"""
    out = Image.fromarray((mask.astype(np.uint8) * 255))
    out.save(str(out_path))


def generate_masks_with_sam(
    image_path: Union[str, Path],
    boxes_json: Union[str, Path, Dict],
    output_mask_path: Union[str, Path],
    per_object_dir: Optional[Union[str, Path]] = None,
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
) -> str:
    """
    使用 SAM 依據 Qwen 輸出的 bbox 產生精確 mask

    Args:
        image_path: 輸入圖片路徑
        boxes_json: Qwen 的 JSON（dict 或檔案路徑）
        output_mask_path: 輸出合併 mask 的 PNG 路徑
        per_object_dir:（可選）若提供，會將每個物件的 mask 另存一份
        checkpoint: SAM 權重（.pth）
        model_type: 模型類型（預設 'vit_h'）
        device: 'cuda' 或 'cpu'

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
        dummy = np.zeros((1, 1), dtype=np.uint8)
        Image.fromarray(dummy).save(str(output_mask_path))
        return str(output_mask_path)

    predictor, device = _load_sam_model(checkpoint=checkpoint, model_type=model_type, device=device, auto_download=True)

    # 載入圖片（numpy，HWC，RGB）
    with Image.open(str(image_path)) as im:
        im_rgb = im.convert("RGB")
    image_np = np.array(im_rgb)

    # 設定圖片到 predictor
    predictor.set_image(image_np)

    H, W = image_np.shape[:2]
    merged_mask = np.zeros((H, W), dtype=np.uint8)

    # 若需要個別輸出
    per_dir = Path(per_object_dir) if per_object_dir else None
    if per_dir:
        per_dir.mkdir(parents=True, exist_ok=True)

    # 逐個 box 產生 mask
    for idx, item in enumerate(boxes):
        box_xyxy = _ensure_xyxy(item.get("box_xyxy", []))
        # 裁切到圖片邊界
        x0, y0, x1, y1 = box_xyxy
        x0 = max(0, min(x0, W))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H))
        y1 = max(0, min(y1, H))
        if x1 <= x0 or y1 <= y0:
            continue

        # SAM predictor 的 box 需要 numpy array（XYXY）
        box_arr = np.array([x0, y0, x1, y1])

        # 使用 SAM 預測
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_arr,
            multimask_output=False,
        )

        # 取最佳 mask
        best_mask = masks[0].astype(bool)

        # 合併到 merged mask
        merged_mask |= best_mask

        if per_dir is not None:
            _save_mask_png(best_mask, per_dir / f"mask_{idx:02d}.png")

    # 輸出合併 mask
    _save_mask_png(merged_mask, output_mask_path)
    
    # 推理完成後清理顯存
    unload_sam_model()
    
    return str(output_mask_path)


if __name__ == "__main__":
    """主程式入口"""
    parser = argparse.ArgumentParser(description="SAM inference: bbox.json -> mask.png")
    parser.add_argument("--image", type=str, default="image.jpg", help="輸入影像路徑")
    parser.add_argument("--boxes", type=str, default="bbox.json", help="輸入 bbox JSON 路徑")
    parser.add_argument("--output", type=str, default="mask.png", help="輸出合併 mask 路徑")
    parser.add_argument("--per_object_dir", type=str, default="masks", help="每物件 mask 輸出資料夾")
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM 權重 .pth（或設定 SAM_CHECKPOINT 環境變數）；若未提供會自動下載")
    parser.add_argument("--model_type", type=str, default="vit_h", help="SAM 模型型號（vit_h, vit_l, vit_b）")
    parser.add_argument("--device", type=str, default=None, help="裝置：cuda/cpu")
    parser.add_argument("--no_auto_download", action="store_true", help="停用自動下載權重功能")
    args = parser.parse_args()

    # 執行推理
    out_path = generate_masks_with_sam(
        image_path=args.image,
        boxes_json=args.boxes,
        output_mask_path=args.output,
        per_object_dir=args.per_object_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
    )

    print(f"完成：{out_path}")
