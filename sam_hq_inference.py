"""
SAM-HQ 推理函數
- 輸入：圖片路徑與 Qwen 輸出的 bbox JSON（XYXY 像素座標，適配 SAM）
- 輸出：精確 mask（PNG），以及可選的每個物件獨立 mask

依序嘗試匯入不同套件名稱：
- segment_anything_hq（常見 SAM-HQ 分支）
- sam_hq
- segment_anything（退而求其次，非 HQ）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
import torch


# 嘗試載入 SAM-HQ/SAM 模組（中文註解 + English nouns）
HQ_REGISTRY = None
HQ_PREDICTOR = None
try:
    # 常見的 SAM-HQ 分支匯入方式
    from segment_anything_hq import sam_model_registry as _hq_registry  # type: ignore
    from segment_anything_hq import SamPredictor as _HQPredictor  # type: ignore
    HQ_REGISTRY, HQ_PREDICTOR = _hq_registry, _HQPredictor
except Exception:
    try:
        from sam_hq import sam_model_registry as _hq_registry  # type: ignore
        from sam_hq import SamPredictor as _HQPredictor  # type: ignore
        HQ_REGISTRY, HQ_PREDICTOR = _hq_registry, _HQPredictor
    except Exception:
        try:
            # 退回原始 SAM（非 HQ）
            from segment_anything import sam_model_registry as _hq_registry  # type: ignore
            from segment_anything import SamPredictor as _HQPredictor  # type: ignore
            HQ_REGISTRY, HQ_PREDICTOR = _hq_registry, _HQPredictor
        except Exception:
            HQ_REGISTRY, HQ_PREDICTOR = None, None


def _load_sam_model(
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
):
    """載入 SAM-HQ/SAM 模型
    - checkpoint：SAM-HQ 權重（.pth）。若未提供，嘗試讀環境變數 SAM_HQ_CHECKPOINT
    - model_type：sam model type（e.g. 'vit_h', 'vit_l'）
    - device：cuda/cpu，自動偵測
    """
    if HQ_REGISTRY is None or HQ_PREDICTOR is None:
        raise RuntimeError(
            "找不到 SAM-HQ/SAM 套件，請安裝 segment_anything_hq 或 sam_hq 或 segment_anything"
        )

    if checkpoint is None:
        checkpoint = os.getenv("SAM_HQ_CHECKPOINT")

    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(
            "請提供 SAM-HQ 權重檔路徑（SAM_HQ_CHECKPOINT 環境變數或函數參數 checkpoint）"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = HQ_REGISTRY[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = HQ_PREDICTOR(sam)
    return predictor, device


def _ensure_xyxy(box: List[int]) -> List[int]:
    """確保 bbox 為 xyxy 格式且為 int。"""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError("box_xyxy 需為 [x0, y0, x1, y1]")
    return [int(round(float(v))) for v in box]


def _save_mask_png(mask: np.ndarray, out_path: Union[str, Path]) -> None:
    """將二值 mask 儲存為 PNG（白=前景，黑=背景）"""
    out = Image.fromarray((mask.astype(np.uint8) * 255))
    out.save(str(out_path))


def generate_masks_with_sam_hq(
    image_path: Union[str, Path],
    boxes_json: Union[str, Path, Dict],
    output_mask_path: Union[str, Path],
    per_object_dir: Optional[Union[str, Path]] = None,
    checkpoint: Optional[str] = None,
    model_type: str = "vit_h",
    device: Optional[str] = None,
    multimask_output: bool = False,
) -> str:
    """
    使用 SAM-HQ 依據 Qwen 輸出的 bbox 產生精確 mask。

    Args:
        image_path: 輸入圖片路徑
        boxes_json: Qwen 的 JSON（dict 或檔案路徑），格式：
          {"image_size":[W,H], "format":"SAM_xyxy_pixel",
           "boxes":[{"label":"obj","box_xyxy":[x0,y0,x1,y1]}, ...]}
        output_mask_path: 輸出合併 mask 的 PNG 路徑
        per_object_dir:（可選）若提供，會將每個物件的 mask 另存一份
        checkpoint: SAM-HQ 權重（.pth）。可改用環境變數 SAM_HQ_CHECKPOINT
        model_type: 模型類型（預設 'vit_h'）
        device: 'cuda' 或 'cpu'
        multimask_output: 是否輸出多重候選（此處取最佳 one-hot）

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

    predictor, device = _load_sam_model(checkpoint=checkpoint, model_type=model_type, device=device)

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

    # 逐個 box 產生 mask（使用單框推理）
    for idx, item in enumerate(boxes):
        box_xyxy = _ensure_xyxy(item.get("box_xyxy", []))
        # 裁切到圖片邊界（避免越界）
        x0, y0, x1, y1 = box_xyxy
        x0 = max(0, min(x0, W))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H))
        y1 = max(0, min(y1, H))
        if x1 <= x0 or y1 <= y0:
            continue

        # SAM predictor 的 box 需要 numpy array（XYXY）
        box_arr = np.array([x0, y0, x1, y1])

        try:
            # 使用 predict（單框），multimask_output=True 會回傳多個候選
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_arr,
                multimask_output=multimask_output,
            )
        except TypeError:
            # 部分版本僅支援不同介面，嘗試 predict_torch
            import torch as _torch
            boxes_t = _torch.tensor(box_arr[None, :], device=device)
            image_sz = image_np.shape[:2]
            boxes_t = predictor.transform.apply_boxes_torch(boxes_t, image_sz)
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_t,
                multimask_output=multimask_output,
            )
            masks = masks.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

        # 取最佳分數的 mask（soft->bool）
        best_idx = int(np.argmax(scores)) if np.ndim(scores) > 0 else 0
        best = masks[best_idx].astype(bool)

        # 合併到 merged mask（logical OR）
        merged_mask |= best

        if per_dir is not None:
            _save_mask_png(best, per_dir / f"mask_{idx:02d}.png")

    # 輸出合併 mask
    _save_mask_png(merged_mask, output_mask_path)
    return str(output_mask_path)



