#!/usr/bin/env python3
"""Grounding DINO 推理腳本。"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
import numpy as np

try:
    from groundingdino.util.inference import annotate, load_image, load_model, predict
except ImportError as exc:  # noqa: BLE001
    raise ImportError(
        "請先安裝 groundingdino 套件，建議執行 `pip install groundingdino` 或依 README 指示安裝。"
    ) from exc

import requests


DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
DEFAULT_WEIGHTS_NAME = "groundingdino_swint_ogc.pth"


def _resolve_config_path() -> str:
    """取出 Grounding DINO 預設 config 路徑。"""

    try:
        cfg_resource = resources.files("groundingdino").joinpath("config/GroundingDINO_SwinT_OGC.py")
    except AttributeError as exc:  # Python < 3.9 fallback（理論上用不到，但保留保險）
        raise RuntimeError("找不到 GroundingDINO 預設 config，請確認安裝完整。") from exc

    with resources.as_file(cfg_resource) as path:
        return str(path)


def _ensure_weights(path: Path) -> Path:
    """確認權重存在，若缺少則從官方網址下載。"""

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("🔽 正在下載 Grounding DINO 權重...", file=sys.stderr)

    with requests.get(DEFAULT_WEIGHTS_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with path.open("wb") as fout:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:  # 保護避免寫入空資料
                    fout.write(chunk)

    print("✅ 權重下載完成", file=sys.stderr)
    return path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析指令列參數。"""

    parser = argparse.ArgumentParser(description="Grounding DINO 推理工具")
    parser.add_argument("--image", required=True, help="輸入影像路徑")
    parser.add_argument("--prompt", required=True, help="Grounding 文字提示，可用句號分隔多個目標")
    parser.add_argument(
        "--output",
        default="groundingdino_output.png",
        help="輸出影像路徑（含標註框）",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="偵測框分數閾值（box threshold）",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="文字匹配閾值（text threshold）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理裝置，例如 cuda 或 cpu",
    )
    parser.add_argument(
        "--weights",
        help="自訂 Grounding DINO 權重檔路徑，未提供則自動下載預設 SwinT 權重",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """主程式：載入模型、執行 Grounding DINO 推理並輸出帶標註影像。"""

    args = _parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"找不到輸入影像：{image_path}")

    if args.weights:
        weights_path = Path(args.weights).expanduser().resolve()
    else:
        weights_path = Path(__file__).resolve().parent / "checkpoints" / DEFAULT_WEIGHTS_NAME

    weights_path = _ensure_weights(weights_path)
    config_path = _resolve_config_path()

    # 載入模型（load_model 內部會負責搬移到指定裝置）
    model = load_model(config_path, str(weights_path), device=args.device)

    # 載入影像 → tensor
    image_source, image_tensor = load_image(str(image_path))

    # 進行 Grounding DINO 推理
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # 在原圖上繪製框與標籤
    annotated = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 兼容 GroundingDINO 不同版本：可能回傳 PIL.Image 或 numpy.ndarray
    if isinstance(annotated, Image.Image):
        pil_img = annotated
    elif isinstance(annotated, np.ndarray):
        arr = annotated
        # 若是浮點且在 [0,1]，轉成 [0,255] 的 uint8
        if issubclass(arr.dtype.type, np.floating):
            if arr.max() <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            # 其他型別一律轉成 uint8（保守處理）
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # GroundingDINO 內部多用 OpenCV，ndarray 常為 BGR；轉成 PIL 期待的 RGB
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]  # BGR -> RGB
        pil_img = Image.fromarray(arr)
    else:
        raise TypeError(f"不支援的輸出影像型別：{type(annotated)}")

    pil_img.save(output_path)
    print(f"✅ 推理完成，結果已輸出至：{output_path}")


if __name__ == "__main__":
    main()

