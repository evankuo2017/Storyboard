#!/usr/bin/env python3
"""
FLUX.1-Kontext-dev：提供函數介面供伺服器呼叫，亦保留 CLI。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

try:
    from diffusers import FluxKontextPipeline
except Exception:
    # 延遲到真正呼叫時再拋出，以便伺服器可先啟動
    FluxKontextPipeline = None  # type: ignore


# 專案內 HF cache 目錄：放在 Storyboard/hf_download
THIS_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = (THIS_DIR / "hf_download").resolve()
os.makedirs(HF_CACHE_DIR, exist_ok=True)


def generate_flux_frame(
    image: str,
    prompt: str,
    output: str,
    guidance_scale: float = 2.5,
    precision: str = "bf16",
    local_files_only: bool = False,
) -> str:
    """
    使用 FLUX.1-Kontext-dev 對單張圖片做編輯，輸出到指定路徑。

    Args:
        image: 輸入圖片路徑
        prompt: 編輯指令文字
        output: 輸出圖片路徑
        guidance_scale: 引導強度（預設 2.5）
        precision: bf16/fp16/fp32（預設 bf16）
        local_files_only: 僅使用本地快取（預設 False）

    Returns:
        寫出的輸出圖片路徑（同 output）
    """
    if FluxKontextPipeline is None:
        raise RuntimeError(
            "需要安裝 diffusers 主分支。建議：pip install \"git+https://github.com/huggingface/diffusers.git\" transformers accelerate safetensors"
        )

    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

    image_path = Path(image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"找不到圖片檔案: {image_path}")
    out_path = Path(output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE_DIR),
        local_files_only=local_files_only,
    )
    pipe.to(device)

    input_image = Image.open(str(image_path)).convert("RGB")
    result = pipe(image=input_image, prompt=prompt, guidance_scale=float(guidance_scale))
    edited_image = result.images[0]
    edited_image.save(str(out_path))
    return str(out_path)


# 保留 CLI 方便單獨測試
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX.1-Kontext-dev editing (function wrapper)")
    parser.add_argument("--image", type=str, required=True, help="輸入圖片路徑（.png/.jpg）")
    parser.add_argument("--prompt", type=str, required=True, help="編輯指令（prompt）")
    parser.add_argument("--output", type=str, required=True, help="輸出圖片檔名")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="引導強度（預設 2.5）")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="計算精度（預設 bf16）")
    parser.add_argument("--local_files_only", action="store_true", help="僅使用本地快取")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = generate_flux_frame(
        image=args.image,
        prompt=args.prompt,
        output=args.output,
        guidance_scale=args.guidance_scale,
        precision=args.precision,
        local_files_only=bool(args.local_files_only),
    )
    print(f"完成！輸出：{out}")


if __name__ == "__main__":
    main()


