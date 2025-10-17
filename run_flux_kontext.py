#!/usr/bin/env python3
"""
使用 FLUX.1-Kontext-dev
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

try:
    from diffusers import FluxKontextPipeline
except Exception as e:
    print("錯誤：需要安裝 diffusers 主分支與相關相依套件。")
    print("建議：pip install \"git+https://github.com/huggingface/diffusers.git\" transformers accelerate safetensors")
    raise


# 預設參數（可直接改動）
DEFAULT_IMAGE = "/home/samjimbe/project/FLUX/example_input.png"  # 請改成你的圖片路徑
DEFAULT_PROMPT = "the scene zoom out showing the girl's long legs."
DEFAULT_OUTPUT = "edited_output.png"

# 把 HF cache 放在專案內，乾淨又可離線
PROJECT_ROOT = Path("/home/samjimbe/project/FLUX").resolve()
HF_CACHE_DIR = PROJECT_ROOT / "hf_download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX.1-Kontext-dev editing with local HF cache (方案A)")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="輸入圖片路徑（.png/.jpg）")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="編輯指令（prompt）")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="輸出圖片檔名")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="引導強度（預設 2.5）")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="計算精度（預設 bf16）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 設定 HF 快取目錄到專案內（方案A）
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    # 若你想強制離線（已下載過權重），可在外部 export HF_HUB_OFFLINE=1

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"錯誤：找不到圖片檔案: {image_path}")
        sys.exit(1)

    out_path = Path(args.output).expanduser().resolve()

    # 選擇 dtype
    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype != torch.float32:
        # CPU 上使用 fp32 最穩定
        dtype = torch.float32

    model_id = "black-forest-labs/FLUX.1-Kontext-dev"

    print("載入管線（Pipeline）…")
    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(HF_CACHE_DIR),      # 專案內快取
        local_files_only=False            # 若第一次未下載，允許連網抓取
    )
    pipe.to(device)

    # 載入圖片
    input_image = Image.open(str(image_path)).convert("RGB")

    print(f"開始編輯：{image_path.name} | prompt: {args.prompt}")
    result = pipe(
        image=input_image,
        prompt=args.prompt,
        guidance_scale=float(args.guidance_scale)
    )
    edited_image = result.images[0]

    # 輸出
    edited_image.save(str(out_path))
    print(f"完成！輸出：{out_path}")


if __name__ == "__main__":
    main()


