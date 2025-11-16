"""
Outpaint 推理函數（圖片擴展生成）
- 與 `diffusers-image-outpaint/app.py` 使用完全相同的模型組合：
  - ControlNet Union SDXL (`xinsir/controlnet-union-sdxl-1.0`)
  - Base 模型 `SG161222/RealVisXL_V5.0_Lightning`
  - VAE `madebyollin/sdxl-vae-fp16-fix`
  - 排程器 `TCDScheduler`
- 功能：將輸入圖片縮小後置中，並生成四周內容完成擴圖
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

_ROOT_DIR = Path(__file__).resolve().parent
_LOCAL_MODULE_DIR = _ROOT_DIR / "Diffusers_Image_Outpaint"
if _LOCAL_MODULE_DIR.exists():
    sys.path.insert(0, str(_LOCAL_MODULE_DIR))

from controlnet_union import ControlNetModel_Union  # type: ignore
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline  # type: ignore

# ---------------------------------------------------------------------------
# 全域設定
# ---------------------------------------------------------------------------
_CACHE_DIR = Path.cwd() / "hf_download" / "hub"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CONTROLNET_REPO = "xinsir/controlnet-union-sdxl-1.0"
_CONTROLNET_CONFIG = "config_promax.json"
_CONTROLNET_CKPT = "diffusion_pytorch_model_promax.safetensors"
_BASE_MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"
_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if _DEVICE == "cuda" else torch.float32

_cached_pipeline: Optional[StableDiffusionXLFillPipeline] = None
_cached_dtype: Optional[torch.dtype] = None


# ---------------------------------------------------------------------------
# 工具函式：與 app.py 保持一致的影像處理邏輯
# ---------------------------------------------------------------------------
def can_expand(source_width: int, source_height: int, target_width: int, target_height: int, alignment: str) -> bool:
    """判斷在指定對齊方式下是否需要擴張（沿用 app.py 的邏輯）。"""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def prepare_image_and_mask(
    image: Image.Image,
    width: int,
    height: int,
    overlap_percentage: int,
    resize_option: str,
    custom_resize_percentage: int,
    alignment: str,
    overlap_left: bool,
    overlap_right: bool,
    overlap_top: bool,
    overlap_bottom: bool,
) -> Tuple[Image.Image, Image.Image]:
    """依 app.py 的邏輯建立 ControlNet 輸入圖與遮罩。"""

    target_size = (width, height)

    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    source = image.resize((new_width, new_height), Image.LANCZOS)

    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:
        resize_percentage = custom_resize_percentage

    resize_factor = resize_percentage / 100
    new_width = max(int(source.width * resize_factor), 64)
    new_height = max(int(source.height * resize_factor), 64)
    source = source.resize((new_width, new_height), Image.LANCZOS)

    overlap_x = max(int(new_width * (overlap_percentage / 100)), 1)
    overlap_y = max(int(new_height * (overlap_percentage / 100)), 1)

    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    else:  # Bottom
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    white_gaps_patch = 2
    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch

    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    mask_draw.rectangle([(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0)

    return background, mask


# ---------------------------------------------------------------------------
# 模型載入與快取（與 app.py 相同組合）
# ---------------------------------------------------------------------------
def get_outpaint_pipeline(dtype_override: Optional[torch.dtype] = None) -> StableDiffusionXLFillPipeline:
    """載入並快取 SDXL Lightning + ControlNet Union 的填充管線。"""

    global _cached_pipeline, _cached_dtype

    if _cached_pipeline is not None and _cached_dtype == (dtype_override or _DTYPE):
        print(f"[Outpaint] 使用快取的 pipeline（dtype={_cached_dtype}）")
        return _cached_pipeline

    effective_dtype = dtype_override or _DTYPE
    print(f"[Outpaint] 載入 ControlNet Union + RealVisXL Lightning，dtype={effective_dtype}, device={_DEVICE}")

    config_path = hf_hub_download(
        _CONTROLNET_REPO,
        filename=_CONTROLNET_CONFIG,
        cache_dir=str(_CACHE_DIR),
    )
    config = ControlNetModel_Union.load_config(config_path)
    controlnet_model = ControlNetModel_Union.from_config(config)

    ckpt_path = hf_hub_download(
        _CONTROLNET_REPO,
        filename=_CONTROLNET_CKPT,
        cache_dir=str(_CACHE_DIR),
    )
    state_dict = load_state_dict(ckpt_path)
    controlnet, *_ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model,
        state_dict,
        ckpt_path,
        _CONTROLNET_REPO,
        list(state_dict.keys()),
    )
    controlnet.to(device=_DEVICE, dtype=effective_dtype)

    vae = AutoencoderKL.from_pretrained(
        _VAE_ID,
        torch_dtype=effective_dtype,
        cache_dir=str(_CACHE_DIR),
    ).to(_DEVICE)

    variant = "fp16" if effective_dtype == torch.float16 else None
    pipeline = StableDiffusionXLFillPipeline.from_pretrained(
        _BASE_MODEL_ID,
        torch_dtype=effective_dtype,
        cache_dir=str(_CACHE_DIR),
        vae=vae,
        controlnet=controlnet,
        variant=variant,
    ).to(_DEVICE)

    pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
    print("[Outpaint] Pipeline 載入完成並已切換 TCDScheduler")

    _cached_pipeline = pipeline
    _cached_dtype = effective_dtype
    return pipeline


def unload_outpaint() -> None:
    """卸載已快取的 pipeline，釋放顯存。"""

    global _cached_pipeline, _cached_dtype
    if _cached_pipeline is None:
        print("[Outpaint] Pipeline 未載入，略過卸載")
        return

    try:
        _cached_pipeline.to("cpu")
    except Exception as exc:  # noqa: BLE001
        print(f"[Outpaint] 將 pipeline 移回 CPU 失敗：{exc}")

    _cached_pipeline = None
    _cached_dtype = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[Outpaint] 已清理 GPU 顯存")


# ---------------------------------------------------------------------------
# 主要推理函式
# ---------------------------------------------------------------------------
def outpaint_center_shrink(
    image: Union[str, Path, Image.Image],
    prompt: str,
    shrink_percent: float,
    output_path: str = "output.jpg",
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 8,
    guidance_scale: float = 2.0,  # 注意：此參數保留但不使用（app.py 中不傳入 guidance_scale）
    seed: Optional[int] = None,
    model_id: str = _BASE_MODEL_ID,  # 同樣保留但實際固定
    canvas_fill_rgb: Tuple[int, int, int] = (255, 255, 255),
    overlap_percentage: int = 2,  # 預設為 1
    resize_option: str = "Custom",  # 改為 Custom 以使用 shrink_percent
    custom_resize_percentage: int = 100,
    alignment: str = "Middle",  # 固定為 Middle 來置中
    overlap_left: bool = True,
    overlap_right: bool = True,
    overlap_top: bool = True,
    overlap_bottom: bool = True,
    return_dict: bool = False,
) -> Union[str, dict]:
    """
    圖片中心縮小＋四周擴圖。
    
    使用與 app.py 相同的邏輯：
    - 原圖大小作為 target size (custom size)
    - 通過 shrink_percent 參數縮小圖片並置中
    - overlap 預設為 1
    - 使用 prepare_image_and_mask 處理所有圖片調整邏輯
    """

    del model_id        # Pipeline 已固定為 RealVisXL Lightning

    if isinstance(image, (str, Path)):
        base_img = Image.open(image).convert("RGB")
        image_path = str(image)
    elif isinstance(image, Image.Image):
        base_img = image.convert("RGB")
        image_path = "PIL.Image"
    else:
        raise TypeError("image 需為檔案路徑或 PIL.Image")

    original_width, original_height = base_img.size
    print(f"[Outpaint] 輸入圖片：{image_path}，原始尺寸：{original_width}x{original_height}")
    
    # 限制最大尺寸以避免 OOM
    MAX_DIMENSION = 1536  # 最大邊長限制
    if max(original_width, original_height) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        base_img = base_img.resize((new_width, new_height), Image.LANCZOS)
        print(f"[Outpaint] 圖片過大，已縮小至 {new_width}x{new_height} (縮放比例: {scale:.2f})")
        original_width, original_height = new_width, new_height
    
    original_img = base_img.copy()  # 保存原圖副本（用於 return_dict）

    # 確保尺寸是 8 的倍數（向上取整，確保生成的圖一定 >= 原圖）
    width = ((original_width + 7) // 8) * 8
    height = ((original_height + 7) // 8) * 8
    if (width, height) != (original_width, original_height):
        print(f"[Outpaint] 向上擴展至 8 的倍數尺寸：{width}x{height}")
        # 需要先調整原圖到 8 的倍數
        base_img = base_img.resize((width, height), resample=Image.LANCZOS)
    else:
        print(f"[Outpaint] 原始尺寸已是 8 的倍數，使用原尺寸：{width}x{height}")

    # 限制 shrink_percent 範圍
    shrink_percent = max(0.0, min(float(shrink_percent), 95.0))
    
    # 計算縮小後的尺寸百分比（shrink_percent 是要縮小的量，所以實際大小是 100 - shrink_percent）
    actual_size_percentage = 100.0 - shrink_percent
    print(f"[Outpaint] 縮小百分比：{shrink_percent}%，實際圖片大小：{actual_size_percentage}%")
    
    # 使用 prepare_image_and_mask 來處理縮小和置中
    # resize_option 固定為 "Custom"，並使用 actual_size_percentage
    ctrl_background, mask = prepare_image_and_mask(
        base_img,                          # 使用原圖（已調整為 8 的倍數）
        width,                             # target width = 原圖寬度
        height,                            # target height = 原圖高度
        overlap_percentage,                # overlap 預設為 1
        "Custom",                          # 使用 Custom 模式
        int(actual_size_percentage),       # 縮小後的實際大小百分比
        alignment,                         # 固定為 "Middle" 置中
        overlap_left,
        overlap_right,
        overlap_top,
        overlap_bottom,
    )

    # 創建 ControlNet 輸入圖片（將遮罩區域填黑）
    control_image = ctrl_background.copy()
    control_image.paste(0, (0, 0), mask)

    # 設定隨機種子
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        print(f"[Outpaint] 使用固定隨機種子：{seed}")

    # 載入 pipeline
    pipeline = get_outpaint_pipeline()

    # 準備 prompt（與 app.py 完全相同）
    final_prompt = f"{prompt} , high quality, 4k"
    print(f"[Outpaint] Prompt: {final_prompt}")
    if negative_prompt:
        print(f"[Outpaint] Negative prompt: {negative_prompt}")

    # 編碼 prompt（與 app.py 完全相同：do_classifier_free_guidance 固定為 True）
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
        final_prompt,
        device=_DEVICE,
        do_classifier_free_guidance=True,  
    )

    # 執行推理生成（與 app.py 完全相同：不傳入 guidance_scale）
    generated_images = []
    for generated in pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=control_image,
        num_inference_steps=num_inference_steps,
        # 不傳入 guidance_scale 參數
    ):
        generated_images.append(generated)

    if not generated_images:
        raise RuntimeError("Pipeline 未產生任何圖片")

    # 取得最終生成結果
    result_image = generated_images[-1]

    # 確保生成尺寸正確
    if result_image.size != (width, height):
        print(f"[警告] 生成圖片尺寸 {result_image.size} 與預期不符，調整為 {width}x{height}")
        result_image = result_image.resize((width, height), resample=Image.LANCZOS)

    # 如果生成的圖片比原圖大，使用中心裁切回原尺寸
    if (width, height) != (original_width, original_height):
        # 計算中心裁切的座標
        left = (width - original_width) // 2
        top = (height - original_height) // 2
        right = left + original_width
        bottom = top + original_height
        
        result_image = result_image.crop((left, top, right, bottom))
        print(f"[Outpaint] 已從 {width}x{height} 中心裁切回原始尺寸 {original_width}x{original_height}")
    
    # 儲存結果
    result_image.save(output_path)
    print(f"[Outpaint] 結果已儲存至：{output_path}")

    # 回傳結果
    if return_dict:
        return {
            "result": result_image,                 # 最終輸出圖片（已裁切回原尺寸）
            "original": original_img,               # 原始輸入圖片
            "control_background": ctrl_background,  # prepare_image_and_mask 的背景
            "control_image": control_image,         # ControlNet 輸入
            "mask": mask,
            "output_path": output_path,
            "info": {
                "shrink_percent": shrink_percent,
                "actual_size_percentage": actual_size_percentage,
                "original_size": (original_width, original_height),
                "processing_size": (width, height),
                "final_size": (result_image.width, result_image.height),
                "overlap_percentage": overlap_percentage,
                "alignment": alignment,
                "seed": seed,
            },
        }
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """命令列介面。"""

    parser = argparse.ArgumentParser(description="Outpaint 圖片擴展生成（ControlNet Union + SDXL Lightning）")
    parser.add_argument("--image", "-i", required=True, type=str, help="輸入圖片路徑")
    parser.add_argument("--prompt", "-p", required=True, type=str, help="生成提示詞")
    parser.add_argument("--shrink_percent", "-s", required=True, type=float, help="縮小百分比（0-95，20 代表縮小到 80%）")
    parser.add_argument("--output", "-o", type=str, default=None, help="輸出圖片路徑")
    parser.add_argument("--negative_prompt", "-n", type=str, default=None, help="反向提示詞，可留空")
    parser.add_argument("--steps", type=int, default=20, help="Diffusion 步數（預設 8）")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子（預設隨機）")
    parser.add_argument("--save_intermediates", action="store_true", help="是否同時輸出 control image 與 mask")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[錯誤] 找不到輸入圖片：{args.image}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        image_path = Path(args.image)
        output_path = str(image_path.with_name(f"{image_path.stem}_outpaint{image_path.suffix}"))

    try:
        result = outpaint_center_shrink(
            image=args.image,
            prompt=args.prompt,
            shrink_percent=args.shrink_percent,
            output_path=output_path,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            seed=args.seed,
            return_dict=args.save_intermediates,
        )

        if args.save_intermediates and isinstance(result, dict):
            result_path = Path(output_path)
            control_path = result_path.with_name(f"{result_path.stem}_control{result_path.suffix}")
            mask_path = result_path.with_name(f"{result_path.stem}_mask.png")
            result["control_image"].save(control_path)
            result["mask"].save(mask_path)
            print(f"[Outpaint] Control image 已儲存至：{control_path}")
            print(f"[Outpaint] Mask 已儲存至：{mask_path}")

        print("\n[Outpaint] 卸載模型...")
        unload_outpaint()
        print("[Outpaint] 完成！")

    except Exception as exc:  # noqa: BLE001
        print(f"\n[錯誤] Outpaint 執行失敗：{exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

