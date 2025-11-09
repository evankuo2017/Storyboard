"""
Qwen Image Edit 2509 
- 提供 get/生成/unload 三個函式，供伺服器端呼叫
- 模型：QwenImageEditPlusPipeline + DFloat11（DF11）
- 自動偵測 VRAM，24GB 以上不啟用 cpu_offload，否則啟用
"""

import os
from typing import Optional

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from dfloat11 import DFloat11Model

# 全域快取模型（global pipeline cache）
_global_pipe_2509 = None


def _check_available_vram() -> float:
    """檢查可用 VRAM（GB）
    
    Returns:
        可用顯存大小（GB），如果無 GPU 則返回 0.0
    """
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        free_memory, _ = torch.cuda.mem_get_info()
        return free_memory / (1024 ** 3)
    except Exception:
        return 0.0


def get_qwen_image_edit_2509_pipeline(
    *,
    cpu_offload: Optional[bool] = None,
    cpu_offload_blocks: int = 20,
    cpu_offload_no_pin_memory: bool = False,
) -> QwenImageEditPlusPipeline:
    """取得或建立 Qwen-Image-Edit-2509 管線
    - 會套用 DFloat11（DF11）優化
    - 自動偵測 VRAM：≥24GB 不啟用 cpu_offload，<24GB 啟用 cpu_offload
    
    Args:
        cpu_offload: 是否啟用 DFloat11 block swapping（None=自動偵測）
        cpu_offload_blocks: CPU offload 的 block 數量
        cpu_offload_no_pin_memory: 是否不使用 pin memory
    """
    global _global_pipe_2509

    if _global_pipe_2509 is not None:
        return _global_pipe_2509

    # 自動偵測是否需要 cpu_offload（DFloat11 block swapping）
    if cpu_offload is None:
        vram_gb = _check_available_vram()
        cpu_offload = (vram_gb < 24.0)
        if vram_gb > 0:
            print(f"偵測到可用 VRAM: {vram_gb:.1f} GB")
            print(f"DFloat11 block swapping: {'啟用' if cpu_offload else '不啟用'}")

    torch_dtype = torch.bfloat16
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
    )

    # 套用 DF11（將 bfloat16 transformer 掛上 DF11 管理）
    DFloat11Model.from_pretrained(
        "DFloat11/Qwen-Image-Edit-2509-DF11",
        bfloat16_model=pipe.transformer,
        device="cpu",
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
        pin_memory=not cpu_offload_no_pin_memory,
    )

    # 啟用 pipeline 的 CPU offload（官方推薦總是啟用）
    pipe.enable_model_cpu_offload()

    _global_pipe_2509 = pipe
    return pipe


def generate_qwen_image_edit_2509(
    *,
    image_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    steps: int = 40,
    seed: Optional[int] = None,
    true_cfg_scale: float = 4.0,
    guidance_scale: float = 1.0,
) -> Optional[str]:
    """呼叫 Qwen-Image-Edit-2509 進行圖片編輯
    
    Args:
        image_path: 輸入圖片路徑
        prompt: 編輯提示詞
        output_path: 輸出圖片路徑
        negative_prompt: 負面提示詞
        steps: 推理步數
        seed: 隨機種子
        true_cfg_scale: True CFG scale
        guidance_scale: Guidance scale
        
    Returns:
        輸出圖片路徑或 None（失敗時）
    """
    try:
        # 自動偵測並載入模型（會根據 VRAM 自動決定是否啟用 cpu_offload）
        pipe = get_qwen_image_edit_2509_pipeline()

        # 載入圖片
        image = Image.open(image_path).convert("RGB")

        # 設定隨機種子
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.manual_seed(seed)

        # 推理
        with torch.inference_mode():
            output = pipe(
                image=[image],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            )
            output_image = output.images[0]
            
            # 保存輸出
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            output_image.save(output_path)
            
        return output_path
        
    except Exception as e:
        print(f"Qwen Image Edit 2509 生成失敗: {e}")
        return None


def unload_qwen_image_edit_2509() -> None:
    """卸載 2509 管線（釋放記憶體/顯存）"""
    global _global_pipe_2509
    if _global_pipe_2509 is not None:
        del _global_pipe_2509
        _global_pipe_2509 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Qwen Image Edit 2509 模型已卸載")


__all__ = [
    "get_qwen_image_edit_2509_pipeline",
    "generate_qwen_image_edit_2509",
    "unload_qwen_image_edit_2509",
]
