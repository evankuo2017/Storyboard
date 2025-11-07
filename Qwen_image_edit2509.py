"""
Qwen Image Edit 2509 版本封裝
- 提供 get/生成/unload 三個函式，供伺服器端呼叫
- 模型：QwenImageEditPlusPipeline + DFloat11（DF11）
"""

import os
from typing import Optional

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from dfloat11 import DFloat11Model

# 全域快取模型（global pipeline cache）
_global_pipe_2509 = None


def get_qwen_image_edit_2509_pipeline(
    *,
    cpu_offload: bool = True,
    cpu_offload_blocks: int = 20,
    cpu_offload_no_pin_memory: bool = False,
) -> QwenImageEditPlusPipeline:
    """取得或建立 Qwen-Image-Edit-2509 管線
    - 會套用 DFloat11（DF11）優化
    - 支援 CPU offload（block swapping）
    """
    global _global_pipe_2509

    if _global_pipe_2509 is not None:
        return _global_pipe_2509

    torch_dtype = torch.bfloat16
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
    )

    # 套用 DF11（將 bfloat16 transformer 掛上 DF11 管理）
    DFloat11Model.from_pretrained(
        "DFloat11/Qwen-Image-Edit-2509-DF11",
        bfloat16_model=pipe.transformer,
        device="cpu",  # DF11 控制搬移策略（不改 pipeline 裝置）
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
        pin_memory=not cpu_offload_no_pin_memory,
    )

    # 啟用 pipeline 的 CPU offload（節省顯存）
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
    cpu_offload: bool = True,
    cpu_offload_blocks: int = 20,
    cpu_offload_no_pin_memory: bool = False,
) -> Optional[str]:
    """呼叫 Qwen-Image-Edit-2509 進行圖片編輯
    - image_path: 輸入圖片（本地路徑）
    - prompt / negative_prompt: 正負面提示詞
    - steps: diffusion 步數
    - true_cfg_scale / guidance_scale: 兩種 CFG 參數
    - 回傳輸出路徑或 None
    """
    try:
        pipe = get_qwen_image_edit_2509_pipeline(
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            cpu_offload_no_pin_memory=cpu_offload_no_pin_memory,
        )

        # 載入圖片（PIL）
        image = Image.open(image_path).convert("RGB")

        # 設定隨機種子（generator 綁定 cuda 或 cpu 皆可，這裡使用可用裝置）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

        # 執行推理
        with torch.inference_mode():
            outputs = pipe(
                image=[image],  # 2509 介面支援 batch 輸入
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            )
            result = outputs.images[0]
            # 輸出
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            result.save(output_path)
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
