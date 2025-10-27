import torch
from PIL import Image
from diffusers import AutoModel, DiffusionPipeline, TorchAoConfig
import os

# 全局變量存儲模型
_global_pipe = None

def get_qwen_image_edit_pipeline():
    """獲取或創建 Qwen Image Edit pipeline"""
    global _global_pipe
    
    if _global_pipe is None:
        model_id = "Qwen/Qwen-Image-Edit"
        torch_dtype = torch.bfloat16
        
        # TorchAO int8 weight-only on transformer
        quantization_config = TorchAoConfig("int8wo")
        
        transformer = AutoModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
        )
        
        _global_pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            transformer=transformer, 
            torch_dtype=torch_dtype,
        )
        _global_pipe.enable_model_cpu_offload()
    
    return _global_pipe

def generate_qwen_image_edit(image_path, prompt, output_path, negative_prompt="", num_inference_steps=25, seed=None):
    """
    使用 Qwen Image Edit 生成編輯後的圖片
    
    Args:
        image_path: 輸入圖片路徑
        prompt: 編輯提示詞
        output_path: 輸出圖片路徑
        negative_prompt: 負面提示詞
        num_inference_steps: 推理步數
        seed: 隨機種子，None 則使用隨機
    
    Returns:
        成功返回輸出路徑，失敗返回 None
    """
    try:
        # 獲取 pipeline
        pipe = get_qwen_image_edit_pipeline()
        
        # 載入圖片
        image = Image.open(image_path).convert("RGB")
        
        # 設置隨機種子
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # 生成圖片
        result = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt
        ).images[0]
        
        # 保存結果
        result.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Qwen Image Edit 生成失敗: {e}")
        return None

def unload_qwen_image_edit():
    """卸載 Qwen Image Edit 模型以釋放記憶體"""
    global _global_pipe
    if _global_pipe is not None:
        del _global_pipe
        _global_pipe = None
        torch.cuda.empty_cache()
        print("Qwen Image Edit 模型已卸載")
