from diffusers_helper.hf_login import login

import os
import sys
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# 導入所需的庫
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# 全局變數 - 移除固定的 video_output 資料夾創建
# outputs_folder 現在由 process_video 函數的 output 參數動態決定

# 模型緩存
_models = None
_high_vram = None

def init_models():
    """初始化模型，只需要執行一次"""
    global _models, _high_vram
    
    if _models is not None:
        return _models
    
    print("初始化FramePack模型...")
    
    # 檢查VRAM並決定模式
    try:
        free_mem_gb = get_cuda_free_memory_gb(gpu)
        _high_vram = free_mem_gb > 60
        print(f'Free VRAM {free_mem_gb} GB')
        print(f'High-VRAM Mode: {_high_vram}')
    except Exception as e:
        print(f"警告: 無法檢測VRAM: {e}")
        print("默認使用低VRAM模式")
        _high_vram = False
    
    # 加載模型
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()
    
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()
    
    if not _high_vram:
        vae.enable_slicing()
        vae.enable_tiling()
    
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')
    
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    
    if not _high_vram:
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)
    
    # 儲存所有模型到全局變數
    _models = {
        'text_encoder': text_encoder,
        'text_encoder_2': text_encoder_2,
        'tokenizer': tokenizer,
        'tokenizer_2': tokenizer_2,
        'vae': vae,
        'feature_extractor': feature_extractor,
        'image_encoder': image_encoder,
        'transformer': transformer
    }
    
    return _models

def unload_framepack():
    """卸載FramePack模型以釋放顯存"""
    global _models, _high_vram
    if _models is not None:
        try:
            # 將所有模型移回CPU
            for name, model in _models.items():
                if hasattr(model, 'to'):
                    try:
                        model.to('cpu')
                        print(f"[FramePack] 已將 {name} 移回 CPU")
                    except Exception as e:
                        print(f"[FramePack] {name} 移回 CPU 失敗: {e}")
            
            # 清除全局變數
            _models = None
            _high_vram = None
            
            # 清理CUDA快取
            torch.cuda.empty_cache()
            print("[FramePack] 模型已卸載，顯存已釋放")
            
        except Exception as e:
            print(f"[FramePack] 卸載過程中發生錯誤: {e}")
    else:
        print("[FramePack] 沒有載入的模型需要卸載")

class ProgressCallback:
    """進度回調類"""
    def __init__(self, external_callback=None):
        self.last_update = 0
        self.external_callback = external_callback
    
    def update(self, percentage, message):
        # 限制更新頻率，避免刷屏
        current_time = time.time()
        if current_time - self.last_update > 0.1 or percentage == 100:
            print(f"\r{message} [{percentage}%]", end="")
            self.last_update = current_time
            
            # 如果有外部回調函數，調用它
            if self.external_callback:
                self.external_callback(percentage, message)
    
    def done(self):
        print()  # 換行

@torch.no_grad()
def process_video(start_image_path, end_image_path=None, progress_callback=None, **kwargs):
    """
    處理視頻生成任務
    
    Args:
        start_image_path: 起始幀圖片路徑
        end_image_path: 結束幀圖片路徑 (可選)
        progress_callback: 進度回調函數 (可選)
        **kwargs: 其他參數，包括:
            - prompt: 提示詞
            - n_prompt: 負面提示詞
            - seed: 隨機種子
            - total_second_length: 影片長度(秒)
            - latent_window_size: 潛在窗口大小
            - steps: 採樣步數
            - cfg: CFG尺度
            - gs: 蒸餾CFG尺度
            - rs: CFG重縮放
            - gpu_memory_preservation: GPU推理保留內存(GB)
            - use_teacache: 使用TeaCache
            - mp4_crf: MP4壓縮質量
            - output: 輸出文件路徑
    
    Returns:
        輸出文件路徑，如果失敗則返回None
    """
    # 初始化模型
    models = init_models()
    
    # 設置默認參數
    params = {
        'prompt': 'Character movements based on storyboard sequence',
        'n_prompt': '',
        'seed': 31337,
        'total_second_length': 5,
        'latent_window_size': 9,
        'steps': 25,
        'cfg': 1.0,
        'gs': 10.0,
        'rs': 0.0,
        'gpu_memory_preservation': 6,
        'use_teacache': True,
        'mp4_crf': 16,
        'output': None,
        # 預覽模式：只生成最後一幀並輸出圖片
        'preview_last_only': False,
        'preview_image_output': None,
    }
    
    # 更新參數
    params.update(kwargs)
    
    # 創建進度回調
    progress = ProgressCallback(external_callback=progress_callback)
    
    # 計算總潛在區段數
    total_latent_sections = (params['total_second_length'] * 30) / (params['latent_window_size'] * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[FramePack] total_second_length={params['total_second_length']}, total_latent_sections={total_latent_sections}")
    
    # 生成任務ID
    job_id = generate_timestamp()
    
    progress.update(0, 'Starting ...')
    
    try:
        # 清理GPU
        if not _high_vram:
            unload_complete_models(
                models['text_encoder'], models['text_encoder_2'], models['image_encoder'], models['vae'], models['transformer']
            )
        
        # 文本編碼
        progress.update(0, 'Text encoding ...')
        
        if not _high_vram:
            fake_diffusers_current_device(models['text_encoder'], gpu)
            load_model_as_complete(models['text_encoder_2'], target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(params['prompt'], models['text_encoder'], models['text_encoder_2'], models['tokenizer'], models['tokenizer_2'])
        
        if params['cfg'] == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(params['n_prompt'], models['text_encoder'], models['text_encoder_2'], models['tokenizer'], models['tokenizer_2'])
        
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # 處理起始幀圖像
        progress.update(0, 'Processing start frame ...')
        
        input_image = Image.open(start_image_path)
        # 確保影像是RGB格式
        if input_image.mode != "RGB":
            print(f"Info: 正在將起始影像 {start_image_path} 從模式 {input_image.mode} 轉換為 RGB。")
            input_image = input_image.convert("RGB")
        input_image_np = np.array(input_image)

        # 在此，input_image_np 應該是 H, W, 3 的形式
        # 作為最後的防護，檢查維度和通道
        if input_image_np.ndim != 3 or input_image_np.shape[-1] != 3:
            error_msg = f"錯誤: 起始影像 {start_image_path} 在RGB轉換後的 NumPy 陣列維度不正確: {input_image_np.shape} (預期 3 維且最後維度為 3)"
            print(error_msg)
            raise ValueError(error_msg)

        H, W, C = input_image_np.shape # 現在這行應該可以正常工作
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
        
        # 不再保存臨時的起始圖片檔案
        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # 處理結束幀圖像(如果提供)
        has_end_image = end_image_path is not None
        if has_end_image:
            progress.update(0, 'Processing end frame ...')
            
            try:
                img_pil = Image.open(end_image_path)
                # 確保影像是RGB格式
                if img_pil.mode != "RGB":
                    print(f"Info: 正在將結束影像 {end_image_path} 從模式 {img_pil.mode} 轉換為 RGB。")
                    img_pil = img_pil.convert("RGB")
                end_image_np = np.array(img_pil)
            except Exception as e:
                print(f"錯誤: 無法開啟或轉換結束影像 {end_image_path} 為 RGB NumPy 陣列: {e}")
                # 可以選擇重新引發異常，或設定 has_end_image = False，或返回錯誤狀態
                # 此處重新引發異常，讓呼叫方 (storyboard_server.py) 處理
                raise 

            # 在此，end_image_np 應該是 H, W, 3 的形式
            # 作為最後的防護，檢查維度和通道
            if end_image_np.ndim != 3 or end_image_np.shape[-1] != 3:
                error_msg = f"錯誤: 結束影像 {end_image_path} 在RGB轉換後的 NumPy 陣列維度不正確: {end_image_np.shape} (預期 3 維且最後維度為 3)"
                print(error_msg)
                raise ValueError(error_msg)

            H_end, W_end, C_end = end_image_np.shape # 現在這行應該可以正常工作
            
            end_image_np = resize_and_center_crop(end_image_np, target_width=width, target_height=height)
            
            # 不再保存臨時的結束圖片檔案
            
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
        
        # VAE編碼
        progress.update(0, 'VAE encoding ...')
        
        if not _high_vram:
            load_model_as_complete(models['vae'], target_device=gpu)
        
        start_latent = vae_encode(input_image_pt, models['vae'])
        
        if has_end_image:
            end_latent = vae_encode(end_image_pt, models['vae'])
        
        # CLIP視覺編碼
        progress.update(0, 'CLIP Vision encoding ...')
        
        if not _high_vram:
            load_model_as_complete(models['image_encoder'], target_device=gpu)
        
        image_encoder_output = hf_clip_vision_encode(input_image_np, models['feature_extractor'], models['image_encoder'])
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, models['feature_extractor'], models['image_encoder'])
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # 結合兩個圖像嵌入或使用加權方法
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2
        
        # 數據類型轉換
        llama_vec = llama_vec.to(models['transformer'].dtype)
        llama_vec_n = llama_vec_n.to(models['transformer'].dtype)
        clip_l_pooler = clip_l_pooler.to(models['transformer'].dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(models['transformer'].dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(models['transformer'].dtype)
        
        # 採樣
        progress.update(0, 'Start sampling ...')
        
        rnd = torch.Generator("cpu").manual_seed(params['seed'])
        num_frames = params['latent_window_size'] * 4 - 3
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        
        # 將迭代器轉換為列表
        latent_paddings = list(reversed(range(total_latent_sections)))
        
        if total_latent_sections > 4:
            # 理論上latent_paddings應該按照上面的序列，但當total_latent_sections > 4時
            # 複製一些項目看起來比擴展它更好
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * params['latent_window_size']
            
            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')
            
            indices = torch.arange(0, sum([1, latent_padding_size, params['latent_window_size'], 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, params['latent_window_size'], 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # 如果提供了結束幀，則在第一個區段使用結束幀潛變量
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            if not _high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=params['gpu_memory_preservation'])
            
            if params['use_teacache']:
                models['transformer'].initialize_teacache(enable_teacache=True, num_steps=params['steps'])
            else:
                models['transformer'].initialize_teacache(enable_teacache=False)
            
            # 進度回調函數
            def callback(d):
                preview = d['denoised']
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / params['steps'])
                hint = f'Sampling {current_step}/{params["steps"]}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                message = f"{hint} - {desc}"
                
                # 更新內部進度
                progress.update(percentage, message)
                
                # 如果有外部回調函數，也調用它
                if progress_callback:
                    progress_callback(percentage, message)
                
                return
            
            # 執行採樣
            generated_latents = sample_hunyuan(
                transformer=models['transformer'],
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=params['cfg'],
                distilled_guidance_scale=params['gs'],
                guidance_rescale=params['rs'],
                num_inference_steps=params['steps'],
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            
            progress.done()  # 換行
            
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            if not _high_vram:
                offload_model_from_device_for_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(models['vae'], target_device=gpu)
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            
            print(f"解碼潛變量為像素...")
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models['vae']).cpu()
            else:
                section_latent_frames = (params['latent_window_size'] * 2 + 1) if is_last_section else (params['latent_window_size'] * 2)
                overlapped_frames = params['latent_window_size'] * 4 - 3
                
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], models['vae']).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            
            if not _high_vram:
                unload_complete_models()
            
            # 確定輸出文件名
            if params['output']:
                output_filename = params['output']
            else:
                # 如果沒有指定輸出路徑，使用臨時文件
                import tempfile
                output_filename = os.path.join(tempfile.gettempdir(), f'{job_id}.mp4')
            
            # 預覽模式：在第一個section（對應最後時間段）完成後立即輸出最後一幀
            if params['preview_last_only'] and is_first_section:
                # 直接輸出最後一幀為圖片
                try:
                    frame = history_pixels[0, :, -1, :, :]  # C,H,W
                    frame = frame.clamp(-1, 1)
                    frame = (frame + 1.0) / 2.0  # [0,1]
                    frame = (frame * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(frame)
                    if params['preview_image_output']:
                        img_out = params['preview_image_output']
                    else:
                        # 使用臨時目錄作為預覽圖片輸出
                        import tempfile
                        img_out = os.path.join(tempfile.gettempdir(), f'{job_id}_preview.png')
                    img.save(img_out)
                    print(f"預覽模式：在第一個section完成後立即輸出最後一幀到 {img_out}")
                    
                    # 單幀生成完成後釋放顯存
                    unload_framepack()
                    
                    return img_out
                except Exception as e:
                    print(f"保存預覽單幀失敗: {e}")
                    # 回退到保存視頻

            print(f"保存視頻到 {output_filename}...")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=params['mp4_crf'])

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            if is_last_section:
                print(f"視頻生成完成！")
                
                # 視頻生成完成後釋放顯存
                unload_framepack()
                
                return output_filename
        
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")
        traceback.print_exc()
        
        if not _high_vram:
            try:
                unload_complete_models(
                    models['text_encoder'], models['text_encoder_2'], models['image_encoder'], models['vae'], models['transformer']
                )
            except:
                pass
        
        # 異常情況下也嘗試釋放顯存
        try:
            unload_framepack()
        except:
            pass
        
        return None

# 命令行入口點
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FramePack - 根據起始幀和結束幀生成影片')
    parser.add_argument('--start_image', type=str, required=True, help='起始幀圖片路徑')
    parser.add_argument('--end_image', type=str, help='結束幀圖片路徑 (可選)')
    parser.add_argument('--prompt', type=str, default='Character movements based on storyboard sequence', help='提示詞')
    parser.add_argument('--n_prompt', type=str, default='', help='負面提示詞')
    parser.add_argument('--seed', type=int, default=31337, help='隨機種子')
    parser.add_argument('--total_second_length', type=float, default=5, help='影片長度(秒)')
    parser.add_argument('--latent_window_size', type=int, default=9, help='潛在窗口大小')
    parser.add_argument('--steps', type=int, default=25, help='採樣步數')
    parser.add_argument('--cfg', type=float, default=1.0, help='CFG尺度')
    parser.add_argument('--gs', type=float, default=10.0, help='蒸餾CFG尺度')
    parser.add_argument('--rs', type=float, default=0.0, help='CFG重縮放')
    parser.add_argument('--gpu_memory_preservation', type=float, default=6, help='GPU推理保留內存(GB)')
    parser.add_argument('--use_teacache', action='store_true', help='使用TeaCache')
    parser.add_argument('--mp4_crf', type=int, default=16, help='MP4壓縮質量')
    parser.add_argument('--output', type=str, help='輸出文件路徑')
    parser.add_argument('--headless', action='store_true', help='無頭模式，用於伺服器')
    parser.add_argument('--share', action='store_true', help='分享Gradio界面（僅用於兼容性）')
    parser.add_argument('--server', type=str, default='0.0.0.0', help='伺服器地址（僅用於兼容性）')
    parser.add_argument('--port', type=int, help='伺服器端口（僅用於兼容性）')
    parser.add_argument('--inbrowser', action='store_true', help='在瀏覽器中打開（僅用於兼容性）')
    
    args = parser.parse_args()
    
    # 檢查輸入文件是否存在
    if not os.path.exists(args.start_image):
        print(f"錯誤: 起始幀圖片不存在: {args.start_image}")
        sys.exit(1)
    
    if args.end_image and not os.path.exists(args.end_image):
        print(f"錯誤: 結束幀圖片不存在: {args.end_image}")
        sys.exit(1)
    
    print(f"FramePack 命令行版本")
    print(f"起始幀: {args.start_image}")
    print(f"結束幀: {args.end_image if args.end_image else '無'}")
    print(f"提示詞: {args.prompt}")
    print(f"影片長度: {args.total_second_length}秒")
    
    # 執行視頻處理
    output_file = process_video(
        args.start_image,
        args.end_image,
        prompt=args.prompt,
        n_prompt=args.n_prompt,
        seed=args.seed,
        total_second_length=args.total_second_length,
        latent_window_size=args.latent_window_size,
        steps=args.steps,
        cfg=args.cfg,
        gs=args.gs,
        rs=args.rs,
        gpu_memory_preservation=args.gpu_memory_preservation,
        use_teacache=args.use_teacache,
        mp4_crf=args.mp4_crf,
        output=args.output
    )
    
    if output_file:
        print(f"視頻生成成功: {output_file}")
    else:
        print("視頻生成失敗") 