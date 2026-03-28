from diffusers_helper.hf_login import login

import json
import os
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import math

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# 導入所需的庫
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# 模型緩存
_models = None
_high_vram = None

def init_models():
    """初始化模型"""
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

def process_storyboard_continuous(segments, output_dir, progress_callback=None):
    """
    真正的連續生成：所有片段在一個大循環裡完成
    
    核心設計：
    - 模型只初始化一次
    - history_latents 和 history_pixels 從頭到尾持續累積（不分段）
    - 在循環中動態更新：start_latent、end_latent、CLIP、prompt
    - latent_paddings 是所有片段的 padding 序列拼接
    
    Args:
        segments: 片段信息列表（按時間順序，倒序處理）
        output_dir: 輸出目錄
        progress_callback: 進度回調函數
    
    Returns:
        final.mp4 的路徑，失敗返回 None
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 初始化模型（只做一次）
    logger.info("初始化模型...")
    models = init_models()
    
    shared_params = {
        'n_prompt': '',
        'latent_window_size': 9,
        'steps': 25,
        'cfg': 1.0,
        'gs': 10.0,
        'rs': 0.0,
        'gpu_memory_preservation': 6,
        'use_teacache': True,
        'mp4_crf': 16,
    }
    
    # 決定 bucket size
    first_seg = segments[-1]  # 倒序處理
    img = Image.open(first_seg['start_image_path'])
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_np = np.array(img)
    H, W, _ = img_np.shape
    height, width = find_nearest_bucket(H, W, resolution=640)
    logger.info(f"使用 bucket 尺寸: {height}x{width}")
    
    try:
        # ===== 1. 預處理所有片段：計算總的 latent_paddings 序列 =====
        all_segment_info = []
        total_sections = 0
        
        for seg_idx, segment in enumerate(reversed(segments)):
            actual_seg_idx = len(segments) - 1 - seg_idx
            total_second_length = segment['total_second_length']
            seg_latent_sections = int(max(round((total_second_length * 30) / (shared_params['latent_window_size'] * 4)), 1))
            
            # 計算這個片段的 padding 序列
            seg_paddings = list(reversed(range(seg_latent_sections)))
            if seg_latent_sections > 4:
                seg_paddings = [3] + [2] * (seg_latent_sections - 3) + [1, 0]
            
            all_segment_info.append({
                'segment': segment,
                'actual_seg_idx': actual_seg_idx,
                'seg_idx': seg_idx,
                'paddings': seg_paddings,
                'start_section_idx': total_sections,
                'end_section_idx': total_sections + len(seg_paddings)
            })
            total_sections += len(seg_paddings)
            
            logger.info(f"片段 {actual_seg_idx}: {seg_latent_sections} latent sections, padding={seg_paddings}")
        
        # 合併所有 padding 序列
        all_paddings = []
        for info in all_segment_info:
            all_paddings.extend(info['paddings'])
        
        logger.info(f"總共 {total_sections} 個 sections，合併的 padding 序列: {all_paddings}")
        
        # ===== 2. 預編碼所有片段的圖片和 prompt =====
        segment_conditions = []
        for info in all_segment_info:
            segment = info['segment']
            actual_seg_idx = info['actual_seg_idx']
            
            logger.info(f"預編碼片段 {actual_seg_idx}...")
            
            # Text encoding
            if not _high_vram:
                fake_diffusers_current_device(models['text_encoder'], gpu)
                load_model_as_complete(models['text_encoder_2'], target_device=gpu)
            
            llama_vec, clip_l_pooler = encode_prompt_conds(
                segment['prompt'],
                models['text_encoder'],
                models['text_encoder_2'],
                models['tokenizer'],
                models['tokenizer_2']
            )
            
            if shared_params['cfg'] == 1:
                llama_vec_n = torch.zeros_like(llama_vec)
                clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                    shared_params['n_prompt'],
                    models['text_encoder'],
                    models['text_encoder_2'],
                    models['tokenizer'],
                    models['tokenizer_2']
                )
            
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            
            # 處理圖片
            input_image = Image.open(segment['start_image_path'])
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            input_image_np = np.array(input_image)
            input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            
            has_end_image = segment['end_image_path'] is not None
            end_image_np = None
            end_image_pt = None
            if has_end_image:
                end_image = Image.open(segment['end_image_path'])
                if end_image.mode != "RGB":
                    end_image = end_image.convert("RGB")
                end_image_np = np.array(end_image)
                end_image_np = resize_and_center_crop(end_image_np, target_width=width, target_height=height)
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
            
            # VAE encode
            if not _high_vram:
                load_model_as_complete(models['vae'], target_device=gpu)
            
            start_latent = vae_encode(input_image_pt, models['vae'])
            end_latent = vae_encode(end_image_pt, models['vae']) if has_end_image else None
            
            # CLIP encode
            if not _high_vram:
                load_model_as_complete(models['image_encoder'], target_device=gpu)
            
            image_encoder_output = hf_clip_vision_encode(input_image_np, models['feature_extractor'], models['image_encoder'])
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            
            if has_end_image:
                end_encoder_output = hf_clip_vision_encode(end_image_np, models['feature_extractor'], models['image_encoder'])
                end_encoder_last_hidden_state = end_encoder_output.last_hidden_state
                image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_encoder_last_hidden_state) / 2
            
            # 轉換類型
            llama_vec = llama_vec.to(models['transformer'].dtype)
            llama_vec_n = llama_vec_n.to(models['transformer'].dtype)
            clip_l_pooler = clip_l_pooler.to(models['transformer'].dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(models['transformer'].dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(models['transformer'].dtype)
            
            segment_conditions.append({
                'start_latent': start_latent,
                'end_latent': end_latent,
                'has_end_image': has_end_image,
                'llama_vec': llama_vec,
                'llama_vec_n': llama_vec_n,
                'llama_attention_mask': llama_attention_mask,
                'llama_attention_mask_n': llama_attention_mask_n,
                'clip_l_pooler': clip_l_pooler,
                'clip_l_pooler_n': clip_l_pooler_n,
                'image_encoder_last_hidden_state': image_encoder_last_hidden_state,
                'seed': segment['seed']
            })
        
        # ===== 3. 統一的大循環：生成所有 sections =====
        logger.info("開始統一的連續生成循環...")
        
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        num_frames = shared_params['latent_window_size'] * 4 - 3
        
        # 記錄每個片段在 history_pixels 中的起始幀位置（pixel 空間）
        segment_pixel_ranges = {}  # {actual_seg_idx: (start_frame, end_frame)}
        
        for section_idx, latent_padding in enumerate(all_paddings):
            # 確定當前屬於哪個片段
            current_seg_info = None
            for info in all_segment_info:
                if info['start_section_idx'] <= section_idx < info['end_section_idx']:
                    current_seg_info = info
                    break
            
            if current_seg_info is None:
                logger.error(f"section {section_idx} 找不到對應片段！")
                continue
            
            actual_seg_idx = current_seg_info['actual_seg_idx']
            seg_idx = current_seg_info['seg_idx']
            is_first_segment = (seg_idx == 0)
            is_first_section_of_segment = (section_idx == current_seg_info['start_section_idx'])
            is_last_section_of_segment = (section_idx == current_seg_info['end_section_idx'] - 1)
            is_last_section_overall = (section_idx == total_sections - 1)
            
            # 獲取當前片段的condition
            cond = segment_conditions[seg_idx]
            
            # 如果是片段的第一個 section，更新 generator seed
            if is_first_section_of_segment:
                rnd = torch.Generator("cpu").manual_seed(cond['seed'])
                logger.info(f"section {section_idx}/{total_sections}: 片段 {actual_seg_idx} 開始，使用 seed={cond['seed']}")
            
            latent_padding_size = latent_padding * shared_params['latent_window_size']
            
            indices = torch.arange(0, sum([1, latent_padding_size, shared_params['latent_window_size'], 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
                [1, latent_padding_size, shared_params['latent_window_size'], 1, 2, 16], dim=1
            )
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            
            clean_latents_pre = cond['start_latent'].to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # 只有第一段的第一 section 使用 end image
            if cond['has_end_image'] and is_first_section_of_segment and is_first_segment:
                clean_latents_post = cond['end_latent'].to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            if not _high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=shared_params['gpu_memory_preservation'])
            
            if shared_params['use_teacache']:
                models['transformer'].initialize_teacache(enable_teacache=True, num_steps=shared_params['steps'])
            else:
                models['transformer'].initialize_teacache(enable_teacache=False)
            
            # 進度回調
            def callback(d):
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / shared_params['steps'])
                section_in_segment = section_idx - current_seg_info['start_section_idx'] + 1
                total_in_segment = current_seg_info['end_section_idx'] - current_seg_info['start_section_idx']
                message = f"片段 {actual_seg_idx}, section {section_in_segment}/{total_in_segment}, step {current_step}/{shared_params['steps']}"
                if progress_callback:
                    progress_callback(actual_seg_idx, percentage, message)
                return
            
            # 執行採樣
            generated_latents = sample_hunyuan(
                transformer=models['transformer'],
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=shared_params['cfg'],
                distilled_guidance_scale=shared_params['gs'],
                guidance_rescale=shared_params['rs'],
                num_inference_steps=shared_params['steps'],
                generator=rnd,
                prompt_embeds=cond['llama_vec'],
                prompt_embeds_mask=cond['llama_attention_mask'],
                prompt_poolers=cond['clip_l_pooler'],
                negative_prompt_embeds=cond['llama_vec_n'],
                negative_prompt_embeds_mask=cond['llama_attention_mask_n'],
                negative_prompt_poolers=cond['clip_l_pooler_n'],
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=cond['image_encoder_last_hidden_state'],
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            
            if is_last_section_overall:
                generated_latents = torch.cat([cond['start_latent'].to(generated_latents), generated_latents], dim=2)
            
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            
            if not _high_vram:
                offload_model_from_device_for_memory_preservation(models['transformer'], target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(models['vae'], target_device=gpu)
            
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
            
            # 解碼與 overlap
            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, models['vae']).cpu()
                # 記錄第一個片段的起始位置
                if is_first_section_of_segment:
                    segment_pixel_ranges[actual_seg_idx] = [0, None]
            else:
                section_latent_frames = (shared_params['latent_window_size'] * 2 + 1) if is_last_section_overall else (shared_params['latent_window_size'] * 2)
                overlapped_frames = shared_params['latent_window_size'] * 4 - 3
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], models['vae']).cpu()
                
                # 記錄片段的起始位置（在第一個 section decode 後）
                if is_first_section_of_segment:
                    segment_pixel_ranges[actual_seg_idx] = [history_pixels.shape[2], None]
                
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            
            # 記錄片段的結束位置（在最後一個 section decode 後）
            if is_last_section_of_segment:
                if actual_seg_idx in segment_pixel_ranges:
                    segment_pixel_ranges[actual_seg_idx][1] = history_pixels.shape[2]
                    logger.info(f"片段 {actual_seg_idx} 像素範圍: {segment_pixel_ranges[actual_seg_idx]}")
                
                if progress_callback:
                    progress_callback(actual_seg_idx, 100, f"片段 {actual_seg_idx} 完成")
            
            if not _high_vram:
                unload_complete_models()
        
        # 4. 保存 final.mp4（完整連續生成的 history_pixels）
        logger.info("保存 final.mp4...")
        if progress_callback:
            progress_callback(-1, 50, "保存 final.mp4...")
        
        final_path = os.path.join(output_dir, "final.mp4")
        logger.info(f"final.mp4 形狀: {history_pixels.shape}，總幀數: {history_pixels.shape[2]}")
        save_bcthw_as_mp4(history_pixels, final_path, fps=30, crf=shared_params['mp4_crf'])
        
        # 保存邊界 frame indices，供實驗比對 concat vs final 時精確對應同一語意邊界
        boundaries_path = os.path.join(output_dir, "final_boundaries.json")
        boundary_frames = []
        for i in range(len(all_segment_info) - 1):
            if i + 1 in segment_pixel_ranges:
                start_next = segment_pixel_ranges[i + 1][0]
                boundary_frames.append(start_next)  # segment i|i+1 的邊界 = segment i+1 的起始幀
        if boundary_frames:
            with open(boundaries_path, "w", encoding="utf-8") as f:
                json.dump({"boundary_frames": boundary_frames, "total_frames": int(history_pixels.shape[2])}, f, indent=2)
            logger.info(f"final_boundaries.json 已保存: {boundary_frames}")
        
        logger.info(f"final.mp4 已保存到 {final_path}")
        
        if progress_callback:
            progress_callback(-1, 100, "全部完成")
        
        # 釋放模型
        unload_framepack()
        
        return final_path
        
    except Exception as e:
        logger.error(f"連續生成過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        
        try:
            unload_framepack()
        except:
            pass
        
        return None