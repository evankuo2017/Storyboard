import os
import argparse
import glob
import torch
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side
from PIL import Image
import numpy as np



def infer_on_two_images(
    sample_image_path: str,
    mask_image_path: str,
    output_path: str | None = None,
    *,
    use_fp16: bool = False,
    steps: int = 20,
    guidance_scale: float = 2.5,
    seed: int = 42,
    cache_dir: str | None = None,
    device: torch.device | None = None,
    pipe: ObjectClearPipeline | None = None,
):
    """對單張 sample 與 mask 做推理並輸出結果。
    - sample_image_path: 來源圖片路徑
    - mask_image_path: 掩膜圖片路徑（白=移除區域，或依模型定義）
    - output_path: 輸出圖片路徑（若為 None，將輸出到 results 目錄）
    其餘參數對應 CLI 旗標。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch_dtype = torch.float16 if use_fp16 else torch.float32
    variant = "fp16" if use_fp16 else None
    generator = torch.Generator(device=device).manual_seed(seed)

    # 允許外部傳入 pipe 以避免重複載入
    if pipe is None:
        pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            cache_dir=cache_dir,
            variant=variant,
        )
        pipe.to(device)

    image = Image.open(sample_image_path).convert("RGB")
    mask = Image.open(mask_image_path).convert("L")
    image_or = image.copy()

    image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
    mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)

    w, h = image.size

    result = pipe(
        prompt="remove the instance of object",
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=h,
        width=w,
        return_attn_map=False,
    )

    fused_img_pil = result.images[0]

    if output_path is None:
        # 預設輸出到當前工作目錄
        basename = os.path.splitext(os.path.basename(sample_image_path))[0]
        output_path = os.path.join(os.getcwd(), f"{basename}.png")

    fused_img_pil = fused_img_pil.resize(image_or.size)
    fused_img_pil.save(output_path)
    return output_path


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    # 新增：直接指定單張 sample 與 mask 的模式
    parser.add_argument('--sample_image', type=str, default='./sample.jpg',
                        help='Path to single sample image (default: ./sample.jpg). If set, runs single-image inference.')
    parser.add_argument('--mask_image', type=str, default='./mask.png',
                        help='Path to single mask image (default: ./mask.png). Used with --sample_image.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path for single-image mode.')

    parser.add_argument('-i', '--input_path', type=str, default='.', 
                        help='Input image or folder. Default: inputs/imgs')
    parser.add_argument('-m', '--mask_path', type=str, default='.',
                        help='Input mask image or folder. Default: inputs/masks')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                        help='Output folder. Default: results/<input_name>')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Path to cache directory")
    parser.add_argument('--use_fp16', action='store_true', 
                        help='Use float16 for inference')
    parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for torch.Generator. Default: 42')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of diffusion inference steps. Default: 20')
    parser.add_argument('--guidance_scale', type=float, default=2.5, 
                        help='CFG guidance scale. Default: 2.5')
    parser.add_argument('--no_agf', action='store_true', 
                        help='Disable Attention Guided Fusion')
    args = parser.parse_args()
    
    # 單張圖片模式優先（若提供 --sample_image 與 --mask_image）
    # 預設以單張模式（若預設檔存在）
    if args.sample_image and args.mask_image and os.path.exists(args.sample_image) and os.path.exists(args.mask_image):
        out_path = infer_on_two_images(
            args.sample_image,
            args.mask_image,
            output_path=args.output,
            use_fp16=args.use_fp16,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            cache_dir=args.cache_dir,
            device=device,
        )
        print(f'Output saved to: {out_path}')
    else:
        
        # ------------------------ input & output ------------------------
        if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            input_img_list = [args.input_path]
            result_root = os.getcwd()
        else: # input img folder
            # scan all the jpg and png images
            input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
            result_root = os.getcwd()
            
        if args.mask_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single mask path
            input_mask_list = [args.mask_path]
        else: # input mask folder
            # scan all the jpg and png masks
            input_mask_list = sorted(glob.glob(os.path.join(args.mask_path, '*.[jpJP][pnPN]*[gG]')))
            
        if len(input_img_list) != len(input_mask_list):
            raise ValueError(f"Mismatch between input images ({len(input_img_list)}) and masks ({len(input_mask_list)}).")

        if not args.output_path is None: # set output path
            result_root = args.output_path
            
        os.makedirs(result_root, exist_ok=True)

        test_img_num = len(input_img_list)
        
        
        # ------------------ set up ObjectClear pipeline -------------------
        torch_dtype = torch.float16 if args.use_fp16 else torch.float32
        variant = "fp16" if args.use_fp16 else None
        generator = torch.Generator(device=device).manual_seed(args.seed)
        use_agf = not args.no_agf
        pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=use_agf,
            cache_dir=args.cache_dir,
            variant=variant,
        )
        pipe.to(device)

        
        # -------------------- start to processing ---------------------
        for i, (img_path, mask_path) in enumerate(zip(input_img_list, input_mask_list)):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            # 使用統一函式進行推理與輸出
            save_path = os.path.join(result_root, f'{basename}.png')
            infer_on_two_images(
                img_path,
                mask_path,
                output_path=save_path,
                use_fp16=args.use_fp16,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                cache_dir=args.cache_dir,
                device=device,
                pipe=pipe,
            )

        print(f'\nAll results are saved in {result_root}')