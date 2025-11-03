"""
測試程序：比較 SAM 和 DIS-SAM 的 mask 生成結果
- 讀取硬編碼的圖片和 prompt
- 使用 Qwen 提取 bbox
- 分別使用 SAM 和 DIS-SAM 生成 mask
- 將 mask 貼回原圖後輸出兩張圖
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

# 確保可以導入模組
# test_sam_mask_comparison.py 在 test/ 目錄下，需要導入父目錄（Storyboard/）中的模組
current_dir = Path(__file__).resolve().parent  # test/ 目錄
STORYBOARD_ROOT = current_dir.parent  # Storyboard/ 根目錄（所有模型和資源的根目錄）

# ========== 設置所有模型的根目錄為 Storyboard/ ==========
# 設置 HuggingFace 下載目錄到 Storyboard/hf_download
os.environ['HF_HOME'] = str(STORYBOARD_ROOT / 'hf_download')
os.environ['HF_HUB_CACHE'] = str(STORYBOARD_ROOT / 'hf_download' / 'hub')
os.environ['TRANSFORMERS_CACHE'] = str(STORYBOARD_ROOT / 'hf_download' / 'transformers')

# 設置 checkpoints 目錄的環境變數（如果有需要的話）
# 注意：dis_sam_inference.py 和 sam_inference.py 會使用 Path(__file__).parent / "checkpoints"
# 這會自動指向 Storyboard/checkpoints（因為它們在 Storyboard/ 目錄下）

# 修改當前工作目錄到 Storyboard/，這樣所有相對路徑都會基於 Storyboard/
# 這確保 Qwen_inference.py 中的 os.getcwd() 會指向 Storyboard/
original_cwd = os.getcwd()
os.chdir(str(STORYBOARD_ROOT))

# 添加父目錄到 sys.path，這樣可以導入 Storyboard/ 下的模組（如 dis_sam_inference.py, isnet.py）
if str(STORYBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(STORYBOARD_ROOT))
# 也添加當前目錄（向後兼容）
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print(f"[配置] Storyboard 根目錄: {STORYBOARD_ROOT}")
print(f"[配置] HuggingFace 緩存: {os.environ.get('HF_HOME')}")
print(f"[配置] Checkpoints 目錄: {STORYBOARD_ROOT / 'checkpoints'}")

try:
    from Qwen_inference import extract_remove_bounding_boxes, unload_qwen
except ImportError:
    print("錯誤：無法導入 Qwen_inference 模組")
    sys.exit(1)

try:
    from sam_inference import generate_masks_with_sam
except ImportError:
    print("錯誤：無法導入 sam_inference 模組")
    sys.exit(1)

try:
    from dis_sam_inference import generate_masks_with_dis_sam
except ImportError:
    print("錯誤：無法導入 dis_sam_inference 模組")
    sys.exit(1)


# ========== 硬編碼的測試數據（可選） ==========
# 如果沒有通過命令行參數提供，將使用這些默認值
DEFAULT_IMAGE_PATH = "test_image.jpg"  # 請修改為實際的圖片路徑
DEFAULT_PROMPT = "remove the soccer net"  # 測試 prompt


def overlay_mask_on_image(
    image_path: str,
    mask_path: str,
    output_path: str,
    mask_color: tuple = (255, 0, 0),  # 紅色，RGBA
    opacity: float = 0.5
) -> str:
    """
    將 mask 疊加到原圖上並保存
    
    Args:
        image_path: 原圖路徑
        mask_path: mask 圖片路徑（白色=前景，黑色=背景）
        output_path: 輸出圖片路徑
        mask_color: mask 的顏色（RGB）
        opacity: mask 的透明度（0-1）
    
    Returns:
        輸出圖片路徑
    """
    # 讀取原圖
    original_img = Image.open(image_path).convert("RGB")
    
    # 讀取 mask（應該是單通道的）
    mask_img = Image.open(mask_path).convert("L")  # 轉為灰度圖
    
    # 將 mask 轉為 numpy array 進行處理
    mask_array = np.array(mask_img)
    
    # 將 mask 轉換為 RGBA，其中白色區域為前景
    # mask_array 中值為 255 的地方是前景（白色），0 是背景（黑色）
    mask_binary = (mask_array > 128).astype(np.uint8)  # 二值化
    
    # 創建彩色 overlay
    overlay = Image.new("RGBA", original_img.size, (0, 0, 0, 0))
    
    # 在 mask 區域繪製顏色
    overlay_array = np.zeros((original_img.height, original_img.width, 4), dtype=np.uint8)
    overlay_array[mask_binary > 0] = [mask_color[0], mask_color[1], mask_color[2], int(255 * opacity)]
    
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    
    # 將 overlay 疊加到原圖
    result = Image.alpha_composite(original_img.convert("RGBA"), overlay)
    
    # 保存結果
    result = result.convert("RGB")
    result.save(output_path)
    
    print(f"已保存疊加圖片: {output_path}")
    return output_path


def draw_bboxes_on_image(
    image_path: str,
    boxes: list,
    output_path: str,
    bbox_color: tuple = (0, 255, 0),  # 綠色
    line_width: int = 3
) -> str:
    """
    在原圖上繪製 bounding boxes
    
    Args:
        image_path: 原圖路徑
        boxes: bbox 列表，格式為 [{"label": str, "box_xyxy": [x0,y0,x1,y1]}, ...]
        output_path: 輸出圖片路徑
        bbox_color: bbox 線條顏色（RGB）
        line_width: 線條寬度
    
    Returns:
        輸出圖片路徑
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    for box_info in boxes:
        box_xyxy = box_info.get("box_xyxy", [])
        label = box_info.get("label", "")
        
        if len(box_xyxy) == 4:
            x0, y0, x1, y1 = [int(v) for v in box_xyxy]
            # 繪製矩形
            draw.rectangle([x0, y0, x1, y1], outline=bbox_color, width=line_width)
            # 繪製標籤
            if label:
                draw.text((x0, y0 - 15), label, fill=bbox_color)
    
    img.save(output_path)
    print(f"已保存 bbox 標註圖片: {output_path}")
    return output_path


def main():
    """主測試流程"""
    # 解析命令行參數
    parser = argparse.ArgumentParser(
        description="測試程序：比較 SAM 和 DIS-SAM 的 mask 生成結果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_sam_mask_comparison.py --image path/to/image.jpg --prompt "remove the person"
  python test_sam_mask_comparison.py --image test.jpg --prompt "remove the car"
        """
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help=f"測試圖片路徑（默認: {DEFAULT_IMAGE_PATH}）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"測試 prompt（默認: {DEFAULT_PROMPT}）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,  # 默認使用 test/ 目錄下的 test_mask_outputs
        help="輸出目錄（默認: test/test_mask_outputs）"
    )
    
    args = parser.parse_args()
    
    TEST_IMAGE_PATH = args.image
    TEST_PROMPT = args.prompt
    
    print("=" * 60)
    print("SAM 與 DIS-SAM Mask 生成對比測試")
    print("=" * 60)
    
    # 檢查測試圖片是否存在（相對於 Storyboard 根目錄）
    # 如果圖片路徑是相對路徑，先嘗試相對於當前工作目錄（Storyboard/），然後嘗試相對於 test/ 目錄
    if not os.path.exists(TEST_IMAGE_PATH):
        # 嘗試相對於 test/ 目錄的路徑
        test_image_path = current_dir / TEST_IMAGE_PATH
        if test_image_path.exists():
            TEST_IMAGE_PATH = str(test_image_path)
        else:
            print(f"錯誤：找不到測試圖片: {TEST_IMAGE_PATH}")
            print(f"已嘗試路徑: {TEST_IMAGE_PATH} 和 {test_image_path}")
            print("請使用 --image 參數指定圖片路徑，或修改腳本中的 DEFAULT_IMAGE_PATH")
            os.chdir(original_cwd)  # 恢復工作目錄
            return
    
    print(f"\n測試圖片: {TEST_IMAGE_PATH}")
    print(f"測試 Prompt: {TEST_PROMPT}")
    
    # 創建輸出目錄（默認在 test/ 目錄下）
    if args.output_dir is None:
        # 默認輸出目錄在 test/ 目錄下的 test_mask_outputs
        output_dir = current_dir / "test_mask_outputs"
    else:
        # 如果提供了相對路徑，相對於 Storyboard/ 根目錄
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            # 相對路徑，基於 Storyboard/ 根目錄
            output_dir = STORYBOARD_ROOT / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n輸出目錄: {output_dir}")
    
    # ========== 步驟 1: 使用 Qwen 提取 bbox ==========
    print("\n[步驟 1] 使用 Qwen 提取 bounding boxes...")
    try:
        boxes_result = extract_remove_bounding_boxes(TEST_IMAGE_PATH, TEST_PROMPT)
        print(f"Qwen 返回結果: {json.dumps(boxes_result, indent=2, ensure_ascii=False)}")
        
        boxes = boxes_result.get("boxes", [])
        if not boxes:
            print("警告：Qwen 沒有返回任何 bounding box，將使用空的 mask")
        else:
            print(f"找到 {len(boxes)} 個 bounding box")
            # 繪製 bbox 到原圖
            bbox_output = output_dir / "original_with_bboxes.jpg"
            draw_bboxes_on_image(TEST_IMAGE_PATH, boxes, str(bbox_output))
        
        # 卸載 Qwen 模型
        try:
            unload_qwen()
            print("Qwen 模型已卸載")
        except Exception as e:
            print(f"卸載 Qwen 模型時發生警告: {e}")
        
    except Exception as e:
        print(f"錯誤：Qwen 提取 bbox 失敗: {e}")
        import traceback
        traceback.print_exc()
        os.chdir(original_cwd)  # 恢復工作目錄
        return
    
    # ========== 步驟 2: 使用 SAM 生成 mask ==========
    print("\n[步驟 2] 使用 SAM 生成 mask...")
    sam_mask_path = output_dir / "sam_mask.png"
    sam_overlay_path = output_dir / "sam_overlay.jpg"
    
    try:
        # 生成 SAM mask
        generate_masks_with_sam(
            image_path=TEST_IMAGE_PATH,
            boxes_json=boxes_result,  # 直接傳入 dict
            output_mask_path=str(sam_mask_path),
            model_type="vit_h",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"SAM mask 已生成: {sam_mask_path}")
        
        # 將 mask 疊加到原圖
        overlay_mask_on_image(
            TEST_IMAGE_PATH,
            str(sam_mask_path),
            str(sam_overlay_path),
            mask_color=(255, 0, 0),  # 紅色
            opacity=0.5
        )
        print(f"SAM 疊加圖片已保存: {sam_overlay_path}")
        
    except Exception as e:
        print(f"錯誤：SAM 生成 mask 失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 步驟 3: 使用 DIS-SAM 生成 mask ==========
    print("\n[步驟 3] 使用 DIS-SAM 生成 mask...")
    dis_sam_mask_path = output_dir / "dis_sam_mask.png"
    dis_sam_overlay_path = output_dir / "dis_sam_overlay.jpg"
    
    try:
        # 生成 DIS-SAM mask（兩階段：SAM + IS-Net 精煉）
        generate_masks_with_dis_sam(
            image_path=TEST_IMAGE_PATH,
            boxes_json=boxes_result,  # 直接傳入 dict
            output_mask_path=str(dis_sam_mask_path),
            sam_model_type="vit_h",  # 使用 vit_l（更快）或 vit_h（更準確）
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_refinement=True,  # 使用 IS-Net 精煉（完整 DIS-SAM）
            auto_download=True
        )
        print(f"DIS-SAM mask 已生成: {dis_sam_mask_path}")
        
        # 將 mask 疊加到原圖
        overlay_mask_on_image(
            TEST_IMAGE_PATH,
            str(dis_sam_mask_path),
            str(dis_sam_overlay_path),
            mask_color=(0, 0, 255),  # 藍色（與 SAM 區分）
            opacity=0.5
        )
        print(f"DIS-SAM 疊加圖片已保存: {dis_sam_overlay_path}")
        
    except Exception as e:
        print(f"錯誤：DIS-SAM 生成 mask 失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 總結 ==========
    print("\n" + "=" * 60)
    print("測試完成！輸出文件：")
    print("=" * 60)
    print(f"1. 原圖 + Bounding Boxes: {output_dir / 'original_with_bboxes.jpg'}")
    print(f"2. SAM Mask: {sam_mask_path}")
    print(f"3. SAM 疊加圖片: {sam_overlay_path}")
    print(f"4. DIS-SAM Mask: {dis_sam_mask_path}")
    print(f"5. DIS-SAM 疊加圖片: {dis_sam_overlay_path}")
    print("=" * 60)
    
    # 恢復原始工作目錄
    os.chdir(original_cwd)


if __name__ == "__main__":
    main()

