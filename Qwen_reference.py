from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import os
import torch
import json
from PIL import Image

# 設定 HuggingFace cache 下載目錄 (hf_download) 於當前工作目錄
cache_dir = os.path.join(os.getcwd(), "hf_download")
os.makedirs(cache_dir, exist_ok=True)

# 全域快取：避免每次呼叫都重新載入模型
_cached_model = None
_cached_processor = None

def get_qwen_model_and_processor():
    """
    取得並快取 Qwen2.5-VL 模型與 Processor。
    - 會使用 device_map="auto" 自動分配裝置
    - 使用當前工作目錄下的 hf_download 作為 cache_dir
    回傳: (model, processor)
    """
    global _cached_model, _cached_processor
    if _cached_model is not None and _cached_processor is not None:
        print("[Qwen] 使用快取的 model/processor")
        return _cached_model, _cached_processor

    print("[Qwen] 載入 Qwen2.5-VL-7B-Instruct 模型與 Processor ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir)
    print("[Qwen] 模型與 Processor 載入完成")

    _cached_model = model
    _cached_processor = processor
    return model, processor

def unload_qwen() -> None:
    """
    卸載 Qwen：將模型移回 CPU 並清除快取引用，避免佔用 GPU。
    """
    global _cached_model, _cached_processor
    try:
        model_ref = _cached_model
        proc_ref = _cached_processor
        # 只做一件事：把模型移回 CPU（忽略 Accelerate 警告）
        try:
            if model_ref is not None:
                model_ref.to('cpu')
                print("[Qwen] 已將模型移回 CPU")
        except Exception as e:
            print(f"[Qwen] 模型移回 CPU 失敗: {e}")
        _cached_model = None
        _cached_processor = None
        try:
            del model_ref
        except Exception:
            pass
        try:
            del proc_ref
        except Exception:
            pass
    except Exception:
        pass

    
## 注意：統一由 get_qwen_model_and_processor() 管理載入與快取；卸載策略交由呼叫端自行決定

def classify_image_edit_task(image: str, user_prompt: str, max_attempts: int = 5) -> int:
    """
    使用 Qwen2.5-VL 對 (image, prompt) 進行分類，回傳 0/1/2。
    - image: 檔案路徑或 URL
    - user_prompt: 文字提示（prompt）
    - max_attempts: 最多重試推理次數（當輸出不為單獨 0/1/2 時重試）
    回傳: int (0/1/2)
    """
    model, processor = get_qwen_model_and_processor()

    image_source = image if image is not None else "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    system_prompt = (
        "You are a specialist designed to distinguish photo-editing tasks.\n"
        "The user_prompt is the text that indicates the way to edit the image.\n"
        "Using the image content and the provided user_prompt to decide the task type.\n"
        "Categories (use the number only):\n"
        "0 — The text asks not to change the background and generate or change a \"single\" object that interacts with the original image.\n"
        "2 — Remove one or many objects from the original image, without making other modifications.\n"
        "otherwise, 1.\n"
        "if the user_prompt asks not to change the background,keeping the background same, or the same meaning, it must be 0, this is the most important rule !!!\n"
        "Strictly output a single number only: 0 or 1 or 2.\n"
        "example: 0"
    )

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_source,
                },
                {"type": "text", "text": "user_prompt: " + (user_prompt or "")},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    attempt = 0
    generated_ids = None
    generated_ids_trimmed = None
    output_text = None
    while attempt < max_attempts:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 僅接受單一數字 0/1/2
        if output_text and output_text[0] and output_text[0][0].isdigit():
            label = int(output_text[0][0])
            print("[Qwen] 分類完成，回傳 label:", label)
            if label in (0, 1, 2):
                return label
        attempt += 1

    # 若多次嘗試仍失敗，預設回傳 1（general case）
    return 1


def extract_remove_bounding_boxes(image: str, user_prompt: str, max_attempts: int = 3):
    """
    解析包含 "remove" 行為的 prompt，找出圖片中需要移除的物件並以 bounding box 輸出。
    回傳格式適配 SAM 的輸入：XYXY 像素座標。

    Returns (dict):
      {
        "image_size": [width, height],
        "format": "SAM_xyxy_pixel",
        "boxes": [ {"label": str, "box_xyxy": [x0,y0,x1,y1]} , ...]
      }
    """
    # 讀取圖片大小（像素）
    try:
        with Image.open(image) as im:
            width, height = im.size
    except Exception:
        width, height = 0, 0

    model, processor = get_qwen_model_and_processor()

    system_prompt = (
        "You are a vision assistant that returns bounding boxes for objects to remove.\n"
        "Task: From the user's prompt, extract all object categories that should be removed if present in the image.\n"
        "Look at the image and return bounding boxes for each matched object you find.\n"
        "Be highly sensitive to very small objects and tiny details; carefully inspect the entire image (background, corners, textures) to avoid missing any target.\n"
        "Prefer high recall: if uncertain, include a plausible bounding box (slightly larger is acceptable) rather than missing the object.\n"
        "Output strictly JSON with keys: image_size, format, boxes.\n"
        "- image_size: [width, height] in pixels.\n"
        "- format: the constant string 'SAM_xyxy_pixel'.\n"
        "- boxes: array of {label: string, box_xyxy: [x0,y0,x1,y1]} where coordinates are integer pixels,\n"
        "  0 <= x0 < x1 <= width, 0 <= y0 < y1 <= height.\n"
        "Do not include any extra text. Return JSON only."
    )

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"user_prompt: {user_prompt}"},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    attempt = 0
    result_obj = {
        "image_size": [width, height],
        "format": "SAM_xyxy_pixel",
        "boxes": []
    }

    while attempt < max_attempts:
        out_ids = model.generate(**inputs, max_new_tokens=256)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        text_out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw = (text_out[0] if text_out else "").strip()
        # 嘗試定位 JSON 區段
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw_json = raw[start:end+1]
            else:
                raw_json = raw
            # 印出 Qwen 回傳之 JSON 內容（可能包含多餘空白），方便在 terminal 觀察
            print("[Qwen][extract_remove_bounding_boxes] 模型回傳 JSON:", raw_json)
            parsed = json.loads(raw_json)
            # 印出解析後 JSON（標準化後），利於除錯
            try:
                print("[Qwen][extract_remove_bounding_boxes] 解析 JSON:", json.dumps(parsed, ensure_ascii=False))
            except Exception:
                pass
            # 正規化與邊界裁切
            boxes = []
            for item in parsed.get("boxes", []):
                label = str(item.get("label", "object"))
                box = item.get("box_xyxy", [])
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    continue
                x0, y0, x1, y1 = [int(round(float(v))) for v in box]
                if width > 0 and height > 0:
                    x0 = max(0, min(x0, width))
                    x1 = max(0, min(x1, width))
                    y0 = max(0, min(y0, height))
                    y1 = max(0, min(y1, height))
                if x1 > x0 and y1 > y0:
                    boxes.append({"label": label, "box_xyxy": [x0, y0, x1, y1]})
            result_obj["boxes"] = boxes
            result_obj["image_size"] = [width, height]
            return result_obj
        except Exception:
            attempt += 1

    # 失敗則回傳空盒
    return result_obj


if __name__ == "__main__":
    # 命令列模式：維持原先介面
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference with optional image path")
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default=None,
        help="影像路徑或URL (image path or URL). 若不指定，使用預設示例影像",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default="",
        help="使用者在命令列輸入的文字 (user prompt). 預設為空字串",
    )
    args = parser.parse_args()

    label = classify_image_edit_task(args.image, args.user_prompt)
    print("QwenClassification label:", label)

