from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import os
import torch
import json
import re
import tempfile
from PIL import Image, ImageDraw, ImageFont

# 設定 HuggingFace cache 下載目錄 (hf_download) 於當前工作目錄
cache_dir = os.path.join(os.getcwd(), "hf_download/hub")
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
    
    # 使用更明確的設備設置，避免 device_map="auto" 導致的設備不匹配
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map=device, 
        cache_dir=cache_dir
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
        
        # 強制清理 GPU 顯存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[Qwen] 已清理 GPU 顯存")
            
    except Exception:
        pass

    
## 注意：統一由 get_qwen_model_and_processor() 管理載入與快取；卸載策略交由呼叫端自行決定


_FREEFORM_BBOX_PATTERN = re.compile(
    r"\s*(?P<label>[^:\n]+):\s*\[(?P<coords>[^\]]+)\]",
    re.MULTILINE,
)


class SecondStageParseError(Exception):
    """第二階段 refinement 輸出無法解析時拋出（Second-stage refinement parse error）。"""
    pass


def classify_with_pal(image: str, user_prompt: str, max_attempts: int = 5) -> int:
    """
    使用 Qwen2.5-VL 對 (image, prompt) 進行分類，回傳 0/1/2。
    採用 PAL (Program-Aided Language Models) 風格的混合方法：
    - 第一階段由模型統計各類任務數量
    - 第二階段由程式依規則決策最終 label
    """
    model, processor = get_qwen_model_and_processor()
    
    image_source = image if image is not None else "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    system_prompt = (
        "You are an image editing task analyzer.\n"
        "Your job is to identify and count different types of editing actions in the user_prompt.\n\n"
        
        "Classify each action into one of these three categories:\n"
        "1. zoom_out_expand: Actions related to zooming out, expanding, or outpainting the image\n"
        "   - IMPORTANT: Recognize ALL synonyms and similar meanings:\n"
        "   - Keywords: zoom out, expand, extend, enlarge, outpaint, show more, make wider, make bigger, "
        "enlarge canvas, widen, broaden, increase canvas size\n"
        "   - ANY phrase meaning 'show more of the image' or 'make the image area larger' counts as zoom_out_expand\n\n"
        "2. remove: Actions that remove or delete objects from the image\n"
        "   - IMPORTANT: Recognize ALL synonyms and similar meanings:\n"
        "   - Keywords: remove, delete, erase, clear, eliminate, get rid of, take out, clean up, "
        "take away, dispose of, strip out, extract\n"
        "   - ANY phrase meaning 'make something disappear from the image' counts as remove\n\n"
        "3. other: All other actions (generate, add, create, modify, change, replace, etc.)\n"
        "   - Keywords: add, generate, create, make, change, modify, replace, switch, swap, exchange, "
        "enhance, adjust, convert, transform, crop, etc.\n\n"
        
        "Count how many actions fall into each category.\n"
        "Output strictly in this format:\n"
        "zoom_out_expand: [number]\n"
        "remove: [number]\n"
        "other: [number]"
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
                {
                    "type": "text", 
                    "text": (
                        "Analyze the user_prompt and count the actions in each category.\n\n"
                        "REMEMBER: Recognize all synonyms!\n"
                        "- zoom_out_expand includes: zoom out, expand, extend, enlarge, show more, etc.\n"
                        "- remove includes: remove, delete, erase, clear, eliminate, get rid of, take out, clean up, etc.\n\n"
                        "Output format:\n"
                        "zoom_out_expand: [number]\n"
                        "remove: [number]\n"
                        "other: [number]\n\n"
                        f"The user_prompt is: {user_prompt or ''}"
                    )
                },
            ],
        },
    ]
    
    print("[Qwen PAL] === 階段 1：語義理解（統計任務類型）===")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    attempt = 0
    zoom_out_expand = 0
    remove = 0
    other = 0
    
    while attempt < max_attempts:
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[Qwen PAL] 模型輸出（第 {attempt + 1} 次嘗試）：\n{output_text}\n")
        
        try:
            zoom_match = re.search(r'zoom_out_expand[:\s]+(\d+)', output_text, re.IGNORECASE)
            remove_match = re.search(r'remove[:\s]+(\d+)', output_text, re.IGNORECASE)
            other_match = re.search(r'other[:\s]+(\d+)', output_text, re.IGNORECASE)
            
            if zoom_match:
                zoom_out_expand = int(zoom_match.group(1))
            if remove_match:
                remove = int(remove_match.group(1))
            if other_match:
                other = int(other_match.group(1))
            
            print(f"[Qwen PAL] 階段 1 完成：zoom_out_expand={zoom_out_expand}, remove={remove}, other={other}")
            break
        except Exception as e:
            print(f"[Qwen PAL] 解析失敗（第 {attempt + 1} 次嘗試）: {e}")
            attempt += 1
    
    if attempt >= max_attempts:
        print("[Qwen PAL] 模型輸出解析失敗，使用關鍵字檢測備選方案...")
        user_prompt_lower = user_prompt.lower() if user_prompt else ""
        
        zoom_out_expand = 1 if any(kw in user_prompt_lower for kw in ['zoom out', 'expand', 'outpaint']) else 0
        remove = 1 if any(kw in user_prompt_lower for kw in ['remove', 'delete', 'erase']) else 0
        other = 1 if any(kw in user_prompt_lower for kw in ['add', 'generate', 'create', 'make', 'change', 'modify', 'replace']) else 0
        
        print(f"[Qwen PAL] 備選統計：zoom_out_expand={zoom_out_expand}, remove={remove}, other={other}")
    
    print("[Qwen PAL] === 階段 2：邏輯判斷（程式執行規則）===")
    
    if zoom_out_expand > 0:
        label = 0
        reason = f"zoom_out_expand={zoom_out_expand} > 0，根據最高優先級規則，回傳 Label 0"
    elif remove > 0 and other == 0:
        label = 2
        reason = f"remove={remove} > 0 且 other={other} == 0，這是純移除任務，回傳 Label 2"
    else:
        label = 1
        reason = f"remove={remove}, other={other}，這是混合或其他操作，回傳 Label 1"
    
    print(f"[Qwen PAL] 程式判斷邏輯：{reason}")
    print(f"[Qwen PAL] === 分類完成，最終 label: {label} ===\n")
    
    return label


def classify_image_edit_task(image: str, user_prompt: str, max_attempts: int = 5) -> int:
    """
    預設分類函數，使用 PAL 方法（與 `Qwen_Prompt_Engineer_inference.py` 對齊）。
    實際調用 `classify_with_pal()`，保留原有介面以向後相容。
    """
    return classify_with_pal(image, user_prompt, max_attempts)


def analyze_zoom_out_ratio(image: str, user_prompt: str, max_attempts: int = 5) -> int:
    """
    分析擴圖任務的縮圖比例建議。
    當任務為「將原圖縮小置中後進行擴圖」時，根據圖片內容與 prompt 判斷最適當的縮圖比例。
    
    Args:
        image: 圖片檔案路徑或 URL
        user_prompt: 文字提示（prompt）
        max_attempts: 最多重試推理次數
    
    Returns:
        int: 建議的縮圖比例（10-50之間的整數，表示百分比）
             例如：回傳 30 代表建議將原圖縮小至 30% 後置中，然後擴圖填充周圍
    """
    model, processor = get_qwen_model_and_processor()

    system_prompt = (
        "You are a task analyst specialized in image outpainting operations.\n"
        "The user wants to zoom out / expand the image, or do outpainting task. This means:\n"
        "1. The original image will be shrunk to a certain percentage\n"
        "2. The shrunk image will be placed in the center\n"
        "3. AI will generate/expand content around the shrunk image to fill the canvas\n\n"
        
        "Your task: Analyze the image content and the user_prompt, then suggest the most appropriate shrink ratio.\n"
        "Consider these factors:\n"
        "- If the user wants to show much more surrounding context or add large new areas, suggest larger ratio (35-50%)\n"
        "- If the user wants moderate expansion to show some additional context, suggest medium ratio (25-35%)\n"
        "- If the user only wants slight expansion or minor additions, suggest smaller ratio (10-25%)\n"
        "- Consider the composition: images with important central subjects may need higher ratios to preserve details\n"
        "- Consider the complexity of generation: more complex new content may benefit from having more space (smaller ratio)\n\n"
        
        "Output strictly a single integer number between 10 and 50 (inclusive).\n"
        "Do not include '%' symbol or any other text. Just the number.\n"
        "Examples:\n"
        "- If user wants 'zoom out and show the full building', output might be: 25\n"
        "- If user wants 'expand slightly to show more sky', output might be: 10\n"
        "- If user wants 'zoom out a lot and add surrounding landscape', output might be: 45\n"
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
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": f"Analyze this zoom-out/expand task. user_prompt: {user_prompt}"
                },
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
    
    # 確保所有輸入張量都在正確的設備上
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    attempt = 0
    while attempt < max_attempts:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 解析輸出，嘗試提取數字
        if output_text and output_text[0]:
            raw_output = output_text[0].strip()
            print(f"[Qwen][analyze_zoom_out_ratio] 第 {attempt + 1} 次嘗試，模型回傳: {raw_output}")
            
            # 提取數字
            numbers = re.findall(r'\d+', raw_output)
            if numbers:
                ratio = int(numbers[0])
                # 驗證範圍
                if 10 <= ratio <= 50:
                    print(f"[Qwen][analyze_zoom_out_ratio] 分析完成，建議縮圖比例: {ratio}%")
                    return ratio
        
        attempt += 1

    # 若多次嘗試仍失敗，預設回傳 30%（中等縮放比例）
    print("[Qwen][analyze_zoom_out_ratio] 多次嘗試失敗，使用預設值: 30%")
    return 30


def _parse_json_with_fallback(raw_text: str):
    """
    嘗試解析 JSON 字串；若遇到多個獨立 JSON 物件（缺少 list 包裹），則自動包成陣列再解析。
    """
    raw_text = (raw_text or "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            wrapped = raw_text if raw_text.startswith('[') else f"[{raw_text}]"
            return json.loads(wrapped)
        except json.JSONDecodeError:
            raise


def _extract_boxes_from_parsed_result(parsed_result):
    """
    從多種可能的模型輸出格式中抽取出 boxes 陣列，並統一為 {label, box_xyxy} 結構。
    支援：
    - {"boxes": [{"label": "...", "box_xyxy": [...]}, ...]}
    - [{"label": "...", "box_xyxy": [...]}, ...]
    - [{"label": "...", "bbox_2d": [...]}, ...]
    - {"label": "...", "bbox_2d": [...]}
    """

    def normalize_entry(entry):
        if not isinstance(entry, dict):
            return None
        label = str(entry.get("label", "object"))
        coords = entry.get("box_xyxy")
        if coords is None:
            coords = (
                entry.get("bbox_2d")
                or entry.get("bbox_xyxy")
                or entry.get("bbox")
            )
        if not (isinstance(coords, (list, tuple)) and len(coords) == 4):
            return None
        normalized_coords = [int(round(float(v))) for v in coords]
        return {"label": label, "box_xyxy": normalized_coords}

    boxes = []
    if isinstance(parsed_result, dict):
        if isinstance(parsed_result.get("boxes"), (list, tuple)):
            for entry in parsed_result["boxes"]:
                normalized = normalize_entry(entry)
                if normalized:
                    boxes.append(normalized)
            return boxes
        normalized = normalize_entry(parsed_result)
        if normalized:
            boxes.append(normalized)
        return boxes

    if isinstance(parsed_result, list):
        for entry in parsed_result:
            normalized = normalize_entry(entry)
            if normalized:
                boxes.append(normalized)
        return boxes

    return boxes


def expand_bbox(box_xyxy, image_size, scale=0.10):
    """
    將 bbox 的寬高增加一定比例（預設 10%），並確保仍在圖片範圍內。
    """
    x0, y0, x1, y1 = box_xyxy
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    expand_w = width * scale / 2
    expand_h = height * scale / 2

    new_x0 = int(round(x0 - expand_w))
    new_y0 = int(round(y0 - expand_h))
    new_x1 = int(round(x1 + expand_w))
    new_y1 = int(round(y1 + expand_h))

    max_w, max_h = image_size
    new_x0 = max(0, new_x0)
    new_y0 = max(0, new_y0)
    new_x1 = min(max_w, new_x1)
    new_y1 = min(max_h, new_y1)

    if new_x1 <= new_x0:
        new_x1 = min(max_w, new_x0 + 1)
    if new_y1 <= new_y0:
        new_y1 = min(max_h, new_y0 + 1)

    return [new_x0, new_y0, new_x1, new_y1]


def draw_bboxes_on_image(image_path: str, boxes: list, output_path: str) -> str:
    """
    在圖片上繪製 bounding boxes，輸出到指定路徑。
    此版本專門搭配臨時檔使用，不在專案資料夾中產生持久輸出。
    """
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        for i, box_info in enumerate(boxes):
            label = box_info.get('label', f'object_{i}')
            box = box_info.get('box_xyxy', [])

            if len(box) == 4:
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], outline='red', width=3)

                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
                    )
                except Exception:
                    font = ImageFont.load_default()

                text = f"{i+1}: {label}"
                bbox = draw.textbbox((x0, y0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                draw.rectangle(
                    [x0, y0 - text_height - 4, x0 + text_width + 4, y0],
                    fill='red'
                )
                draw.text(
                    (x0 + 2, y0 - text_height - 2),
                    text,
                    fill='white',
                    font=font
                )

        img.save(output_path)
        return output_path
    except Exception as e:
        print(f"[Qwen] 繪製 bbox 時發生錯誤: {e}")
        return image_path


def mask_bboxes_on_image(image_path: str, boxes: list, output_path: str) -> str:
    """
    在圖片上將 bounding boxes 區域填黑（遮蔽），輸出到指定路徑。
    用於第二輪推理，遮蔽已檢測到的物件，讓模型專注於尋找遺漏的物件。
    """
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        for box_info in boxes:
            box = box_info.get('box_xyxy', [])
            if len(box) == 4:
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], fill='black')

        img.save(output_path)
        return output_path
    except Exception as e:
        print(f"[Qwen] 遮蔽 bbox 時發生錯誤: {e}")
        return image_path


def extract_remove_bounding_boxes(
    image: str,
    user_prompt: str,
    max_attempts: int = 3,
    enable_refinement: bool = True,
):
    """
    使用新版兩階段演算法解析包含 remove 行為的 prompt，找出圖片中需要移除的物件並輸出 bounding boxes。
    與 `Qwen_Prompt_Engineer_inference.py` 對齊，但所有中間圖片一律使用臨時檔並在結束後刪除，不在專案資料夾輸出。

    Returns (dict):
      {
        "image_size": [width, height],
        "format": "SAM_xyxy_pixel",
        "boxes": [ {"label": str, "box_xyxy": [x0,y0,x1,y1]} , ...]
      }
    """
    # 讀取圖片大小（像素）- 圖片必須存在
    try:
        with Image.open(image) as im:
            width, height = im.size
    except Exception as e:
        raise ValueError(f"無法讀取圖片 {image}: {e}")

    model, processor = get_qwen_model_and_processor()

    system_prompt = (
        "You are a vision assistant that returns bounding boxes for objects to remove.\n"
        "Task: From the user's prompt, extract all object categories that should be removed if present in the image.\n"
        "Look at the image and return bounding boxes for each matched object you find.\n"
        "Be highly sensitive to very small objects and tiny details.\n"
        "Prefer high recall: if uncertain, include a plausible bounding box (slightly larger is acceptable) rather than missing the object.\n"
        "Output strictly JSON with key 'boxes'.\n"
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    attempt = 0
    result_obj = {
        "image_size": [width, height],
        "format": "SAM_xyxy_pixel",
        "boxes": []
    }

    # 用來追蹤所有臨時檔案，方便 finally 清理
    tmp_files = []

    try:
        while attempt < max_attempts:
            out_ids = model.generate(**inputs, max_new_tokens=256)
            trimmed = [o[len(i):] for i, o in zip(inputs['input_ids'], out_ids)]
            text_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw = (text_out[0] if text_out else "").strip()

            try:
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    raw_json = raw[start:end+1]
                else:
                    raw_json = raw
                print("[Qwen][extract_remove_bounding_boxes] 模型回傳 JSON:", raw_json)
                parsed = json.loads(raw_json)

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

                print(f"[Qwen][extract_remove_bounding_boxes] 第一次推理完成，找到 {len(boxes)} 個物件")

                if not enable_refinement or len(boxes) == 0:
                    return result_obj

                # ===== 第二階段：使用臨時檔進行 refinement =====
                # 建立臨時檔路徑（bbox 圖）
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
                    round1_image_path = tmp1.name
                    tmp_files.append(round1_image_path)
                draw_bboxes_on_image(image, boxes, output_path=round1_image_path)

                # 建立臨時檔路徑（masked 圖）
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                    masked_image_path = tmp2.name
                    tmp_files.append(masked_image_path)
                image_masked = mask_bboxes_on_image(image, boxes, output_path=masked_image_path)

                print("[Qwen][extract_remove_bounding_boxes] === 開始第二次推理（檢查遺漏）===")

                first_round_summary = "\n".join([
                    f"{i+1}. {box['label']}: [{box['box_xyxy'][0]}, {box['box_xyxy'][1]}, {box['box_xyxy'][2]}, {box['box_xyxy'][3]}]"
                    for i, box in enumerate(boxes)
                ])

                system_prompt_round2 = (
                    f"Original removal request: \"{user_prompt}\"\n\n"
                    f"First round detected {len(boxes)} object(s):\n{first_round_summary}\n\n"
                    "Please carefully check if there are any MISSED objects, especially:\n"
                    "- Small objects\n"
                    "- Objects in corners or edges\n"
                    "- Partially visible objects\n"
                    "- Objects that blend with the background\n\n"
                    "Return bounding boxes ONLY for NEW missed objects."
                )

                user_prompt_round2 = (
                    "The image you are viewing now has BLACK AREAS where the previously detected objects were located. "
                    "These black areas represent objects that were already identified in the first round. "
                    "Please focus on the NON-BLACK areas and check if there are any additional objects that were missed. "
                    "This masking helps you concentrate on finding new objects that haven't been detected yet."
                )

                messages_round2 = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt_round2}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_masked},
                            {"type": "text", "text": user_prompt_round2},
                        ],
                    },
                ]

                text_round2 = processor.apply_chat_template(
                    messages_round2, tokenize=False, add_generation_prompt=True
                )
                image_inputs_round2, video_inputs_round2 = process_vision_info(messages_round2)
                inputs_round2 = processor(
                    text=[text_round2],
                    images=image_inputs_round2,
                    videos=video_inputs_round2,
                    padding=True,
                    return_tensors="pt",
                )
                inputs_round2 = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs_round2.items()}

                attempt_round2 = 0
                new_boxes = []
                round2_parse_success = False

                while attempt_round2 < max_attempts:
                    out_ids_round2 = model.generate(**inputs_round2, max_new_tokens=256)
                    trimmed_round2 = [
                        o[len(i):] for i, o in zip(inputs_round2['input_ids'], out_ids_round2)
                    ]
                    text_out_round2 = processor.batch_decode(
                        trimmed_round2,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    raw_round2 = (text_out_round2[0] if text_out_round2 else "").strip()

                    try:
                        start2 = raw_round2.find('{')
                        end2 = raw_round2.rfind('}')
                        if start2 != -1 and end2 != -1 and end2 > start2:
                            raw_json_round2 = raw_round2[start2:end2+1]
                        else:
                            raw_json_round2 = raw_round2

                        print(f"[Qwen][extract_remove_bounding_boxes] 第二次推理輸出: {raw_json_round2}")
                        parsed_round2 = _parse_json_with_fallback(raw_json_round2)
                        candidate_boxes = _extract_boxes_from_parsed_result(parsed_round2)
                        if candidate_boxes:
                            new_boxes = candidate_boxes
                            round2_parse_success = True
                            break
                    except Exception as e:
                        print(f"[Qwen][extract_remove_bounding_boxes] 第二次推理解析失敗: {e}")
                        attempt_round2 += 1

                if not round2_parse_success:
                    raise SecondStageParseError("第二次推理輸出格式無法解析")

                print(f"[Qwen][extract_remove_bounding_boxes] 第二次推理找到 {len(new_boxes)} 個新物件")

                all_boxes = boxes + new_boxes

                # 最後再把所有 bbox 擴大 10%
                expanded_boxes = []
                for box in all_boxes:
                    expanded = expand_bbox(box["box_xyxy"], (width, height), scale=0.10)
                    expanded_boxes.append({"label": box["label"], "box_xyxy": expanded})

                result_obj["boxes"] = expanded_boxes

                # 為了除錯可以另外產生一張最終 bbox 圖，但仍使用臨時檔並在 finally 刪除
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp3:
                    final_image_path = tmp3.name
                    tmp_files.append(final_image_path)
                draw_bboxes_on_image(image, expanded_boxes, output_path=final_image_path)

                print(
                    f"[Qwen][extract_remove_bounding_boxes] 總共找到 {len(expanded_boxes)} 個物件 "
                    f"(第一次: {len(boxes)}, 第二次新增: {len(new_boxes)})"
                )

                return result_obj

            except SecondStageParseError as e:
                print(f"[Qwen][extract_remove_bounding_boxes] 第二次推理失敗：{e}")
                raise
            except Exception as e:
                print(f"[Qwen][extract_remove_bounding_boxes] 第一次推理發生錯誤: {e}")
                attempt += 1

        print("[Qwen][extract_remove_bounding_boxes] 所有嘗試均失敗，回傳空結果")
        return result_obj
    finally:
        # 清理所有臨時檔，避免在磁碟留下中間圖片
        for p in tmp_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


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


