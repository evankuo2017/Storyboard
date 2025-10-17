from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import os
import torch

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
    try:
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
    finally:
        unload_qwen()

    # 若多次嘗試仍失敗，預設回傳 1（general case）
    return 1

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