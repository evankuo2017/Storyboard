import os
from datetime import datetime
from typing import Callable, Optional
from torchvision.io import read_video

# 不依賴套件；加入當前檔案目錄到 sys.path 後本地匯入
framepack_process_video = None
_import_error_detail = None
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from framepack_start_end import process_video as framepack_process_video
except Exception as e:
    _import_error_detail = str(e)
    framepack_process_video = None

def _ensure_process_video_loaded():
    """若 framepack_process_video 尚未可用，嘗試再次載入。"""
    global framepack_process_video
    if framepack_process_video is not None:
        return
    try:
        import sys as _sys
        _sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from framepack_start_end import process_video as _pv
        framepack_process_video = _pv
        return
    except Exception:
        pass


def _ensure_dirs(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _extract_last_frame_without_ffmpeg(video_path: str, out_image_path: str) -> None:
    """使用 torchvision.io 讀取最後一幀，避免系統 ffmpeg 依賴。"""
    import torch
    from PIL import Image

    video, _, info = read_video(video_path, pts_unit='sec')  # video: [T, H, W, C] uint8
    if video.numel() == 0 or video.shape[0] == 0:
        raise RuntimeError('無法讀取生成的視頻內容')

    last = video[-1]  # [H, W, C]
    if last.dtype != torch.uint8:
        last = last.to(torch.uint8)
    img = Image.fromarray(last.numpy())
    _ensure_dirs(out_image_path)
    img.save(out_image_path)


def generate_one_frame(
    prev_image_path: str,
    prompt: str,
    out_image_path: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    duration_seconds: float = 1.0,
) -> None:
    """
    以 demo_gradio/framepack 的方式，從前一張圖與 prompt 生成短片，然後抽取單幀圖片輸出。
    - prev_image_path: 前一張圖片的本地路徑（必填）
    - prompt: 可為空字串
    - out_image_path: 輸出圖片路徑（.png）
    """
    if not framepack_process_video:
        _ensure_process_video_loaded()
    if not framepack_process_video:
        detail = _import_error_detail or 'unknown'
        raise RuntimeError(f"framepack_process_video 不可用，請確認 framepack_start_end.py 匯出 process_video；import 錯誤: {detail}")

    if not prev_image_path or not os.path.exists(prev_image_path):
        raise FileNotFoundError(f'找不到前一張圖片: {prev_image_path}')

    _ensure_dirs(out_image_path)

    # 視訊暫存位置（存到系統暫存目錄，不保存到輸出資料夾）
    import tempfile
    tmp_video_name = f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    tmp_video_path = os.path.join(tempfile.gettempdir(), tmp_video_name)

    # 生成 1 秒短片（30fps）
    def progress_callback(percentage, message):
        # 對外回調
        try:
            if progress_cb:
                progress_cb(int(percentage), str(message))
        except Exception:
            pass

        # 取消檢查
        try:
            if cancel_cb and cancel_cb():
                raise RuntimeError('任務已取消')
        except RuntimeError:
            raise
        except Exception:
            pass

    # 調用高階流程（預覽模式：只產生最後一幀並直接輸出圖片）
    import random
    preview_image_out = out_image_path
    # 打印 prompt 與 seed（seed 使用 0..2^31-1）
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[Preview] prompt='{prompt or ''}'")
    logger.info(f"[Preview] duration_seconds={duration_seconds}")
    _seed = random.randint(0, (2**31) - 1)
    logger.info(f"[Preview] seed={_seed}")
    result = framepack_process_video(
        start_image_path=prev_image_path,
        end_image_path=None,
        progress_callback=progress_callback,
        prompt=prompt or "",
        n_prompt="",
        seed=_seed,
        total_second_length=float(duration_seconds),
        latent_window_size=9,
        steps=25,
        cfg=1.0,
        gs=10.0,
        rs=0.0,
        gpu_memory_preservation=6,
        use_teacache=True,
        mp4_crf=16,
        output=tmp_video_path,
        preview_last_only=True,
        preview_image_output=preview_image_out,
    )
    
    # 在預覽模式下，後端會回傳圖片路徑；若失敗則嘗試回退抽幀
    if result and os.path.exists(result):
        return
    if os.path.exists(tmp_video_path):
        _extract_last_frame_without_ffmpeg(tmp_video_path, out_image_path)
        return
    raise RuntimeError('生成單幀失敗')


