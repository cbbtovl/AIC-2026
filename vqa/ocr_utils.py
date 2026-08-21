# ocr_utils.py
"""
TẦNG 3.5 (MỚI): OCR — CHỈ chạy trên top 5-10 candidate SAU Object Rerank +
Grouping, KHÔNG chạy trên toàn bộ 177k keyframe (quá chậm, không cần thiết).

Dùng EasyOCR (không phải Tesseract) vì đọc tốt tiếng Việt CÓ DẤU hơn hẳn —
quan trọng để bắt được caption/tên hiển thị trên video (vd. tên nhân vật
trong bản tin, phụ đề, biển hiệu...).

Cài đặt:
    pip install easyocr

Lần chạy đầu tiên EasyOCR sẽ tự tải model detection + recognition (~100-200MB),
cần mạng để tải 1 lần rồi cache lại.
"""

import threading
from config import OCR_LANGS, OCR_MIN_CONF, OCR_USE_GPU

_reader = None
_load_lock = threading.Lock()


def _load_reader():
    global _reader
    if _reader is not None:
        return _reader
    with _load_lock:
        if _reader is None:  # double-check sau khi giành được lock
            import easyocr
            import torch

            use_gpu = OCR_USE_GPU and torch.cuda.is_available()
            print(f"⌛ Đang tải EasyOCR reader (langs={OCR_LANGS}, gpu={use_gpu})...")
            _reader = easyocr.Reader(OCR_LANGS, gpu=use_gpu)
    return _reader


def extract_text(image_path: str) -> str:
    """
    Đọc text trên 1 ảnh (keyframe gốc hoặc ảnh grid 2x2). Trả về chuỗi rỗng
    nếu không đọc được text nào hoặc lỗi (không làm crash pipeline).
    """
    if not image_path:
        return ""
    try:
        reader = _load_reader()
        results = reader.readtext(image_path, detail=1)
        lines = [text.strip() for (_bbox, text, conf) in results
                 if conf >= OCR_MIN_CONF and text.strip()]
        return " | ".join(lines)
    except Exception as e:
        print(f"⚠️ [ocr_utils] Lỗi OCR ({image_path}): {type(e).__name__}: {e}")
        return ""


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        print("Dùng: python ocr_utils.py <đường_dẫn_ảnh>")
    else:
        print(f"OCR text: {extract_text(path)!r}")