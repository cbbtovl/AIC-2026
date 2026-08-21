# video_utils.py
"""
Tiện ích tìm file video .mp4 gốc theo video_id.

Cấu trúc thư mục thực tế của bạn:
    D:\\aic-vqa-project\\data\\videos\\Videos_L21_a\\L21_V001.mp4
    D:\\aic-vqa-project\\data\\videos\\Videos_L21_a\\L21_V002.mp4
    D:\\aic-vqa-project\\data\\videos\\Videos_L21_b\\L21_V0xx.mp4
    ...

Vì video được chia theo nhiều batch folder (Videos_L21_a, Videos_L21_b, ...),
không thể đoán trực tiếp đường dẫn từ video_id -> phải quét (glob) 1 lần,
build thành dict {video_id: full_path}, rồi tra cứu O(1) cho mọi record.
Tránh việc gọi glob.glob() lặp lại hàng ngàn lần (rất chậm khi build metadata
cho toàn bộ dataset).
"""

import os
import glob
from pathlib import Path
from config import VIDEOS_ROOT_DIR

_video_index: dict[str, str] | None = None


def build_video_index(force_rebuild: bool = False) -> dict[str, str]:
    global _video_index
    if _video_index is not None and not force_rebuild:
        return _video_index

    index: dict[str, str] = {}
    pattern = os.path.join(VIDEOS_ROOT_DIR, "**", "*.mp4")
    for path in glob.glob(pattern, recursive=True):
        video_id = Path(path).stem  # "L21_V001.mp4" -> "L21_V001"
        index[video_id] = path

    _video_index = index
    print(f"🎞️  [video_utils] Đã index {len(index)} file video .mp4 trong {VIDEOS_ROOT_DIR}")
    return index


def find_video_path(video_id: str) -> str:
    """Trả về đường dẫn tuyệt đối file mp4 cho video_id, hoặc '' nếu không tìm thấy."""
    index = build_video_index()
    return index.get(video_id, "")


def format_timestamp(seconds: float) -> str:
    """12.5 -> '00:12' ; 725.3 -> '12:05' (dùng để hiển thị / đặt tên khi cần)."""
    try:
        seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        seconds = 0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"