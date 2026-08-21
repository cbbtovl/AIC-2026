# build_metadata.py (Bản cập nhật: OBJECTS + video_path cho file mp4 gốc)
import os
import csv
import json
import glob
from pathlib import Path

from config import (
    MAP_KEYFRAMES_DIR, MEDIA_INFO_DIR, IMAGES_ROOT_DIR, METADATA_JSONL,
    OBJECTS_DIR, OBJECT_SCORE_THRESHOLD, OBJECT_NMS_IOU_THRESHOLD,
)
from video_utils import find_video_path
from image_utils import find_image_path

# Alias giữ tương thích ngược: debug_missing.py có `from build_metadata import _find_image_path`
_find_image_path = find_image_path


def _load_media_info(video_id: str) -> dict:
    path = os.path.join(MEDIA_INFO_DIR, f"{video_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    return {
        "video_title": info.get("title", ""),
        "video_description": (info.get("description") or "")[:500],
        "video_keywords": info.get("keywords", []),
        "video_author": info.get("author", ""),
        "publish_date": info.get("publish_date", ""),
        "watch_url": info.get("watch_url", ""),
    }


def _iou(box_a, box_b) -> float:
    """Intersection-over-Union giữa 2 box [ymin, xmin, ymax, xmax] chuẩn hoá
    0..1. Dùng để phát hiện 2 detection thực ra là CÙNG 1 instance vật thể
    (model detect chồng lấn nhiều lần lên cùng 1 toà nhà/người), thay vì 2
    instance khác nhau đứng cạnh nhau."""
    try:
        ya0, xa0, ya1, xa1 = [float(v) for v in box_a]
        yb0, xb0, yb1, xb1 = [float(v) for v in box_b]
    except (TypeError, ValueError, IndexError):
        return 0.0

    inter_y0, inter_x0 = max(ya0, yb0), max(xa0, xb0)
    inter_y1, inter_x1 = min(ya1, yb1), min(xa1, xb1)
    inter_area = max(0.0, inter_y1 - inter_y0) * max(0.0, inter_x1 - inter_x0)

    area_a = max(0.0, ya1 - ya0) * max(0.0, xa1 - xa0)
    area_b = max(0.0, yb1 - yb0) * max(0.0, xb1 - xb0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _box_to_position(box) -> tuple[str, str]:
    """Từ 1 bounding box chuẩn hoá [ymin, xmin, ymax, xmax] (0..1, định dạng
    TF Object Detection API — chuẩn dùng trong data/objects/*.json), suy ra
    vị trí ngang thô (trái/giữa/phải) + kích thước thô (gần/xa camera) dựa
    trên tâm và diện tích box. Trả về ("", "") nếu box không hợp lệ.

    ĐỘNG LỰC: trước đây OBJECTS chỉ là danh sách tên class -> không phân biệt
    được "1 người bên trái" với "3 người dàn hàng ngang" hay "người ở gần/xa
    camera". Object Rerank vì vậy chỉ so khớp được LOẠI vật thể, không so
    khớp được ĐẶC ĐIỂM không gian mà câu hỏi thường mô tả."""
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in box]
    except (TypeError, ValueError, IndexError):
        return "", ""

    cx = (xmin + xmax) / 2.0
    area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

    if cx < 0.33:
        h_pos = "trái"
    elif cx > 0.66:
        h_pos = "phải"
    else:
        h_pos = "giữa"

    if area > 0.15:
        size = "gần"   # chiếm phần lớn khung hình -> gần camera
    elif area < 0.02:
        size = "xa"    # rất nhỏ -> xa camera / hậu cảnh
    else:
        size = ""

    return h_pos, size


def _load_objects_string(video_id: str, n: int) -> str:
    """Đọc file JSON object tương ứng (vd: objects/L21_V001/011.json) và trả
    về chuỗi mô tả vật thể ĐÃ LÀM GIÀU: tên class + số lượng + vị trí thô.

    BỔ SUNG (thay vì chỉ liệt kê tên class duy nhất 1 lần như trước): dùng
    thêm "detection_boxes" (nếu file JSON có) để:
      - đếm số lượng mỗi class xuất hiện trong frame (vd "person x3")
      - gộp vị trí ngang thô của các instance cùng class (vd "(trái, phải)")
    Nếu file JSON không có "detection_boxes" (một số bộ dữ liệu cũ hơn),
    tự động fallback về hành vi CŨ (chỉ tên class, không lỗi/crash)."""
    candidates = [
        os.path.join(OBJECTS_DIR, video_id, f"{n:03d}.json"),
        os.path.join(OBJECTS_DIR, video_id, f"{n:04d}.json"),
        os.path.join(OBJECTS_DIR, f"{video_id}_{n:03d}.json"),
    ]

    json_path = None
    for p in candidates:
        if os.path.exists(p):
            json_path = p
            break

    if not json_path:
        return ""

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("detection_scores", [])
        entities = data.get("detection_class_entities", [])
        boxes = data.get("detection_boxes", [])  # có thể thiếu/rỗng ở dataset cũ

        # Lọc theo threshold trước, giữ nguyên thứ tự gốc (model xuất theo
        # điểm giảm dần) — cần điểm CAO đứng trước để NMS giữ lại đúng box
        # tự tin nhất làm đại diện cho mỗi instance.
        filtered = []  # list of (score, entity, box)
        for i, (score_str, entity) in enumerate(zip(scores, entities)):
            try:
                score = float(score_str)
            except ValueError:
                continue
            if score < OBJECT_SCORE_THRESHOLD:
                continue
            box = boxes[i] if i < len(boxes) else None
            filtered.append((score, entity, box))
        filtered.sort(key=lambda t: t[0], reverse=True)  # phòng khi input chưa sort sẵn

        # NMS theo entity: 1 vật thể thật có thể bị model detect chồng lấn
        # NHIỀU LẦN (nhiều box gần trùng nhau) -> nếu đếm thô sẽ ra "x7" dù
        # thực tế chỉ có 1-2 vật thể. Chỉ tính là instance MỚI nếu box không
        # chồng lấn cao (IoU >= ngưỡng) với bất kỳ box nào cùng entity đã
        # được giữ lại trước đó (ưu tiên giữ box điểm cao hơn làm đại diện).
        order: list[str] = []          # giữ thứ tự xuất hiện đầu tiên, không trùng
        counts: dict[str, int] = {}
        positions: dict[str, list[str]] = {}
        kept_boxes: dict[str, list] = {}  # box của các instance ĐÃ giữ, theo entity

        for score, entity, box in filtered:
            kept = kept_boxes.setdefault(entity, [])
            is_duplicate = box is not None and any(
                _iou(box, kb) >= OBJECT_NMS_IOU_THRESHOLD for kb in kept if kb is not None
            )
            if is_duplicate:
                continue  # cùng 1 instance đã được đếm rồi, bỏ qua

            if entity not in counts:
                order.append(entity)
                counts[entity] = 0
                positions[entity] = []
            counts[entity] += 1
            kept.append(box)

            if box:
                h_pos, size = _box_to_position(box)
                tag = h_pos if not size else f"{h_pos}-{size}"
                if tag and tag not in positions[entity]:
                    positions[entity].append(tag)

        parts = []
        for entity in order:
            cnt = counts[entity]
            label = entity if cnt <= 1 else f"{entity} x{cnt}"
            pos_tags = positions.get(entity) or []
            if pos_tags:
                label += f" ({'/'.join(pos_tags)})"
            parts.append(label)

        return ", ".join(parts)
    except Exception:
        return ""


def build_metadata():
    csv_files = sorted(glob.glob(os.path.join(MAP_KEYFRAMES_DIR, "*.csv")))
    if not csv_files:
        print(f"❌ Không tìm thấy file CSV nào trong {MAP_KEYFRAMES_DIR}")
        return

    print(f"🔍 Tìm thấy {len(csv_files)} video. Đang build metadata tích hợp OBJECTS + video_path...")

    n_total = 0
    n_missing_video = 0
    with open(METADATA_JSONL, "w", encoding="utf-8") as out_f:
        for csv_path in csv_files:
            video_id = Path(csv_path).stem
            media_info = _load_media_info(video_id)
            video_path = find_video_path(video_id)  # đường dẫn file .mp4 gốc, đã index 1 lần
            if not video_path:
                n_missing_video += 1

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    n = int(float(row["n"]))
                    image_path = _find_image_path(video_id, n)

                    if image_path is None:
                        continue

                    # Lấy chuỗi vật thể từ thư mục objects/
                    objects_str = _load_objects_string(video_id, n)

                    pts_time = float(row["pts_time"])

                    record = {
                        "id": f"{video_id}_{n:03d}",
                        "video_id": video_id,
                        "keyframe_n": n,
                        "pts_time": pts_time,
                        "fps": float(row["fps"]),
                        "frame_idx": int(float(row["frame_idx"])),
                        "image_path": image_path,
                        "video_path": video_path,       # <-- MỚI: đường dẫn video mp4 gốc
                        **media_info,
                        "OBJECTS": objects_str,
                        "description": "",
                        "tags": [],
                        "source": "aic2026",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_total += 1

    print(f"✅ Đã ghi thành công {n_total} keyframe metadata vào {METADATA_JSONL}")
    if n_missing_video:
        print(f"⚠️  {n_missing_video}/{len(csv_files)} video KHÔNG tìm thấy file .mp4 tương ứng "
              f"trong {os.path.dirname('')}. Kiểm tra lại VIDEOS_ROOT_DIR trong config.py và tên file.")


if __name__ == "__main__":
    build_metadata()