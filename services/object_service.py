"""Read AIC object detections and expose frame-level evidence."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import os

# Lazy YOLO support (ultralytics). We import only when needed.
_yolo_model = None

def _yolo_detect(image_path: str, conf_thresh: float = 0.25) -> List[Dict]:
    global _yolo_model
    try:
        from ultralytics import YOLO
    except Exception:
        print("[Objects] ultralytics not available; install ultralytics to enable local detection.")
        return []

    if _yolo_model is None:
        try:
            # Load smallest model by default for CPU-friendly behavior
            _yolo_model = YOLO(os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt"))
        except Exception:
            try:
                _yolo_model = YOLO('yolov8n')
            except Exception:
                return []

    try:
        results = _yolo_model(image_path, imgsz=640, conf=conf_thresh)[0]
        detections = []
        # boxes.cls and boxes.conf
        if hasattr(results, 'boxes') and results.boxes is not None:
            cls_list = results.boxes.cls.tolist() if hasattr(results.boxes, 'cls') else []
            confs = results.boxes.conf.tolist() if hasattr(results.boxes, 'conf') else []
            xys = results.boxes.xyxy.tolist() if hasattr(results.boxes, 'xyxy') else []
            for cls_idx, conf, xy in zip(cls_list, confs, xys):
                try:
                    label = _yolo_model.names[int(cls_idx)]
                except Exception:
                    label = str(int(cls_idx))
                detections.append({
                    "label": str(label),
                    "score": float(conf),
                    "box": xy,
                })
        if not detections:
            print(f"[Objects] YOLO ran but found no detections for {image_path} (conf={conf_thresh}).")
        return detections
    except Exception:
        print(f"[Objects] YOLO inference failed for {image_path}.")
        return []


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBJECT_ROOTS = [
    PROJECT_ROOT / "uploads" / "objects-aic25-b1_extracted" / "objects",
    PROJECT_ROOT / "objects",
]

# Cache lưu trữ tạm các frame đã gọi qua để không phải đọc lại file nhiều lần
_object_cache: Dict[str, Dict] = {}


def get_object_index() -> Dict[str, Dict]:
    """Giữ lại hàm này để tương thích ngược với các module khác nếu cần."""
    return _object_cache


def get_frame_objects(video_id: str, ordinal: Optional[int] = None) -> List[Dict]:
    if ordinal is None:
        return []
    
    cache_key = f"{video_id}:{int(ordinal)}"
    if cache_key in _object_cache:
        return _object_cache[cache_key].get("detections", [])

    # TỐI ƯU: Thay vì quét rglob toàn bộ hàng trăm ngàn file, ta truy cập trực tiếp file JSON theo cấu trúc thư mục
    for root in OBJECT_ROOTS:
        if not root.is_dir():
            continue
        json_path = root / video_id / f"{int(ordinal)}.json"
        if json_path.is_file():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                labels = payload.get("detection_class_entities", [])
                scores = payload.get("detection_scores", [])
                boxes = payload.get("detection_boxes", [])
                detections = []
                for label, score, box in zip(labels, scores, boxes):
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError):
                        continue
                    if numeric_score >= 0.30:
                        detections.append({
                            "label": str(label),
                            "score": numeric_score,
                            "box": box,
                        })
                
                # Lưu vào cache để các lần gọi sau siêu mượt
                frame_data = {
                    "video_id": video_id,
                    "ordinal": int(ordinal),
                    "detections": detections,
                }
                _object_cache[cache_key] = frame_data
                return detections
            except (OSError, json.JSONDecodeError):
                continue

    # Nếu không tìm thấy file precomputed từ BTC, thực hiện tìm ảnh và chạy YOLO fallback như cũ
    keyframe_roots = [PROJECT_ROOT / "uploads", PROJECT_ROOT / "map-keyframes", PROJECT_ROOT]
    image_names = [f"{int(ordinal):03d}.jpg", f"{int(ordinal)}.jpg"]
    found_image = None
    for root in keyframe_roots:
        for image_name in image_names:
            candidate = root / "keyframes" / video_id / image_name
            if candidate.is_file():
                found_image = candidate
                break
            candidate2 = root / video_id / image_name
            if candidate2.is_file():
                found_image = candidate2
                break
        if found_image:
            break

    if found_image:
        detections = _yolo_detect(str(found_image))
        if detections:
            frame_data = {
                "video_id": video_id,
                "ordinal": int(ordinal),
                "detections": detections,
            }
            _object_cache[cache_key] = frame_data
            return detections

    return []


def summarize_frame_objects(video_id: str, ordinal: Optional[int] = None) -> str:
    detections = get_frame_objects(video_id, ordinal)
    if not detections:
        return "Không có object detection phù hợp."
    counts = Counter(item["label"] for item in detections)
    return ", ".join(
        f"{label}: {count}"
        for label, count in counts.most_common()
    )