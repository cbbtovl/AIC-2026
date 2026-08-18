"""Build a verifiable manifest for the AIC video/keyframe/feature dataset."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


VIDEO_ID_PATTERN = re.compile(r"^(L\d+_V\d+|V\d+)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_feature_pairs(upload_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for csv_path in sorted(upload_dir.rglob("*.csv")):
        npy_path = csv_path.with_suffix(".npy")
        if npy_path.is_file():
            yield csv_path, npy_path


def discover_keyframe_roots(upload_dir: Path, project_root: Path) -> List[Path]:
    roots = sorted(upload_dir.rglob("keyframes"))
    roots.extend([upload_dir, upload_dir / "keyframes", project_root / "map-keyframes"])
    return [root for root in roots if root.is_dir()]


def resolve_image_path(video_id: str, ordinal: int, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        for image_name in (f"{ordinal:03d}.jpg", f"{ordinal}.jpg"):
            image_path = root / video_id / image_name
            if image_path.is_file():
                return image_path.resolve()
        for video_dir in root.rglob(video_id):
            if video_dir.is_dir():
                for image_name in image_name:
                    image_path = video_dir / image_name
                    if image_path.is_file():
                        return image_path.resolve()
    return None


def discover_video_paths(project_root: Path) -> Dict[str, Path]:
    video_paths: Dict[str, Path] = {}
    for video_path in (project_root / "data" / "videos").rglob("*"):
        if video_path.is_file() and video_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            video_paths.setdefault(video_path.stem, video_path.resolve())
    return video_paths


def parse_number(row: Dict[str, str], field_name: str, default: Optional[float] = None) -> Optional[float]:
    value = row.get(field_name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def build_manifest(project_root: Path) -> Tuple[List[Dict], Dict]:
    upload_dir = project_root / "uploads"
    keyframe_roots = discover_keyframe_roots(upload_dir, project_root)
    video_paths = discover_video_paths(project_root)
    records: List[Dict] = []
    pair_count = 0
    csv_row_count = 0
    feature_shape_counts: Dict[str, int] = {}
    missing_images = 0
    missing_videos = 0
    mismatched_pairs = 0

    for csv_path, npy_path in discover_feature_pairs(upload_dir):
        pair_count += 1
        feature_array = np.load(npy_path, mmap_mode="r")
        feature_shape = list(feature_array.shape)
        feature_shape_key = "x".join(str(value) for value in feature_shape)
        feature_shape_counts[feature_shape_key] = feature_shape_counts.get(feature_shape_key, 0) + 1

        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        if len(rows) != len(feature_array):
            mismatched_pairs += 1
        usable_count = min(len(rows), len(feature_array))
        video_id = csv_path.stem
        video_path = video_paths.get(video_id)
        if video_path is None:
            missing_videos += 1

        for row_index in range(usable_count):
            row = rows[row_index]
            ordinal_value = parse_number(row, "n", row_index + 1)
            ordinal = int(ordinal_value) if ordinal_value is not None else row_index + 1
            image_path = resolve_image_path(video_id, ordinal, keyframe_roots)
            if image_path is None:
                missing_images += 1

            csv_row_count += 1
            records.append({
                "video_id": video_id,
                "ordinal": ordinal,
                "frame_idx": int(parse_number(row, "frame_idx", 0) or 0),
                "pts_time": parse_number(row, "pts_time"),
                "fps": parse_number(row, "fps"),
                "image_path": str(image_path) if image_path else None,
                "video_path": str(video_path) if video_path else None,
                "csv_path": str(csv_path.resolve()),
                "feature_path": str(npy_path.resolve()),
                "feature_index": row_index,
                "feature_dim": int(feature_array.shape[1]) if feature_array.ndim > 1 else 1,
                "image_exists": image_path is not None,
                "video_exists": video_path is not None,
            })

    summary = {
        "pair_count": pair_count,
        "record_count": csv_row_count,
        "feature_shape_counts": feature_shape_counts,
        "mismatched_pairs": mismatched_pairs,
        "missing_images": missing_images,
        "missing_videos": missing_videos,
        "keyframe_roots": [str(root.resolve()) for root in keyframe_roots],
        "video_count": len(video_paths),
        "valid_records": sum(record["image_exists"] and record["video_exists"] for record in records),
    }
    return records, summary


def write_manifest(output_path: Path, records: List[Dict], summary: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump({"summary": summary, "records": records}, output_file, ensure_ascii=False, indent=2)


def main() -> None:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project_root / "dataset_manifest.json")
    args = parser.parse_args()

    records, summary = build_manifest(project_root)
    write_manifest(args.output, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
