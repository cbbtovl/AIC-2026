"""Backfill legacy database metadata from the canonical CSV/keyframe layout."""

import argparse
import csv
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


LEGACY_FILENAME = re.compile(
    r"^(L\d+_V\d+)[_-](\d+(?:\.\d+)?)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
ORDINAL_ONLY_FILENAME = re.compile(r"^(?:Keyframes_[^_-]+\s*-\s*)?(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
VIDEO_PATH_PATTERN = re.compile(r"(L\d+_V\d+|V\d+)", re.IGNORECASE)
VIDEO_DIRECTORY = re.compile(r"^(L\d+_V\d+|V\d+)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def discover_image(video_id: str, ordinal: int, roots: list[Path]) -> Optional[str]:
    for root in roots:
        for image_name in (f"{ordinal:03d}.jpg", f"{ordinal}.jpg"):
            image_path = root / video_id / image_name
            if image_path.is_file():
                return image_path.resolve().as_posix()
        for video_dir in root.rglob(video_id):
            if video_dir.is_dir():
                for image_name in (f"{ordinal:03d}.jpg", f"{ordinal}.jpg"):
                    image_path = video_dir / image_name
                    if image_path.is_file():
                        return image_path.resolve().as_posix()
    return None


def build_image_index(roots: list[Path]) -> Dict[Tuple[str, int], str]:
    """Scan image assets once instead of recursively searching for every CSV row."""
    image_index: Dict[Tuple[str, int], str] = {}
    for root in roots:
        for image_path in root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            ordinal_match = re.fullmatch(r"(\d+)", image_path.stem)
            video_id = image_path.parent.name
            if not ordinal_match or not VIDEO_DIRECTORY.fullmatch(video_id):
                continue
            key = (video_id, int(ordinal_match.group(1)))
            image_index.setdefault(key, image_path.resolve().as_posix())
    return image_index


def load_csv_metadata(project_root: Path) -> Dict[Tuple[str, int], Dict]:
    upload_dir = project_root / "uploads"
    image_roots = list(upload_dir.rglob("keyframes"))
    image_roots.extend([upload_dir, upload_dir / "keyframes", project_root / "map-keyframes"])
    image_index = build_image_index(image_roots)
    print(f"[repair] Đã lập chỉ mục {len(image_index):,} ảnh một lần.", flush=True)
    metadata: Dict[Tuple[str, int], Dict] = {}

    for csv_path in sorted(upload_dir.rglob("*.csv")):
        video_id = csv_path.stem
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            for row_index, row in enumerate(csv.DictReader(csv_file), start=1):
                try:
                    ordinal = int(float(row.get("n") or row_index))
                    frame_idx = int(float(row.get("frame_idx") or 0))
                    pts_time = float(row["pts_time"]) if row.get("pts_time") else None
                except (TypeError, ValueError):
                    continue
                metadata[(video_id, ordinal)] = {
                    "frame_idx": frame_idx,
                    "pts_time": pts_time,
                    "image_path": image_index.get((video_id, ordinal)),
                }
    return metadata


def repair_database(database_path: Path, project_root: Path, apply_changes: bool) -> Dict:
    started_at = time.perf_counter()
    metadata = load_csv_metadata(project_root)
    print(f"[repair] Đã đọc {len(metadata):,} dòng metadata từ CSV.", flush=True)
    connection = sqlite3.connect(str(database_path))
    try:
        rows = connection.execute(
            "SELECT id, filename, filepath FROM items WHERE filetype = 'image'"
        ).fetchall()
        print(f"[repair] Đang kiểm tra {len(rows):,} record trong database...", flush=True)
        updates = []
        unmatched = 0
        image_matches = 0
        for row_number, (item_id, filename, old_filepath) in enumerate(rows, start=1):
            match = LEGACY_FILENAME.match(Path(filename).name)
            if match:
                video_id, ordinal_text = match.groups()
            else:
                ordinal_match = ORDINAL_ONLY_FILENAME.match(Path(filename).name)
                path_match = VIDEO_PATH_PATTERN.search(str(old_filepath))
                if not ordinal_match or not path_match:
                    unmatched += 1
                    continue
                video_id = path_match.group(1)
                ordinal_text = ordinal_match.group(1)
            ordinal = int(float(ordinal_text))
            item_metadata = metadata.get((video_id, ordinal))
            if item_metadata is None:
                unmatched += 1
                continue
            image_path = item_metadata["image_path"]
            if image_path:
                image_matches += 1
            updates.append((
                f"{video_id}_{ordinal:03d}.jpg",
                video_id,
                ordinal,
                item_metadata["frame_idx"],
                item_metadata["pts_time"],
                str(item_metadata["pts_time"]) if item_metadata["pts_time"] is not None else None,
                image_path or old_filepath,
                item_id,
            ))
            if row_number % 10000 == 0:
                print(f"[repair] Đã lập kế hoạch {row_number:,}/{len(rows):,} record...", flush=True)

        if apply_changes and updates:
            print(f"[repair] Bắt đầu ghi {len(updates):,} record vào SQLite...", flush=True)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(items)")}
            for column_name, definition in {
                "ordinal": "INTEGER",
                "pts_time": "REAL",
                "embedding_type": "TEXT DEFAULT 'text'",
            }.items():
                if column_name not in columns:
                    connection.execute(f"ALTER TABLE items ADD COLUMN {column_name} {definition}")
            connection.executemany(
                """
                UPDATE items
                SET filename = ?, video_id = ?, ordinal = ?, frame_idx = ?,
                    pts_time = ?, timestamp_ms = ?, filepath = ?
                WHERE id = ?
                """,
                updates,
            )
            connection.commit()
            print(f"[repair] Đã commit xong trong {time.perf_counter() - started_at:.1f}s.", flush=True)

        return {
            "database_records": len(rows),
            "metadata_entries": len(metadata),
            "candidate_updates": len(updates),
            "image_path_matches": image_matches,
            "unmatched_records": unmatched,
            "applied": apply_changes,
            "elapsed_seconds": round(time.perf_counter() - started_at, 1),
        }
    finally:
        connection.close()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "database.db")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--apply", action="store_true", help="Write the backfilled metadata")
    args = parser.parse_args()
    print(repair_database(args.database, args.project_root, args.apply))


if __name__ == "__main__":
    main()
