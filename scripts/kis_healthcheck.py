"""Offline readiness checks for the KIS retrieval pipeline."""

import sys
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EXPECTED_DIMENSION = 512


def read_embedding_dimension(embedding_text: Optional[str]) -> Optional[int]:
    if not embedding_text:
        return None
    try:
        return len(json.loads(embedding_text))
    except (TypeError, json.JSONDecodeError):
        return None


def run_healthcheck(database_path: Path, sample_limit: int = 500) -> Dict:
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    try:
        available_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(items)").fetchall()
        }
        required_columns = {"video_id", "frame_idx", "pts_time", "ordinal"}
        missing_schema_columns = sorted(required_columns - available_columns)
        total_records = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        image_records = connection.execute(
            "SELECT COUNT(*) FROM items WHERE filetype = 'image'"
        ).fetchone()[0]
        missing_assets = connection.execute(
            "SELECT COUNT(*) FROM items WHERE filepath IS NULL OR filepath = ''"
        ).fetchone()[0]
        pts_time_expression = "pts_time" if "pts_time" in available_columns else "NULL"
        rows = connection.execute(
            f"""
            SELECT filepath, video_id, frame_idx, {pts_time_expression} AS pts_time, embedding
            FROM items
            WHERE embedding IS NOT NULL
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
    finally:
        connection.close()

    dimensions = Counter()
    missing_files = 0
    missing_frame_metadata = 0
    for row in rows:
        dimension = read_embedding_dimension(row["embedding"])
        dimensions[str(dimension)] += 1
        if not row["filepath"] or not Path(row["filepath"]).is_file():
            missing_files += 1
        if row["video_id"] is None or row["frame_idx"] is None or row["pts_time"] is None:
            missing_frame_metadata += 1

    return {
        "database": str(database_path.resolve()),
        "total_records": total_records,
        "image_records": image_records,
        "sampled_embeddings": len(rows),
        "embedding_dimensions": dict(dimensions),
        "expected_dimension": EXPECTED_DIMENSION,
        "sampled_missing_files": missing_files,
        "sampled_missing_frame_metadata": missing_frame_metadata,
        "missing_schema_columns": missing_schema_columns,
        "empty_filepaths": missing_assets,
        "ready_for_kis": (
            total_records > 0
            and not missing_schema_columns
            and dimensions.get(str(EXPECTED_DIMENSION), 0) == len(rows)
            and missing_assets == 0
        ),
    }


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "database.db")
    parser.add_argument("--sample-limit", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run_healthcheck(args.database, args.sample_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
