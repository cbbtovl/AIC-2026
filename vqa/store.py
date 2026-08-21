"""
BƯỚC 2 — Embed ảnh bằng SigLIP + lưu vector kèm metadata vào ChromaDB.

Chạy: python store.py
"""

import json

from config import (
    METADATA_JSONL, CHROMA_DB_DIR, COLLECTION_NAME, EMBED_BATCH_SIZE,
)
from embed import embed_images_batch


def get_collection():
    """Trả về ChromaDB collection — dùng chung cho cả store.py và retrieve.py."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _record_to_metadata(r: dict) -> dict:
    """Chuyển 1 record trong metadata_all.jsonl thành metadata phẳng cho Chroma
    (Chroma chỉ chấp nhận str/int/float/bool, không nhận list/dict lồng nhau).

    BUG ĐÃ SỬA: bản gốc thiếu "id" và "keyframe_n" -> khi lấy candidate từ vector
    search (retrieve.py) ra thì grid_processor.build_temporal_grid() không có
    keyframe_n để dùng, và pipeline không match được frame_id."""
    return {
        "id": r["id"],
        "video_id": r["video_id"],
        "keyframe_n": r["keyframe_n"],
        "pts_time": r["pts_time"],
        "fps": r["fps"],
        "frame_idx": r["frame_idx"],
        "image_path": r["image_path"],
        "video_path": r.get("video_path", ""),
        "video_title": r.get("video_title", ""),
        "video_description": r.get("video_description", ""),
        "video_keywords": ",".join(r.get("video_keywords", [])) if isinstance(r.get("video_keywords"), list) else r.get("video_keywords", ""),
        "video_author": r.get("video_author", ""),
        "publish_date": r.get("publish_date", ""),
        "watch_url": r.get("watch_url", ""),
        "OBJECTS": r.get("OBJECTS", ""),
        "description": r.get("description", ""),
        "tags": ",".join(r.get("tags", [])) if isinstance(r.get("tags"), list) else r.get("tags", ""),
        "source": r.get("source", ""),
    }


def _get_existing_ids(collection, page_size: int = 5000) -> set:
    """Lấy toàn bộ ID đã có trong collection theo TỪNG TRANG.

    BUG: collection.get() không giới hạn (lấy hết 1 lần) sẽ crash với lỗi
    'too many SQL variables' khi collection có nhiều record (đúng tình huống
    của bạn — vài chục nghìn keyframe). ChromaDB (SQLite backend) giới hạn số
    biến ràng buộc trong 1 câu query. Giải pháp: phân trang bằng limit/offset,
    mỗi lần chỉ lấy page_size id, gộp lại dần."""
    existing_ids = set()
    total = collection.count()
    offset = 0
    while offset < total:
        batch = collection.get(limit=page_size, offset=offset, include=[])
        ids = batch.get("ids", [])
        if not ids:
            break
        existing_ids.update(ids)
        offset += len(ids)
    return existing_ids


def ingest():
    import os
    from tqdm import tqdm

    if not os.path.exists(METADATA_JSONL):
        print(f"❌ Chưa có {METADATA_JSONL}. Chạy 'python build_metadata.py' trước.")
        return

    collection = get_collection()
    existing_ids = _get_existing_ids(collection)  # phân trang, không còn crash 'too many SQL variables'
    print(f"Đã có {len(existing_ids)} keyframe trong DB từ trước, sẽ bỏ qua các id này.")

    records = []
    with open(METADATA_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] not in existing_ids:
                records.append(r)

    print(f"Cần embed {len(records)} keyframe mới...")

    for i in tqdm(range(0, len(records), EMBED_BATCH_SIZE)):
        batch = records[i:i + EMBED_BATCH_SIZE]
        image_paths = [r["image_path"] for r in batch]
        by_path = {r["image_path"]: r for r in batch}
        try:
            vectors, ok_paths = embed_images_batch(image_paths)
        except Exception:
            import traceback
            print(f"⚠️  Lỗi embed batch {i} — bỏ qua batch này")
            traceback.print_exc()
            continue

        if not vectors:
            continue

        ok_records = [by_path[p] for p in ok_paths]
        collection.add(
            ids=[r["id"] for r in ok_records],
            embeddings=vectors,
            metadatas=[_record_to_metadata(r) for r in ok_records],
        )

    print("✅ Ingest hoàn tất. Vector DB đã sẵn sàng tại", CHROMA_DB_DIR)


if __name__ == "__main__":
    ingest()