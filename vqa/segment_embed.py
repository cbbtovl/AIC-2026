# segment_embed.py
"""
(MỚI) Embed field 'summary' của mỗi segment (xem segment_topics.py) bằng
BGE-M3 — dense embedding đa ngôn ngữ, tốt tiếng Việt — để retrieval theo
NGỮ NGHĨA thay vì chỉ khớp anchor CHÍNH XÁC (org/province).

VÌ SAO CẦN THÊM TẦNG NÀY: anchor extraction (identity_rescan.py) chỉ bắt
được đúng CHỮ xuất hiện trên hình (tên tổ chức viết HOA, tên tỉnh khớp
gazetteer). Nhưng câu hỏi thường DIỄN GIẢI LẠI nội dung bằng từ khác hẳn —
ví dụ câu hỏi "trao quà tại một xã thuộc Khánh Hòa" không trùng 1 từ khoá cụ
thể nào với summary "CLB FANA trao quà từ thiện tại xã Giang Ly, huyện Khánh
Vinh, tỉnh Khánh Hòa" nếu chỉ so khớp org/province đã trích riêng lẻ — NHƯNG
2 câu này RẤT GẦN NGHĨA. Vì summary là câu văn hoàn chỉnh (khác OCR thô toàn
tên riêng/số rời rạc), dense embedding bắt được sự tương đồng này tốt.

CHI PHÍ: số segment/video ít hơn số frame RẤT NHIỀU (vài chục/video so với
vài nghìn frame/video) -> embed toàn bộ segment rẻ hơn hẳn embed từng frame,
và chạy 100% LOCAL (không tốn quota OpenRouter).

CÀI ĐẶT:
    pip install sentence-transformers

CHẠY (offline, 1 lần, resumable — chỉ embed segment MỚI mỗi lần chạy lại):
    python segment_embed.py

YÊU CẨU: đã chạy `python segment_topics.py` trước (cân segment_topics.jsonl
có field 'summary' khác rỗng).
"""

import os
import json
import pickle
import threading

from config import (
    SEGMENT_TOPICS_PATH, SEGMENT_EMBEDDINGS_PATH, BGE_MODEL_NAME,
    SEGMENT_SIMILARITY_MIN_SCORE, SEGMENT_SIMILARITY_TOP_K,
)

_model = None
_load_lock = threading.Lock()


def _load_model():
    """Lazy-load BGE-M3 (giống pattern embed.py/rerank_vlm.py trong project)
    -- chỉ tải khi thực sự cần, tự chọn GPU nếu có, fallback CPU. Lỗi (chưa
    cài sentence-transformers, hoặc chưa tải được model) -> raise ra ngoài
    để hàm gọi tự quyết định fallback, KHÔNG nuốt lỗi ở đây."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⌛ Đang tải BGE-M3 ({BGE_MODEL_NAME}, device={device})... (chỉ tải 1 lần đầu)")
            _model = SentenceTransformer(BGE_MODEL_NAME, device=device)
    return _model


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_segment_summaries(force_rebuild: bool = False):
    """Đọc segment_topics.jsonl, embed field 'summary' của MỌI segment có
    summary khác rỗng, lưu ra SEGMENT_EMBEDDINGS_PATH (pickle).

    RESUMABLE: nếu file đã có và force_rebuild=False, chỉ embed thêm segment
    MỚI (segment_id chưa có trong file cũ), giữ nguyên vector cũ -- tránh
    embed lại toàn bộ mỗi khi segment_topics.py chạy thêm video mới."""
    if not os.path.exists(SEGMENT_TOPICS_PATH):
        print("❌ Chưa có segment_topics.jsonl. Chạy `python segment_topics.py` trước.")
        return

    segments = []
    with open(SEGMENT_TOPICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("summary"):
                segments.append(rec)

    if not segments:
        print("⚠️ Không có segment nào có 'summary' khác rỗng -- không có gì để embed.")
        return

    existing = {"segment_id": [], "video_id": [], "member_ids": [], "summary": [], "vector": []}
    if os.path.exists(SEGMENT_EMBEDDINGS_PATH) and not force_rebuild:
        with open(SEGMENT_EMBEDDINGS_PATH, "rb") as f:
            existing = pickle.load(f)
        print(f"📦 Đã có {len(existing['segment_id'])} segment embedding từ lân chạy trước (resume).")

    done_ids = set(existing["segment_id"])
    todo = [s for s in segments if s["segment_id"] not in done_ids]
    print(f"📊 Cân embed thêm {len(todo)}/{len(segments)} segment mới.")

    if not todo:
        print("✅ Không có gì mới để embed.")
        return

    model = _load_model()
    texts = [s["summary"] for s in todo]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True).tolist()

    existing["segment_id"].extend(s["segment_id"] for s in todo)
    existing["video_id"].extend(s["video_id"] for s in todo)
    existing["member_ids"].extend(s["member_ids"] for s in todo)
    existing["summary"].extend(s["summary"] for s in todo)
    existing["vector"].extend(vectors)

    with open(SEGMENT_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(existing, f)
    print(f"✅ Đã lưu {len(existing['segment_id'])} segment embedding vào {SEGMENT_EMBEDDINGS_PATH}")


# ---------------------------------------------------------------------------
# (Dùng bởi segment_topics.segment_boost_candidates) Semantic search
# ---------------------------------------------------------------------------

_index_cache: dict | None = None


def _load_index() -> dict | None:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    if not os.path.exists(SEGMENT_EMBEDDINGS_PATH):
        return None
    with open(SEGMENT_EMBEDDINGS_PATH, "rb") as f:
        _index_cache = pickle.load(f)
    return _index_cache


def search_segments_by_similarity(
    query: str, top_k: int = SEGMENT_SIMILARITY_TOP_K, min_score: float = SEGMENT_SIMILARITY_MIN_SCORE,
) -> list[dict]:
    """Embed `query` bằng BGE-M3, so cosine với TOÀN BỘ segment đã embed sẵn
    (số lượng nhỏ nên brute-force đủ nhanh, không cân ANN index), trả vê
    top_k segment có cosine >= min_score, sắp giảm dân theo điểm.

    Lỗi (model chưa tải được / chưa embed segment nào) -> trả vê [] AN TOÀN,
    KHÔNG raise, để pipeline.py tự fallback vê anchor-boost/BM25 thường."""
    index = _load_index()
    if not index or not index.get("vector"):
        return []

    try:
        model = _load_model()
        query_vec = model.encode([query], normalize_embeddings=True).tolist()[0]
    except Exception as e:
        print(f"⚠️ [segment_embed] Lỗi embed câu hỏi, bỏ qua semantic segment search: {type(e).__name__}: {e}")
        return []

    scored = []
    for i in range(len(index["segment_id"])):
        sim = _cosine(query_vec, index["vector"][i])
        if sim >= min_score:
            scored.append({
                "segment_id": index["segment_id"][i],
                "video_id": index["video_id"][i],
                "member_ids": index["member_ids"][i],
                "summary": index["summary"][i],
                "similarity": round(sim, 4),
            })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    embed_segment_summaries()