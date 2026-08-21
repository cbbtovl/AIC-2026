# retrieve.py
"""
Vector search (SigLIP) qua ChromaDB — cặp với store.py (ingest).

BUGFIX: file này trước đây bị thay nhầm bằng bản retrieve_qdrant.py (import
`from store_qdrant import get_client` -> ModuleNotFoundError vì project
KHÔNG có store_qdrant.py, toàn bộ ingest/store thực tế dùng ChromaDB qua
store.py). Khôi phục lại bản Chroma đúng, dùng chung collection với
store.py (get_collection()) để đảm bảo ingest và retrieval luôn trỏ vào
cùng 1 DB.

Trả về ĐÚNG shape mà pipeline._hybrid_retrieve() mong đợi:
    {"ids": [[...]], "metadatas": [[...]], "distances": [[...]]}
(distance ở đây là khoảng cách thật do Chroma trả về, càng nhỏ càng gần —
công thức `vector_score = 1.0 / (1.0 + dist)` trong pipeline.py dùng đúng
quy ước này, không cần đổi gì thêm).
"""
from config import CHROMA_DB_DIR, COLLECTION_NAME
from embed import embed_text
from store import get_collection

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


def retrieve(query: str, top_k: int = 20) -> dict:
    collection = _get_collection()

    if collection.count() == 0:
        print("⚠️  [retrieve] ChromaDB collection rỗng (chưa chạy `python store.py` để ingest). "
              "Bỏ qua vector search, chỉ dùng BM25.")
        return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    query_vec = embed_text(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    return results


if __name__ == "__main__":
    # test nhanh: python retrieve.py "câu hỏi"
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "người đàn ông mặc áo đỏ"
    res = retrieve(q, top_k=5)
    for doc_id, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        print(f"{doc_id} | video={meta.get('video_id')} t={meta.get('pts_time')} | dist={dist:.4f}")