"""
BƯỚC TEMPORAL GROUPING — sau rerank, top-15 candidate thường có nhiều frame
đến từ CÙNG 1 cảnh quay liên tục trong cùng 1 video (vì các frame liền kề
nhau về mặt thời gian thường trông rất giống nhau về mặt hình ảnh, nên
model retrieval/rerank cũng chấm điểm chúng gần bằng nhau).

Nếu đưa thẳng top-15 này cho Gemini, có thể 6-7 frame trong đó chỉ là
cùng 1 khoảnh khắc lặp lại — vừa tốn token, vừa làm Gemini "thấy" ít
khoảnh khắc/video khác nhau hơn thực tế, dễ bỏ sót đáp án đúng nằm ở
1 candidate khác bị lép vế do trùng lặp.

Cách xử lý: gom các frame CÙNG video_id và cách nhau dưới
TEMPORAL_WINDOW_SEC giây thành 1 nhóm, chỉ giữ lại frame có điểm rerank
cao nhất trong nhóm làm đại diện — rồi lấy top-N nhóm (theo điểm đại diện)
để đưa tiếp cho Gemini.
"""

from config import (
    TEMPORAL_WINDOW_SEC, FINAL_TOP_K,
    USE_EMBEDDING_GROUPING, GROUP_EMBEDDING_SIMILARITY_THRESHOLD,
)


def _cosine_sim(a, b) -> float:
    """Cosine similarity giữa 2 vector cùng chiều. Trả 0.0 nếu lỗi/vector rỗng
    (không import numpy ở module-level để grouping.py vẫn nhẹ khi
    USE_EMBEDDING_GROUPING=False và không ai gọi hàm này)."""
    if not a or not b:
        return 0.0
    try:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return 0.0


def _compute_embeddings(metadatas: list[dict]) -> dict[int, list[float]]:
    """Embed ảnh gốc (image_path) của từng candidate bằng SigLIP — dùng LẠI
    model đã được load sẵn từ Tầng 1 (embed.py cache global), nên KHÔNG tốn
    thêm chi phí tải model, chỉ tốn 1 lần forward pass cho danh sách nhỏ
    (top_k_final*2 candidate, xem pipeline.py) -> rất rẻ.

    Trả về dict {index_trong_metadatas: embedding_vector}, bỏ qua candidate
    thiếu image_path hoặc embed lỗi (không làm crash grouping)."""
    try:
        from embed import embed_images_batch
    except Exception as e:
        print(f"⚠️ [grouping] Không thể import embed_images_batch, tắt embedding-grouping: {e}")
        return {}

    idx_by_path: dict[str, list[int]] = {}
    paths = []
    for i, m in enumerate(metadatas):
        p = m.get("image_path")
        if p:
            idx_by_path.setdefault(p, []).append(i)
            if p not in paths:
                paths.append(p)

    if not paths:
        return {}

    try:
        vectors, ok_paths = embed_images_batch(paths)
    except Exception as e:
        print(f"⚠️ [grouping] Lỗi embed ảnh cho embedding-grouping, fallback về time-only: {e}")
        return {}

    result: dict[int, list[float]] = {}
    for path, vec in zip(ok_paths, vectors):
        for i in idx_by_path.get(path, []):
            result[i] = vec
    return result


def group_temporal(
    ranked_metadatas: list[dict],
    top_n: int = FINAL_TOP_K,
    window_sec: float = TEMPORAL_WINDOW_SEC,
    use_embedding: bool | None = None,
    similarity_threshold: float = GROUP_EMBEDDING_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    `ranked_metadatas`: list đã sort giảm dần theo độ liên quan (ví dụ
    output của object_rerank.py / rerank_vlm.py — mỗi item có "pts_time").

    GỘP theo 2 điều kiện: (1) cùng video_id + cách nhau <= window_sec giây
    NHƯ CŨ, VÀ (2, MỚI — xem config.USE_EMBEDDING_GROUPING) cosine similarity
    giữa SigLIP embedding của 2 ảnh >= similarity_threshold. Chỉ gộp khi CẢ
    HAI điều kiện đều thoả, để tránh 2 vấn đề của bản cũ (chỉ dựa thời gian):
        - Gộp NHẦM 2 cảnh khác nhau chỉ vì tình cờ cách nhau vài giây (cắt
          cảnh nhanh) -> mất 1 candidate khác biệt thật sự.
        - KHÔNG gộp 2 frame giống hệt nhau nhưng cách nhau hơn window_sec
          giây (cảnh quay tĩnh kéo dài) -> lãng phí 1 "suất" FINAL_TOP_K cho
          nội dung trùng lặp.
    Nếu tắt use_embedding hoặc embed lỗi -> tự động fallback về hành vi CŨ
    (chỉ so thời gian), KHÔNG crash pipeline.

    Trả về tối đa `top_n` metadata đại diện, mỗi cái được thêm field mới:
        - "group_size": số frame đã bị gộp vào (kể cả chính nó)
        - "group_members": list [{"pts_time":..., "frame_idx":...}, ...]
          của các frame cùng nhóm (để debug/hiển thị nếu cần)
    Thứ tự trả về giữ nguyên thứ tự liên quan giảm dần (không sort lại
    theo thời gian), vì đây là thứ tự sẽ đưa cho VLM cuối.
    """
    if not ranked_metadatas:
        return []

    if use_embedding is None:
        use_embedding = USE_EMBEDDING_GROUPING

    embeddings = _compute_embeddings(ranked_metadatas) if use_embedding else {}

    groups = []  # list of {"rep": metadata, "video_id":..., "time":..., "emb":..., "members":[...]}

    for idx, m in enumerate(ranked_metadatas):
        video_id = m.get("video_id")
        pts_time = m.get("pts_time")
        emb = embeddings.get(idx)

        matched = None
        for g in groups:
            if g["video_id"] != video_id or pts_time is None or g["time"] is None:
                continue
            if abs(pts_time - g["time"]) > window_sec:
                continue
            # Điều kiện thời gian đã thoả. Nếu CÓ embedding cho cả 2 phía,
            # yêu cầu thêm điều kiện thị giác; nếu thiếu embedding (lỗi/không
            # có image_path), fallback về chỉ-thời-gian như bản cũ.
            if emb is not None and g["emb"] is not None:
                if _cosine_sim(emb, g["emb"]) < similarity_threshold:
                    continue
            matched = g
            break

        if matched is not None:
            # frame đang xét có điểm rerank thấp hơn hoặc bằng đại diện hiện
            # tại (vì ranked_metadatas đã sort giảm dần) -> chỉ cần thêm vào
            # danh sách member, KHÔNG đổi đại diện của nhóm.
            matched["members"].append({
                "pts_time": pts_time,
                "frame_idx": m.get("frame_idx"),
            })
        else:
            groups.append({
                "rep": m,
                "video_id": video_id,
                "time": pts_time,
                "emb": emb,
                "members": [{"pts_time": pts_time, "frame_idx": m.get("frame_idx")}],
            })

    result = []
    for g in groups[:top_n]:
        m2 = dict(g["rep"])
        m2["group_size"] = len(g["members"])
        m2["group_members"] = g["members"]
        result.append(m2)
    return result


if __name__ == "__main__":
    # test nhanh: python grouping.py "câu hỏi"
    # BUG ĐÃ SỬA: trước đây import `from rerank import rerank` — module này
    # không tồn tại trong project (rerank thật nằm ở object_rerank.py /
    # rerank_vlm.py) -> khối test này luôn crash ImportError nếu chạy trực
    # tiếp. Đổi sang object_rerank.rerank_by_objects() cho khớp pipeline thật.
    import sys
    from retrieve import retrieve
    from object_rerank import rerank_by_objects

    q = sys.argv[1] if len(sys.argv) > 1 else "một người đang nói chuyện trước camera"
    res = retrieve(q, top_k=50)
    reranked = rerank_by_objects(q, res["metadatas"][0], top_n=15)
    grouped = group_temporal(reranked, top_n=8)
    for m in grouped:
        print(f"{m['video_id']} | t={m['pts_time']:.2f}s | object_score={m.get('object_rerank_score', 0):.4f} "
              f"| gộp {m['group_size']} frame")