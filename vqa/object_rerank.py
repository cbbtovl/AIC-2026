# object_rerank.py
"""
TẦNG 2: OBJECT RERANK.

SigLIP (Tầng 1) tìm theo độ tương đồng ẢNH~SEMANTIC TỔNG QUÁT, không đảm bảo
đúng VẬT THỂ cụ thể có trong câu hỏi. Ở đây, ta so khớp câu hỏi với trường
`OBJECTS` — danh sách class đã detect sẵn cho mỗi keyframe (xem
`data/objects/*.json` -> `build_metadata.py` -> field "OBJECTS" trong
metadata_all.jsonl / ChromaDB) — để ưu tiên đúng frame có vật thể liên quan.

VẤN ĐỀ NGÔN NGỮ: OBJECTS là tên class tiếng Anh (COCO/OpenImages), còn câu
hỏi thường là tiếng Việt -> so khớp trực tiếp gần như luôn trượt.

THAY ĐỔI (bản này): trước đây hàm này TỰ gọi 1 lần Nemotron
(generate_query_keywords) để trích từ khóa tiếng Anh — TỐN THÊM 1 request
OpenRouter RIÊNG, ngay cả khi Tầng 1 (pipeline.py) đã dịch câu hỏi sang tiếng
Anh cho SigLIP (translate_query_en trong qwen_vqa.py). Giờ nhận thẳng bản
dịch đó qua tham số `english_query` và TOKENIZE CỤC BỘ (không gọi LLM lại) —
tiết kiệm 1 request/câu hỏi, đồng thời đảm bảo Object Rerank và SigLIP luôn
"hiểu" câu hỏi theo ĐÚNG 1 cách diễn giải tiếng Anh giống nhau. Nếu không có
`english_query` (vd gọi độc lập `python object_rerank.py "..."` để test) ->
tự fallback gọi generate_query_keywords() (LLM) như hành vi cũ.
"""

import re

from qwen_vqa import generate_query_keywords

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
# OBJECTS giờ có thể ở dạng làm giàu "person x3 (trái/phải-gần), car (giữa)"
# (xem build_metadata.py) — cần bóc phần đếm số lượng "xN" và phần vị trí
# trong ngoặc TRƯỚC khi tokenize, để so khớp entity vẫn hoạt động đúng như
# trước (so khớp "person" với "person", không bị dính rác "x3"/"tri"/"gan").
_PAREN_RE = re.compile(r"\([^)]*\)")
_COUNT_RE = re.compile(r"\bx\d+\b")

# Stopword tiếng Anh RẤT phổ biến — lọc ra khỏi bản dịch câu hỏi trước khi
# dùng làm keyword_set, để không lãng phí "suất" so khớp cho các từ chức năng
# vô nghĩa với object-detection (vd "a", "the", "is", "with"...).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "in", "on", "at", "of", "with", "and", "or", "to", "this", "that",
    "these", "those", "he", "she", "it", "they", "his", "her", "its",
    "their", "there", "here", "has", "have", "had", "for", "by", "from",
    "as", "who", "what", "where", "when", "how", "which", "than", "then",
    "some", "any", "into", "near", "front", "behind", "wearing", "shows",
    "showing", "appears", "scene", "video", "image", "picture",
}


def _normalize_word(w: str) -> str:
    """Chuẩn hoá 1 từ tiếng Anh: lowercase + bỏ ký tự không phải chữ/số."""
    return _NON_ALNUM_RE.sub("", w.lower().strip())


def _singularize(w: str) -> str:
    """Số ít hoá RẤT đơn giản (không dùng thư viện ngoài) — đủ để khớp các
    cặp phổ biến như "cars"~"car", "glasses"~"glass", "buildings"~"building"."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _tokenize_objects(objects_str: str) -> set[str]:
    result: set[str] = set()
    for segment in (objects_str or "").split(","):
        if not segment.strip():
            continue
        cleaned = _PAREN_RE.sub(" ", segment)
        cleaned = _COUNT_RE.sub(" ", cleaned)
        for w in cleaned.split():
            token = _singularize(_normalize_word(w))
            if token:
                result.add(token)
    return result


def _keywords_from_english_phrase(english_query: str) -> list[str]:
    """Trích danh sách từ khóa TỪ BẢN DỊCH TIẾNG ANH đã có sẵn (không gọi LLM
    lại) — tách từ, bỏ stopword, chuẩn hoá số ít, khử trùng lặp giữ thứ tự."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for raw in re.split(r"[^A-Za-z]+", english_query or ""):
        if not raw:
            continue
        token = _singularize(raw.lower())
        if not token or token in _STOPWORDS or len(token) < 2:
            continue
        if token not in seen_set:
            seen_set.add(token)
            seen.append(token)
    return seen[:10]


# ----- BỔ SUNG (fix bug hoà điểm giữa các video khác nhau) -----
# build_metadata.py đã tính sẵn VỊ TRÍ (trái/phải/giữa) + KÍCH THƯỚC
# (gần/xa) cho mỗi entity trong OBJECTS, vd "car (trái-gần)" — nhưng
# _tokenize_objects() (ở trên) CỐ TÌNH bỏ hẳn phần trong ngoặc trước khi so
# khớp (để tránh dính token rác như "x3"/"tri"/"gan" vào entity matching).
# Hậu quả: 2 candidate CÙNG chứa "person, car" nhưng thực ra khác hẳn về vị
# trí/kích thước trong khung hình (thậm chí ở 2 VIDEO khác nhau) sẽ LUÔN
# hoà điểm tuyệt đối — đúng bug user báo cáo (object_rerank_score=0.32 y hệt
# nhau giữa L30_V028 và L25_V027, xem repro_bug.py).
#
# Fix: khi câu hỏi (tiếng Việt) có nhắc từ chỉ vị trí, cộng thêm 1 bonus nhỏ
# cho candidate nào có entity KHỚP CHÍNH XÁC vị trí đó — tái sử dụng dữ liệu
# build_metadata.py đã tính, không cần gọi thêm LLM/tài nguyên nào.
_POSITION_WORDS_VI = {
    "bên trái": "trái", "phía trái": "trái", "trái": "trái",
    "bên phải": "phải", "phía phải": "phải", "phải": "phải",
    "chính giữa": "giữa", "ở giữa": "giữa", "giữa": "giữa",
    "phía trước": "gần", "gần camera": "gần", "cận cảnh": "gần",
    "phía sau": "xa", "đằng xa": "xa", "hậu cảnh": "xa", "xa": "xa",
}
_OBJECTS_ENTITY_RE = re.compile(r"^([a-zA-Z][a-zA-Z ]*?)(?:\s*x\d+)?\s*(?:\(([^)]*)\))?$")


def _extract_query_position_hints(query_vi: str) -> set[str]:
    """Trích các tag vị trí (trái/phải/giữa/gần/xa) được nhắc trong câu hỏi
    tiếng Việt gốc — dùng chuỗi DÀI HƠN trước để "bên trái" không bị match
    nhầm rời rạc là "trái" trước (dù ở đây ra cùng tag nên không ảnh hưởng,
    nhưng giữ đúng thứ tự cho rõ ràng/dễ mở rộng sau này)."""
    q = (query_vi or "").lower()
    hits = set()
    for phrase in sorted(_POSITION_WORDS_VI, key=len, reverse=True):
        if phrase in q:
            hits.add(_POSITION_WORDS_VI[phrase])
    return hits


def _entity_position_tags(objects_str: str) -> dict[str, set[str]]:
    """{entity_token: {tag,...}} — parse LẠI trực tiếp từ OBJECTS gốc (CHƯA
    bị _PAREN_RE/_COUNT_RE xoá) để lấy đúng tag vị trí/kích thước gắn với
    từng entity, phục vụ so khớp bonus ở rerank_by_objects()."""
    result: dict[str, set[str]] = {}
    for segment in (objects_str or "").split(","):
        seg = segment.strip()
        if not seg:
            continue
        m = _OBJECTS_ENTITY_RE.match(seg)
        if not m:
            continue
        entity_token = _singularize(_normalize_word(m.group(1)))
        if not entity_token:
            continue
        tags = set()
        paren = m.group(2)
        if paren:
            for group in paren.split("/"):
                for tag in group.split("-"):
                    tag = tag.strip()
                    if tag:
                        tags.add(tag)
        result.setdefault(entity_token, set()).update(tags)
    return result


def _compute_idf_weights(candidates: list[dict]) -> dict[str, float]:
    """FIX (không phân biệt candidate): token càng hiếm trong pool càng có
    trọng số cao (kiểu IDF), tính NGAY TRÊN pool candidate hiện tại (top-30
    của Tầng 1) — khớp đúng "umbrella" (hiếm) đóng góp điểm NHIỀU HƠN HẲN
    khớp "person" (phổ biến), tạo ra sự khác biệt điểm số thật sự giữa các
    candidate thay vì hòa điểm."""
    import math
    n = len(candidates) or 1
    doc_freq: dict[str, int] = {}
    for c in candidates:
        for t in _tokenize_objects(c.get("OBJECTS", "")):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return {t: math.log((n + 1) / (df + 0.5)) + 1.0 for t, df in doc_freq.items()}


def rerank_by_objects(
    query: str,
    candidates: list[dict],
    top_n: int = 10,
    english_query: str | None = None,
) -> list[dict]:
    """
    `candidates`: list metadata (output của Tầng 1 - SigLIP/hybrid retrieve),
    ĐÃ theo thứ tự liên quan giảm dần (dùng làm tie-break khi object-score bằng nhau).

    `query`: phần MÔ TẢ CẢNH của câu hỏi (tiếng Việt) — chỉ dùng làm fallback
    nếu `english_query` không được cung cấp (xem bên dưới).

    `english_query`: bản dịch tiếng Anh NGẮN GỌN của câu hỏi, THƯỜNG được
    truyền vào từ pipeline.py (đã tính 1 lần ở Tầng 1 qua
    qwen_vqa.translate_query_en(), dùng chung với SigLIP) — khi có, hàm này
    tokenize CỤC BỘ, KHÔNG gọi LLM. Nếu None (vd gọi độc lập để test), tự
    fallback gọi generate_query_keywords() (LLM, tốn 1 request).

    Trả về tối đa `top_n` candidate, mỗi cái được thêm:
        - "object_rerank_score": điểm khớp CÓ TRỌNG SỐ IDF (0..~1)
        - "object_rerank_matched": danh sách từ khóa thực sự khớp (debug/hiển thị)
    """
    if not candidates:
        return []

    if english_query:
        keywords = _keywords_from_english_phrase(english_query)
        source_note = "bản dịch tiếng Anh (tái dùng, không tốn thêm request)"
    else:
        keywords = generate_query_keywords(query)  # fallback LLM riêng (dùng khi gọi độc lập/test)
        source_note = "LLM riêng (fallback — không có english_query truyền vào)"

    keyword_set = {_singularize(_normalize_word(k)) for k in keywords if k.strip()}
    keyword_set.discard("")

    idf_weights = _compute_idf_weights(candidates)
    position_hints = _extract_query_position_hints(query)  # rỗng nếu câu hỏi không nhắc vị trí -> không đổi hành vi cũ

    scored = []
    for idx, c in enumerate(candidates):
        obj_set = _tokenize_objects(c.get("OBJECTS", ""))

        exact_overlap = keyword_set & obj_set
        remaining = keyword_set - exact_overlap
        partial_overlap = set()
        if remaining:
            for k in remaining:
                if len(k) < 3:
                    continue
                for o in obj_set:
                    if len(o) >= 3 and (k in o or o in k):
                        partial_overlap.add(k)
                        break

        if keyword_set:
            weighted = sum(idf_weights.get(k, 1.0) for k in exact_overlap) \
                + 0.5 * sum(idf_weights.get(k, 1.0) for k in partial_overlap)
            max_possible = sum(idf_weights.get(k, 1.0) for k in keyword_set) or 1.0
            score = weighted / max_possible
        else:
            score = 0.0

        # BỔ SUNG: bonus vị trí — PHÁ VỠ tie giữa các candidate chỉ khớp
        # đúng những entity CHUNG CHUNG (person/car...) nhưng thực ra khác
        # nhau về vị trí/kích thước trong khung hình (kể cả khác video hẳn),
        # tái dùng dữ liệu build_metadata.py đã tính, KHÔNG tốn thêm request.
        matched_position_entities = []
        if position_hints and exact_overlap:
            entity_tags = _entity_position_tags(c.get("OBJECTS", ""))
            matched_position_entities = [
                e for e in exact_overlap if position_hints & entity_tags.get(e, set())
            ]
        position_bonus = 0.15 * len(matched_position_entities)
        score = min(1.0, score + position_bonus)

        item = dict(c)
        item["object_rerank_score"] = round(score, 4)
        item["object_rerank_matched"] = sorted(exact_overlap | partial_overlap)
        item["object_rerank_position_bonus"] = round(position_bonus, 4)
        item["_siglip_rank"] = idx  # tie-break: giữ thứ tự SigLIP gốc khi điểm bằng nhau
        scored.append(item)

    scored.sort(key=lambda x: (-x["object_rerank_score"], x["_siglip_rank"]))

    if keywords:
        print(f"🔎 [object_rerank] Từ khóa từ {source_note}: {keywords}")
    else:
        print("⚠️ [object_rerank] Không có từ khóa — giữ nguyên thứ tự SigLIP.")

    return scored[:top_n]


if __name__ == "__main__":
    # test nhanh: python object_rerank.py "câu hỏi"
    import sys
    from retrieve import retrieve

    q = sys.argv[1] if len(sys.argv) > 1 else "người đàn ông đeo kính đứng trước toà nhà"
    res = retrieve(q, top_k=30)
    metadatas = res.get("metadatas", [[]])[0]
    top = rerank_by_objects(q, metadatas, top_n=10)  # không truyền english_query -> fallback LLM
    for m in top:
        print(f"{m.get('video_id')} | t={m.get('pts_time')} | score={m['object_rerank_score']:.2f} "
              f"| matched={m['object_rerank_matched']} | OBJECTS={m.get('OBJECTS','')[:80]}")