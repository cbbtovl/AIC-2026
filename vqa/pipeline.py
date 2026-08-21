# pipeline.py
"""
Pipeline (bản cập nhật — thêm Segment-Topic Boost + Anchor-Entity Boost +
Open-Text Extraction; và BẢN NÀY: BỎ HẲN OBJECT RERANK):

    177k keyframes
        -> Dịch câu hỏi sang tiếng Anh 1 lần (qwen_vqa.translate_query_en)   Tầng 1
        -> Trích anchor entities (person/org/province) — MỚI                Tầng 1
        -> SigLIP + BM25 (+ Anchor Boost + Segment-Topic Boost — MỚI)        top-30
        -> Identity Rescan (Tầng 1.5, identity_rescan.py)                    nếu cần
        -> Open-Text Rescan + Extraction (identity_rescan.py + open_text_qa.py)
        -> Florence-2 rerank cục bộ (rerank_vlm.py)                          top-5..10
        -> Temporal Grouping (grouping.py)                                   top-5 (FINAL_TOP_K)
        -> OCR trên ẢNH KEYFRAME THẬT của candidate đã chọn (ocr_utils.py)   Tầng 3
        -> Nemotron 3 Nano Omni CoT + Self-Consistency (qwen_vqa.py)         Tầng 4-5

=== BUGFIX LOG (BẢN NÀY) — BỎ OBJECT RERANK + FIX "retrieval đúng nhưng VLM
    chấm sai video hoàn toàn" ===

TRIỆU CHỨNG: log thực tế cho thấy Tầng 1 đã đúng — cả segment-boost lẫn
anchor-boost đều bắt được đúng segment/video FANA (org=['FANA'],
province=Khánh Hòa). Nhưng candidate CUỐI CÙNG đưa cho VLM lại là 1 video
hoàn toàn khác (OCR ra "Vợ Nhặt — Kim Lân", chẳng liên quan gì) — nghĩa là
tín hiệu đúng bị RỚT MẤT ở đâu đó giữa Tầng 1 và Tầng 4.

NGUYÊN NHÂN GỐC: Object Rerank (rerank_by_objects, object_rerank.py) chấm
điểm dựa trên field "OBJECTS" (nhãn object-detection tự động: person, car,
table...). Với các clip tin tức/sự kiện (trao quà, hội nghị...), OBJECTS
gần như KHÔNG có nhãn nào khớp từ khóa câu hỏi ("fana", "gift", "commune"...)
-> object_rerank_score = 0.00 cho HẦU HẾT candidate, kể cả candidate ĐÚNG vừa
được anchor/segment-boost kéo lên. Code cũ chỉ bảo vệ candidate có
"identity_match" (Identity Rescan) khỏi bị Object Rerank loại — KHÔNG bảo vệ
candidate có "_anchor_boosted"/"_segment_boosted" -> candidate đúng bị hoà
điểm 0.00 với hàng chục candidate sai khác, bị văng khỏi top_k_object_rerank
trước khi kịp tới Florence/VLM.

GIẢI PHÁP (theo yêu cầu — sau khi đã có OCR cache + anchor/segment-boost
mạnh hơn nhiều, Object Rerank dựa nhãn object-detection thô không còn đáng
tin cậy): BỎ HẲN bước Object Rerank. Candidate Tầng 1 (đã sort theo combined
score SigLIP+BM25, xem _sort_by_combined_score) được cắt còn top_k_object_rerank
(giữ tên tham số để không phải đổi chữ ký hàm run_pipeline/app.py) rồi đưa
THẲNG vào Florence-2 rerank (thị giác THẬT trên ảnh thật, local, không tốn
quota) — vẫn giữ NGUYÊN cơ chế bảo vệ candidate có bằng chứng thật
(identity_match / _anchor_boosted / _segment_boosted) khỏi bị cắt ở CẢ 2 chỗ:
lúc cắt pool ban đầu VÀ lúc Florence tự rerank xong.

=== BUGFIX / NÂNG CẤP LOG (giữ nguyên từ bản trước) ===

VẤN ĐỀ: câu hỏi dạng "CLB FANA trao quà tại 1 xã thuộc Khánh Hòa, xã đó tên
gì?" bị pipeline cũ trả lời sai vì 2 lỗ hổng CHỒNG NHAU:

  1) Tầng 1 (SigLIP + BM25) KHÔNG có tín hiệu nào cho "FANA"/"Khánh Hòa" —
     SigLIP chỉ so ngữ nghĩa thị giác chung (người, quà, trao tặng...), BM25
     chỉ khớp được nếu media-info (title/description) của đúng video đó tình
     cờ chứa sẵn 2 từ này (thường KHÔNG, vì media-info tự động sinh, sơ sài).
     -> Candidate pool tầng 1 sai video NGAY TỪ ĐẦU, mọi tầng sau chỉ đang
     tối ưu trên rác.

  2) Ngay cả khi tầng 1 tình cờ đúng video: identity_rescan.py (bản cũ) chỉ
     giải được bài toán "ĐÃ BIẾT target string, xác nhận đúng frame" (tên
     người). Với câu hỏi FANA, CHÍNH ĐÁP ÁN (tên xã) là ẩn số — không có gì
     để so khớp trước -> identity_rescan không kích hoạt được cho loại câu
     hỏi này.

GIẢI PHÁP (giữ nguyên, xem identity_rescan.py + open_text_qa.py):

  (a) ANCHOR-ENTITY BOOST: trích các loại anchor tổng quát hơn (không chỉ tên
      người) — tên tổ chức viết HOA ("FANA"), tên tỉnh/thành (gazetteer
      "Khánh Hòa") — rồi chạy 1 lượt BM25 RIÊNG bằng các anchor này, boost
      thẳng bm25_score để candidate ĐÚNG (nếu media-info có nhắc anchor) chắc
      chắn lọt top mà không phụ thuộc vào độ khớp ngữ nghĩa SigLIP.

  (b) OPEN-TEXT RESCAN + EXTRACTION: khi câu hỏi thuộc dạng "đáp án là 1
      chuỗi text chưa biết trước" (is_open_text_query) VÀ candidate đã hội tụ
      về ít video_id — quét OCR RỘNG (không so khớp gì, chỉ thu thập) trên
      các video nghi vấn, rồi dùng 1 LLM call TEXT-ONLY (rẻ hơn nhiều so với
      chấm N ảnh bằng VLM) để trích xuất đáp án trực tiếp từ khối OCR gộp.
      Nếu tìm được -> dùng NGAY làm kết quả cuối, bỏ qua bước VLM chấm ảnh
      (vừa rẻ hơn, vừa chính xác hơn vì đọc thẳng chữ trên hình thay vì suy
      đoán thị giác). Nếu không tìm được -> fallback nguyên vẹn về luồng VLM
      cũ, KHÔNG mất gì so với hành vi trước đây.

  Cả 3 (anchor-boost, segment-boost, open-text rescan) đều CHỈ kích hoạt khi
  có tín hiệu phù hợp — câu hỏi thường (mô tả cảnh/màu sắc/hành động) chạy y
  hệt luồng cũ, không tốn thêm chi phí.

=== BUGFIX LOG (giữ nguyên từ bản trước — bỏ grid 2x2, sửa SigLIP/OCR) ===
  - Không còn ghép lưới 2x2; OCR/VLM luôn dùng đúng `image_path` thật của
    candidate, khớp 100% với pts_time/frame_id trả về.
  - SigLIP luôn nhận câu hỏi ĐÃ DỊCH sang tiếng Anh (translate_query_en),
    dùng chung cho SigLIP + BM25 (lượt 2), không tốn thêm request LLM so với
    bản gốc.
"""
import time
from config import (
    RETRIEVAL_TOP_K, OBJECT_RERANK_TOP_K, FINAL_TOP_K, USE_QUERY_TRANSLATION,
    USE_FLORENCE_RERANK, FLORENCE_RERANK_POOL, USE_OCR,
    HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT,
    USE_ANSWER_TARGET_SPLIT, OCR_TRIGGER_KEYWORDS,
)
from retrieve import retrieve
from bm25_search import bm25_engine
from grouping import group_temporal
from ocr_utils import extract_text
from qwen_vqa import (
    translate_query_en, translate_and_extract_anchors, ask_gemini_self_consistency,
    predict_single_vote, split_scene_and_question, OpenRouterQuotaExhausted,
)
from identity_rescan import (
    maybe_rescan, extract_anchor_entities, maybe_open_text_rescan, is_plausible_org_name,
)
from open_text_qa import is_open_text_query, extract_answer_from_ocr_context
from video_utils import find_video_path, format_timestamp
from vn_gazetteer import find_province
from segment_topics import segment_boost_candidates  # MỚI


def _question_needs_ocr(question: str) -> bool:
    """Chỉ trả True nếu câu hỏi có dấu hiệu cần đọc chữ/text/biển hiệu/caption
    trong khung hình — xem config.OCR_TRIGGER_KEYWORDS. Câu hỏi thuần mô tả
    hành động/màu sắc/vật thể sẽ bỏ qua OCR, tiết kiệm thời gian EasyOCR mà
    không đánh đổi độ chính xác."""
    q_lower = (question or "").lower()
    return any(kw in q_lower for kw in OCR_TRIGGER_KEYWORDS)


def _hybrid_retrieve(query_vi: str, query_en: str, top_k: int) -> list[dict]:
    """Tầng 1: SigLIP vector search LUÔN chạy trên bản dịch tiếng Anh
    (`query_en` — xem translate_query_en() trong qwen_vqa.py; SigLIP-base
    không đa ngôn ngữ nên nhúng thẳng câu tiếng Việt khiến vector search gần
    như ngẫu nhiên, đây là NGUYÊN NHÂN GỐC của bug "SigLIP retrieve sai" ->
    kéo theo OCR/VLM chạy nhầm keyframe không liên quan).

    BM25 chạy SONG SONG trên CẢ câu gốc tiếng Việt (`query_vi`, khớp tốt với
    video_title/description thường tiếng Việt) VÀ bản dịch tiếng Anh
    (`query_en`, khớp tốt với field OBJECTS luôn là tên class tiếng Anh) — gộp
    kết quả theo 'id', CỘNG DỒN bm25_score nếu 1 frame khớp ở cả 2 lượt.

    Merge theo 'id' để không trùng lặp. Nếu ChromaDB chưa được ingest, vector
    search trả rỗng và pipeline vẫn chạy được nhờ BM25."""
    merged: dict[str, dict] = {}

    # --- Vector (SigLIP) — LUÔN dùng câu tiếng Anh ---
    vec_res = retrieve(query_en, top_k=top_k)
    ids = vec_res.get("ids", [[]])[0]
    metadatas = vec_res.get("metadatas", [[]])[0]
    distances = vec_res.get("distances", [[]])[0]
    for doc_id, meta, dist in zip(ids, metadatas, distances):
        item = dict(meta) if meta else {}
        item["id"] = item.get("id") or doc_id
        item["vector_score"] = 1.0 / (1.0 + dist)  # distance càng nhỏ -> score càng cao
        merged[item["id"]] = item

    # --- BM25 (text/objects) — chạy cả câu gốc lẫn bản dịch (nếu khác nhau) ---
    bm25_queries = [query_vi]
    if query_en and query_en.strip().lower() != query_vi.strip().lower():
        bm25_queries.append(query_en)

    for bq in bm25_queries:
        for item in bm25_engine.search(bq, top_k=top_k):
            key = item.get("id")
            if not key:
                continue
            if key in merged:
                merged[key]["bm25_score"] = merged[key].get("bm25_score", 0.0) + item.get("bm25_score", 0.0)
            else:
                item["vector_score"] = 0.0
                merged[key] = item

    return list(merged.values())


def _anchor_boost_retrieve(anchors: dict, top_k: int) -> list[dict]:
    """Chạy 1 lượt BM25 RIÊNG bằng các anchor entity (org_names + province +
    person_name) đã trích được từ câu hỏi.

    BUG/THIẾU SÓT ĐÃ SỬA: bản trước CỐ TÌNH bỏ person_name ra khỏi lượt boost
    này với lý do "đã có identity_rescan xử lý riêng" — nhưng identity_rescan
    (khi chưa có OCR cache ấm) CẦN candidate đã hội tụ về ít video_id TRƯỚC
    thì mới rescan được, còn rescan lại chính là thứ tạo ra sự hội tụ đó ->
    vòng luẩn quẩn con-gà-quả-trứng: với câu hỏi có TÊN NGƯỜI nhưng KHÔNG có
    org/province đi kèm, _anchor_boost_retrieve() trả về [] ngay (terms rỗng)
    -> không có gì giúp hội tụ -> identity_rescan gate (len(video_ids) >
    IDENTITY_RESCAN_MAX_VIDEOS) gần như luôn fail -> rescan không bao giờ
    chạy. Giờ đưa person_name vào CÙNG lượt boost BM25 như org/province: nếu
    media-info (title/description) của đúng video tình cờ nhắc tên đó, video
    sẽ được kéo lên ngay ở Tầng 1 — không cần đợi OCR/rescan. Việc này KHÔNG
    thay thế identity_rescan (vẫn cần cho trường hợp media-info không nhắc
    tên nhưng tên hiển thị dạng chữ TRÊN HÌNH) mà bổ sung thêm 1 đường hội tụ
    độc lập, rẻ (chỉ BM25, không tốn quota).

    ĐỘNG LỰC (giữ nguyên từ bản gốc): nếu media-info (title/description) của
    đúng video có nhắc tới "FANA"/"Khánh Hòa"/"Hồng Nhung" nhưng phần còn lại
    của câu hỏi (mô tả cảnh chung chung: người, quà, trao tặng...) lại không
    đặc trưng, combined score ở `_sort_by_combined_score()` có thể vẫn xếp
    video đúng khá thấp vì SigLIP/BM25-thường không có tín hiệu mạnh. Ta CỘNG
    THÊM 1 điểm bm25 rất cao (tương tự cách identity_rescan gán
    vector_score=1.0 cho identity hits) để đảm bảo candidate có nhắc anchor
    LUÔN lọt vào top pool, không bị lu mờ.

    Trả về list các item BM25 hit (đã gán bm25_score cực cao qua field
    '_anchor_boosted') — hàm gọi (run_pipeline) sẽ merge vào candidates chính
    và cộng dồn qua _sort_by_combined_score() như bình thường.

    Nếu không có anchor nào -> trả về [] (không tốn gì thêm)."""
    terms = list(anchors.get("org_names") or [])
    if anchors.get("province"):
        terms.append(anchors["province"])
    if anchors.get("person_name"):
        terms.append(anchors["person_name"])
    if not terms:
        return []

    anchor_query = " ".join(terms)
    hits = bm25_engine.search(anchor_query, top_k=top_k)
    for item in hits:
        # Điểm boost cực cao nhưng KHÔNG "vô hạn" (identity_rescan dùng 1.0
        # tuyệt đối cho vector_score vì đó là bằng chứng OCR THẬT — ở đây mới
        # chỉ là gợi ý từ BM25 nên vẫn cộng dồn thay vì ghi đè, để nhiều anchor
        # khớp cùng lúc -> điểm càng cao, đúng tinh thần BM25 gốc).
        item["bm25_score"] = item.get("bm25_score", 0.0) + 20.0
        item["_anchor_boosted"] = True
    if hits:
        print(f"🎯 [anchor_boost] Boost {len(hits)} candidate theo anchor: {terms}")
    return hits


def _sort_by_combined_score(candidates: list[dict]) -> list[dict]:
    """Sắp xếp theo combined score (vector + BM25, min-max normalize) — đây
    giờ là thứ tự CUỐI CÙNG quyết định candidate nào vào pool Florence-2
    (Object Rerank đã bị bỏ, xem BUGFIX LOG ở đầu file)."""
    if not candidates:
        return []

    vec_scores = [c.get("vector_score", 0.0) for c in candidates]
    bm25_scores = [c.get("bm25_score", 0.0) for c in candidates]
    v_max = max(vec_scores) or 1.0
    b_max = max(bm25_scores) or 1.0

    def combined(c):
        v = c.get("vector_score", 0.0) / v_max
        b = c.get("bm25_score", 0.0) / b_max
        return HYBRID_VECTOR_WEIGHT * v + HYBRID_BM25_WEIGHT * b

    return sorted(candidates, key=combined, reverse=True)


def _empty_result(search_query_en: str, message: str, t_start: float, extra_timings: dict | None = None) -> dict:
    """Kết quả rỗng dùng chung cho các trường hợp lỗi/không có dữ liệu — luôn có
    ĐỦ key mà app.py/main.py mong đợi, để tránh KeyError/crash ở tầng hiển thị."""
    timings = {"total": round(time.time() - t_start, 2)}
    if extra_timings:
        timings.update(extra_timings)
    return {
        "video_id": "", "frame_id": "", "pts_time": 0.0,
        "answer": message,
        "reasoning": "", "confidence": 0.0,
        "search_query_en": search_query_en, "top_candidates": [], "image_paths": [],
        "video_path": "", "timestamp_str": "00:00",
        "gemini_output": {}, "timings": timings,
    }


def _build_open_text_result(
    ocr_result: dict, grouped: list[dict], image_paths: list[dict],
    english_query: str, timings: dict, t_start: float,
) -> dict:
    """Đóng gói kết quả khi Open-Text Extraction TÌM ĐƯỢC đáp án — bỏ qua
    bước VLM chấm ảnh, dùng đúng format output mà app.py/main.py mong đợi
    (giống hệt output của solve_vqa_with_self_consistency)."""
    timings["total"] = round(time.time() - t_start, 2)
    video_path = find_video_path(ocr_result.get("video_id", "")) if ocr_result.get("video_id") else ""

    out = {
        "video_id": ocr_result.get("video_id", ""),
        "frame_id": ocr_result.get("frame_id", ""),
        "pts_time": ocr_result.get("pts_time", 0.0),
        "answer": ocr_result.get("answer", ""),
        "reasoning": "[Open-Text Extraction từ OCR] " + ocr_result.get("reasoning", ""),
        "confidence": 0.85,  # đọc trực tiếp từ chữ trên hình -> tự tin cao hơn suy đoán thị giác thường
        "_source": "open_text_extraction",
    }
    out["video_path"] = video_path
    out["timestamp_str"] = format_timestamp(out["pts_time"])
    out["search_query_en"] = english_query
    out["top_candidates"] = grouped
    out["image_paths"] = image_paths
    out["gemini_output"] = dict(out)
    out["timings"] = timings
    return out


def run_pipeline(
    question: str,
    use_self_consistency: bool = True,
    top_k_retrieval: int = RETRIEVAL_TOP_K,
    top_k_object_rerank: int = OBJECT_RERANK_TOP_K,
    top_k_final: int = FINAL_TOP_K,
) -> dict:
    timings = {}
    t_start = time.time()

    # ---- Tầng 1: (tuỳ chọn) Tách scene/question + Dịch tiếng Anh + Hybrid Retrieval -> top-30 ----
    t0 = time.time()
    if USE_ANSWER_TARGET_SPLIT:
        split = split_scene_and_question(question)
        scene_query = split["scene"]
    else:
        scene_query = question

    # ---- Dịch tiếng Anh + trích anchor entity CHUNG 1 LLM call (MỚI —
    # translate_and_extract_anchors, xem qwen_vqa.py) khi USE_QUERY_TRANSLATION
    # bật -- TỔNG QUÁT hơn hẳn heuristic regex cũ (Title-Case/viết-Hoa): LLM
    # hiểu ngữ cảnh nên vẫn trích đúng anchor dù câu hỏi không viết hoa chuẩn
    # hoặc lỗi chính tả, không tốn thêm request (gộp chung với bước dịch vốn
    # đã luôn chạy). Luôn MERGE thêm anchor từ regex heuristic
    # (extract_anchor_entities, miễn phí, 0 chi phí) làm lưới an toàn — hết
    # quota hoặc LLM bỏ sót vẫn còn nguồn tín hiệu thứ 2 độc lập. ----
    if USE_QUERY_TRANSLATION:
        llm_out = translate_and_extract_anchors(scene_query)
        english_query = llm_out["english"]
    else:
        english_query = scene_query
        llm_out = {"person_name": None, "org_names": [], "province": None}

    candidates = _hybrid_retrieve(scene_query, english_query, top_k=top_k_retrieval)

    regex_anchors = extract_anchor_entities(question)  # miễn phí, luôn chạy làm lưới an toàn

    # BUGFIX: org_names có thể lẫn từ chức năng tiếng Việt phổ biến (vd "đó")
    # bị nhận nhầm thành tên tổ chức — từ CẢ 2 NGUỒN (LLM hallucinate lẫn
    # regex _ORG_CAPS_RE khớp caps tình cờ), không chỉ riêng 1 nguồn. Lọc
    # SAU KHI merge bằng is_plausible_org_name() (xem identity_rescan.py) để
    # chặn triệt để trước khi anchor rác này lan ra segment_boost_candidates/
    # _anchor_boost_retrieve/maybe_open_text_rescan và kéo theo hàng chục
    # candidate KHÔNG liên quan (vd mọi frame OCR có chữ "đó").
    raw_org_names = sorted(set((llm_out.get("org_names") or []) + (regex_anchors.get("org_names") or [])))

    # BUGFIX (đối xứng với org_names ở trên): LLM (translate_and_extract_anchors)
    # cũng có thể trả person_name = 1 tên tỉnh/thành (hallucinate), không chỉ
    # riêng identity_rescan.extract_target_name() mới mắc lỗi này. Nếu lọt qua,
    # maybe_rescan()/maybe_open_text_rescan() coi đây là "tên người cần xác
    # nhận" -> quét OCR cache toàn database, khớp mọi video tình cờ nhắc tên
    # tỉnh, gán identity_match=True (vector_score=1.0 tối đa) cho hàng loạt
    # candidate KHÔNG liên quan, đè lên candidate đúng chỉ có bm25 boost
    # thường (xem BUGFIX LOG). extract_target_name() (regex) đã tự lọc rồi,
    # ở đây chỉ cần chặn thêm phía LLM.
    raw_person_name = llm_out.get("person_name") or regex_anchors.get("person_name")
    person_name = None if (raw_person_name and find_province(raw_person_name) == raw_person_name) else raw_person_name

    anchors = {
        "person_name": person_name,
        "org_names": [o for o in raw_org_names if is_plausible_org_name(o)],
        "province": llm_out.get("province") or regex_anchors.get("province"),
    }

    # ---- (MỚI) Segment-Topic Boost — xem segment_topics.py + segment_embed.py.
    # Chạy TRƯỚC anchor-boost frame-level vì đây là nguồn tín hiệu MẠNH HƠN:
    # kéo NGUYÊN CỤM frame cùng chủ đề/sự kiện vào candidate pool (không chỉ
    # 1 frame trùng chữ tình cờ), qua 2 cơ chế kết hợp:
    #   (1) anchor match — org/province của segment khớp CHÍNH XÁC anchor
    #       vừa trích ở trên (rẻ, không cần model).
    #   (2) similarity match — cosine BGE-M3 giữa câu hỏi và segment summary,
    #       bắt được câu hỏi DIỄN GIẢI LẠI nội dung mà không trùng từ khoá
    #       nào với anchor (vd "trao quà tại 1 xã thuộc Khánh Hòa" khớp
    #       ngữ nghĩa với summary "CLB FANA trao quà... tỉnh Khánh Hòa" dù
    #       không chung từ nào).
    # Khác identity_rescan.maybe_open_text_rescan (cần candidate đã hội tụ
    # TRƯỚC mới rescan được — vòng luẩn quẩn con-gà-quả-trứng đã ghi ở đầu
    # file): segment-boost hoạt động NGAY LẬP TỨC, không phụ thuộc candidate
    # ban đầu đúng hay sai, vì tra cứu trực tiếp trên dữ liệu đã tiền xử lý
    # offline. Nếu chưa chạy segment_topics.py/segment_embed.py -> trả về []
    # an toàn ở phần tương ứng, KHÔNG crash pipeline. ----
    segment_hits = segment_boost_candidates(question, anchors, max_segments=5)
    if segment_hits:
        existing_ids = {c.get("id") for c in candidates}
        for hit in segment_hits:
            # Boost cao hơn cả anchor thường (20.0, xem _anchor_boost_retrieve
            # bên dưới) vì đây là bằng chứng theo CỤM đã xác nhận offline
            # (nhiều frame cùng đồng ý), không chỉ 1 dòng media-info tình cờ
            # trùng chữ. Match "both" (cả anchor lẫn similarity cùng chọn)
            # được cộng thêm để càng chắc chắn hơn nữa.
            base_boost = 30.0
            if hit.get("_segment_match_type") == "both":
                base_boost += 10.0
            hit["bm25_score"] = max(hit.get("bm25_score", 0.0), base_boost)
            hit["vector_score"] = hit.get("vector_score", 0.0)
            if hit.get("id") in existing_ids:
                for c in candidates:
                    if c.get("id") == hit["id"]:
                        c["bm25_score"] = max(c.get("bm25_score", 0.0), hit["bm25_score"])
                        c["segment_summary"] = hit.get("segment_summary", "")
                        c["_segment_boosted"] = True
                        c["_segment_match_type"] = hit.get("_segment_match_type")
                        break
            else:
                candidates.append(hit)
                existing_ids.add(hit.get("id"))
        candidates = _sort_by_combined_score(candidates)

    # ---- Anchor-Entity Boost — trích tên tổ chức viết HOA / tên tỉnh từ câu
    # hỏi (KHÔNG gọi LLM, chi phí ~0) và boost thẳng bm25_score cho các
    # candidate CÓ MEDIA-INFO nhắc tới anchor đó (bổ sung độc lập với
    # segment-boost ở trên -- segment-boost dựa trên OCR THẤY TRÊN HÌNH,
    # anchor-boost dưới đây dựa trên title/description của video). ----
    anchor_hits = _anchor_boost_retrieve(anchors, top_k=top_k_retrieval)
    if anchor_hits:
        existing_ids = {c.get("id") for c in candidates}
        for hit in anchor_hits:
            key = hit.get("id")
            if not key:
                continue
            if key in existing_ids:
                for c in candidates:
                    if c.get("id") == key:
                        c["bm25_score"] = max(c.get("bm25_score", 0.0), hit["bm25_score"])
                        c["_anchor_boosted"] = True
                        break
            else:
                candidates.append(hit)
                existing_ids.add(key)

    candidates = _sort_by_combined_score(candidates)
    timings["retrieval"] = round(time.time() - t0, 2)

    # ---- Tầng 1.5a: Identity Rescan — fix "chọn sai vùng thời gian" khi câu
    # hỏi nhắc TÊN NGƯỜI cụ thể (giữ nguyên hành vi cũ, xem identity_rescan.py) ----
    t0 = time.time()
    identity_hits = maybe_rescan(question, candidates)
    if identity_hits:
        existing_ids = {c.get("id") for c in candidates}
        for hit in identity_hits:
            hit["vector_score"] = 1.0
            hit["bm25_score"] = max((c.get("bm25_score", 0.0) for c in candidates), default=1.0) or 1.0
            if hit.get("id") in existing_ids:
                for c in candidates:
                    if c.get("id") == hit["id"]:
                        c.update(hit)
                        break
            else:
                candidates.append(hit)
                existing_ids.add(hit.get("id"))
        candidates = _sort_by_combined_score(candidates)
    timings["identity_rescan"] = round(time.time() - t0, 2)

    if not candidates:
        return _empty_result(
            english_query,
            "⚠️ Không tìm thấy candidate nào. Kiểm tra: (1) đã chạy build_metadata.py? "
            "(2) đã chạy store.py để ingest ảnh vào ChromaDB? (3) đã chạy bm25_search.py build index?",
            t_start, timings,
        )

    # ---- Tầng 1.5b (MỚI): Open-Text Rescan + Extraction — cho câu hỏi dạng
    # "xã này tên gì?" nơi ĐÁP ÁN LÀ ẨN SỐ, không có target string để so khớp
    # trước như identity rescan. Quét OCR RỘNG trên các video nghi vấn rồi hỏi
    # LLM (text-only, rẻ) trích xuất trực tiếp. Nếu TÌM ĐƯỢC -> trả kết quả
    # NGAY, bỏ qua toàn bộ Florence/VLM phía sau (tiết kiệm quota + chính xác
    # hơn). Nếu KHÔNG tìm được -> fallback nguyên vẹn về luồng cũ bên dưới,
    # không mất gì. ----
    t0 = time.time()
    if is_open_text_query(question):
        open_text_frames = maybe_open_text_rescan(question, candidates, anchors=anchors)
        if open_text_frames:
            try:
                ocr_answer = extract_answer_from_ocr_context(question, open_text_frames)
            except OpenRouterQuotaExhausted as e:
                print(f"⛔ [open_text] {e} — bỏ qua open-text extraction, fallback về luồng VLM thường.")
                ocr_answer = {"found": False}

            timings["open_text_rescan"] = round(time.time() - t0, 2)

            if ocr_answer.get("found"):
                print(f"✅ [open_text] Tìm được đáp án trực tiếp từ OCR: "
                      f"'{ocr_answer['answer']}' tại video={ocr_answer.get('video_id')} "
                      f"t={ocr_answer.get('pts_time')}s")
                # image_paths chỉ để hiển thị debug — dùng frame đã tìm thấy + vài
                # candidate gốc, KHÔNG cần chạy full grouping/OCR/VLM phía dưới nữa.
                matched_frame = next(
                    (f for f in open_text_frames if f.get("id") == ocr_answer.get("frame_id")), None,
                )
                grouped_preview = [matched_frame] if matched_frame else []
                image_paths_preview = [f.get("image_path", "") for f in grouped_preview]
                return _build_open_text_result(
                    ocr_answer, grouped_preview, image_paths_preview,
                    english_query, timings, t_start,
                )
            else:
                print("ℹ️ [open_text] Không trích xuất được đáp án rõ ràng từ OCR "
                      "— tiếp tục luồng VLM chấm ảnh như bình thường.")
        else:
            timings["open_text_rescan"] = round(time.time() - t0, 2)
    else:
        timings["open_text_rescan"] = 0.0

    # ---- Tầng 2 (BỎ OBJECT RERANK — xem BUGFIX LOG đầu file): candidate ở
    # đây đã được sort theo combined score (SigLIP + BM25 + anchor/segment
    # boost, xem _sort_by_combined_score). Cắt còn top_k_object_rerank ứng
    # viên (giữ tên tham số cũ để không phải đổi chữ ký hàm/app.py) làm pool
    # đưa vào Florence-2 — NHƯNG luôn "protected_ids" gồm mọi candidate có
    # bằng chứng thật (identity_match / anchor_boosted / segment_boosted) để
    # KHÔNG BAO GIỜ bị cắt mất chỉ vì thứ hạng combined score thấp hơn 1 chút
    # so với candidate khác (đây chính là bug đã fix: trước đây các candidate
    # này bị Object Rerank/OBJECTS-matching cho điểm 0.00 rồi văng khỏi pool). ----
    t0 = time.time()
    protected_ids = {
        c.get("id") for c in candidates
        if c.get("identity_match") or c.get("_anchor_boosted") or c.get("_segment_boosted")
    }
    pool = candidates[:top_k_object_rerank]
    if protected_ids:
        kept_ids = {c.get("id") for c in pool}
        missing = [c for c in candidates if c.get("id") in protected_ids and c.get("id") not in kept_ids]
        if missing:
            # Ưu tiên identity_match (bằng chứng OCR xác nhận danh tính) lên
            # đầu, rồi tới bm25_score — chỉ để thứ tự ổn định, không ảnh hưởng
            # việc có được GIỮ LẠI hay không (đã đảm bảo ở trên).
            missing.sort(key=lambda c: (bool(c.get("identity_match")), c.get("bm25_score", 0.0)), reverse=True)
            pool = missing + pool
    timings["object_rerank"] = round(time.time() - t0, 2)  # giữ key cũ để app.py không phải sửa

    # ---- Florence-2 rerank THỊ GIÁC THẬT trên ảnh thật (image_path), chạy
    # LOCAL, không tốn quota OpenRouter. Đây giờ là bước rerank DUY NHẤT
    # trước khi grouping/VLM. ----
    t0 = time.time()
    if USE_FLORENCE_RERANK:
        from rerank_vlm import rerank_with_florence
        visual_pool = min(len(pool), max(top_k_final, FLORENCE_RERANK_POOL))
        object_reranked = rerank_with_florence(english_query, pool, top_n=visual_pool)
        if protected_ids:
            kept_ids = {c.get("id") for c in object_reranked}
            missing = [c for c in pool if c.get("id") in protected_ids and c.get("id") not in kept_ids]
            if missing:
                object_reranked = missing + object_reranked
            object_reranked.sort(key=lambda c: not (
                c.get("identity_match") or c.get("_anchor_boosted") or c.get("_segment_boosted")
            ))
    else:
        object_reranked = pool
    timings["florence_rerank"] = round(time.time() - t0, 2)

    # ---- Gom nhóm theo thời gian + độ giống thị giác, cắt còn top_k_final ----
    grouped = group_temporal(object_reranked, top_n=top_k_final)

    # ---- Tầng 3: đường dẫn ảnh keyframe THẬT của từng candidate (KHÔNG còn
    # ghép lưới 2x2). Đây CHÍNH LÀ ảnh sẽ đưa cho cả OCR lẫn VLM, đảm bảo cả 2
    # luôn "nhìn" đúng 1 khung hình khớp với pts_time/frame_id trả về. ----
    image_paths = [c.get("image_path", "") for c in grouped]

    # ---- OCR — CHỈ trên các candidate đã chọn (top 5-10), VÀ CHỈ khi câu hỏi
    # thực sự có dấu hiệu cần đọc chữ — xem _question_needs_ocr(). ----
    t0 = time.time()
    run_ocr = USE_OCR and _question_needs_ocr(question)
    if run_ocr:
        for cand in grouped:
            # Nếu đã có ocr_text sẵn từ Identity Rescan (Tầng 1.5, cũng OCR
            # trên chính image_path) -> tái dùng, khỏi OCR lại lần nữa.
            if cand.get("ocr_text"):
                continue
            img_path = cand.get("image_path", "")
            cand["ocr_text"] = extract_text(img_path) if img_path else ""
    else:
        for cand in grouped:
            cand.setdefault("ocr_text", "")
        if USE_OCR:
            print("ℹ️ [OCR] Bỏ qua — câu hỏi không có dấu hiệu cần đọc chữ/text trong ảnh.")
    timings["ocr"] = round(time.time() - t0, 2)

    # ---- Tầng 4 + 5: VLM CoT (Nemotron, + Self-Consistency) ----
    t0 = time.time()
    try:
        if use_self_consistency:
            gemini_out = ask_gemini_self_consistency(question, image_paths, grouped)
        else:
            gemini_out = predict_single_vote(question, image_paths, grouped)
    except Exception as e:
        import traceback
        traceback.print_exc()
        timings["gemini"] = round(time.time() - t0, 2)
        timings["total"] = round(time.time() - t_start, 2)
        result = _empty_result(
            english_query,
            f"⚠️ Lỗi ở tầng VQA (OpenRouter): {type(e).__name__}: {e}",
            t_start, timings,
        )
        result["top_candidates"] = grouped
        result["image_paths"] = image_paths
        return result
    timings["gemini"] = round(time.time() - t0, 2)
    timings["total"] = round(time.time() - t_start, 2)

    # ---- Gắn video_path + timestamp dễ đọc cho kết quả cuối (dùng khi có file .mp4 gốc) ----
    final_video_id = gemini_out.get("video_id", "")
    video_path = ""
    for cand in grouped:
        if cand.get("video_id") == final_video_id and cand.get("video_path"):
            video_path = cand["video_path"]
            break
    if not video_path and final_video_id:
        video_path = find_video_path(final_video_id)  # fallback: tra cứu trực tiếp nếu metadata thiếu field này

    gemini_out["video_path"] = video_path
    gemini_out["timestamp_str"] = format_timestamp(gemini_out.get("pts_time", 0))
    gemini_out["search_query_en"] = english_query
    gemini_out["top_candidates"] = grouped
    gemini_out["image_paths"] = image_paths
    gemini_out["gemini_output"] = dict(gemini_out)  # tương thích app.py: res["gemini_output"]
    gemini_out["timings"] = timings

    if gemini_out.get("_quota_exhausted"):
        gemini_out["answer"] = (
            f"⚠️ [Hết quota OpenRouter free-tier — kết quả có thể chưa đầy đủ] "
            f"{gemini_out.get('answer', '')}"
        )

    return gemini_out