# identity_rescan.py
"""
TẦNG 1.5 — Fix "Retrieval chọn sai vùng thời gian / sai video" khi câu hỏi
nhắc tới các THỰC THỂ RIÊNG mà SigLIP/BM25 (Tầng 1, dựa ngữ nghĩa thị giác +
text tổng quát) không đủ sức phân biệt.

=== BUGFIX LOG (BẢN NÀY) — extract_target_name() bắt nhầm từ đánh dấu câu hỏi ===

TRIỆU CHỨNG: với câu hỏi dạng "...tỉnh Khánh Hòa. Hỏi: xã đó tên gì?",
heuristic Title-Case cũ nối liền "Khánh Hòa" với "Hỏi" (chữ hoa đầu câu hỏi)
thành 1 cụm "Khánh Hòa Hỏi" -> tưởng đây là tên người -> tốn hàng trăm giây
rescan tìm 1 cái tên không tồn tại (chắc chắn fail), đồng thời làm loãng
anchor_boost/open_text_rescan bằng 1 cụm từ rác.

NGUYÊN NHÂN: is_title chỉ kiểm tra "chữ đầu hoa, còn lại thường" — không loại
trừ các từ ĐÁNH DẤU CẤU TRÚC câu hỏi (thường đứng đầu câu/mệnh đề, viết hoa
theo NGỮ PHÁP chứ không phải vì là tên riêng): "Hỏi", "Đáp", "Câu"...

GIẢI PHÁP: thêm _QUESTION_MARKER_WORDS — nếu 1 từ Title-Case trùng (không
phân biệt hoa/thường sau khi strip dấu câu) với danh sách này, KHÔNG coi nó
là phần của tên riêng, cắt run tại đó (giống hệt gặp 1 từ không phải
Title-Case). Danh sách cố tình BAO PHỦ ít nhưng ĐÚNG — không cố loại các từ
Title-Case khác (rủi ro loại nhầm tên người thật bắt đầu giống 1 từ thường).
"""

import os
import re
import json
import unicodedata

from config import (
    IDENTITY_RESCAN_ENABLED, IDENTITY_RESCAN_MAX_FRAMES,
    IDENTITY_RESCAN_SAMPLE_STRIDE, IDENTITY_RESCAN_MAX_VIDEOS,
    OCR_CACHE_PATH,
)
from ocr_utils import extract_text
from vn_gazetteer import find_province

# Cache OCR đã precompute offline (ocr_precompute.py), nạp 1 LẦN/process dạng
# {frame_id: ocr_text}. Đọc trực tiếp từ OCR_CACHE_PATH (KHÔNG qua
# bm25_engine.metadatas) để luôn phản ánh đúng cache mới nhất trên đĩa, kể cả
# khi BM25 index chưa được rebuild lại.
_ocr_cache: dict[str, str] | None = None


def _load_ocr_cache() -> dict[str, str]:
    global _ocr_cache
    if _ocr_cache is not None:
        return _ocr_cache
    cache: dict[str, str] = {}
    if os.path.exists(OCR_CACHE_PATH):
        with open(OCR_CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id") and rec.get("ocr_text"):
                    cache[rec["id"]] = rec["ocr_text"]
    _ocr_cache = cache
    return cache

# ----- (MỚI) Giới hạn riêng cho open-text sweep — thường cần quét NHIỀU frame
# hơn identity rescan (vì banner/backdrop ghi tên xã có thể chỉ xuất hiện
# thoáng qua ở 1 vài giây, không lặp lại xuyên suốt như tên người trên caption
# cố định) — nhưng vẫn phải có trần để không quét vô hạn video dài. Có thể đưa
# lên config.py nếu bạn muốn chỉnh dễ hơn; để tạm ở đây cho gọn patch. -----
OPEN_TEXT_RESCAN_MAX_FRAMES = 250
OPEN_TEXT_RESCAN_SAMPLE_STRIDE = 1

# Cache toàn bộ metadata theo video_id, build 1 lần từ BM25 engine (đã load sẵn
# metadata_all.jsonl vào RAM) -> KHÔNG đọc lại file, không tốn thêm I/O đáng kể.
_frames_by_video: dict[str, list[dict]] | None = None
# Map phẳng id -> metadata, dùng cho tra cứu toàn database qua OCR cache
# (search_precomputed_ocr_for_name/terms) mà không cần biết trước video_id.
_id_to_frame: dict[str, dict] | None = None


def _build_video_index() -> dict[str, list[dict]]:
    global _frames_by_video, _id_to_frame
    if _frames_by_video is not None:
        return _frames_by_video

    from bm25_search import bm25_engine
    if not bm25_engine.metadatas:
        bm25_engine.build_or_load()

    index: dict[str, list[dict]] = {}
    id_map: dict[str, dict] = {}
    for m in bm25_engine.metadatas:
        vid = m.get("video_id")
        if not vid:
            continue
        index.setdefault(vid, []).append(m)
        if m.get("id"):
            id_map[m["id"]] = m

    for vid, frames in index.items():
        frames.sort(key=lambda f: f.get("pts_time", 0.0))

    _frames_by_video = index
    _id_to_frame = id_map
    return index


# ----- (MỚI, xem BUGFIX LOG đầu file) Các từ Title-Case là DẤU HIỆU CẤU TRÚC
# câu hỏi/văn bản, KHÔNG phải tên riêng — dù viết hoa chữ đầu (do đứng đầu
# câu/mệnh đề). So khớp KHÔNG phân biệt hoa-thường sau khi đã strip dấu câu
# (giống cách `core` được tính trong extract_target_name). -----
_QUESTION_MARKER_WORDS = {
    "hỏi", "đáp", "câu", "video", "đoạn", "ảnh", "hình", "ghi", "chú",
}


def extract_target_name(query: str) -> str | None:
    """Trích TÊN RIÊNG khả nghi từ câu hỏi bằng heuristic (KHÔNG tốn quota
    OpenRouter): tìm chuỗi >= 2 từ liên tiếp viết Hoa-Đầu-Từ (Title Case) —
    cách viết tên người tiếng Việt luôn giữ NGUYÊN dù nằm giữa câu, khác với
    danh từ thường. VD: "cô giáo Hồng Nhung đang dạy..." -> "Hồng Nhung".

    BUGFIX (xem BUGFIX LOG đầu file): loại trừ các từ đánh dấu cấu trúc câu
    hỏi ("Hỏi", "Đáp", "Câu"...) khỏi run Title-Case — trước đây câu dạng
    "...tỉnh Khánh Hòa. Hỏi: ..." bị nối nhầm thành tên người "Khánh Hòa Hỏi".

    Đây là heuristic, có thể trượt/sai một số trường hợp hiếm — nhưng chi phí
    bằng 0 (không gọi LLM) và đủ tốt cho phần lớn câu hỏi có tên người rõ ràng
    kiểu AIC. Trả về None nếu không tìm được ứng viên hợp lý."""
    words = query.strip().split()
    runs: list[list[str]] = []
    current: list[str] = []
    for w in words:
        core = w.strip(",.?!:;\"'()“”‘’")
        is_title = len(core) > 1 and core[0].isupper() and core[1:].islower()
        is_marker = core.lower() in _QUESTION_MARKER_WORDS
        if is_title and not is_marker:
            current.append(core)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)

    if not runs:
        return None
    best = max(runs, key=len)
    candidate = " ".join(best)

    # (MỚI, BUGFIX) Loại nếu TOÀN BỘ cụm vừa trích trùng khít 1 tên tỉnh/thành
    # (vd "Khánh Hòa") — 2 từ Title-Case liên tiếp khớp y hệt điều kiện nhận
    # diện tên người, nhưng đây là ĐỊA DANH, không phải người. Nếu không loại,
    # maybe_rescan()/search_precomputed_ocr_for_name() sẽ quét TOÀN BỘ OCR
    # cache tìm chữ "Khánh Hòa" -- khớp bất kỳ video nào tình cờ nhắc tên tỉnh
    # (vd phóng sự địa lý, thời sự khác), gán "identity_match": True +
    # vector_score=1.0 (điểm TỐI ĐA) cho hàng loạt candidate SAI, đè hẳn lên
    # candidate ĐÚNG chỉ có bm25 boost thường (xem BUGFIX LOG pipeline.py).
    # Chỉ so khớp TOÀN BỘ chuỗi (không phải substring) để không loại nhầm tên
    # người thật có chứa tên tỉnh bên trong (vd "Nguyễn Khánh Hòa").
    if find_province(candidate) == candidate:
        return None

    return candidate


# ---------------------------------------------------------------------------
# (MỚI) Tổng quát hoá: trích NHIỀU loại anchor entity, không chỉ tên người.
# ---------------------------------------------------------------------------

# Từ viết HOA toàn bộ 2-6 ký tự — heuristic bắt tên viết tắt/tổ chức kiểu
# "FANA", "CLB ABC", "UBND"... Loại trừ vài từ viết hoa phổ biến không mang
# nghĩa "tên riêng tổ chức" hay gặp trong câu hỏi AIC (đơn vị đo, số La Mã...)
# để giảm false positive.
_ORG_CAPS_RE = re.compile(r"\b[A-ZÀ-Ỹ]{2,6}\b")
_ORG_CAPS_STOPWORDS = {"TP", "TV", "MC", "VN", "USD", "VND", "OK"}


# (MỚI, BUGFIX) Từ chức năng tiếng Việt (đại từ/liên từ/trạng từ) phổ biến,
# thường xuất hiện viết-Hoa do nhấn mạnh, đầu câu, hoặc lỗi transcribe —
# TRÙNG độ dài 2-6 ký tự với heuristic org-name (_ORG_CAPS_RE) nhưng KHÔNG
# phải tên tổ chức thật. Trường hợp thực tế gây lỗi: câu hỏi có "...xã ĐÓ
# tên gì?" -> "ĐÓ" bị nhận nhầm là org, kéo theo search_precomputed_ocr_for_terms
# khớp GẦN NHƯ MỌI frame có chữ "đó" trong OCR (từ cực phổ biến) -> ngập
# candidate rác. So khớp KHÔNG PHÂN BIỆT HOA-THƯỜNG (dùng bởi
# is_plausible_org_name() bên dưới để lọc CẢ nguồn LLM lẫn nguồn regex).
_COMMON_VN_WORDS_NOT_ORG = {
    "ĐÓ", "NÀY", "ĐÂY", "LÀ", "VÀ", "CÓ", "KHÔNG", "ĐÃ", "SẼ", "RẤT",
    "CŨNG", "NHƯ", "VỚI", "CHO", "TỪ", "TẠI", "TRONG", "NGOÀI", "SAU",
    "TRƯỚC", "AI", "GÌ", "NÀO", "SAO", "VẬY", "THẾ", "NHÉ", "NHA", "MÀ",
    "THÌ", "NHƯNG", "HAY", "HOẶC", "VỀ", "ĐANG", "VẪN", "CHỈ", "CÒN",
}


def is_plausible_org_name(name: str) -> bool:
    """(MỚI) Kiểm tra 1 chuỗi có HỢP LÝ để coi là tên tổ chức/CLB hay không —
    dùng CHUNG cho cả anchor trích bằng regex (_extract_org_names, bên dưới)
    LẪN anchor do LLM trả về (translate_and_extract_anchors trong qwen_vqa.py)
    — vì LLM cũng có thể hallucinate 1 từ chức năng thường (vd "đó") thành
    org_name, KHÔNG chỉ riêng heuristic regex mới mắc lỗi này. pipeline.py
    nên lọc qua hàm này SAU KHI merge org_names từ cả 2 nguồn."""
    if not name or not name.strip():
        return False
    normalized = name.strip().upper()
    if normalized in _ORG_CAPS_STOPWORDS or normalized in _COMMON_VN_WORDS_NOT_ORG:
        return False
    return True


def _extract_org_names(query: str) -> list[str]:
    hits = []
    for m in _ORG_CAPS_RE.finditer(query):
        token = m.group(0)
        if not is_plausible_org_name(token):
            continue
        if token not in hits:
            hits.append(token)
    return hits


def extract_anchor_entities(query: str) -> dict:
    """Tổng quát hoá `extract_target_name()`: trả về TẤT CẢ loại anchor tìm
    được trong câu hỏi, không chỉ tên người, để pipeline.py dùng boost
    retrieval Tầng 1 bằng nhiều tín hiệu khác nhau cùng lúc.

    Trả về:
        {
          "person_name": str | None,   # vd "Hồng Nhung"
          "org_names": list[str],      # vd ["FANA"]
          "province": str | None,      # vd "Khánh Hòa"
        }
    Không gọi LLM — chi phí bằng 0, an toàn để gọi ở MỌI câu hỏi."""
    return {
        "person_name": extract_target_name(query),
        "org_names": _extract_org_names(query),
        "province": find_province(query),
    }


def _normalize_for_match(text: str) -> str:
    """Chuẩn hoá bỏ dấu tiếng Việt + lowercase, để so khớp OCR (thường lỗi/thiếu
    dấu) với tên trích từ câu hỏi một cách khoan dung hơn so khớp tuyệt đối."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _name_matches(ocr_text: str, name: str) -> bool:
    if not ocr_text or not name:
        return False
    return _normalize_for_match(name) in _normalize_for_match(ocr_text)


def rescan_videos_for_identity(name: str, video_ids: list[str]) -> list[dict]:
    """Quét (các) video trong `video_ids` tìm frame có caption/chyron chứa
    `name`, ƯU TIÊN đọc từ OCR CACHE đã precompute offline (gần như miễn phí
    — chỉ tra dict) thay vì luôn gọi EasyOCR sống như trước:

        - Frame ĐÃ có trong cache: quét TOÀN BỘ (không giới hạn/không lấy
          mẫu stride nữa — cache rẻ nên phủ được hết cả video).
        - Frame CHƯA có trong cache: fallback OCR SỐNG như hành vi cũ, vẫn
          giữ giới hạn IDENTITY_RESCAN_MAX_FRAMES + lấy mẫu stride (vì OCR
          sống tốn thời gian thật), để không làm treo pipeline khi cache còn
          thưa (build_ocr_precompute.py chưa chạy xong toàn bộ).

    Trả về list metadata của các frame KHỚP TÊN, mỗi cái được gắn thêm:
        - "ocr_text": text OCR đã trích (tái dùng luôn, khỏi OCR lại ở Tầng 3.5)
        - "identity_match": True
    Đây là BẰNG CHỨNG THẬT (tên hiển thị đúng trên hình), nên pipeline.py sẽ
    ưu tiên các frame này lên đầu, bất kể điểm SigLIP/BM25 ra sao."""
    index = _build_video_index()
    cache = _load_ocr_cache()
    matches = []

    for vid in video_ids:
        frames = index.get(vid, [])
        if not frames:
            continue

        cached_frames = [f for f in frames if f.get("id") in cache]
        uncached_frames = [f for f in frames if f.get("id") not in cache]

        if cached_frames:
            print(f"🔍 [identity_rescan] {vid}: {len(cached_frames)}/{len(frames)} frame "
                  f"có sẵn trong OCR cache -> quét TOÀN BỘ (gần như miễn phí) để tìm "
                  f"tên '{name}'...")
        for m in cached_frames:
            text = cache[m["id"]]
            if _name_matches(text, name):
                item = dict(m)
                item["ocr_text"] = text
                item["identity_match"] = True
                matches.append(item)

        if uncached_frames:
            if len(uncached_frames) > IDENTITY_RESCAN_MAX_FRAMES:
                stride = max(1, len(uncached_frames) // IDENTITY_RESCAN_MAX_FRAMES)
            else:
                stride = max(1, IDENTITY_RESCAN_SAMPLE_STRIDE)
            sampled = uncached_frames[::stride][:IDENTITY_RESCAN_MAX_FRAMES]
            print(f"🔍 [identity_rescan] {vid}: {len(sampled)}/{len(uncached_frames)} frame "
                  f"CHƯA có cache -> OCR sống (có giới hạn) để tìm tên '{name}'...")
            for m in sampled:
                img_path = m.get("image_path", "")
                if not img_path:
                    continue
                text = extract_text(img_path)
                if _name_matches(text, name):
                    item = dict(m)
                    item["ocr_text"] = text
                    item["identity_match"] = True
                    matches.append(item)

    if matches:
        found_times = ", ".join(f"{m.get('pts_time', 0):.1f}s" for m in matches[:10])
        print(f"✅ [identity_rescan] Tìm thấy {len(matches)} frame có caption khớp tên "
              f"'{name}': {found_times}")
    else:
        print(f"⚠️ [identity_rescan] Không tìm thấy frame nào có caption khớp tên '{name}' "
              f"trong (các) video đã quét — có thể tên không xuất hiện dạng chữ trên hình, "
              f"hoặc nằm ngoài phạm vi đã quét được.")

    return matches


# ---------------------------------------------------------------------------
# (MỚI) Tra cứu TRỰC TIẾP trên OCR CACHE đã precompute OFFLINE (xem
# ocr_precompute.py -> ocr_cache.jsonl) — KHÔNG phụ thuộc candidate đã hội tụ
# về ít video_id hay chưa (giải quyết đúng vấn đề "con gà quả trứng": rescan
# live cần hội tụ trước, nhưng hội tụ đúng lại cần retrieval Tầng 1 có tín
# hiệu cho tên riêng, vốn KHÔNG BAO GIỜ có). Chi phí: 100% local, không tốn
# quota LLM. Đọc thẳng OCR_CACHE_PATH qua `_load_ocr_cache()` — không cần đợi
# `python bm25_search.py` rebuild lại; nếu cache chưa có/rỗng -> trả về []
# ngay, AN TOÀN, pipeline.py sẽ tự fallback về luồng live-rescan cũ bên dưới.
# ---------------------------------------------------------------------------

def search_precomputed_ocr_for_name(name: str, max_results: int = 20) -> list[dict]:
    """Quét TOÀN BỘ OCR cache đã precompute sẵn tìm frame có caption khớp
    `name`. Trả về list metadata (kèm 'ocr_text', 'identity_match': True),
    rỗng nếu cache chưa có hoặc không khớp gì."""
    cache = _load_ocr_cache()
    if not cache:
        return []
    _build_video_index()  # đảm bảo _id_to_frame đã sẵn sàng
    matches = []
    for frame_id, ocr_text in cache.items():
        if _name_matches(ocr_text, name):
            meta = _id_to_frame.get(frame_id)
            if not meta:
                continue
            item = dict(meta)
            item["ocr_text"] = ocr_text
            item["identity_match"] = True
            matches.append(item)
            if len(matches) >= max_results:
                break
    return matches


def search_precomputed_ocr_for_terms(terms: list[str], max_results: int = 40) -> list[dict]:
    """Như trên nhưng khớp NHIỀU term (OR) — dùng cho open-text rescan khi chỉ
    có anchor tổ chức/tỉnh (không phải tên người cụ thể để so khớp 1-1)."""
    if not terms:
        return []
    cache = _load_ocr_cache()
    if not cache:
        return []
    _build_video_index()
    matches = []
    for frame_id, ocr_text in cache.items():
        if any(_name_matches(ocr_text, t) for t in terms):
            meta = _id_to_frame.get(frame_id)
            if not meta:
                continue
            item = dict(meta)
            item["ocr_text"] = ocr_text
            matches.append(item)
            if len(matches) >= max_results:
                break
    return matches


def maybe_rescan(query: str, candidates: list[dict]) -> list[dict]:
    """Hàm tiện ích gọi từ pipeline.py: quyết định có nên rescan hay không, và
    nếu có thì trả về list frame bổ sung tìm được (rỗng nếu không rescan/không
    tìm thấy gì).

    THỨ TỰ ƯU TIÊN (bản cập nhật):
        1. Tra cứu TRỰC TIẾP trên OCR cache đã precompute sẵn (nếu có, xem
           ocr_precompute.py) — không cần candidate hội tụ, bao phủ TOÀN BỘ
           database, không tốn quota.
        2. Nếu cache chưa có/không khớp gì -> fallback rescan LIVE (OCR trên
           phần chưa cache) như bản cũ, nhưng CHỈ khi candidate đã hội tụ về
           <= IDENTITY_RESCAN_MAX_VIDEOS video_id."""
    if not IDENTITY_RESCAN_ENABLED or not candidates:
        return []

    name = extract_target_name(query)
    if not name:
        return []

    precomputed = search_precomputed_ocr_for_name(name)
    if precomputed:
        print(f"✅ [identity_rescan] Tìm thấy {len(precomputed)} frame khớp tên "
              f"'{name}' từ OCR cache đã precompute sẵn (bỏ qua rescan hội tụ).")
        return precomputed

    video_ids = list({c.get("video_id") for c in candidates[:10] if c.get("video_id")})
    if not video_ids or len(video_ids) > IDENTITY_RESCAN_MAX_VIDEOS:
        return []

    return rescan_videos_for_identity(name, video_ids)


# ---------------------------------------------------------------------------
# (MỚI) Open-text sweep — KHÔNG so khớp target đã biết, chỉ THU THẬP OCR để
# LLM tự trích xuất đáp án (xem open_text_qa.py). Dùng cho câu hỏi kiểu
# "xã này tên gì?" nơi chính đáp án là ẩn số, không có gì để match trước.
# ---------------------------------------------------------------------------

def rescan_videos_open_text(
    video_ids: list[str],
    max_frames: int = OPEN_TEXT_RESCAN_MAX_FRAMES,
) -> list[dict]:
    """Thu thập OCR text (KHÔNG lọc theo điều kiện khớp nào) trên các video
    trong `video_ids`, ƯU TIÊN đọc từ OCR CACHE precompute sẵn (gần miễn phí
    -> quét TOÀN BỘ frame có cache của video, không giới hạn max_frames),
    phần frame CHƯA có cache mới OCR sống (có giới hạn max_frames như cũ).
    Trả về TẤT CẢ frame có OCR text khác rỗng, kèm 'ocr_text', để nơi gọi
    (thường là open_text_qa.extract_answer_from_ocr_context) tự suy luận
    đáp án.

    Khác `rescan_videos_for_identity`: không có tham số "cần so khớp gì" vì
    bài toán ở đây là TRÍCH XUẤT thông tin chưa biết trước, không phải XÁC
    NHẬN thông tin đã biết."""
    index = _build_video_index()
    cache = _load_ocr_cache()
    collected: list[dict] = []

    for vid in video_ids:
        frames = index.get(vid, [])
        if not frames:
            continue

        cached_frames = [f for f in frames if f.get("id") in cache]
        uncached_frames = [f for f in frames if f.get("id") not in cache]

        if cached_frames:
            print(f"🔍 [open_text_rescan] {vid}: {len(cached_frames)}/{len(frames)} frame "
                  f"có sẵn trong OCR cache -> lấy TOÀN BỘ (gần như miễn phí).")
        for m in cached_frames:
            text = cache[m["id"]]
            if text.strip():
                item = dict(m)
                item["ocr_text"] = text
                collected.append(item)

        if uncached_frames:
            if len(uncached_frames) > max_frames:
                stride = max(1, len(uncached_frames) // max_frames)
            else:
                stride = max(1, OPEN_TEXT_RESCAN_SAMPLE_STRIDE)
            sampled = uncached_frames[::stride][:max_frames]
            print(f"🔍 [open_text_rescan] {vid}: {len(sampled)}/{len(uncached_frames)} frame "
                  f"CHƯA có cache -> OCR sống (có giới hạn).")
            for m in sampled:
                img_path = m.get("image_path", "")
                if not img_path:
                    continue
                text = extract_text(img_path)
                if text.strip():
                    item = dict(m)
                    item["ocr_text"] = text
                    collected.append(item)

    print(f"✅ [open_text_rescan] Thu thập được {len(collected)} frame có OCR text "
          f"từ {len(video_ids)} video.")
    return collected


def maybe_open_text_rescan(
    query: str,
    candidates: list[dict],
    anchors: dict | None = None,
    max_video_ids: int = 3,
) -> list[dict]:
    """Tương tự `maybe_rescan()` nhưng cho luồng open-text — quyết định có nên
    quét OCR rộng hay không.

    Điều kiện kích hoạt (do pipeline.py truyền `anchors` đã tính sẵn từ
    `extract_anchor_entities()`, tránh tính lại):
        1. Câu hỏi có ít nhất 1 anchor (org_name HOẶC province HOẶC
           person_name) — nếu không có anchor nào, câu hỏi quá chung chung,
           quét toàn bộ OCR sẽ vừa tốn thời gian vừa dễ nhiễu.
        2. Candidate ban đầu hội tụ vào SỐ ÍT video_id (<= max_video_ids) —
           cùng logic với maybe_rescan(): nếu retrieval đã sai cả video thì
           quét OCR sâu hơn cũng vô ích.

    Lưu ý: hàm này CHỦ ĐỘNG quét NHIỀU FRAME HƠN identity rescan thường dùng
    (xem OPEN_TEXT_RESCAN_MAX_FRAMES) vì banner/backdrop ghi địa danh có thể
    chỉ xuất hiện thoáng qua — nên chỉ nên bật khi thực sự cần (pipeline.py
    kiểm tra `is_open_text_query()` trước khi gọi hàm này)."""
    if not IDENTITY_RESCAN_ENABLED or not candidates:
        return []

    if anchors is None:
        anchors = extract_anchor_entities(query)

    terms = list(anchors.get("org_names") or [])
    if anchors.get("province"):
        terms.append(anchors["province"])
    if anchors.get("person_name"):
        terms.append(anchors["person_name"])
    if not terms:
        return []

    # Ưu tiên tra cứu OCR index đã build sẵn — không cần hội tụ video, quét
    # toàn database, không tốn quota (xem search_precomputed_ocr_for_terms).
    precomputed = search_precomputed_ocr_for_terms(terms)
    if precomputed:
        print(f"✅ [open_text_rescan] Tìm thấy {len(precomputed)} frame khớp anchor "
              f"{terms} từ OCR cache đã precompute sẵn (bỏ qua rescan hội tụ).")
        return precomputed

    video_ids = list({c.get("video_id") for c in candidates[:10] if c.get("video_id")})
    if not video_ids or len(video_ids) > max_video_ids:
        return []

    return rescan_videos_open_text(video_ids)