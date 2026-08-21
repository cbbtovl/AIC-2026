# segment_topics.py
"""
TIÊN XỬ LÝ OFFLINE (mới) — Phân đoạn (segment) mỗi video theo CHỦ ĐÊ dựa trên
độ tương đồng OCR giữa các frame liên tiếp, sau đó tóm tắt + trích anchor
entity (org/province/location) cho từng đoạn bằng 1 LLM call/đoạn.

ĐỘNG LỰC: ocr_cache.jsonl (từ ocr_precompute.py) cho OCR text THEO TỪNG FRAME
RIÊNG LẺ — banner tít trong video bản tin lặp lại giống hệt nhau qua nhiều
frame liên tiếp (cùng 1 tin), trong khi ở video sự kiện đơn lẻ (vd L30_V072)
chỉ 1 cụm frame ngắn có banner sự kiện, phần còn lại không có chữ. Nếu
retrieval chỉ dựa trên OCR RỜI RẠC của từng frame (như bm25_search.py hiện
tại), 2 vấn đê xảy ra:
  1) Anchor entity (tên tổ chức/địa danh) chỉ boost được ĐÚNG frame chứa nó;
     các frame lân cận cùng sự kiện nhưng không lọt đúng góc quay chứa chữ
     lại KHÔNG được boost -> object rerank/grouping có thể chọn nhầm frame.
  2) Câu hỏi thường diễn giải lại nội dung ("trao quà tại 1 xã thuộc Khánh
     Hòa") chứ không lặp đúng ticker OCR -> BM25 exact-match yếu.

GIẢI PHÁP (module MỚI, HOÀN TOÀN OFFLINE, không ảnh hưởng luồng query cũ):
  (a) Gom các frame liên tiếp CÙNG VIDEO có OCR text GIỐNG NHAU (Jaccard
      overlap trên token, tái dùng bm25_search._tokenize để không có 2 nguồn
      sự thật khác nhau vê cách tokenize) thành 1 "segment".
  (b) Với mỗi segment có OCR khác rỗng, gọi 1 LLM call (rẻ, text-only) để
      tóm tắt NGẮN GỌN nội dung + trích anchor (org/province/location).
  (c) Ghi ra segment_topics.jsonl. bm25_search.py (cân patch thêm, xem cuối
      file) nạp field 'segment_summary' vào corpus của MỌI frame thuộc
      segment đó; pipeline.py (cân patch thêm) dùng anchor của segment để
      boost TOÀN BỘ member_ids ngay từ Tâng 1 -- không cân đợi hội tụ video
      như identity_rescan.maybe_open_text_rescan() (giải quyết đúng vấn đê
      "con gà quả trứng" đã ghi trong pipeline.py).

CHẠY:
    python segment_topics.py                       # toàn bộ database
    python segment_topics.py --video-id L30_V072    # test nhanh 1 video

YÊU CẨU: đã chạy `python ocr_precompute.py` (cân ocr_cache.jsonl) VÀ
`python build_metadata.py` (cân metadata_all.jsonl để biết pts_time theo
đúng thứ tự thời gian mỗi video). KHÔNG thay thế ocr_precompute.py -- vẫn
cân chạy trước, module này chỉ đọc lại cache đã có, không tự OCR gì cả.

RESUMABLE: giống ocr_precompute.py, ghi APPEND vào segment_topics.jsonl,
chạy lại sẽ tự bỏ qua segment đã xử lý (khớp theo (video_id, segment_id)).
"""

import os
import re
import json
import argparse

from config import METADATA_JSONL, OCR_CACHE_PATH, SEGMENT_TOPICS_PATH, SEGMENT_SIMILARITY_THRESHOLD, SEGMENT_MAX_GAP_SEC
from bm25_search import bm25_engine, _tokenize  # tái dùng tokenizer đã có, tránh 2 nguồn sự thật
from segment_llm import chat_segment as _chat_with_fallback, SegmentQuotaExhausted as OpenRouterQuotaExhausted
from qwen_vqa import _extract_json
from vn_gazetteer import find_province



def _load_ocr_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if not os.path.exists(OCR_CACHE_PATH):
        return cache
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
    return cache


def _load_frames_by_video(video_id_filter: str | None) -> dict[str, list[dict]]:
    """Đọc metadata_all.jsonl, group theo video_id, sort theo pts_time."""
    by_video: dict[str, list[dict]] = {}
    with open(METADATA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if video_id_filter and r.get("video_id") != video_id_filter:
                continue
            by_video.setdefault(r["video_id"], []).append(r)
    for frames in by_video.values():
        frames.sort(key=lambda x: x.get("pts_time", 0.0))
    return by_video


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def _build_segments_for_video(frames: list[dict], ocr_cache: dict[str, str]) -> list[dict]:
    """Gom các frame liên tiếp có OCR giống nhau (Jaccard >= threshold) VÀ
    cách nhau <= SEGMENT_MAX_GAP_SEC giây thành 1 segment.

    Frame KHÔNG có OCR (rỗng) bị BỎ QUA hoàn toàn -- không tốn LLM call cho
    đoạn không chữ, và KHÔNG làm gãy segment đang mở (coi như "không có tín
    hiệu", giữ nguyên segment hiện tại thay vì đóng sớm) -- đúng cho trường
    hợp video sự kiện đơn lẻ, nơi banner chỉ xuất hiện lúc máy quay hướng
    đúng vào nó, các frame xen giữa không có chữ vẫn thuộc cùng 1 cảnh."""
    segments: list[dict] = []
    current: dict | None = None

    for f in frames:
        text = ocr_cache.get(f.get("id", ""), "")
        if not text.strip():
            continue
        tokens = set(_tokenize(text))

        if current is not None:
            gap = f.get("pts_time", 0.0) - current["end_pts"]
            sim = _jaccard(tokens, current["_last_tokens"])
            if gap <= SEGMENT_MAX_GAP_SEC and sim >= SEGMENT_SIMILARITY_THRESHOLD:
                current["members"].append(f)
                current["end_pts"] = f.get("pts_time", 0.0)
                current["_last_tokens"] = tokens
                continue
            segments.append(current)  # không nối được -- đóng segment cũ

        current = {
            "members": [f],
            "start_pts": f.get("pts_time", 0.0),
            "end_pts": f.get("pts_time", 0.0),
            "_last_tokens": tokens,
        }

    if current is not None:
        segments.append(current)
    return segments


_ORG_CAPS_RE = re.compile(r"\b[A-ZÀ-Ỹ]{2,6}\b")
_ORG_CAPS_STOPWORDS = {"TP", "TV", "MC", "VN", "USD", "VND", "OK"}


def _summarize_segment(video_id: str, seg_idx: int, ocr_texts: list[str]) -> dict:
    """1 LLM call/segment -- tóm tắt + trích anchor. Lỗi bình thường -> trả
    vê kết quả rỗng an toàn (KHÔNG raise ra ngoài); hết quota -> raise để
    hàm gọi phía trên dừng sớm toàn bộ batch (resume lại sau)."""
    context = " | ".join(t for t in ocr_texts if t.strip())[:2000]
    if not context.strip():
        # Không có OCR để tóm tắt -- đây KHÔNG phải lỗi LLM, không đánh dấu
        # "_llm_failed" (nếu không, --redo-failed sẽ cố gọi LLM lại vô ích
        # cho những segment vốn dĩ chẳng có gì để tóm tắt).
        return {"summary": "", "org": [], "province": None, "location_detail": "", "_llm_failed": False}

    base_prompt = f"""Đoạn text OCR dưới đây trích từ các khung hình LIÊN TIẾP của cùng 1 video
(cùng 1 cảnh/tin/sự kiện). OCR có thể lỗi chính tả, thiếu dấu, dư ký tự rác.

OCR: {context}

Nhiệm vụ: tóm tắt NGẮN GỌN (1 câu, tiếng Việt) nội dung/sự kiện đoạn này nói vê, và
trích các thực thể THỰC SỰ CÓ trong OCR (không suy đoán, không bịa nếu không thấy rõ).

TRẢ LỜI NGAY, không viết lời dẫn/diễn giải lại đề bài, không suy luận thành lời trước
khi trả lời. Ký tự ĐẦU TIÊN của câu trả lời PHẢI là dấu '{{'. CHỈ trả vê đúng 1 object
JSON hợp lệ (không markdown, không giải thích thêm):
{{
  "summary": "câu tóm tắt ngắn gọn, hoặc chuỗi rỗng nếu OCR quá rác không hiểu được",
  "org": ["tên tổ chức/CLB/đơn vị viết HOA nếu có, có thể rỗng"],
  "province": "tên tỉnh/thành nếu OCR có nhắc, ngược lại null",
  "location_detail": "địa danh cụ thể hơn (xã/huyện/phường) nếu có, ngược lại chuỗi rỗng"
}}"""

    def _local_only_fallback() -> dict:
        """Dùng khi LLM lỗi/không trả JSON hợp lệ SAU khi đã thử lại — vẫn cố
        lấy anchor bằng regex/gazetteer LOCAL (miễn phí, không tốn LLM), y hệt
        fallback đang áp dụng ở nhánh THÀNH CÔNG bên dưới.

        "_llm_failed": True -- ĐÁNH DẤU để build_segment_topics() (option
        --redo-failed) nhận diện được đây là segment bị lỗi PARSE (không phải
        thật sự "không có gì đáng tóm tắt"), và có thể chọn tóm tắt lại sau
        khi đã tăng max_tokens/sửa prompt, THAY VÌ bị coi là "đã xong" mãi mãi
        chỉ vì đã có 1 dòng (rỗng) trong segment_topics.jsonl -- đây chính là
        lý do các segment lỗi trong log cũ (#83, #84, ...) không tự khỏi dù
        chạy lại nhiều lần: done_keys chỉ xét CÓ MẶT trong file hay không,
        không phân biệt "xong thật" với "lỗi nhưng đã ghi fallback rỗng"."""
        return {
            "summary": "",
            "org": [m for m in _ORG_CAPS_RE.findall(context) if m not in _ORG_CAPS_STOPWORDS],
            "province": find_province(context),
            "location_detail": "",
            "_llm_failed": True,
        }

    # Retry 1 lần với prompt nhấn mạnh hơn nếu lượt đầu không ra JSON hợp lệ
    # (model reasoning như nemotron-3-ultra hay viết câu dẫn/suy luận trước
    # khi in JSON -- lượt 2 cảnh báo thẳng lỗi vừa mắc để model tự sửa).
    max_retries = 1
    raw_text = ""
    result = None
    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if attempt > 0:
            prompt += (
                "\n\nLƯU Ý: lượt trả lời TRƯỚC KHÔNG HỢP LỆ (bạn đã viết lời dẫn/giải thích "
                "thay vì JSON ngay, hoặc JSON bị cắt cụt giữa chừng). LẦN NÀY: TUYỆT ĐỐI không "
                "viết gì trước hoặc sau object JSON, ký tự đầu tiên phải là '{'."
            )
        try:
            raw_text, _ = _chat_with_fallback(
                [{"role": "user", "content": prompt}],
                # Tăng từ 450 -> 900: 450 vẫn không đủ cho model reasoning (vd
                # nemotron-3-ultra) hay sinh chain-of-thought DÀI trước khi in
                # JSON cuối -- output vẫn bị cắt cụt giữa chừng suy luận, KHÔNG
                # kịp in dấu "{" nào, hoặc in JSON rồi bị cắt dở dang giữa 1
                # field string ("Unterminated string...") -- xem log lỗi gốc.
                max_tokens=900,
                temperature=0.1,
            )
            result = _extract_json(raw_text)
        except OpenRouterQuotaExhausted:
            raise
        except Exception as e:
            preview = raw_text[:150].replace("\n", " ") if raw_text else "(không nhận được phản hồi)"
            if attempt < max_retries:
                print(f"  ⚠️ [segment_topics] JSON không hợp lệ ở lượt {attempt+1} cho segment "
                      f"{video_id}#{seg_idx} ({type(e).__name__}: {e} | raw='{preview}') — thử lại...")
                continue
            print(f"  ⚠️ [segment_topics] Lỗi tóm tắt segment {video_id}#{seg_idx} sau "
                  f"{max_retries+1} lượt thử: {type(e).__name__}: {e} | raw='{preview}'")
            return _local_only_fallback()
        else:
            break  # parse JSON thành công -> thoát vòng lặp retry

    summary = (result.get("summary") or "").strip()
    org = [o.strip() for o in (result.get("org") or []) if isinstance(o, str) and o.strip()]
    province = result.get("province") or None

    # Fallback LOCAL (không tốn LLM thêm): nếu model bỏ sót, tự dò lại bằng
    # gazetteer + regex viết-hoa ngay trên context gốc -- rẻ, bổ sung thêm
    # cho chắc, không hại gì nếu model đã trả đúng rồi.
    if not province:
        province = find_province(context)
    if not org:
        org = [m for m in _ORG_CAPS_RE.findall(context) if m not in _ORG_CAPS_STOPWORDS]

    return {
        "summary": summary,
        "org": org,
        "province": province,
        "location_detail": (result.get("location_detail") or "").strip(),
        "_llm_failed": False,
    }


def _record_needs_redo(r: dict) -> bool:
    """True nếu record này nên được tóm tắt LẠI khi chạy --redo-failed.

    2 trường hợp:
      1) Record MỚI (được ghi bởi bản vá này trở đi) có "_llm_failed": true
         -- biết chắc chắn là lỗi parse JSON, không phải OCR rác hợp lệ.
      2) Record CŨ (ghi TRƯỚC bản vá này -- hoàn toàn KHÔNG có field
         "_llm_failed" trong dict, khác None) VÀ summary rỗng. Record cũ
         không thể phân biệt "OCR rác nên model trả rỗng hợp lệ" với "LLM
         lỗi parse JSON, ghi fallback rỗng" (2 trường hợp cho ra data giống
         hệt nhau ở bản code cũ) -- coi summary rỗng là ứng viên redo AN
         TOÀN: nếu thật sự chỉ là OCR rác, tóm tắt lại cũng chỉ ra rỗng lần
         nữa (không hại gì, không tốn quota đáng kể vì số này thường ít);
         còn nếu là lỗi parse thì lần này sẽ được sửa đúng.
      Record MỚI nhưng "_llm_failed": false thì summary rỗng là HỢP LỆ THẬT
      (model đã tóm tắt thành công, chỉ là không có gì đáng nói) -- không
      redo, tránh tốn quota vô ích."""
    if r.get("_llm_failed"):
        return True
    if "_llm_failed" not in r and not (r.get("summary") or "").strip():
        return True
    return False


def build_segment_topics(video_id_filter: str | None = None, flush_every: int = 50, redo_failed: bool = False):
    ocr_cache = _load_ocr_cache()
    if not ocr_cache:
        print("❌ Chưa có OCR cache. Chạy `python ocr_precompute.py` trước.")
        return
    if not os.path.exists(METADATA_JSONL):
        print(f"❌ Chưa có {METADATA_JSONL}. Chạy `python build_metadata.py` trước.")
        return

    by_video = _load_frames_by_video(video_id_filter)
    print(f"🔍 Xử lý {len(by_video)} video...")

    # BUG ĐÃ SỬA: resume trước đây chỉ xét "(video_id, segment_id) đã CÓ MẶT
    # trong file hay chưa" -- không phân biệt "đã tóm tắt THÀNH CÔNG" với "LLM
    # lỗi parse JSON, ghi fallback rỗng (summary='', org từ regex local)".
    # Hậu quả đúng như log bạn thấy: các segment lỗi (#83, #84, ...) VẪN được
    # ghi 1 dòng vào segment_topics.jsonl (qua _local_only_fallback()) rồi
    # nghiễm nhiên bị coi là "xong" -- chạy lại lệnh này (kể cả sau khi đã
    # tăng max_tokens/sửa prompt như bản vá này) sẽ KHÔNG BAO GIỜ tóm tắt lại
    # chúng, vì done_keys vẫn thấy chúng "đã có trong file".
    #
    # `redo_failed=True` (cờ --redo-failed): xoá khỏi file các record được
    # _record_needs_redo() nhận diện là lỗi (xem hàm đó — bao gồm CẢ record
    # cũ ghi từ trước khi bản vá này tồn tại, không chỉ record có field
    # "_llm_failed" mới), rồi ghi lại file KHÔNG có chúng -- để vòng lặp bên
    # dưới coi các segment đó là CHƯA xử lý và tóm tắt lại bằng prompt/
    # max_tokens mới.
    # rồi ghi lại file KHÔNG có chúng -- để vòng lặp bên dưới coi các segment
    # đó là CHƯA xử lý và tóm tắt lại bằng prompt/max_tokens mới.
    existing_records: list[dict] = []
    if os.path.exists(SEGMENT_TOPICS_PATH):
        with open(SEGMENT_TOPICS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if redo_failed:
            n_before = len(existing_records)
            existing_records = [r for r in existing_records if not _record_needs_redo(r)]
            n_removed = n_before - len(existing_records)
            if n_removed:
                print(f"🔁 [--redo-failed] Bỏ {n_removed} segment đã lưu trước đây nhưng LLM lỗi/summary "
                      f"rỗng (kể cả record ghi từ trước bản vá, chưa có field _llm_failed) -- sẽ tóm tắt "
                      f"lại từ đầu cho các segment này.")
                with open(SEGMENT_TOPICS_PATH, "w", encoding="utf-8") as out_f:
                    for r in existing_records:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                print("🔁 [--redo-failed] Không có segment nào bị đánh dấu lỗi trong file hiện tại.")

    done_keys = {(r.get("video_id"), r.get("segment_id")) for r in existing_records}
    print(f"📊 Đã có {len(done_keys)} segment từ lân chạy trước (resume, không làm lại).")

    n_written = 0
    quota_hit = False
    with open(SEGMENT_TOPICS_PATH, "a", encoding="utf-8") as out_f:
        for video_id, frames in by_video.items():
            if quota_hit:
                break
            segments = _build_segments_for_video(frames, ocr_cache)
            for idx, seg in enumerate(segments):
                segment_id = f"{video_id}_seg{idx:03d}"
                if (video_id, segment_id) in done_keys:
                    continue

                member_ids = [m["id"] for m in seg["members"]]
                ocr_texts = [ocr_cache.get(mid, "") for mid in member_ids]

                try:
                    meta = _summarize_segment(video_id, idx, ocr_texts)
                except OpenRouterQuotaExhausted as e:
                    print(f"⛔ [segment_topics] {e} -- dừng sớm, chạy lại lệnh này để resume.")
                    quota_hit = True
                    break

                record = {
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "start_pts": round(seg["start_pts"], 2),
                    "end_pts": round(seg["end_pts"], 2),
                    "member_ids": member_ids,
                    **meta,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % flush_every == 0:
                    out_f.flush()
                    print(f"  ⏳ Đã ghi {n_written} segment...")

    status = "dừng giữa chừng do hết quota (đã lưu tiến độ)" if quota_hit else "hoàn tất"
    print(f"✅ {n_written} segment mới đã ghi vào {SEGMENT_TOPICS_PATH} ({status}).")
    if not quota_hit:
        print("👉 Chạy tiếp: python bm25_search.py   (rebuild index, nạp segment_summary vào corpus)")


# ---------------------------------------------------------------------------
# (Dùng bởi pipeline.py) Boost candidate theo segment — 2 CƠ CHẾ KẾT HỢP:
#   (1) ANCHOR match: org/province của segment khớp CHÍNH XÁC anchor trích
#       từ câu hỏi (rẻ, không cần model nào, xem _segment_boost_by_anchor).
#   (2) SIMILARITY match (MỚI, xem segment_embed.py): cosine giữa embedding
#       BGE-M3 của câu hỏi và embedding của segment summary -- bắt được câu
#       hỏi DIỄN GIẢI LẠI nội dung (vd "trao quà tại 1 xã thuộc Khánh Hòa")
#       mà không trùng 1 từ khoá cụ thể nào với anchor đã trích riêng lẻ.
# Cả 2 ĐỘC LẬP, không cái nào thay thế cái kia -- anchor match chính xác
# tuyệt đối khi có, similarity match cứu các trường hợp câu hỏi paraphrase
# mà anchor extraction (regex/gazetteer) không bắt được hoặc bắt thiếu.
# ---------------------------------------------------------------------------

_segment_index_cache: list[dict] | None = None


def _load_segment_index() -> list[dict]:
    global _segment_index_cache
    if _segment_index_cache is not None:
        return _segment_index_cache
    records: list[dict] = []
    if os.path.exists(SEGMENT_TOPICS_PATH):
        with open(SEGMENT_TOPICS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    _segment_index_cache = records
    return records


def _frames_from_segment(seg: dict, match_type: str, extra: dict | None = None) -> list[dict]:
    """Tra full metadata (qua bm25_engine.get_by_id) cho mọi member_id của 1
    segment, gắn thêm segment_summary + cờ debug. Dùng CHUNG cho cả 2 cơ chế
    match bên dưới để tránh 2 nguồn sự thật khác nhau."""
    if not bm25_engine.metadatas:
        bm25_engine.build_or_load()
    items = []
    for mid in seg.get("member_ids", []):
        meta = bm25_engine.get_by_id(mid)
        if not meta:
            continue
        item = dict(meta)
        item["segment_summary"] = seg.get("summary", "")
        item["_segment_boosted"] = True
        item["_segment_match_type"] = match_type
        if extra:
            item.update(extra)
        items.append(item)
    return items


def _segment_boost_by_anchor(anchors: dict, max_segments: int = 5) -> list[dict]:
    """CƠ CHẾ 1: org/province của segment khớp CHÍNH XÁC anchors đã trích
    từ câu hỏi (identity_rescan.extract_anchor_entities). Rẻ, không cần
    model nào, chạy được ngay cả khi chưa embed segment (segment_embed.py)."""
    segments = _load_segment_index()
    if not segments:
        return []

    org_names = {o.upper() for o in (anchors.get("org_names") or [])}
    province = anchors.get("province")
    if not org_names and not province:
        return []

    hits: list[dict] = []
    matched_segments = 0
    for seg in segments:
        seg_org = {o.upper() for o in (seg.get("org") or [])}
        org_match = bool(org_names & seg_org)
        province_match = bool(province and seg.get("province") == province)
        if not (org_match or province_match):
            continue
        matched_segments += 1
        hits.extend(_frames_from_segment(seg, match_type="anchor"))
        if matched_segments >= max_segments:
            break

    if hits:
        print(f"🎯 [segment_boost] (anchor) {matched_segments} segment khớp "
              f"org={sorted(org_names)}, province={province} -> {len(hits)} frame.")
    return hits


def _segment_boost_by_similarity(question: str, max_segments: int = 5) -> list[dict]:
    """CƠ CHẾ 2 (MỚI): cosine similarity giữa embedding BGE-M3 của câu hỏi
    và embedding của segment summary (xem segment_embed.py). Bắt được câu
    hỏi DIỄN GIẢI LẠI nội dung dù không trùng 1 từ khoá cụ thể nào với
    summary gốc -- ví dụ câu hỏi "trao quà tại 1 xã thuộc Khánh Hòa" khớp
    ngữ nghĩa với summary "CLB FANA trao quà từ thiện tại xã Giang Ly,
    huyện Khánh Vinh, tỉnh Khánh Hòa" dù không chung từ khoá.

    Cân đã chạy `python segment_embed.py` trước (cân segment_embeddings.pkl).
    Nếu chưa -> search_segments_by_similarity() tự trả vê [] an toàn."""
    try:
        from segment_embed import search_segments_by_similarity
    except Exception as e:
        print(f"⚠️ [segment_boost] Không import được segment_embed ({type(e).__name__}: {e}) "
              f"-- bỏ qua semantic match, chỉ dùng anchor match.")
        return []

    matches = search_segments_by_similarity(question, top_k=max_segments)
    if not matches:
        return []

    segments_by_id = {s["segment_id"]: s for s in _load_segment_index()}
    hits: list[dict] = []
    for m in matches:
        seg = segments_by_id.get(m["segment_id"])
        if not seg:
            continue
        hits.extend(_frames_from_segment(
            seg, match_type="similarity", extra={"_segment_similarity": m["similarity"]},
        ))

    if hits:
        top_scores = [m["similarity"] for m in matches]
        print(f"🧠 [segment_boost] (similarity/BGE-M3) {len(matches)} segment khớp "
              f"(cosine cao nhất={max(top_scores):.3f}) -> {len(hits)} frame.")
    return hits


def segment_boost_candidates(question: str, anchors: dict, max_segments: int = 5) -> list[dict]:
    """API CHÍNH dùng bởi pipeline.py — kết hợp CẢ 2 cơ chế trên, dedup theo
    frame id (nếu 1 frame được cả 2 cơ chế cùng chọn, gộp cờ thành "both" để
    debug, giữ similarity score cao hơn khi có xung đột).

    Khác _anchor_boost_retrieve trong pipeline.py (chỉ boost DUY NHẤT frame
    có media-info trùng chữ, không theo cụm) và khác identity_rescan.
    maybe_open_text_rescan (cân candidate đã hội tụ TRƯỚC mới rescan được —
    vòng luẩn quẩn con-gà-quả-trứng): hàm này hoạt động NGAY ở Tâng 1, không
    phụ thuộc candidate ban đâu đúng hay sai, vì tra cứu trực tiếp trên dữ
    liệu đã tiên xử lý offline (segment_topics.jsonl + segment_embeddings.pkl).

    Nếu chưa chạy segment_topics.py và/hoặc segment_embed.py -> trả vê []
    an toàn ở phân tương ứng, KHÔNG crash pipeline."""
    anchor_hits = _segment_boost_by_anchor(anchors, max_segments=max_segments)
    similarity_hits = _segment_boost_by_similarity(question, max_segments=max_segments)

    merged: dict[str, dict] = {}
    for hit in anchor_hits + similarity_hits:
        key = hit.get("id")
        if not key:
            continue
        if key in merged:
            existing = merged[key]
            if existing.get("_segment_match_type") != hit.get("_segment_match_type"):
                existing["_segment_match_type"] = "both"
            if "_segment_similarity" in hit:
                existing["_segment_similarity"] = max(
                    existing.get("_segment_similarity", 0.0), hit["_segment_similarity"],
                )
        else:
            merged[key] = hit

    return list(merged.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân đoạn video theo chủ đê + tóm tắt bằng LLM (offline, resumable)")
    parser.add_argument("--video-id", type=str, default=None, help="Chỉ xử lý 1 video cụ thể (test nhanh)")
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--redo-failed", action="store_true",
                         help="Tóm tắt lại các segment mà lần chạy trước LLM lỗi (JSON không hợp lệ, "
                              "summary rỗng) -- xem BUG ĐÃ SỬA trong build_segment_topics().")
    args = parser.parse_args()
    build_segment_topics(video_id_filter=args.video_id, flush_every=args.flush_every, redo_failed=args.redo_failed)