# qwen_vqa.py
"""
Dùng OpenRouter (https://openrouter.ai) để gọi các model VLM FREE — không tốn
phí trong suốt mùa thi (2-3 tháng). Có danh sách FALLBACK_MODELS: nếu model
đầu tiên bị rate-limit / lỗi / bị gỡ khỏi tier free, tự động thử model kế tiếp
thay vì crash pipeline giữa buổi thi.

=== BUGFIX LOG ===
[Lần 1] predict_single_vote() chỉ gửi 1 ảnh (candidate #1) nhưng liệt kê metadata
của N candidate, bắt model "chọn" trong số đó -> model đoán mù candidate chưa
từng thấy ảnh. Đã sửa: gửi N ảnh 1 lượt kèm nhãn thứ tự.

[Lần 2] Nhiều model free xử lý nhiều ảnh trong 1 turn không đáng tin cậy -> vẫn
hallucinate. FIX: chấm điểm TỪNG candidate ĐỘC LẬP (1 ảnh / 1 lần gọi), video_id/
pts_time/frame_id luôn lấy THẲNG từ metadata thật, model không còn cơ hội bịa.

[Lần 3] Hết quota NGÀY (50 req/ngày free-tier, dùng chung mọi model) -> thêm
OpenRouterQuotaExhausted để dừng sớm thay vì lặp lỗi cho từng candidate/vote.

[Lần 4] Vẫn tốn quota QUÁ NHANH -> giảm số request CẦN THIẾT ở tầng pipeline
(config.py) + _throttle() chủ động giới hạn tốc độ gọi client-side.

[Lần 5] Thêm generate_query_keywords() cho Tầng 2 (object_rerank.py).

[Lần 6 — bản này] 2 THAY ĐỔI LỚN theo yêu cầu "bỏ grid, SigLIP retrieve bị
sai khiến OCR sai tệp keyframe":

  (a) BỎ generate_hyde(). Nguyên nhân gốc của "SigLIP retrieve sai": SigLIP
      không đa ngôn ngữ, nhúng thẳng câu hỏi TIẾNG VIỆT vào embed_text() gần
      như vô nghĩa với model. Thay generate_hyde() (mô tả dài, max_tokens=400,
      TẮT mặc định vì tốn quota) bằng translate_query_en() MỚI — dịch NGẮN
      GỌN (max_tokens=80), LUÔN BẬT theo mặc định (config.USE_QUERY_TRANSLATION),
      và bản dịch này được DÙNG CHUNG cho cả SigLIP lẫn Object Rerank (Tầng 2)
      thay vì gọi LLM riêng 2 lần cho cùng 1 việc "hiểu câu hỏi bằng tiếng
      Anh" — xem object_rerank.rerank_by_objects(english_query=...).
      generate_query_keywords() vẫn được GIỮ LẠI (không xoá) để
      object_rerank.py vẫn dùng được độc lập (vd chạy `python
      object_rerank.py "..."` để test), nhưng pipeline.py chính KHÔNG còn
      gọi nó nữa (đã có bản dịch dùng chung, tokenize cục bộ, không tốn
      thêm request).

  (b) BỎ toàn bộ logic "lưới 2x2 + ô đen placeholder" khỏi _score_candidate().
      Trước đây ảnh gửi cho VLM (và OCR ở ocr_utils.py, gọi từ pipeline.py)
      là ẢNH GHÉP 4 keyframe liền kề (n-1, n, n+1, n+2) — khiến OCR/VLM có
      thể "thấy" nội dung/chữ thuộc về 1 trong 3 khung LÂN CẬN chứ không phải
      đúng khung n mà pts_time/frame_id trả về đại diện -> bug "OCR đọc nhầm
      keyframe" đúng như user báo cáo. Giờ mỗi candidate chỉ gửi ĐÚNG 1 ảnh
      keyframe thật của chính nó (metadata["image_path"]) — đảm bảo những gì
      VLM/OCR "nhìn thấy" LUÔN khớp 100% với pts_time/frame_id trả về.
"""

import os
import json
import base64
import time as _time
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from config import VLM_MODELS, SELF_CONSISTENCY_VOTES, OPENROUTER_FREE_RPM_LIMIT

load_dotenv()
_client = None


class OpenRouterQuotaExhausted(Exception):
    """Quota free-tier OpenRouter (ngày HOẶC phút) đã hết — dùng CHUNG cho mọi
    model :free trong VLM_MODELS, thử model khác trong danh sách cũng vô ích."""
    pass


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("❌ Không tìm thấy OPENROUTER_API_KEY trong file .env! Vui lòng kiểm tra lại cấu hình.")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/aic2026-vqa",  # OpenRouter yêu cầu header nhận diện app
                "X-Title": "AIC2026 VQA Pipeline",
            },
        )
    return _client


def _image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


# ---------------------------------------------------------------------------
# Throttle chủ động: KHÔNG BAO GIỜ để tổng số request trong 60s gần nhất chạm
# ngưỡng OPENROUTER_FREE_RPM_LIMIT (20), dù gọi model nào — vì quota này dùng
# CHUNG cho cả tài khoản (mọi model :free), không phải riêng từng model.
# ---------------------------------------------------------------------------
_call_timestamps: list[float] = []
_RATE_WINDOW_SEC = 60.0
_RATE_SAFETY_MARGIN = 2  # chừa dư 2 request để tránh sát ngưỡng do trễ mạng


def _throttle():
    now = _time.time()
    while _call_timestamps and now - _call_timestamps[0] > _RATE_WINDOW_SEC:
        _call_timestamps.pop(0)
    limit = max(1, OPENROUTER_FREE_RPM_LIMIT - _RATE_SAFETY_MARGIN)
    if len(_call_timestamps) >= limit:
        sleep_for = _RATE_WINDOW_SEC - (now - _call_timestamps[0]) + 0.5
        if sleep_for > 0:
            print(f"⏳ Chủ động chờ {sleep_for:.1f}s để tránh chạm rate-limit "
                  f"{OPENROUTER_FREE_RPM_LIMIT} request/phút...")
            _time.sleep(sleep_for)
    _call_timestamps.append(_time.time())


def _is_account_wide_quota_error(exc: Exception) -> bool:
    """Lỗi quota DÙNG CHUNG toàn tài khoản (ngày hoặc phút) — thử model khác
    trong VLM_MODELS chắc chắn cũng 429, không có ích gì."""
    msg = str(exc).lower()
    return (
        "free-models-per-day" in msg
        or "free-models-per-min" in msg
        or "openrouter_free_tier_daily" in msg
        or "openrouter_free_tier_per_minute" in msg
    )


def _chat_with_fallback(messages: list[dict], **kwargs):
    """Thử LẦN LƯỢT từng model trong VLM_MODELS (config.py). Trước MỖI lần gọi
    thật sự, chủ động throttle (_throttle()) để không bao giờ chạm rate-limit.

    - Lỗi quota DÙNG CHUNG toàn tài khoản (ngày/phút) -> raise
      OpenRouterQuotaExhausted NGAY, không thử các model còn lại.
    - Lỗi khác (model rỗng, model cụ thể quá tải ở nhà cung cấp, v.v.) -> vẫn
      fallback sang model kế tiếp như cũ."""
    client = _get_client()
    last_err = None
    for model_name in VLM_MODELS:
        _throttle()
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)
            # BUGFIX: 1 số model free của OpenRouter (vd nvidia/nemotron-3-nano-omni)
            # đôi khi trả về response object với resp.choices = None (lỗi/rate-limit
            # ở phía nhà cung cấp cho ĐÚNG model đó, không phải lỗi quota tài khoản)
            # thay vì raise exception bình thường -> resp.choices[0] ném TypeError
            # "'NoneType' object is not subscriptable", vẫn bị except Exception bên
            # dưới bắt được và fallback sang model kế tiếp NHƯNG bị gộp chung với
            # "model trả lời rỗng"/"lỗi mạng" trong log, khó phân biệt khi debug.
            # Kiểm tra tường minh trước khi subscript để log đúng nguyên nhân.
            if not resp or not getattr(resp, "choices", None):
                last_err = f"Model {model_name} trả về response rỗng/không có choices."
                print(f"  ⚠️ {last_err} — thử model kế tiếp...")
                continue
            content = resp.choices[0].message.content
            if content and content.strip():
                return content, model_name
            last_err = f"Model {model_name} trả về nội dung rỗng."
        except Exception as e:
            if _is_account_wide_quota_error(e):
                raise OpenRouterQuotaExhausted(
                    "Đã chạm giới hạn free-tier OpenRouter DÙNG CHUNG cho mọi model :free "
                    "(20 request/phút hoặc 50 request/ngày). Nạp $10 tại "
                    "openrouter.ai/settings/credits để nâng lên 1000 request/ngày, hoặc đợi "
                    "reset (phút: ~60s; ngày: 00:00 UTC ~7h sáng giờ VN)."
                ) from e
            last_err = f"Model {model_name} lỗi: {type(e).__name__}: {e}"
            print(f"  ⚠️ {last_err} — thử model kế tiếp...")
            continue
    raise RuntimeError(last_err or "Không có model nào trong VLM_MODELS khả dụng.")


def _extract_json(raw_text: str) -> dict:
    """Parse JSON từ output model, chịu được các kiểu 'rác' model free hay thêm
    (```json fences, text thừa trước/sau object)."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = raw_text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned.strip())


def translate_and_extract_anchors(query: str) -> dict:
    """TỔNG QUÁT HOÁ `translate_query_en()` + `identity_rescan.extract_anchor_entities()`
    (heuristic regex dựa viết-Hoa) THÀNH 1 LLM CALL DUY NHẤT — KHÔNG tốn thêm
    quota so với bản cũ (trước đây pipeline.py gọi translate_query_en() 1 lần
    RỒI extract_anchor_entities() (regex, miễn phí) RIÊNG — giờ gộp bước dịch
    + trích anchor vào CHUNG prompt này, vẫn 1 request/câu hỏi).

    TẠI SAO CẦN: heuristic regex (Title-Case, viết-Hoa 2-6 ký tự) chỉ đúng khi
    câu hỏi được gõ ĐÚNG CHUẨN chính tả tiếng Việt (viết hoa tên riêng). Với
    câu hỏi gõ thường/không dấu/lỗi transcribe (rất phổ biến với data thực
    tế), heuristic bỏ sót gần hết. LLM hiểu NGỮ CẢNH nên trích đúng dù câu hỏi
    không viết hoa, không cần liệt kê tay các "từ đánh dấu mệnh đề" như
    "Hỏi:"/"Đáp:" (heuristic regex ở identity_rescan.py vẫn giữ made để làm
    fallback local, xem bên dưới).

    Trả về:
        {
          "english": str,                # bản dịch, dùng cho SigLIP/object_rerank y hệt cũ
          "person_name": str | None,
          "org_names": list[str],
          "province": str | None,
        }
    Lỗi/hết quota -> trả về {"english": query, "person_name": None, "org_names": [], "province": None}
    (pipeline.py sẽ tự fallback sang extract_anchor_entities() regex — xem
    pipeline.py, KHÔNG mất khả năng nhận anchor, chỉ kém tổng quát hơn khi
    không có LLM)."""
    empty = {"english": query, "person_name": None, "org_names": [], "province": None}
    try:
        prompt = f"""Câu hỏi tìm kiếm video (tiếng Việt) dưới đây có thể chứa các THỰC THỂ RIÊNG
(tên người, tên tổ chức/CLB/đơn vị, tên tỉnh/thành/xã) — kể cả khi KHÔNG viết hoa đúng
chuẩn hoặc bị lỗi chính tả/thiếu dấu.

Câu hỏi: {query}

CHỈ trả về đúng 1 object JSON hợp lệ (không markdown, không giải thích thêm):
{{
  "english": "bản dịch/mô tả TIẾNG ANH ngắn gọn (tối đa 20 từ) phần MÔ TẢ CẢNH: vật thể, người, màu sắc, hành động",
  "person_name": "tên người cụ thể được nhắc, hoặc null nếu không có",
  "org_names": ["tên tổ chức/CLB/đơn vị được nhắc, có thể rỗng"],
  "province": "tên tỉnh/thành/xã được nhắc, hoặc null nếu không có"
}}"""
        text, _ = _chat_with_fallback(
            [{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        result = _extract_json(text)
        english = (result.get("english") or "").strip() or query
        person_name = (result.get("person_name") or "").strip() or None
        org_names = [o.strip() for o in (result.get("org_names") or []) if isinstance(o, str) and o.strip()]
        province = (result.get("province") or "").strip() or None
        return {
            "english": english, "person_name": person_name,
            "org_names": org_names, "province": province,
        }
    except OpenRouterQuotaExhausted as e:
        print(f"⛔ [translate_and_extract_anchors] {e} — fallback dịch câu gốc + regex heuristic (extract_anchor_entities).")
        return empty
    except Exception as e:
        print(f"⚠️ Lỗi translate_and_extract_anchors: {e} — fallback dịch câu gốc + regex heuristic.")
        return empty


def translate_query_en(query: str) -> str:
    """THAY THẾ generate_hyde() cũ — dịch NGẮN GỌN câu hỏi (thường tiếng Việt)
    sang 1 câu tiếng Anh đơn giản, dùng LÀM CHUNG cho:
        (1) SigLIP vector search (retrieve.py qua pipeline._hybrid_retrieve) —
            SigLIP-base KHÔNG đa ngôn ngữ, nhúng thẳng tiếng Việt gần như vô
            nghĩa với model -> đây là NGUYÊN NHÂN GỐC của bug "SigLIP retrieve
            sai" mà user báo cáo.
        (2) Object Rerank (Tầng 2, object_rerank.py) — tokenize CỤC BỘ câu
            tiếng Anh này (không gọi LLM thêm lần nữa) để so khớp field
            OBJECTS (luôn là tên class tiếng Anh).

    Chỉ tốn max_tokens=80 (rẻ hơn nhiều generate_hyde() cũ dùng max_tokens=400
    cho 2-3 câu mô tả), và THAY THẾ HẲN cho generate_query_keywords() từng
    được gọi RIÊNG trong object_rerank.py -> tổng chi phí quota KHÔNG đổi
    (vẫn ~1 request/câu hỏi như bản cũ), chỉ đổi CÁCH DÙNG cho hiệu quả hơn.

    Lỗi/hết quota -> fallback về NGUYÊN VĂN câu hỏi gốc (KHÔNG crash pipeline;
    BM25 vẫn chạy song song trên câu gốc, chỉ SigLIP sẽ kém chính xác hơn)."""
    try:
        prompt = f"""Translate/describe the following Vietnamese video-search query into ONE short, plain English sentence (max 20 words) describing the scene: objects, people, colors, clothing, actions mentioned. Use simple common nouns (person, car, building, red shirt, motorbike...). Return ONLY the English sentence — no explanation, no quotes, no markdown.

Query: {query}
English:"""
        text, _ = _chat_with_fallback(
            [{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.1,
        )
        text = text.strip().strip('"').strip()
        return text if text else query
    except OpenRouterQuotaExhausted as e:
        print(f"⛔ Dịch câu hỏi: {e} — dùng thẳng câu hỏi gốc (SigLIP sẽ kém chính xác hơn).")
        return query
    except Exception as e:
        print(f"⚠️ Lỗi dịch câu hỏi sang tiếng Anh: {e}")
        return query


def split_scene_and_question(query: str) -> dict:
    """TẦNG 1: tách câu hỏi AIC (thường dạng "<mô tả cảnh>. <câu hỏi>") thành
    2 phần riêng biệt:
        - "scene": phần MÔ TẢ CẢNH/ĐỐI TƯỢNG — dùng cho retrieval (BM25/SigLIP)
          và trích từ khóa cho Object Rerank (Tầng 2).
        - "question": câu HỎI CỤ THỂ cần trả lời — dùng khi hỏi VLM (Tầng 5).

    TỐN THÊM 1 REQUEST OpenRouter/câu hỏi — xem config.USE_ANSWER_TARGET_SPLIT
    để bật/tắt tuỳ ngân sách quota. Nếu lỗi/hết quota -> fallback an toàn:
    scene = question = câu gốc."""
    try:
        prompt = f"""Câu hỏi AIC dưới đây thường có 2 phần: (1) MÔ TẢ CẢNH/ĐỐI TƯỢNG xuất
hiện trong video, và (2) CÂU HỎI CỤ THỂ cần trả lời về cảnh đó. Hãy tách 2 phần này.

Câu hỏi gốc: {query}

CHỈ trả về 1 object JSON hợp lệ (không markdown, không giải thích) với 2 trường:
- "scene": phần mô tả cảnh/đối tượng (tiếng Việt, bỏ phần câu hỏi cụ thể). Nếu không tách rõ được, để nguyên câu gốc.
- "question": câu hỏi cụ thể cần trả lời (tiếng Việt). Nếu không tách rõ được, để nguyên câu gốc."""

        text, _ = _chat_with_fallback(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        result = _extract_json(text)
        scene = (result.get("scene") or "").strip() or query
        question = (result.get("question") or "").strip() or query
        return {"scene": scene, "question": question}
    except OpenRouterQuotaExhausted as e:
        print(f"⛔ {e} — bỏ qua tách scene/question, dùng nguyên câu hỏi gốc.")
        return {"scene": query, "question": query}
    except Exception as e:
        print(f"⚠️ Lỗi tách scene/question: {e}")
        return {"scene": query, "question": query}


def generate_query_keywords(query: str) -> list[str]:
    """Trích 4-6 từ khóa TIẾNG ANH từ câu hỏi qua LLM.

    GIỮ LẠI để object_rerank.py vẫn chạy được ĐỘC LẬP (vd `python
    object_rerank.py "câu hỏi"` để test nhanh, không qua pipeline.py đầy đủ).
    Trong pipeline.py CHÍNH, hàm này KHÔNG còn được gọi nữa — object_rerank
    giờ nhận thẳng bản dịch tiếng Anh đã có sẵn từ translate_query_en() (Tầng
    1) qua tham số `english_query`, tránh tốn thêm 1 request trùng lặp cho
    cùng 1 việc "dịch câu hỏi sang tiếng Anh"."""
    try:
        prompt = f"""Liệt kê tối đa 6 danh từ TIẾNG ANH mô tả vật thể/đối tượng/hành động chính có thể xuất hiện trong cảnh video trả lời câu hỏi dưới đây. Dùng từ đơn giản, phổ biến (kiểu nhãn object detection: person, car, building, phone, glasses, table...). CHỈ trả về các từ cách nhau bởi dấu phẩy, KHÔNG giải thích, KHÔNG đánh số.

Câu hỏi: {query}
Từ khóa (English, comma-separated):"""

        text, _ = _chat_with_fallback(
            [{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        words = [w.strip().lower().strip(".") for w in text.split(",") if w.strip()]
        return words[:6]
    except OpenRouterQuotaExhausted as e:
        print(f"⛔ {e} — bỏ qua trích từ khóa object-rerank, giữ nguyên thứ tự SigLIP.")
        return []
    except Exception as e:
        print(f"⚠️ Lỗi trích từ khóa object-rerank: {e}")
        return []


def _score_candidate(query: str, image_path: str, candidate: dict) -> dict:
    """Chấm điểm ĐỘC LẬP 1 candidate duy nhất (1 ẢNH KEYFRAME THẬT / 1 lần gọi
    API — KHÔNG còn ghép lưới 2x2, xem BUGFIX LOG [Lần 6] ở đầu file).

    video_id/pts_time/frame_id của candidate được gán thẳng từ dữ liệu có sẵn,
    model chỉ đánh giá "ảnh NÀY có khớp câu hỏi không, tự tin bao nhiêu, trả
    lời gì". Vì `image_path` LUÔN là đúng 1 keyframe thật ứng với đúng
    pts_time/frame_id của candidate, không còn khả năng model "thấy" nội dung
    của 1 khung hình LÂN CẬN rồi gán nhầm cho candidate này.

    candidate["ocr_text"] (do ocr_utils.py trích trên CHÍNH image_path này,
    xem pipeline.py) được đưa vào prompt làm ngữ cảnh bổ sung — hữu ích cho
    câu hỏi có tên riêng/chữ hiển thị trên màn hình.

    Raise OpenRouterQuotaExhausted (KHÔNG bắt ở đây) để hàm gọi phía trên (
    predict_single_vote) có thể dừng sớm toàn bộ vòng lặp candidate còn lại,
    thay vì lãng phí request cho từng candidate khi đã biết chắc quota hết."""
    try:
        data_url = _image_to_data_url(image_path)
        objects_hint = candidate.get("OBJECTS", "")
        ocr_hint = candidate.get("ocr_text", "")

        prompt = f"""
Ảnh đính kèm là 1 keyframe THẬT trích ra từ 1 video tại 1 thời điểm cụ thể.
Vật thể đã detect được tự động (chỉ tham khảo, có thể sai/thiếu): {objects_hint or "(không có)"}
Chữ/text đọc được trên ảnh bằng OCR (chỉ tham khảo, có thể sai chính tả/thiếu dấu,
nhưng có thể chứa tên người, caption, biển hiệu hữu ích): {ocr_hint or "(không có)"}

Câu hỏi: {query}

Đánh giá CHỈ dựa trên nội dung THỰC SỰ NHÌN THẤY trong ảnh — không suy đoán chỉ vì
gợi ý vật thể/OCR ở trên nghe "hợp" với câu hỏi trong khi ảnh không thể hiện điều đó.
Nếu OCR có tên riêng khớp với câu hỏi, có thể dùng làm căn cứ xác nhận đúng người/cảnh,
nhưng câu trả lời cuối (vd: màu áo) vẫn phải mô tả ĐÚNG những gì nhìn thấy trong ảnh.

LƯU Ý QUAN TRỌNG: "matches=true, confidence cao" chỉ có nghĩa là ẢNH NÀY TRÔNG HỢP
với câu hỏi theo những gì bạn nhìn thấy — KHÔNG có nghĩa đây chắc chắn là ĐÚNG THỜI
ĐIỂM/ĐÚNG NGƯỜI mà câu hỏi nhắc tới. Nếu câu hỏi nhắc TÊN RIÊNG một người cụ thể mà
bạn KHÔNG thấy tên đó xuất hiện dạng chữ/caption trên ảnh (dù ảnh có 1 người phụ nữ
trông giống "hợp lý"), hãy HẠ confidence xuống rõ rệt (vd <= 0.5) để phản ánh việc
bạn đang ĐOÁN dựa trên hình ảnh chung chung, không có bằng chứng danh tính xác thực.

Trả lời NGAY, không suy luận dài dòng. CHỈ trả về đúng 1 object JSON hợp lệ (không
markdown, không giải thích thêm) với các trường theo ĐÚNG THỨ TỰ sau:
- "matches": true nếu ảnh khớp/liên quan trực tiếp tới câu hỏi, false nếu không.
- "confidence": số thực từ 0.0 đến 1.0 — mức độ tự tin vào đánh giá "matches".
- "answer": nếu matches=true, trả lời trực tiếp câu hỏi bằng tiếng Việt; nếu false thì để chuỗi rỗng "".
- "reasoning": lý do, viết trong TỐI ĐA 1 câu ngắn.
"""

        # Retry tối đa 1 lần nếu JSON hỏng hoặc matches=true nhưng answer rỗng
        # (mâu thuẫn). Không retry khi hết quota (raise ngay để
        # predict_single_vote() dừng sớm toàn bộ vòng lặp).
        max_retries = 1
        last_parse_error = None
        for attempt in range(max_retries + 1):
            prompt_to_send = prompt
            if attempt > 0:
                prompt_to_send += (
                    "\n\nLƯU Ý: lượt trả lời trước đó KHÔNG HỢP LỆ (JSON sai định dạng, "
                    "hoặc matches=true nhưng answer để trống). Hãy trả lời LẠI, đảm bảo "
                    "ĐÚNG định dạng JSON, và nếu matches=true thì answer PHẢI có nội dung."
                )
            try:
                raw_text, used_model = _chat_with_fallback(
                    [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_to_send},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }],
                    max_tokens=300,
                    temperature=0.2,
                )
                result = _extract_json(raw_text)
            except OpenRouterQuotaExhausted:
                raise
            except (json.JSONDecodeError, ValueError) as e:
                last_parse_error = e
                if attempt < max_retries:
                    print(f"  ⚠️ JSON không hợp lệ ở lượt {attempt+1}, thử lại...")
                    continue
                return {
                    "matches": False, "confidence": 0.0, "answer": "",
                    "reasoning": f"Lỗi parse JSON sau {max_retries+1} lượt thử: {last_parse_error}",
                    "_model_used": "",
                }

            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            matches = bool(result.get("matches", False))
            answer = (result.get("answer") or "").strip()

            if matches and not answer and attempt < max_retries:
                print(f"  ⚠️ matches=true nhưng answer rỗng ở lượt {attempt+1}, thử lại...")
                continue

            return {
                "matches": matches,
                "confidence": confidence,
                "answer": answer,
                "reasoning": (result.get("reasoning") or "").strip(),
                "_model_used": used_model,
            }
    except OpenRouterQuotaExhausted:
        raise  # để predict_single_vote() xử lý dừng sớm, không nuốt lỗi ở đây
    except Exception as e:
        return {
            "matches": False, "confidence": 0.0, "answer": "",
            "reasoning": f"Lỗi chấm điểm candidate: {type(e).__name__}: {e}",
            "_model_used": "",
        }


def predict_single_vote(query: str, image_paths: list[str], candidates_info: list[dict]) -> dict:
    """Thực hiện 1 lượt: chấm điểm TỪNG candidate độc lập (đúng 1 ảnh keyframe
    thật / candidate — KHÔNG còn ghép lưới) rồi chọn candidate tốt nhất trong
    danh sách đã được Object Rerank (Tầng 2) + Grouping chọn ra (mặc định
    FINAL_TOP_K=5, xem config.py).

    Nếu hết quota giữa chừng -> DỪNG ngay vòng lặp, trả về kết quả rõ ràng dựa
    trên những gì đã chấm điểm được (nếu có), thay vì crash hoặc lặp lỗi.
    """
    valid_pairs = [
        (c, p) for c, p in zip(candidates_info, image_paths)
        if p and os.path.exists(p)
    ]
    if not valid_pairs:
        return {
            "reasoning": "Không có ảnh keyframe hợp lệ nào cho các candidate.",
            "video_id": "", "frame_id": "", "pts_time": 0.0, "answer": "",
        }

    scored = []
    quota_exhausted = False
    for c, p in valid_pairs:
        try:
            r = _score_candidate(query, p, c)
        except OpenRouterQuotaExhausted as e:
            print(f"⛔ {e}\n   -> dừng chấm điểm các candidate còn lại trong vote này.")
            quota_exhausted = True
            break
        scored.append((c, r))
        print(f"  · candidate id={c.get('id')} video={c.get('video_id')} t={c.get('pts_time')} "
              f"-> matches={r['matches']} conf={r['confidence']:.2f} model={r.get('_model_used') or '(lỗi)'}")

    if not scored:
        return {
            "video_id": "", "frame_id": "", "pts_time": 0.0, "answer": "",
            "reasoning": "Hết quota OpenRouter free-tier — chưa chấm điểm được candidate nào."
                         if quota_exhausted else "Không chấm điểm được candidate nào (xem log).",
            "_quota_exhausted": quota_exhausted,
        }

    matched = [(c, r) for c, r in scored if r["matches"]]
    pool = matched if matched else scored

    # FIX (evidence correctness ≠ answer correctness — VLM có thể trả lời đúng
    # trên MỘT FRAME SAI): nếu trong pool có candidate mang "identity_match"
    # (đã được Identity Rescan xác nhận bằng OCR caption tên thật, xem
    # identity_rescan.py) — ưu tiên nhóm này làm ứng viên tối hậu.
    identity_pool = [(c, r) for c, r in pool if c.get("identity_match")]
    if identity_pool:
        pool = identity_pool

    best_c, best_r = max(pool, key=lambda cr: cr[1]["confidence"])

    # video_id/pts_time/frame_id lấy THẲNG từ metadata thật -> không thể hallucinate.
    return {
        "video_id": best_c.get("video_id", ""),
        "pts_time": best_c.get("pts_time", 0.0),
        "frame_id": best_c.get("id", ""),
        "answer": best_r.get("answer") or "(model không đưa ra câu trả lời rõ ràng)",
        "reasoning": best_r.get("reasoning", ""),
        "confidence": best_r.get("confidence", 0.0),
        "_model_used": best_r.get("_model_used", ""),
        "_matched_any": bool(matched),
        "_identity_confirmed": bool(best_c.get("identity_match")),
        "_quota_exhausted": quota_exhausted,
    }


def solve_vqa_with_self_consistency(query: str, image_paths: list[str], candidates_info: list[dict]) -> dict:
    """Tầng 4: Self-Consistency Voting qua N lượt (config SELF_CONSISTENCY_VOTES,
    mặc định = 1 -> thực chất KHÔNG voting, chỉ 1 lượt duy nhất, xem
    config.py để tăng lại nếu đã nạp credit OpenRouter).

    Nếu 1 vote báo hết quota -> DỪNG hẳn sớm (không thử vote kế tiếp)."""
    votes = []
    last_error_reasoning = ""
    if SELF_CONSISTENCY_VOTES > 1:
        print(f"🧠 Đang thực hiện Self-Consistency ({SELF_CONSISTENCY_VOTES} votes)...")

    for i in range(SELF_CONSISTENCY_VOTES):
        if SELF_CONSISTENCY_VOTES > 1:
            print(f" Vote {i+1}/{SELF_CONSISTENCY_VOTES}:")
        res = predict_single_vote(query, image_paths, candidates_info)

        if res.get("_quota_exhausted") and not res.get("video_id"):
            last_error_reasoning = res.get("reasoning", "Hết quota OpenRouter.")
            print(f"⛔ {last_error_reasoning} — dừng sớm ở vote {i+1}.")
            break

        vid = (res.get("video_id") or "").strip()
        if vid:
            try:
                pts_time = float(res.get("pts_time"))
            except (TypeError, ValueError):
                pts_time = None

            if pts_time is not None:
                try:
                    model_conf = max(0.0, min(1.0, float(res.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    model_conf = 0.0
                votes.append((
                    vid,
                    pts_time,
                    res.get("answer", ""),
                    res.get("reasoning", ""),
                    res.get("frame_id", ""),
                    model_conf,
                ))
            else:
                last_error_reasoning = f"Vote {i+1}: pts_time không hợp lệ: {res.get('pts_time')!r}"
                print(f"  ⚠️ {last_error_reasoning}")
        else:
            last_error_reasoning = res.get("reasoning", "")
            print(f"  ⚠️ Vote {i+1}/{SELF_CONSISTENCY_VOTES} thất bại: {last_error_reasoning}")

        if res.get("_quota_exhausted"):
            print("⛔ Quota hết giữa chừng vote này — dừng sớm, không vote tiếp.")
            break

    if not votes:
        return {
            "video_id": "", "frame_id": "", "pts_time": 0.0,
            "answer": "Không tìm thấy câu trả lời",
            "reasoning": last_error_reasoning or "Không rõ nguyên nhân — xem log console.",
            "confidence": 0.0,
        }

    video_counts = Counter([v[0] for v in votes])
    best_video_id, top_count = video_counts.most_common(1)[0]

    matching_votes = [v for v in votes if v[0] == best_video_id]
    avg_pts = sum([v[1] for v in matching_votes]) / len(matching_votes)
    best_ans = matching_votes[0][2]
    best_reasoning = matching_votes[0][3]
    best_frame_id = matching_votes[0][4]

    avg_model_conf = sum(v[5] for v in matching_votes) / len(matching_votes)
    vote_agreement = top_count / len(votes)
    final_confidence = vote_agreement * avg_model_conf

    return {
        "video_id": best_video_id,
        "frame_id": best_frame_id,
        "pts_time": round(avg_pts, 2),
        "answer": best_ans,
        "reasoning": best_reasoning,
        "confidence": round(final_confidence, 2),
        "vote_agreement": round(vote_agreement, 2),
        "avg_model_confidence": round(avg_model_conf, 2),
    }


# Alias tương thích với pipeline.py
ask_gemini_self_consistency = solve_vqa_with_self_consistency