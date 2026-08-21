# segment_llm.py
"""
Client LLM RIÊNG dùng cho tiền xử lý offline (segment_topics.py) — TÁCH BIỆT
HOÀN TOÀN quota với qwen_vqa.py (dùng cho pipeline lúc thi đấu thật).

LÝ DO CẦN FILE NÀY:
    OpenRouter free-tier giới hạn quota THEO TÀI KHOẢN (20 req/phút, 50
    req/ngày), KHÔNG PHẢI theo từng model. Vì vậy đổi model trong
    config.VLM_MODELS (vd nvidia -> gemini) mà vẫn dùng CHUNG 1 API key thì
    KHÔNG có tác dụng tiết kiệm quota gì cả — 429 vẫn 429.

    Giải pháp thật sự: dùng 1 TÀI KHOẢN OpenRouter THỨ HAI (email khác, key
    khác) CHỈ để chạy segment_topics.py — hoàn toàn không đụng tới quota của
    tài khoản chính (dùng cho qwen_vqa.py / pipeline.py lúc thi).

CÁCH DÙNG:
    1. Tạo tài khoản OpenRouter thứ 2 (email khác) -> lấy API key mới.
    2. Thêm vào .env:
           OPENROUTER_API_KEY_SEGMENT=sk-or-v1-xxxxxxxx
    3. Trong segment_topics.py, đổi:
           from qwen_vqa import _chat_with_fallback, _extract_json, OpenRouterQuotaExhausted
       thành:
           from segment_llm import chat_segment as _chat_with_fallback
           from qwen_vqa import _extract_json, OpenRouterQuotaExhausted
       (_extract_json là hàm parse JSON thuần, KHÔNG gọi API -> tái dùng an
       toàn, không tốn quota gì thêm.)
    4. Chạy như cũ: python segment_topics.py (resumable — hết quota giữa
       chừng vẫn ghi được tiến độ, chạy lại sau sẽ tự bỏ qua phần đã xong).

Toàn bộ logic throttle/retry/quota-detection COPY NGUYÊN từ qwen_vqa.py để
hành vi nhất quán (không tạo ra 2 cách xử lý lỗi khác nhau), chỉ đổi:
    - Tên biến môi trường API key (KEY_SEGMENT thay vì KEY chính).
    - Danh sách model riêng (SEGMENT_MODELS, có thể khác VLM_MODELS).
    - Không có phần xử lý ảnh (segment_topics.py chỉ cần text-only).
"""

import os
import time as _time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ----- Model dùng cho segment_topics.py — ĐỔI Ở ĐÂY nếu muốn thử model khác.
# Vẫn nên giữ vài fallback phòng khi 1 model bị OpenRouter tạm gỡ khỏi free
# tier hoặc quá tải ở nhà cung cấp (KHÔNG phải lỗi quota tài khoản). -----
SEGMENT_MODELS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-26b-a4b-it:free",
]

# Rate limit tự khai báo cho tài khoản THỨ HAI này — giữ an toàn giống
# qwen_vqa.py (mặc định OpenRouter free = 20 req/phút dùng chung mọi model).
SEGMENT_RPM_LIMIT = 20

_client = None


class SegmentQuotaExhausted(Exception):
    """Quota free-tier của TÀI KHOẢN THỨ HAI (dành riêng cho segment_topics.py)
    đã hết — KHÔNG liên quan gì tới quota của qwen_vqa.py/pipeline chính."""
    pass


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY_SEGMENT", "").strip()
        if not api_key:
            raise ValueError(
                "❌ Không tìm thấy OPENROUTER_API_KEY_SEGMENT trong file .env! "
                "Tạo 1 tài khoản OpenRouter THỨ HAI (email khác), lấy API key, "
                "rồi thêm dòng OPENROUTER_API_KEY_SEGMENT=... vào .env "
                "(tách biệt hoàn toàn với OPENROUTER_API_KEY dùng cho pipeline thi)."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/aic2026-vqa",
                "X-Title": "AIC2026 VQA Pipeline - Segment Preprocessing",
            },
        )
    return _client


# ----- Throttle riêng cho tài khoản này — KHÔNG dùng chung timestamp list
# với qwen_vqa._call_timestamps (2 tài khoản khác nhau, không liên quan). -----
_call_timestamps: list[float] = []
_RATE_WINDOW_SEC = 60.0
_RATE_SAFETY_MARGIN = 2


def _throttle():
    now = _time.time()
    while _call_timestamps and now - _call_timestamps[0] > _RATE_WINDOW_SEC:
        _call_timestamps.pop(0)
    limit = max(1, SEGMENT_RPM_LIMIT - _RATE_SAFETY_MARGIN)
    if len(_call_timestamps) >= limit:
        sleep_for = _RATE_WINDOW_SEC - (now - _call_timestamps[0]) + 0.5
        if sleep_for > 0:
            print(f"⏳ [segment_llm] Chủ động chờ {sleep_for:.1f}s để tránh chạm "
                  f"rate-limit {SEGMENT_RPM_LIMIT} request/phút...")
            _time.sleep(sleep_for)
    _call_timestamps.append(_time.time())


def _is_account_wide_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "free-models-per-day" in msg
        or "free-models-per-min" in msg
        or "openrouter_free_tier_daily" in msg
        or "openrouter_free_tier_per_minute" in msg
    )


def chat_segment(messages: list[dict], **kwargs):
    """Cùng SIGNATURE với qwen_vqa._chat_with_fallback(messages, **kwargs) ->
    segment_topics.py chỉ cần đổi 1 dòng import, không cần sửa gì khác.

    Trả về (raw_text: str, model_name: str). Raise SegmentQuotaExhausted nếu
    quota tài khoản THỨ HAI hết (KHÔNG phải OpenRouterQuotaExhausted của
    qwen_vqa.py — cố tình tách class để không lẫn lộn 2 tài khoản khi debug)."""
    client = _get_client()
    last_err = None
    for model_name in SEGMENT_MODELS:
        _throttle()
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages, **kwargs)
            content = resp.choices[0].message.content
            if content and content.strip():
                return content, model_name
            last_err = f"Model {model_name} trả về nội dung rỗng."
        except Exception as e:
            if _is_account_wide_quota_error(e):
                raise SegmentQuotaExhausted(
                    "Đã chạm giới hạn free-tier của tài khoản OpenRouter THỨ HAI "
                    "(dành riêng cho segment_topics.py). Đợi reset (phút: ~60s; "
                    "ngày: 00:00 UTC ~7h sáng giờ VN) hoặc tạo thêm 1 tài khoản khác."
                ) from e
            last_err = f"Model {model_name} lỗi: {type(e).__name__}: {e}"
            print(f"  ⚠️ [segment_llm] {last_err} — thử model kế tiếp...")
            continue
    raise RuntimeError(last_err or "Không có model nào trong SEGMENT_MODELS khả dụng.")


if __name__ == "__main__":
    # Test nhanh: python segment_llm.py
    text, model = chat_segment(
        [{"role": "user", "content": "Trả lời đúng 1 chữ: OK"}],
        max_tokens=10, temperature=0.0,
    )
    print(f"[{model}] -> {text!r}")