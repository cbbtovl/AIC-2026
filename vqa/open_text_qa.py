# open_text_qa.py
"""
TẦNG BỔ SUNG — "Open-Text Extraction".

VẤN ĐỀ: identity_rescan.py (bản gốc) chỉ giải được bài toán "tôi ĐÃ BIẾT
target string (tên người), tìm frame OCR khớp đúng chuỗi đó". Nhưng có 1 lớp
câu hỏi khác, phổ biến không kém trong AIC: đáp án CHÍNH LÀ 1 chuỗi text
chưa biết trước (tên xã, tên biển hiệu, số liệu trên bảng...) — không có gì
để "so khớp", chỉ có thể tìm bằng cách đọc OCR trên nhiều frame rồi để LLM
tự suy luận đáp án nằm ở đâu trong đó.

Ví dụ: "CLB FANA trao quà tại 1 xã thuộc Khánh Hòa. Xã đó tên gì?"
  - Không có "tên xã" nào để so khớp trước (đó chính là ẩn số).
  - Nhưng nếu quét OCR đủ nhiều frame của (các) video nghi vấn, khả năng cao
    sẽ bắt được 1 khung hình có banner/backdrop ghi rõ "UBND XÃ ... - HUYỆN
    ... - TỈNH KHÁNH HÒA" hoặc phông chữ chương trình có in tên xã.

CÁCH DÙNG (xem pipeline.py đã cập nhật):
    1. Nếu câu hỏi thuộc dạng open-text (is_open_text_query) VÀ candidate đã
       hội tụ về ít video_id -> quét OCR RỘNG hơn (nhiều frame hơn identity
       rescan thường dùng) bằng identity_rescan.rescan_videos_open_text().
    2. Gộp toàn bộ text OCR thu được (kèm mốc thời gian) thành 1 khối context
       NGẮN GỌN, đưa cho LLM (text-only, KHÔNG cần ảnh -> rẻ hơn nhiều so với
       chấm từng ảnh bằng VLM) để trích xuất đáp án + xác định frame chứa nó.
    3. Nếu tìm được (found=True) -> dùng ngay kết quả này làm câu trả lời
       cuối, KHÔNG cần chờ VLM chấm ảnh nữa (tiết kiệm quota + chính xác hơn
       vì đáp án đọc trực tiếp từ chữ trên hình, không phải suy đoán thị
       giác). Nếu không tìm được -> fallback về luồng VLM chấm ảnh như cũ.

Chi phí: 1 request LLM text-only/câu hỏi (rẻ hơn hẳn N request VLM có ảnh của
predict_single_vote) — chỉ tốn thêm khi câu hỏi thực sự thuộc dạng open-text.
"""

import re

# Tái dùng hạ tầng gọi OpenRouter đã có sẵn trong qwen_vqa.py — KHÔNG viết
# lại client/throttle/quota-handling để tránh 2 nguồn sự thật khác nhau.
from qwen_vqa import _chat_with_fallback, _extract_json, OpenRouterQuotaExhausted


# Các mẫu câu hỏi "đáp án là 1 chuỗi text cần đọc trên màn hình", tổng quát
# hơn OCR_TRIGGER_KEYWORDS trong config.py (vốn chỉ dùng để BẬT/TẮT OCR, không
# phân biệt được đây có phải câu hỏi "open-text" hay chỉ tình cờ có chữ "tên").
_OPEN_TEXT_PATTERNS = [
    r"tên là gì", r"tên gì", r"tên của .*là gì", r"gọi là gì",
    r"ghi (chữ )?gì", r"viết (chữ )?gì", r"hiển thị (chữ )?gì",
    r"\bxã (nào|này|gì)\b", r"\bhuyện (nào|này|gì)\b",
    r"\bphường (nào|này|gì)\b", r"\bthôn (nào|này|gì)\b",
    r"biển (hiệu|báo|số) (ghi|là) gì", r"số (bao nhiêu|mấy)",
    r"con số (nào|gì)", r"dòng chữ (nào|gì)",
]
_OPEN_TEXT_RE = re.compile("|".join(_OPEN_TEXT_PATTERNS), re.IGNORECASE)


def is_open_text_query(question: str) -> bool:
    """True nếu câu hỏi có dạng 'đáp án là 1 chuỗi text/tên riêng cần đọc
    trên màn hình mà ta CHƯA BIẾT TRƯỚC' — khác với identity_rescan (biết
    trước tên người, chỉ cần xác nhận đúng frame)."""
    return bool(_OPEN_TEXT_RE.search(question or ""))


def extract_answer_from_ocr_context(
    question: str,
    ocr_snippets: list[dict],
    max_context_chars: int = 6000,
) -> dict:
    """Gộp OCR text của nhiều frame (mỗi phần tử cần có 'pts_time', 'video_id',
    'id', 'ocr_text') thành 1 khối context, hỏi LLM (text-only) trích xuất
    đáp án trực tiếp.

    Trả về dict:
        {"found": bool, "answer": str, "frame_id": str, "video_id": str,
         "pts_time": float, "reasoning": str}
    Không raise ra ngoài (trừ OpenRouterQuotaExhausted, để pipeline.py biết
    dừng sớm) — lỗi khác trả về found=False để pipeline fallback sang VLM.
    """
    usable = [s for s in ocr_snippets if (s.get("ocr_text") or "").strip()]
    if not usable:
        return {"found": False, "answer": "", "frame_id": "", "video_id": "",
                "pts_time": 0.0, "reasoning": "Không có OCR text nào để trích xuất."}

    # Sắp theo thời gian cho LLM dễ theo mạch, và cắt bớt nếu quá dài (tránh
    # vượt context của model free-tier) — ưu tiên GIỮ NGUYÊN các đoạn ĐẦU vì
    # thường banner/backdrop chứa tên địa danh xuất hiện sớm trong cảnh trao quà.
    usable.sort(key=lambda s: (s.get("video_id", ""), s.get("pts_time", 0.0)))

    lines = []
    total_len = 0
    for s in usable:
        line = f"[id={s.get('id','')} | video={s.get('video_id','')} | t={s.get('pts_time',0):.1f}s] {s['ocr_text']}"
        if total_len + len(line) > max_context_chars:
            break
        lines.append(line)
        total_len += len(line)
    context = "\n".join(lines)

    prompt = f"""Dưới đây là các đoạn text đọc được bằng OCR từ nhiều khung hình khác nhau
của (các) video ứng viên. OCR có thể lỗi chính tả, thiếu dấu, hoặc đọc sai vài ký tự
— hãy dùng suy luận ngữ cảnh để sửa các lỗi nhỏ hợp lý (vd thiếu dấu tiếng Việt).

Câu hỏi cần trả lời: {question}

Các đoạn OCR (định dạng [id=... | video=... | t=...s] nội dung):
{context}

Nhiệm vụ: tìm trong các đoạn OCR trên đoạn nào chứa CÂU TRẢ LỜI TRỰC TIẾP cho câu hỏi
(ví dụ tên xã/phường/huyện, tên biển hiệu, con số cụ thể...). CHỈ dùng thông tin THỰC
SỰ CÓ trong OCR — không suy đoán, không bịa nếu không thấy rõ.

CHỈ trả về đúng 1 object JSON hợp lệ (không markdown, không giải thích thêm):
{{
  "found": true/false,
  "answer": "đáp án ngắn gọn tiếng Việt nếu found=true, ngược lại chuỗi rỗng",
  "matched_id": "giá trị id của dòng OCR chứa bằng chứng, nếu found=true, ngược lại rỗng",
  "reasoning": "tối đa 1 câu ngắn giải thích vì sao chọn đoạn đó"
}}"""

    try:
        raw_text, _ = _chat_with_fallback(
            [{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.1,
        )
        result = _extract_json(raw_text)
    except OpenRouterQuotaExhausted:
        raise
    except Exception as e:
        return {"found": False, "answer": "", "frame_id": "", "video_id": "",
                "pts_time": 0.0, "reasoning": f"Lỗi trích xuất OCR-QA: {type(e).__name__}: {e}"}

    found = bool(result.get("found", False))
    answer = (result.get("answer") or "").strip()
    matched_id = (result.get("matched_id") or "").strip()

    if not found or not answer:
        return {"found": False, "answer": "", "frame_id": "", "video_id": "",
                "pts_time": 0.0, "reasoning": (result.get("reasoning") or "").strip()}

    # Map matched_id -> metadata thật (video_id/pts_time lấy từ dữ liệu có sẵn,
    # KHÔNG tin id do model tự "nhớ lại" nếu nó không khớp danh sách thật).
    by_id = {s.get("id"): s for s in usable}
    src = by_id.get(matched_id)
    if src is None:
        # model không trả đúng id đã cho -> vẫn coi found nhưng thiếu định vị frame chính xác
        return {
            "found": True, "answer": answer, "frame_id": "", "video_id": "",
            "pts_time": 0.0,
            "reasoning": (result.get("reasoning") or "").strip()
            + " (⚠️ không map được matched_id về frame cụ thể)",
        }

    return {
        "found": True,
        "answer": answer,
        "frame_id": src.get("id", ""),
        "video_id": src.get("video_id", ""),
        "pts_time": src.get("pts_time", 0.0),
        "reasoning": (result.get("reasoning") or "").strip(),
    }