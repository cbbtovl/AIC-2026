"""
VQA & KISC Service - Phiên bản Multimodal RAG (Tối thượng cho AIC 2026):
  - Ép phạm vi dữ liệu (Grounding) dựa trên kết quả lọc không gian từ FAISS.
  - Hỗ trợ hội thoại tương tác (Conversational KIS) để thu hẹp phạm vi qua từng lượt.
"""

import os
import sys
from google import genai
from google.genai.errors import ClientError
from PIL import Image
from config import config
from indexer import query_search_text
from services.object_service import summarize_frame_objects

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def _safe_print(*values):
    message = " ".join(str(v) for v in values)
    try:
        print(message)
    except UnicodeEncodeError:
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write((message + "\n").encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            else:
                print(message.encode("cp1252", errors="replace").decode("cp1252"))
        except Exception:
            print(message.encode("ascii", errors="replace").decode("ascii"))

_gemini_client_instance = None

def _get_gemini_client():
    global _gemini_client_instance
    if _gemini_client_instance is None and config.gemini_api_key:
        _gemini_client_instance = genai.Client(api_key=config.gemini_api_key)
    return _gemini_client_instance

def interact_kisc(query_text: str, chat_history: list = None, filetype_filter: str = "Hình ảnh") -> dict:
    _safe_print(f"[VQA] Start; query='{query_text[:50]}'")
    client = _get_gemini_client()
    
    # 1. Lọc dữ liệu
    raw_candidates = query_search_text(query_text, filetype_filter=filetype_filter, limit=15)
    if not raw_candidates:
        return {"answer": "Không tìm thấy manh mối nào khớp.", "candidates": []}

    # 2. Xử lý ngữ cảnh (Context)
    context_items = []
    valid_images = []
    for cand in raw_candidates:
        filepath = cand.get("filepath", "")
        # ... (giữ nguyên logic tạo info)
        info = f"{cand.get('filename')} | {cand.get('description')} | {cand.get('extracted_text')}"
        context_items.append(info)
        
        # Thêm ảnh an toàn
        if filepath and os.path.exists(filepath) and len(valid_images) < 3:
            try:
                img = Image.open(filepath)
                img.thumbnail((1024, 1024)) # Resize để tránh lỗi quá tải
                valid_images.append(img)
            except: pass

    # 3. Tạo Prompt
    prompt = f"Người dùng hỏi: '{query_text}'. Dựa vào đây: \n" + "\n".join(context_items)
    
    # 4. Gửi lên Gemini (Bản mới)
    try:
        # Cách gửi đúng chuẩn: tạo một danh sách [anh1, anh2, text]
        contents = valid_images + [prompt]
        
        # Gọi model
        response = client.models.generate_content(
            model=config.gemini_model,  # 👈 Lấy trực tiếp từ config
            contents=contents
        )
        return {"answer": response.text, "candidates": raw_candidates[:5]}

    except Exception as e:
        _safe_print(f"❌ LỖI VQA: {str(e)}")
        # Fallback text
        try:
            resp_fallback = client.models.generate_content(
                model=config.gemini_model,  # 👈 Lấy trực tiếp từ config
                contents=[f"Dữ liệu: {context_items}. Câu hỏi: {query_text}"]
            )
            return {"answer": resp_fallback.text, "candidates": raw_candidates[:5]}
        except:
            return {"answer": "Hệ thống đang bảo trì, vui lòng thử lại sau.", "candidates": raw_candidates[:5]}