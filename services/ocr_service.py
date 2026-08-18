"""
OCR Service - Bản tối giản sạch import thừa + Chống sập lỗi thư viện CLIP
"""
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
from google import genai
from config import config

# --- MODEL & FEATURE CACHES CỤC BỘ ---
_paddle_ocr_engine = None
_cached_text_features = None
_candidate_descriptions = [
    "a document or text page", "a screenshot of code or software",
    "a photo of people", "an infographic or diagram",
    "a natural landscape", "an indoor room scene", "a product or object photo"
]

# --- ĐỊNH VỊ THIẾT BỊ CHẠY ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- TỰ ĐỘNG KẾT NỐI EMBEDDING SERVICE ---
def _clip_provider():
    try:
        from services.embedding_service import _load_clip
        return _load_clip()
    except ImportError:
        try:
            from embedding_service import _load_clip
            return _load_clip()
        except ImportError:
            raise ImportError("Không thể tìm thấy embedding_service")

def _get_paddle_ocr():
    global _paddle_ocr_engine
    if _paddle_ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr_engine = PaddleOCR(use_angle_cls=True, lang='vi')
            print("✅ Đã khởi tạo thành công PaddleOCR v4 Tiếng Việt!")
        except ImportError:
            print("⚠️ Chưa cài paddleocr. Hãy chạy lệnh: pip install paddlepaddle paddleocr")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo PaddleOCR: {e}")
    return _paddle_ocr_engine

# --- CHẾ ĐỘ CLOUD (GEMINI SDK MỚI - SẠCH SẼ) ---
def _process_image_gemini(image_path: str) -> dict:
    if not config.gemini_api_key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY")
        
    client = genai.Client(api_key=config.gemini_api_key)
    img = Image.open(image_path)
    
    prompt = (
        "Bạn là hệ thống phân tích hình ảnh thông minh. Hãy thực hiện 2 nhiệm vụ:\n"
        "1. OCR: Trích xuất CHÍNH XÁC toàn bộ văn bản xuất hiện trong ảnh.\n"
        "2. Mô tả: Mô tả ngắn gọn nội dung cốt lõi của ảnh.\n"
        "Trả về JSON chính xác dạng: {\"extracted_text\": \"...\", \"description\": \"...\"}"
    )
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[img, prompt],
        config={'response_mime_type': 'application/json'}
    )
    
    parsed = json.loads(response.text)
    return {
        "extracted_text": parsed.get("extracted_text", ""),
        "description": parsed.get("description", "")
    }

# --- CHẾ ĐỘ OFFLINE CỤC BỘ (TỐI ƯU TOÀN DIỆN) ---
def _process_image_local_paddle(image_path: str) -> dict:
    global _cached_text_features
    filename = Path(image_path).name
    extracted_text = ""
    description = f"Local Scan: {filename}."

    # 1. Quét chữ bằng PaddleOCR
    try:
        ocr_engine = _get_paddle_ocr()
        if ocr_engine is not None:
            result = ocr_engine.ocr(image_path, cls=True)
            if result and result[0]:
                text_lines = [line[1][0] for line in result[0] if line and line[1]]
                extracted_text = " ".join(text_lines).strip()
                description += f" Trích xuất thành công {len(text_lines)} dòng chữ."
    except Exception as e:
        extracted_text = f"[Lỗi local OCR: {e}]"

    # 2. Phân tích ngữ cảnh bằng CLIP (Nếu thiếu thư viện, hệ thống vẫn chạy mượt mà)
    try:
        model, preprocess, tokenizer = _clip_provider()
        
        # Chỉ mã hóa danh sách text mẫu đúng 1 lần trên RAM
        if _cached_text_features is None:
            text_tokens = tokenizer(_candidate_descriptions).to(DEVICE)
            with torch.no_grad():
                txt_feat = model.encode_text(text_tokens)
                _cached_text_features = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        
        # Trích xuất tính năng ảnh
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ _cached_text_features.T).squeeze().cpu().numpy()

        top_idx = np.argsort(sims)[::-1][:2]
        top_tags = [_candidate_descriptions[i] for i in top_idx]
        description += f" Ngữ cảnh (CLIP): {', '.join(top_tags)}."
    except Exception as e:
        # Giảm cấp thông minh: Ghi log nhẹ và bỏ qua bước CLIP chứ không làm sập ứng dụng
        description += " [Ngữ cảnh: Không khả dụng do thiếu cấu hình CLIP]"
        
    return {"extracted_text": extracted_text, "description": description}

def process_image(image_path: str) -> dict:
    if config.provider == "gemini" and config.gemini_api_key:
        try:
            return _process_image_gemini(image_path)
        except Exception as e:
            print(f"[OCR] Gemini lỗi ({e}). Tự động hạ cấp sang Local.")
    return _process_image_local_paddle(image_path)