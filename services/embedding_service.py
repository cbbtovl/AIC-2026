"""
Embedding Service - Đã đồng bộ chuẩn Không gian Vector CLIP 512 Chiều (512D):
  - Image Search: Dùng CLIP ViT-B/32 (512D) [open_clip_torch] - Khớp 100% với file .npy của BTC
  - Text Search (Đa ngôn ngữ / Tiếng Việt): Dùng clip-ViT-B-32-multilingual-v1 (512D) [sentence_transformers]
"""
import sys
import torch
from typing import List
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─── BỘ NHỚ ĐỆM MÔ HÌNH (CACHE) ─────────────────────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_multilingual_text_model = None

# Tự động nhận diện GPU để xử lý siêu tốc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip():
    """Tải mô hình CLIP ViT-B/32 chuẩn 512D (Đồng bộ với dataset .npy của BTC)"""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        try:
            import open_clip
        except ImportError:
            raise ImportError("Chưa cài thư viện open_clip_torch. Chạy: pip install open_clip_torch")

        model_name = "ViT-B-32"
        print(f"[Embedding] Đang tải mô hình thị giác CLIP {model_name} (512D)...")
        
        try:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
        except Exception:
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="laion2b_s34b_b79k"
            )
            
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        _clip_model = _clip_model.to(DEVICE).eval()
        print(f"✅ [Embedding] Hệ thống Vector CLIP (512D) đã sẵn sàng trên: {DEVICE}.")
    
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _load_multilingual_text_model():
    """Tải mô hình Text CLIP Đa Ngôn Ngữ (Hiểu sâu Tiếng Việt, ép ra Vector 512D chuẩn CLIP)"""
    global _multilingual_text_model
    if _multilingual_text_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Chưa cài sentence_transformers. Chạy: pip install sentence_transformers")

        model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        print(f"[Embedding] Đang tải mô hình Văn bản Tiếng Việt Đa ngôn ngữ: {model_name}...")
        try:
            _multilingual_text_model = SentenceTransformer(model_name, local_files_only=True).to(DEVICE)
        except Exception:
            _multilingual_text_model = SentenceTransformer(model_name).to(DEVICE)
        print("✅ [Embedding] Mô hình Văn bản Đa ngôn ngữ (512D) đã sẵn sàng.")
    
    return _multilingual_text_model


# ─── CÁC HÀM API CHÍNH DÙNG CHO INDEXER & SEARCH ─────────────────────────

def get_image_embedding(image_path: str) -> List[float]:
    """Tạo Vector đại diện cho ảnh (512D) bằng CLIP ViT-B/32"""
    try:
        model, preprocess, _ = _load_clip()
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # Chuẩn hóa L2
        return feat.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ [Embedding] Lỗi tạo vector hình ảnh {image_path}: {e}")
        return []


def get_image_embeddings_batch(image_paths: List[str]) -> List[List[float]]:
    """Tạo Vector đại diện cho một lô ảnh (512D) xử lý song song trên GPU"""
    if not image_paths:
        return []
    try:
        model, preprocess, _ = _load_clip()
        imgs = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                imgs.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️ [Embedding] Lỗi đọc ảnh {path}: {e}")
        
        if not imgs:
            return [[] for _ in image_paths]
            
        batch_tensor = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            
        features_list = features.cpu().numpy().tolist()
        
        final_results = [[] for _ in image_paths]
        for idx, feat in zip(valid_indices, features_list):
            final_results[idx] = feat
            
        return final_results
    except Exception as e:
        print(f"❌ [Embedding] Lỗi tạo vector lô hình ảnh: {e}")
        return [[] for _ in image_paths]


def get_text_embedding(text: str) -> List[float]:
    """Tạo Vector 512D từ câu truy vấn Tiếng Việt/Anh (Ép vào Không gian CLIP)"""
    try:
        model = _load_multilingual_text_model()
        emb = model.encode(text, normalize_embeddings=True)
        return emb.tolist()
    except Exception as e:
        print(f"❌ [Embedding Error]: {e}")
        return []


def get_clip_text_embedding(query: str) -> List[float]:
    """
    Tạo Vector 512D từ câu truy vấn (Hỗ trợ Tiếng Việt mượt mà).
    Thay vì dùng Tokenizer gốc bị lỗi Tiếng Việt, hàm này chuyển hướng sang Multilingual CLIP.
    """
    return get_text_embedding(query)


def get_embedding(text: str) -> List[float]:
    """Hàm tương thích ngược cho các file cũ"""
    return get_text_embedding(text)