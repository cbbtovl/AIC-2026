"""
Embedding bằng SigLIP (chạy local, không tốn API).
Dùng chung cho cả lúc ingest (embed ảnh) và lúc query (embed câu hỏi text).
"""

import os
import threading
# Nếu vẫn crash sau khi đã dùng đúng GPU, bật dòng dưới để CUDA báo lỗi
# đúng vị trí thay vì crash lệch chỗ do thực thi bất đồng bộ:
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

from config import SIGLIP_MODEL_NAME

_model = None
_processor = None
_device = "cpu"  # được cập nhật đúng trong _load_model()
_load_lock = threading.Lock()  # chống nhiều luồng cùng tải model 1 lúc (Streamlit rerun)


def _to_tensor(output):
    """
    model.get_image_features()/get_text_features() bình thường trả thẳng
    torch.Tensor, nhưng ở 1 số phiên bản transformers lại trả về object
    (BaseModelOutputWithPooling) chứa tensor bên trong. Hàm này xử lý cả
    2 trường hợp để không phụ thuộc phiên bản cài đặt.
    """
    if hasattr(output, "numpy"):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "last_hidden_state"):
        # fallback cuối: mean-pool qua chiều token
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Không nhận diện được kiểu output: {type(output)}")


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is None:  # double-check sau khi giành được lock
            import torch
            from transformers import AutoProcessor, AutoModel

            global _device
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[embed] Device được chọn: {_device}")
            if _device == "cuda":
                print(f"[embed] GPU: {torch.cuda.get_device_name(0)}")
                print(f"[embed] VRAM tổng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                n_threads = os.cpu_count() or 4
                torch.set_num_threads(n_threads)
                print(f"[embed] torch.set_num_threads({n_threads})")

            print(f"Đang tải model SigLIP ({SIGLIP_MODEL_NAME})... (chỉ tải 1 lần đầu)")
            _processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            _model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(_device)
            _model.eval()
    return _model, _processor


def _embed_one_image(path: str):
    """Embed 1 ảnh. Trả về vector, hoặc None nếu ảnh lỗi/không mở được."""
    import torch
    from PIL import Image

    model, processor = _load_model()
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"  Anh loi, bo qua: {path} ({e})")
        return None

    inputs = processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        features = _to_tensor(model.get_image_features(**inputs))
    return features[0].cpu().numpy().tolist()


def embed_images_batch(image_paths: list[str]):
    """
    Nhận danh sách đường dẫn ảnh, trả về (vectors, ok_paths).
    Nếu cả batch lỗi, tự động fallback sang embed từng ảnh một để cô lập
    đúng ảnh gây lỗi, không làm mất nguyên batch.
    """
    import torch
    from PIL import Image

    model, processor = _load_model()

    images = []
    valid_paths = []
    for p in image_paths:
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_paths.append(p)
        except Exception as e:
            print(f"  Anh loi, bo qua: {p} ({e})")

    if not images:
        return [], []

    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            features = _to_tensor(model.get_image_features(**inputs))
        return features.cpu().numpy().tolist(), valid_paths
    except Exception:
        import traceback
        print("  Loi khi embed ca batch, chuyen sang embed tung anh de co lap loi:")
        traceback.print_exc()
        if _device == "cuda":
            torch.cuda.empty_cache()  # dọn VRAM trước khi thử lại từng ảnh, tránh crash lặp lại
        vectors, ok_paths = [], []
        for p in valid_paths:
            vec = _embed_one_image(p)
            if vec is not None:
                vectors.append(vec)
                ok_paths.append(p)
        return vectors, ok_paths


def embed_text(text: str) -> list[float]:
    """Nhận 1 câu text (câu hỏi/mô tả), trả về 1 vector.

    SigLIP giới hạn cứng max_position_embeddings=64 token — câu hỏi dài hơn
    64 token (thường gặp với câu tiếng Việt nhiều dấu, nhiều từ) sẽ lỗi nếu
    không truncate. truncation=True + max_length=64 đảm bảo luôn cắt đúng
    giới hạn của model, không phụ thuộc độ dài câu hỏi nhập vào.
    """
    import torch

    model, processor = _load_model()
    inputs = processor(
        text=[text],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        features = _to_tensor(model.get_text_features(**inputs))
    return features[0].cpu().numpy().tolist()


def count_tokens(text: str) -> int:
    """Đếm số token thật sự (chưa pad/cắt) của 1 câu text theo tokenizer
    của SigLIP. Dùng để kiểm tra trước xem câu hỏi có vượt quá 64 token
    (giới hạn cứng của model) hay không, trước khi gọi embed_text()."""
    _load_model()  # đảm bảo _processor đã được tải
    tokens = _processor.tokenizer(text=text)
    return len(tokens["input_ids"])