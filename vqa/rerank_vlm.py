# rerank_vlm.py
import os
import math
import threading
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    PretrainedConfig,
    RobertaTokenizer,
    RobertaTokenizerFast,
)
from config import FLORENCE_RERANK_MODEL

# --- 1. PATCH CẤP CLASS CHO TRANSFORMERS MỚI ---

if not hasattr(PretrainedConfig, "forced_bos_token_id"):
    setattr(PretrainedConfig, "forced_bos_token_id", None)


def _get_add_spec_tokens(self):
    if hasattr(self, "_additional_special_tokens"):
        return self._additional_special_tokens
    return self.special_tokens_map.get("additional_special_tokens", [])


def _set_add_spec_tokens(self, val):
    self._additional_special_tokens = val


for tok_cls in [RobertaTokenizer, RobertaTokenizerFast]:
    if not hasattr(tok_cls, "additional_special_tokens"):
        setattr(tok_cls, "additional_special_tokens", property(_get_add_spec_tokens, _set_add_spec_tokens))

# --- 2. QUẢN LÝ MÔ HÌNH FLORENCE-2 ---

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_load_lock = threading.Lock()  # chống nhiều luồng cùng tải model 1 lúc (Streamlit rerun)


def _load_florence():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is None:  # double-check sau khi giành được lock
            print(f"⌛ Đang tải Florence-2 Reranker ({FLORENCE_RERANK_MODEL})...")
            _processor = AutoProcessor.from_pretrained(FLORENCE_RERANK_MODEL, trust_remote_code=True)
            _model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_RERANK_MODEL,
                trust_remote_code=True,
                attn_implementation="eager",
                torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            ).to(_device).eval()
    return _model, _processor


def _score_one(model, processor, image_path: str, prompt: str) -> tuple[float, str]:
    """Trả về (score liên tục, generated_text).

    Trước đây: score = 1.0 nếu có "yes" trong text, ngược lại 0.1 (NHỊ PHÂN).
    -> nhiều candidate cùng "yes" bị HÒA ĐIỂM TUYỆT ĐỐI, sort không phân biệt
    được ai match tốt hơn -> rerank gần như ngẫu nhiên trong nhóm "yes".

    Giờ: dùng beam search (num_beams=3) + sequences_scores (log-prob trung
    bình mỗi token) làm độ tự tin liên tục, cộng vào điểm nền yes/no ->
    "yes" tự tin cao luôn xếp trên "yes" tự tin thấp.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(_device)
    if _device == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=5,
            num_beams=3,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].lower()

    try:
        # sequences_scores: log-likelihood trung bình/token của chuỗi được beam-search chọn, thường <= 0.
        confidence = math.exp(float(outputs.sequences_scores[0].item()))  # ~ (0, 1]
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.5  # fallback trung tính nếu bản transformers không trả sequences_scores

    is_yes = "yes" in generated_text
    score = (1.0 + confidence) if is_yes else (confidence * 0.5)
    return score, generated_text


def rerank_with_florence(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Tầng 2: Rerank các ứng viên từ Tầng 1 bằng Florence-2.
    Nếu gặp lỗi tải mô hình, tự động trả về top_n ứng viên ban đầu để không đứt gãy pipeline.
    """
    try:
        model, processor = _load_florence()
    except Exception as e:
        print(f"⚠️ Bỏ qua bước Florence-2 Rerank do lỗi tải mô hình: {e}")
        return candidates[:top_n]

    scored_candidates = []
    prompt = f"<VQA> Does this image visually depict: {query}? Answer YES or NO."

    for item in candidates:
        image_path = item.get("image_path")
        if not image_path or not os.path.exists(image_path):
            item["florence_score"] = 0.0
            item["florence_ans"] = ""
            scored_candidates.append(item)
            continue

        try:
            score, generated_text = _score_one(model, processor, image_path, prompt)
            item["florence_score"] = round(score, 4)
            item["florence_ans"] = generated_text
        except Exception:
            item["florence_score"] = 0.0
            item["florence_ans"] = ""
        scored_candidates.append(item)

    scored_candidates.sort(key=lambda x: x.get("florence_score", 0.0), reverse=True)
    return scored_candidates[:top_n]