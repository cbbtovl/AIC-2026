import os
import re
import json
import pickle
from rank_bm25 import BM25Okapi
from config import METADATA_JSONL, BM25_INDEX_PATH, OCR_CACHE_PATH, SEGMENT_TOPICS_PATH


def _load_ocr_cache() -> dict[str, str]:
    """Nạp OCR cache đã precompute sẵn OFFLINE (xem ocr_precompute.py) — dict
    {frame_id: ocr_text}. Trả về {} nếu chưa precompute (an toàn, không crash)."""
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


def _load_segment_summaries() -> dict[str, str]:
    """(MỚI) Nạp segment_topics.jsonl (xem segment_topics.py) — trả về dict
    {frame_id: segment_summary}. Mỗi frame thuộc 1 segment được gán CÙNG 1
    câu tóm tắt (vd "CLB FANA trao quà từ thiện tại xã Giang Ly, huyện Khánh
    Vinh, tỉnh Khánh Hòa"), giúp BM25 khớp được câu hỏi DIỄN GIẢI LẠI nội
    dung — thứ mà OCR ticker rời rạc theo từng frame ("06:30:11 | TIN CHÍNH
    | ...") không bao giờ khớp được bằng exact-match từ.

    Trả về {} nếu chưa chạy segment_topics.py (an toàn, không crash — BM25
    vẫn hoạt động y hệt bản cũ, chỉ thiếu tín hiệu segment_summary)."""
    mapping: dict[str, str] = {}
    if not os.path.exists(SEGMENT_TOPICS_PATH):
        return mapping
    with open(SEGMENT_TOPICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary = rec.get("summary", "")
            if not summary:
                continue
            for mid in rec.get("member_ids", []):
                mapping[mid] = summary
    return mapping


# Khớp mọi ký tự KHÔNG PHẢI chữ/số (giữ Unicode nên tiếng Việt có dấu vẫn
# nguyên vẹn) — dùng để dọn dấu câu trước khi tokenize.
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def _keywords_to_str(kw) -> str:
    """video_keywords có thể là list (từ metadata_all.jsonl) hoặc str (từ Chroma).
    Trước đây nhét thẳng list vào f-string -> ra "['a', 'b']" làm rác BM25 index."""
    if isinstance(kw, list):
        return " ".join(str(k) for k in kw)
    return str(kw or "")


def _tokenize(text: str) -> list[str]:
    """Tokenize dùng CHUNG cho cả lúc build corpus và lúc query.

    BUG ĐÃ SỬA: trước đây chỉ `.lower().split()` — với OBJECTS dạng
    "person, car, glasses" (phân tách bởi dấu phẩy, xem build_metadata.py),
    split theo khoảng trắng để lại token "person," "car," CÒN DÍNH DẤU PHẨY,
    không bao giờ khớp với token "person"/"car" sạch của câu truy vấn ->
    làm yếu hẳn tín hiệu BM25 từ trường vật thể quan trọng nhất. Giờ thay
    mọi ký tự không phải chữ/số bằng khoảng trắng trước khi split, để
    "person," và "person" luôn tokenize giống nhau."""
    cleaned = _NON_WORD_RE.sub(" ", text.lower())
    return [t for t in cleaned.split() if t]


class BM25Engine:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self._id_index = None  # (MỚI) cache {id: metadata}, build lười ở get_by_id()

    def build_or_load(self, force_rebuild: bool = False):
        self._id_index = None  # metadatas sắp đổi -> hủy cache id_index cũ

        if os.path.exists(BM25_INDEX_PATH) and not force_rebuild:
            print("📦 Đang tải BM25 Index từ disk...")
            with open(BM25_INDEX_PATH, "rb") as f:
                saved = pickle.load(f)
                self.bm25 = saved["bm25"]
                self.documents = saved["documents"]
                self.metadatas = saved["metadatas"]
            print(f"✅ Đã tải BM25 Index cho {len(self.metadatas)} frames.")
            return

        if not os.path.exists(METADATA_JSONL):
            print(f"❌ Chưa có {METADATA_JSONL}. Chạy `python build_metadata.py` trước.")
            return

        print("🔨 Đang khởi tạo BM25 Index mới từ metadata...")
        ocr_cache = _load_ocr_cache()
        if ocr_cache:
            print(f"📝 Đã nạp OCR cache offline ({len(ocr_cache)} frame có text) — "
                  f"đưa vào corpus BM25 để khớp chữ/tên hiển thị trên hình ngay ở Tầng 1.")
        else:
            print("ℹ️ Chưa có OCR cache offline (chạy `python ocr_precompute.py` "
                  "để bật khớp theo chữ trên hình) — build BM25 như bình thường.")

        # (MỚI) segment_summary — xem _load_segment_summaries() ở trên.
        segment_summaries = _load_segment_summaries()
        if segment_summaries:
            print(f"🧩 Đã nạp segment summary ({len(segment_summaries)} frame có tóm tắt cụm, "
                  f"xem segment_topics.py) — giúp BM25 khớp câu hỏi diễn giải lại nội dung, "
                  f"không chỉ ticker OCR rời rạc.")
        else:
            print("ℹ️ Chưa có segment_topics.jsonl (chạy `python segment_topics.py` "
                  "để bật khớp theo tóm tắt cụm) — build BM25 như bình thường.")

        corpus = []
        self.metadatas = []
        self.documents = []

        with open(METADATA_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                ocr_text = ocr_cache.get(data.get("id", ""), "")
                if ocr_text:
                    data["ocr_text_precomputed"] = ocr_text

                # (MỚI) gán segment_summary nếu frame này thuộc 1 segment đã
                # được segment_topics.py tóm tắt — lưu cả vào metadata (để
                # app.py/pipeline.py có thể hiển thị/dùng lại) lẫn vào
                # text_content (để BM25 index được câu tóm tắt).
                seg_summary = segment_summaries.get(data.get("id", ""), "")
                if seg_summary:
                    data["segment_summary"] = seg_summary

                text_content = (
                    f"{data.get('video_title', '')} "
                    f"{data.get('OBJECTS', '')} "
                    f"{_keywords_to_str(data.get('video_keywords', []))} "
                    f"{data.get('video_description', '')} "
                    f"{ocr_text} "
                    f"{seg_summary}"
                )
                tokens = _tokenize(text_content)
                corpus.append(tokens)
                self.documents.append(text_content)
                self.metadatas.append(data)

        self.bm25 = BM25Okapi(corpus)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f)
        print(f"✅ Đã lưu BM25 Index hoàn chỉnh cho {len(self.metadatas)} frames!")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            self.build_or_load()
        if not self.bm25:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                item = dict(self.metadatas[idx])
                item["bm25_score"] = float(scores[idx])
                results.append(item)
        return results

    def get_by_id(self, doc_id: str) -> dict | None:
        """(MỚI) Tra cứu 1 metadata đầy đủ theo id — dùng bởi
        segment_topics.segment_boost_candidates() để lấy full metadata
        (image_path, video_id, pts_time...) cho các frame được segment-boost,
        KHÔNG cần quét lại toàn bộ self.metadatas mỗi lần gọi.

        Cache lười theo _id_index (rebuild khi build_or_load() chạy lại, xem
        chỗ reset self._id_index = None ở đầu build_or_load())."""
        if not self.metadatas:
            self.build_or_load()
        if self._id_index is None:
            self._id_index = {m.get("id"): m for m in self.metadatas if m.get("id")}
        return self._id_index.get(doc_id)


bm25_engine = BM25Engine()

if __name__ == "__main__":
    bm25_engine.build_or_load(force_rebuild=True)