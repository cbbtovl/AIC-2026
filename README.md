🚀 AI Multimodal Video Retrieval System (AIC)
Hệ thống Truy xuất Video & Keyframe Đa phương thức thông minh, phục vụ các bài toán tìm kiếm khung hình, hỏi đáp thị giác và truy vết chuỗi sự kiện theo trục thời gian trong các kỳ thi AI Challenge (AIC).

📂 1. Cấu trúc Thư mục Dự án
multimodal-retrieval/
│
├── 🚀 FILE CHẠY CHÍNH & CỐT LÕI (Core)
│   ├── server.py                # FastAPI Backend Server chính (Chạy: python server.py)
│   ├── index.html               # Giao diện Web HTML chính
│   ├── style.css                # File định dạng giao diện Web (CSS)
│   └── config.py                # Quản lý cài đặt, đường dẫn và Model
│
├── 🧠 DỊCH VỤ AI CHUYÊN BIỆT (services/)
│   ├── semantic_service.py      # Trích xuất Vector ngữ nghĩa bằng SigLIP
│   ├── ocr_service.py           # Trích xuất OCR văn bản bằng Florence-2 Large & BGE-M3
│   ├── asr_service.py           # Trích xuất giọng nói bằng Whisper Large V3 & BGE-M3
│   ├── vqa_service.py           # Hỏi đáp ảnh bằng Multimodal RAG (SigLIP + Florence-2 + BM25)
│   └── trake_service.py         # Tìm kiếm chuỗi sự kiện theo thời gian bằng OpenCLIP ViT-B/32
│
├── 🛠️ CÔNG CỤ DÒNG LỆNH & BẢO TRÌ (scripts/)
│   ├── dataset_manifest.py      # Quét và tạo file thống kê manifest toàn bộ dataset
│   ├── kis_healthcheck.py       # Kiểm tra sức khỏe Qdrant và độ dài vector của các model
│   ├── kis_evaluation.py        # Chấm điểm độ chính xác Recall@k (R@1, 5, 20, 50, 100)
│   └── make_embeddings.py       # Gom và tạo file JSON duy nhất cho từng tính năng
│
├── 🧪 BỘ KIỂM THỬ HỆ THỐNG (tests/)
│   ├── test_database.py         # Kiểm tra kết nối và vector search của Qdrant (SigLIP, OpenCLIP, BGE-M3)
│   ├── test_services.py         # Kiểm tra trích xuất vector và tính năng các model chuyên biệt
│   └── test_vqa.py              # Kiểm tra luồng hỏi đáp VQA
│
├── 📦 DỮ LIỆU & CẤU HÌNH (Data & Storage)
│   ├── mapkeyframe/             # Thư mục chứa file CSV mapkeyframe từ BTC (pts_time, n, frame_idx)
│   ├── uploads/                 # Thư mục lưu ảnh keyframes và video gốc
│   └── *.json                   # 4 file JSON gộp cuối cùng (semantic, ocr, asr, trake) để đẩy lên Qdrant

---

## ⚡ 2. Hướng dẫn Cài đặt & Khởi động Nhanh

### Bước 1: Kích hoạt môi trường ảo (Virtualenv)
2. Hướng dẫn Cài đặt & Khởi động Qdrant Local (Không kèm Docker)
Bước 1: Cài đặt Qdrant độc lập (Standalone Qdrant Binary trên Windows/Linux)
Tải bản release mới nhất của Qdrant từ trang GitHub chính thức: Qdrant GitHub Releases.

Giải nén file tải về vào một thư mục trên máy (ví dụ: C:\qdrant hoặc /opt/qdrant).

Mở Terminal / PowerShell tại thư mục đó và khởi chạy trực tiếp file thực thi:
``` *.\qdrant.exe*
Sau đó Qdrant sẽ chạy ngầm tại cổng mặc định: http://localhost:6333.
```

Bước 2: Kích hoạt môi trường ảo và cài đặt thư viện Python
```python -m venv venv
```.\venv\Scripts\Activate.ps1

```pip install torch torchvision transformers sentence-transformers qdrant-client fastapi uvicorn pillow open-clip-torch openai-whisper rank_bm25
```

### Bước 3: Khởi chạy Ứng dụng Backend Server
Chạy ```python server.py
Mở trình duyệt và truy cập file index.html hoặc chạy thông qua server để trải nghiệm giao diện.
---

3. Các Tính năng Chính trên Giao diện Web
Giao diện được thiết kế tối ưu bằng HTML & CSS tùy chỉnh, chia bố cục trực quan với các tab chức năng chuyên biệt:

🔍 Tab 1 - Semantic Search:

Nhập từ khóa truy vấn ngữ nghĩa (hỗ trợ mô hình SigLIP).

Hệ thống trả về danh sách khung hình kèm độ tương đồng.

🔤 Tab 2 - OCR Search:

Hỗ trợ nhập văn bản text hoặc kéo thả ảnh (Image-to-Image) để tìm kiếm các khung hình có nội dung tương đồng văn bản/hình ảnh được nhận diện bởi Florence-2 Large và embedding BGE-M3.

🎙️ Tab 3 - ASR Search:

Nhập từ khóa lời thoại nhân vật được trích xuất tự động qua Whisper Large V3 và mã hóa vector bằng BGE-M3.

🤖 Tab 4 - VQA (Hỏi đáp thị giác):

Nhập câu hỏi về hình ảnh. Hệ thống sử dụng Multimodal RAG kết hợp SigLIP, Florence-2 Large nhận diện đặc trưng (biển báo, chữ số) và thuật toán BM25 Reranking để đưa ra câu trả lời chính xác dựa trên chủ đề dataset đã được gán nhãn sẵn.

⏱️ Tab 5 - TraKE (Truy vết chuỗi sự kiện):

Nhập các mốc sự kiện liên tiếp theo thời gian (E1, E2, E3, E4) sử dụng mô hình OpenCLIP ViT-B/32 truy xuất các khoảnh khắc tăng dần trong cùng một video.

🎥 Keyframe Inspector (Bên phải):

Xem chi tiết thông số Frame ID, thời gian pts_time, đường dẫn ảnh và Trình phát video HTML5 tự động tua đến đúng giây khi click vào khung hình.
---

## 🛠️ 4. Hướng dẫn Chạy các Công cụ & Kiểm thử

### Kiểm tra sức khỏe dữ liệu:
```Kiểm tra sức khỏe độ dài vector các model:
```python scripts/kis_healthcheck.py

### Chạy các bài test:
```powershell
python tests/test_database.py      # Kiểm tra kết nối Qdrant và collection (semantic, ocr, asr, trake)
python tests/test_services.py      # Kiểm tra trích xuất đặc trưng từ các model chuyên biệt
python tests/test_vqa.py           # Kiểm tra luồng RAG và Reranking BM25
```

---

## 💡 Workflow Hệ thống (Bảng Quy trình)

DATAFLOW HỆ THỐNG:

[ 1. DỮ LIỆU ĐẦU VÀO (DATA SOURCE) ]
   └── Đọc file CSV mapkeyframe do Ban Tổ Chức cung cấp (lấy các trường: pts_time, n, frame_idx)
   └── file dataset webp: https://www.kaggle.com/datasets/thinhunguyn/aic2026
   └── file video : https://drive.google.com/file/d/171nciyprX-2V00WSKhi85m2GRKM-XY7c/view?fbclid=IwY2xjawTyG3lwZG9mA2V4dG4DYWVtAjExAHNydGMGYXBwX2lkATAAAR6XOmpm2Nj00QO0TfTHJzYVTxwskyqvQ4P3_caMy-pQOiGFMwWWVXsPTI3v3w_aem_w7uFdLDiaP07jxxyUoDFww
[ 2. TRÍCH XUẤT ĐẶC TRƯNG & EMBEDDING (SERVICES) ]
   ├── Semantic Service: Dùng mô hình SigLIP để trích xuất ngữ nghĩa hình ảnh.
   ├── OCR Service: Dùng Florence-2 Large trích xuất văn bản (txt) -> Chuyển thành vector bằng BGE-M3.
   ├── ASR Service: Dùng Whisper Large V3 trích xuất audio thành các đoạn lời thoại (ngắt quãng tối đa 1 giây, gán asr_start/asr_end) -> Chuyển thành vector bằng BGE-M3.
   └── TraKE Service: Dùng OpenCLIP ViT-B/32 để trích xuất chuỗi khoảnh khắc sự kiện theo thời gian.

[ 3. GỘP NHÓM FILE JSON (AGGREGATION) ]
   └── Gom toàn bộ các file embedding lẻ thành ĐÚNG 4 FILE JSON DUY NHẤT:
       ├── semantic.json
       ├── ocr.json
       ├── asr.json
       └── trake.json

[ 4. LƯU TRỮ DỮ LIỆU (QDRANT LOCAL DATABASE) ]
   └── Đẩy 4 file JSON lên Qdrant (không cần Docker) vào các collection riêng biệt:
       ├── Collection "semantic" (Vector SigLIP)
       ├── Collection "ocr" (Vector Florence-2 + BGE-M3)
       ├── Collection "asr" (Vector Whisper + BGE-M3)
       ├── Collection "trake" (Vector OpenCLIP ViT-B/32)
       └── (VQA Service sử dụng kết hợp 2 collection semantic & ocr kèm Reranking BM25)

[ 5. TRUY VẤN & GIAO DIỆN (FRONTEND & BACKEND) ]
   ├── Người dùng thao tác trên giao diện Web tùy chỉnh (HTML & CSS).
   └── Gửi request tới FastAPI Server (server.py) để query trực tiếp từ Qdrant Local và trả kết quả realtime về giao diện.