# config.py

# ----- Đường dẫn dữ liệu -----
IMAGES_ROOT_DIR   = r"D:\aic-vqa-project\data\keyframes"
MAP_KEYFRAMES_DIR = r"D:\aic-vqa-project\data\map-keyframes"
MEDIA_INFO_DIR    = r"D:\aic-vqa-project\data\media-info"
OBJECTS_DIR       = r"D:\aic-vqa-project\data\objects"
VIDEOS_ROOT_DIR   = r"D:\aic-vqa-project\data\videos"   # chứa các batch: Videos_L21_a\L21_V001.mp4, Videos_L21_b\..., v.v.

# ----- Output & Databases -----
METADATA_JSONL    = "./metadata_all.jsonl"
CHROMA_DB_DIR     = "./chroma_db"
COLLECTION_NAME   = "aic2026_keyframes"
BM25_INDEX_PATH   = "./bm25_index.pkl"

# ----- (MỚI) Qdrant — CHỈ dùng nếu bạn chuyển sang store_qdrant.py/retrieve_qdrant.py
# thay cho store.py/retrieve.py (ChromaDB). Không ảnh hưởng gì nếu vẫn dùng Chroma. -----
# QDRANT_MODE = "local": chạy Qdrant EMBEDDED ngay trong process Python (giống
#   ChromaDB PersistentClient) -- KHÔNG cần server/Docker, dữ liệu lưu thẳng vào
#   thư mục QDRANT_LOCAL_PATH trên đĩa. Dùng khi không cài/chạy được Docker.
# QDRANT_MODE = "server": kết nối tới Qdrant server đang chạy tại QDRANT_URL
#   (vd qua Docker, hoặc Qdrant Cloud) -- dùng khi cần nhiều process/máy cùng
#   truy cập chung 1 DB (embedded chỉ cho phép 1 process mở cùng lúc).
QDRANT_MODE       = "local"
QDRANT_LOCAL_PATH = "./qdrant_db"
QDRANT_URL        = "http://localhost:6333"   # server mode (Docker) HOẶC Qdrant Cloud endpoint
QDRANT_API_KEY    = ""                        # để trống nếu server local không auth; điền key nếu dùng Qdrant Cloud
QDRANT_COLLECTION = "aic2026_keyframes"

# BỎ: GRID_OUTPUT_DIR (không còn ghép lưới 2x2 — xem BUGFIX LOG trong
# pipeline.py / qwen_vqa.py để biết lý do bỏ hẳn bước ghép lưới: nó làm mất
# tương ứng 1-1 giữa ảnh mà OCR/VLM nhìn thấy và pts_time/frame_id thật sự
# được trả về, gây ra bug "OCR đọc nhầm keyframe" mà user báo cáo).
# File grid_processor.py (dùng biến này) đã bị XOÁ hẳn khỏi project — nó
# không còn được pipeline.py gọi tới nữa và sẽ crash ngay khi import vì
# GRID_OUTPUT_DIR không còn tồn tại. Nếu bạn thấy grid_processor.py xuất
# hiện lại (vd copy từ bản cũ), hãy xoá nó, đừng thêm lại biến này.

# ----- Embedding & Reranking Models -----
SIGLIP_MODEL_NAME        = "google/siglip-base-patch16-224"
FLORENCE_RERANK_MODEL    = "microsoft/Florence-2-base"

# ----- Candidate Counters (pipeline mới: SigLIP top30 -> Object Rerank top5-10 -> OCR -> Nemotron) -----
RETRIEVAL_TOP_K = 30      # Tầng 1: SigLIP + BM25 lấy top-30 (theo đúng sơ đồ pipeline mới)
# BUGFIX (dọn dẹp): RERANK_TOP_K từng được định nghĩa ở đây nhưng KHÔNG có
# nơi nào trong code thực sự đọc nó — rerank_vlm.rerank_with_florence() nhận
# top_n trực tiếp từ pipeline.py (qua FLORENCE_RERANK_POOL), không qua biến
# này. Đã xoá để tránh gây hiểu lầm "đổi giá trị này sẽ ảnh hưởng Florence
# rerank" — thực tế đổi FLORENCE_RERANK_POOL bên dưới mới có tác dụng.

# BUGFIX (quota) CŨ: trước đây FINAL_TOP_K=1 để tối thiểu hoá request OpenRouter.
# Giờ ưu tiên ĐỘ CHÍNH XÁC theo yêu cầu -> nâng lên 5 (mỗi candidate = 1 request
# VQA độc lập ở qwen_vqa.py, xem _score_candidate/predict_single_vote).
# LƯU Ý QUOTA: free-tier OpenRouter = 20 req/phút, 50 req/ngày DÙNG CHUNG mọi
# model :free. FINAL_TOP_K=5 x SELF_CONSISTENCY_VOTES=1 = 5 request VQA +
# 1 request dịch câu hỏi (Tầng 1, DÙNG CHUNG cho cả SigLIP lẫn Object Rerank,
# xem USE_QUERY_TRANSLATION bên dưới) = 6 request/câu hỏi. Nếu có credit
# OpenRouter (>= $10 -> 1000 req/ngày) có thể tăng FINAL_TOP_K lên 8-10 để bám
# sát sơ đồ "Top 5-10" và tăng thêm độ chính xác.
FINAL_TOP_K     = 5

# ----- BỔ SUNG (bị thiếu, gây ImportError ở grouping.py / store.py) -----
TEMPORAL_WINDOW_SEC = 5.0   # gộp các frame cùng video, cách nhau <= 5s thành 1 nhóm
EMBED_BATCH_SIZE    = 16    # batch size khi embed ảnh bằng SigLIP (store.py)

# ----- Grouping theo EMBEDDING (grouping.py), bổ sung cho temporal window -----
USE_EMBEDDING_GROUPING = True
GROUP_EMBEDDING_SIMILARITY_THRESHOLD = 0.90   # cosine similarity, 0..1 — càng cao càng khắt khe

# ----- Tầng 2 MỚI: Object Rerank (object_rerank.py) -----
OBJECT_RERANK_TOP_K = FINAL_TOP_K * 2   # giữ dư một chút trước khi group_temporal cắt còn FINAL_TOP_K

USE_FLORENCE_RERANK = True
FLORENCE_RERANK_POOL = FINAL_TOP_K * 2

# ----- Tầng 3 (trước đây "3.5"): OCR (ocr_utils.py) -----
# CHỈ chạy OCR trên candidate đã qua Object Rerank + Grouping (top 5-10),
# KHÔNG chạy trên toàn bộ 177k keyframe. Dùng EasyOCR vì đọc tốt tiếng Việt
# có dấu (quan trọng để bắt được caption/tên hiển thị trên video).
#
# QUAN TRỌNG: OCR giờ LUÔN chạy trên chính ảnh keyframe THẬT của candidate
# (metadata["image_path"]) — KHÔNG còn ghép lưới 2x2 trước khi OCR. Điều này
# đảm bảo text đọc được LUÔN thuộc đúng frame mà pts_time/frame_id trả về đại
# diện, sửa dứt điểm bug "OCR đọc nhầm keyframe" (trước đây do ghép ảnh n-1,
# n, n+1, n+2 vào 1 ảnh, OCR có thể đọc được chữ từ 1 trong 3 khung lân cận
# chứ không phải đúng khung n).
USE_OCR       = True
OCR_LANGS     = ["vi", "en"]   # EasyOCR: danh sách ngôn ngữ cần nhận diện
OCR_MIN_CONF  = 0.40           # bỏ qua dòng text OCR có độ tin cậy thấp hơn ngưỡng này
OCR_USE_GPU   = True           # tự động fallback CPU nếu không có CUDA (xem ocr_utils.py)

# ----- VLM Model (qua OpenRouter, FREE tier) & Self-Consistency -----
VLM_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]

SELF_CONSISTENCY_VOTES = 1

# ----- Tầng 1.5: Identity Rescan (identity_rescan.py) -----
IDENTITY_RESCAN_ENABLED = True
IDENTITY_RESCAN_MAX_FRAMES = 120
IDENTITY_RESCAN_SAMPLE_STRIDE = 1

# BUGFIX: trước đây gate hội tụ video bị hard-code "> 2 thì bỏ qua rescan"
# ngay trong identity_rescan.py, và KHÔNG log gì khi bị chặn ở bước này ->
# nhìn ngoài giống hệt "identity_rescan không tồn tại". Với câu hỏi nhắc tên
# riêng (vốn gần như KHÔNG có tín hiệu thị giác cho SigLIP/BM25 tầng 1), rất
# dễ candidate ban đầu trải rộng > 2 video -> đúng lúc rescan hữu ích nhất
# thì lại bị tắt. Nới ngưỡng lên 6 (đưa ra config để chỉnh không cần sửa
# code) — quét OCR sâu trên tối đa 6 video thay vì chỉ 2, vẫn có giới hạn để
# không quét vô tội vạ hàng trăm video khi retrieval tầng 1 sai be bét.
IDENTITY_RESCAN_MAX_VIDEOS = 6

# ----- OCR CACHE OFFLINE (ocr_precompute.py) — MỚI -----
# Giải quyết vấn đề "con gà quả trứng" của identity_rescan/open_text_qa: rescan
# OCR trực tiếp (live) CẦN candidate đã hội tụ về ít video trước, nhưng candidate
# chỉ hội tụ đúng khi Tầng 1 (SigLIP/BM25 — không có tín hiệu cho tên riêng/địa
# danh) đã may mắn đúng. Chạy OCR 1 LẦN OFFLINE cho (phần lớn) database, lưu vào
# OCR_CACHE_PATH, rồi (a) đưa vào corpus BM25 để Tầng 1 khớp được luôn "Hồng
# Nhung"/"Khánh Hòa" mà KHÔNG cần rescan, và (b) rescan (nếu vẫn cần) đọc từ
# cache gần như miễn phí thay vì luôn gọi EasyOCR sống. Chi phí: 100% LOCAL
# (EasyOCR), KHÔNG tốn 1 request OpenRouter nào.
OCR_CACHE_PATH = "./ocr_cache.jsonl"

# CPU không có GPU -> OCR toàn bộ ~177k ảnh rất tốn thời gian. Các keyframe liền kề
# cùng video thường trông rất giống nhau (banner/caption tồn tại xuyên suốt vài
# giây) -> chỉ cần OCR 1/STRIDE frame/video vẫn bắt được hầu hết text, giảm tải
# đúng theo tỉ lệ này mà không mất nhiều recall. Có thể chạy lại với stride nhỏ
# hơn sau (ocr_precompute.py tự resume, không OCR lại phần đã có).
OCR_PRECOMPUTE_STRIDE = 3

# Số process song song khi precompute OCR cache — mỗi process chỉ load EasyOCR
# reader MỘT LẦN (không phải mỗi ảnh), tận dụng hết core CPU. Chỉnh theo số core
# máy bạn.
OCR_PRECOMPUTE_WORKERS = 4

# ----- Tách "mô tả cảnh" khỏi "câu hỏi cụ thể" (qwen_vqa.split_scene_and_question) -----
USE_ANSWER_TARGET_SPLIT = False

# ----- Chỉ chạy OCR khi câu hỏi CÓ dấu hiệu cần đọc chữ trong ảnh -----
OCR_TRIGGER_KEYWORDS = [
    "chữ", "chu", "biển", "bien", "tên", "ten", "viết", "viet",
    "hiển thị", "hien thi", "phụ đề", "phu de", "logo", "nhãn", "nhan",
    "bảng", "bang", "số", "so", "caption", "text", "dòng chữ", "dong chu",
    "tiêu đề", "tieu de", "ghi", "in trên", "in tren",
]

# ----- BỔ SUNG/THAY THẾ HyDE: DỊCH CÂU HỎI SANG TIẾNG ANH 1 LẦN, DÙNG CHUNG -----
# BUG GỐC (đã được user chẩn đoán đúng — "SigLIP retrieve bị sai"): SigLIP
# ("google/siglip-base-patch16-224") được train hầu như HOÀN TOÀN trên cặp
# ảnh-text TIẾNG ANH, KHÔNG đa ngôn ngữ. Trước đây khi USE_HYDE=False (mặc
# định), pipeline nhúng THẲNG câu hỏi TIẾNG VIỆT vào embed_text() cho SigLIP
# -> model gần như "mù" với câu hỏi (không hiểu ngôn ngữ đầu vào), khiến
# vector search trả về candidate gần như ngẫu nhiên -> mọi tầng phía sau
# (Object Rerank, Grouping, OCR, VLM) đều làm việc trên 1 pool candidate sai
# ngay từ đầu. Đây chính là NGUYÊN NHÂN GỐC, không phải lỗi logic retrieval.
#
# GIẢI PHÁP: bỏ hẳn generate_hyde() (mô tả dài dòng, tốn max_tokens=400) và
# generate_query_keywords() gọi RIÊNG trong object_rerank.py (tốn thêm 1
# request/câu hỏi). Thay bằng translate_query_en() (qwen_vqa.py) — DỊCH NGẮN
# GỌN (max_tokens=80) câu hỏi sang tiếng Anh MỘT LẦN DUY NHẤT ở Tầng 1, rồi
# DÙNG CHUNG bản dịch này cho:
#   (a) SigLIP vector search (sửa đúng bug ngôn ngữ ở trên)
#   (b) BM25 lượt 2 (khớp field OBJECTS vốn luôn là tên class tiếng Anh)
#   (c) Object Rerank (Tầng 2) — tokenize cục bộ (không gọi LLM lại)
# => TỔNG CHI PHÍ QUOTA KHÔNG ĐỔI (vẫn đúng 1 request dịch/câu hỏi như bản cũ
# object_rerank luôn tốn), nhưng SigLIP giờ nhận đúng ngôn ngữ nó hiểu, và
# không còn gọi LLM 2 lần lãng phí cho cùng 1 việc "hiểu câu hỏi bằng tiếng
# Anh". Lỗi/hết quota -> tự fallback về câu hỏi gốc (KHÔNG crash), chỉ mất đi
# lợi ích dịch, y hệt hành vi HyDE cũ.
USE_QUERY_TRANSLATION = True

OBJECT_SCORE_THRESHOLD = 0.20

# ----- NMS cho đếm số lượng vật thể (build_metadata.py) -----
OBJECT_NMS_IOU_THRESHOLD = 0.5

# ----- Trọng số fusion Tầng 1 (SigLIP vector vs BM25 keyword) -----
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT   = 0.4

# ----- Rate limit chủ động (tránh 429, xem _throttle() trong qwen_vqa.py) -----
OPENROUTER_FREE_RPM_LIMIT = 20

# Alias tương thích với check_embed_health.py / extract_frames.py
VIDEOS_DIR = VIDEOS_ROOT_DIR
NORMALIZE_EMBEDDINGS = False   # embed.py hiện không chuẩn hoá vector, giữ khớp với DB đã ingest

# ----- Segment Topic Classification (segment_topics.py) — MỚI -----
# Phân đoạn video theo chủ đề dựa trên độ tương đồng OCR giữa các frame liên
# tiếp, dùng để boost retrieval Tầng 1 theo CỤM thay vì từng frame rời rạc
# (xem segment_topics.segment_boost_candidates(), gọi từ pipeline.py).
SEGMENT_TOPICS_PATH = "./segment_topics.jsonl"

# Jaccard tối thiểu (trên tập token OCR) giữa 2 frame liên tiếp để coi là
# CÙNG 1 segment. Thấp hơn -> gộp dễ hơn (segment dài hơn, ít gọi LLM hơn,
# nhưng dễ gộp nhầm 2 cảnh khác nhau). Cao hơn -> tách segment nhạy hơn.
SEGMENT_SIMILARITY_THRESHOLD = 0.35

# Khoảng cách thời gian tối đa (giây) giữa 2 frame CÓ OCR để vẫn coi là còn
# thuộc cùng 1 segment (dù ở giữa có frame không chữ bị bỏ qua). Cao hơn ->
# chịu được khoảng trống dài hơn (hợp cho video sự kiện, banner xuất hiện
# rời rạc); thấp hơn -> tách segment nhạy hơn (hợp cho ticker bản tin đổi
# liên tục).
SEGMENT_MAX_GAP_SEC = 6.0

# ----- Segment Semantic Search (segment_embed.py) — MỚI, dùng BGE-M3 -----
# Embed 'summary' của mỗi segment bằng BGE-M3 (dense, đa ngôn ngữ, tốt tiếng
# Việt) để retrieval theo NGỮ NGHĨA — bắt được câu hỏi diễn giải lại nội
# dung mà không trùng từ khoá với anchor (org/province) đã trích riêng lẻ.
# Chạy 100% LOCAL, không tốn quota OpenRouter. Cần: pip install sentence-transformers
BGE_MODEL_NAME = "BAAI/bge-m3"
SEGMENT_EMBEDDINGS_PATH = "./segment_embeddings.pkl"

# Cosine tối thiểu để coi 1 segment là "khớp ngữ nghĩa" với câu hỏi. Thấp
# hơn -> bắt được nhiều paraphrase hơn nhưng dễ nhiễu; cao hơn -> chỉ boost
# khi thực sự chắc chắn.
SEGMENT_SIMILARITY_MIN_SCORE = 0.55

# Số segment tối đa lấy theo similarity mỗi câu hỏi (giữ nhỏ để không kéo
# quá nhiều frame không liên quan vào candidate pool).
SEGMENT_SIMILARITY_TOP_K = 5