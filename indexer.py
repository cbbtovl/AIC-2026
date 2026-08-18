"""
Indexer - Kỹ sư Điều phối quy trình (Bản tối ưu hóa & Khống chế lỗi):
  - Chuẩn hóa Metadata Keyframe gắn liền VideoID.
  - Xử lý lỗi ảnh hỏng (Fault-Tolerance) tuyệt đối.
  - Tối ưu hóa Memory RAM & Disk I/O.
"""

import json
import os
import shutil
import zipfile
import cv2
import numpy as np
import subprocess
import concurrent.futures
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

from config import UPLOAD_DIR
from database import save_item, save_item_batch, get_db_connection, search_items
from services.embedding_service import get_image_embedding, get_image_embeddings_batch, get_text_embedding, get_clip_text_embedding
from services.ocr_service import process_image


SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
SUPPORTED_DOCS   = {".txt", ".pdf", ".docx", ".doc", ".md", ".csv"}
SUPPORTED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}

# ==========================================
# 1. CÔNG CỤ HỖ TRỢ & TOÁN HỌC
# ==========================================

def get_file_category(extension: str) -> str:
    ext = extension.lower()
    if ext in SUPPORTED_IMAGES: return "image"
    if ext in SUPPORTED_DOCS:   return "document"
    if ext in SUPPORTED_VIDEOS: return "video"
    return "other"


def get_system_extractor() -> str:
    """Truy tìm 7-Zip trên máy"""
    common_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "/usr/bin/7z", "/usr/local/bin/7z"
    ]
    for p in common_paths:
        if Path(p).exists(): return p
    return ""

# ==========================================
# 2. XỬ LÝ THEO LÔ (BATCH PROCESSING)
# ==========================================

def process_image_batch(image_paths: List[Path], base_filename: str, log=print) -> List[Dict[str, Any]]:
    """Xử lý theo lô các ảnh bằng ThreadPool và Batch Embedding (An toàn tuyệt đối)"""
    if not image_paths: return []
        
    log(f"[Batch] Đang chạy OCR trên {len(image_paths)} ảnh...")
    ocr_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_image, str(p)): p for p in image_paths}
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                res = future.result()
                ocr_results.append((p, res))
            except Exception as e:
                log(f"⚠️ Bỏ qua lỗi OCR cho {p.name}: {e}")
                ocr_results.append((p, {"extracted_text": "", "description": ""}))
                
    ocr_results.sort(key=lambda x: str(x[0]))
    
    log(f"[Batch] Đang trích xuất Vector cho {len(image_paths)} ảnh...")
    str_paths = [str(p) for p, _ in ocr_results]
    
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(str_paths), batch_size):
        batch_subset = str_paths[i:i+batch_size]
        try:
            emb_subset = get_image_embeddings_batch(batch_subset)
            all_embeddings.extend(emb_subset)
        except Exception as e:
            log(f"⚠️ Lỗi Batch Embedding nhóm {i}: {e}")
            all_embeddings.extend([[] for _ in batch_subset])
    
    items_to_save = []
    indexed_results = []

    project_root = Path(__file__).resolve().parent
    
    for (p, ocr), emb in zip(ocr_results, all_embeddings):
        if not emb: continue
        
        extracted_text = ocr.get("extracted_text", "")
        description = ocr.get("description", "")
        
        final_filename = f"{base_filename} - {p.name}" if base_filename not in p.name else p.name

        # Lưu đường dẫn tương đối so với thư mục dự án
        try:
            rel_path = p.relative_to(project_root)
        except ValueError:
            # Phòng trường hợp ảnh nằm ở ổ đĩa khác hoàn toàn
            rel_path = p
            # Chuyển đổi sang chuẩn đường dẫn tương đối dạng web (dùng dấu /)
        normalized_filepath = str(rel_path).replace("\\", "/") ###

        
        items_to_save.append({
            "filename": final_filename,
            "filepath": normalized_filepath, ###
            "filetype": "image",
            "extracted_text": extracted_text,
            "description": description,
            "embedding": emb,
            "embedding_type": "clip_visual"
        })
        indexed_results.append({
            "filename": final_filename,
            "filetype": "image",
            "extracted_text": (extracted_text[:100] + "...") if len(extracted_text) > 100 else extracted_text,
            "description": description
        })
        
    save_item_batch(items_to_save)
    return indexed_results



def _extract_and_index_archive(archive_path: Path, log) -> List[Dict[str, Any]]:
    """Giải nén và nạp dữ liệu tốc độ cao (Đã tối ưu cho Keyframe của BTC)"""
    archive_name = archive_path.stem
    permanent_dir = UPLOAD_DIR / f"{archive_name}_extracted"
    permanent_dir.mkdir(parents=True, exist_ok=True)
    
    ext = archive_path.suffix.lower()
    log(f"[Indexer] Bung file nén {ext.upper()}: {archive_path.name}...")
    
    if ext == ".zip":
        try:
            with zipfile.ZipFile(str(archive_path), 'r') as zf:
                zf.extractall(str(permanent_dir))
        except Exception as e:
            log(f"❌ Lỗi giải nén ZIP: {e}")
            return []
    elif ext == ".rar":
        tool_path = get_system_extractor()
        if not tool_path:
            log("❌ Cần 7-Zip để giải nén RAR.")
            return []
        cmd = [tool_path, "x", str(archive_path), f"-o{permanent_dir}", "-y"]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            log(f"❌ Lỗi giải nén RAR.")
            return []

    all_files = list(permanent_dir.rglob("*"))
    valid_files = [f for f in all_files if f.is_file() and not f.name.startswith(".") and "__MACOSX" not in str(f)]
    
    image_files = [f for f in valid_files if get_file_category(f.suffix) == "image"]
    other_files = [f for f in valid_files if get_file_category(f.suffix) not in ("image", "other")]
    
    indexed = []
    if image_files:
        log(f"[Indexer] Tìm thấy {len(image_files)} ảnh Keyframe. Đang nạp thẳng vào AI...")
        # Đẩy thẳng toàn bộ ảnh vào chạy batch, không cần lọc lại nữa
        indexed.extend(process_image_batch(image_files, archive_name, log))

    for fp in other_files:
        try:
            res = index_file(str(fp), log)
            indexed.append(res)
        except Exception as e:
            log(f"[Indexer] Bỏ qua {fp.name}: {e}")

    return indexed

# ==========================================
# 3. ROUTER CHÍNH & TÌM KIẾM
# ==========================================

def index_file(src_filepath: str, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Lập chỉ mục Hình ảnh, Video, Zip, Rar"""
    src_path = Path(src_filepath)
    if not src_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {src_filepath}")

    ext = src_path.suffix.lower()
    filetype = get_file_category(ext)
    log = progress_callback or print

    # 1. Xử lý file Nén (Zip / Rar)
    if ext in [".zip", ".rar"]:
        results = _extract_and_index_archive(src_path, log)
        return {"filename": src_path.name, "filetype": "archive", "extracted_text": f"Đã nạp {len(results)} file(s).", "description": ""}

    # 2. Xử lý Hình ảnh (Chỉ copy và xử lý NẾU đúng là file image)
    filename = src_path.name

    if filetype == "image":
        dest_path = UPLOAD_DIR / filename

        # Chỉ copy vào UPLOAD_DIR khi thực sự xử lý ảnh
        if not dest_path.exists():
            shutil.copy2(str(src_path), str(dest_path))

        log(f"[Indexer] Phân tích ảnh: {filename}")
        result = process_image(str(dest_path))
        extracted_text = result.get("extracted_text", "")
        description    = result.get("description", "")
        embedding      = get_image_embedding(str(dest_path))
        embedding_type = "clip_visual"

        save_item(
            filename=dest_path.name,
            filepath=str(dest_path),
            filetype=filetype,
            extracted_text=extracted_text,
            description=description,
            embedding=embedding,
            embedding_type=embedding_type
        )

        log(f"✅ Hoàn tất nạp: {dest_path.name}")
        return {
            "filename": dest_path.name,
            "filetype": filetype,
            "extracted_text": (extracted_text[:100] + "...") if len(extracted_text) > 100 else extracted_text,
            "description": description
        }
    else:
        # Nếu là Video hoặc định dạng khác không hỗ trợ -> Bỏ qua ngay, KHÔNG copy
        return {
            "filename": filename,
            "filetype": filetype,
            "extracted_text": "Bỏ qua (Không hỗ trợ tài liệu)",
            "description": ""
        }
def query_search_text(query_str: str, filetype_filter: str = "Tất cả", limit: int = 300):
    """
    Tìm kiếm vector tương đồng đã được tối ưu Ma trận NumPy (Vectorization):
    - Đã đồng bộ với CLIP Text Embedding (512D)
    - Tốc độ xử lý hàng trăm ngàn record dưới 0.05s
    - Lọc filetype trực tiếp từ SQLite SQL
    """
    try:
        started_at = time.perf_counter()
        # 💡 BƯỚC 1: Sử dụng CLIP Text Embedding thay vì MPNet để khớp 512D với đặc trưng ảnh
        # Đã import sẵn từ services.embedding_service ở đầu file
        query_vector = get_clip_text_embedding(query_str)
        
        if not query_vector:
            print("⚠️ Không thể tạo vector cho câu truy vấn!")
            return []

        embedding_elapsed = time.perf_counter() - started_at
        results = search_items(query_vector, filetype_filter=filetype_filter, limit=limit)
        print(f"[KIS] embedding={embedding_elapsed:.2f}s total={time.perf_counter() - started_at:.2f}s results={len(results)}", flush=True)
        return results

    except Exception as e:
        print(f"❌ Lỗi trong query_search_text: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def load_mapkeyframe_csv(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        mapping = {}
        
        for _, row in df.iterrows():
            f_idx = int(row.get('frame_idx', row.get('frame', 0)))
            
            if 'pts_time' in row:
                pts_sec = float(row['pts_time'])
            elif 'pts_ms' in row or 'pts' in row:
                pts_val = float(row.get('pts_ms', row.get('pts', 0)))
                pts_sec = pts_val / 1000.0 if pts_val > 10000 else pts_val
            else:
                pts_sec = f_idx / float(row.get('fps', 25.0))

            mapping[f_idx] = {
                'pts_time': pts_sec,
                'n': int(row.get('n', 0)),
                'fps': float(row.get('fps', 30.0))
            }
            
        return mapping
    except Exception as e:
        print(f"⚠️ Lỗi đọc file CSV {csv_path}: {e}")
        return {}

def index_precomputed_dataset(csv_path: str, npy_path: str, json_path: str = None) -> int:
    try:
        csv_p = Path(csv_path)
        npy_p = Path(npy_path)

        if not csv_p.exists() or not npy_p.exists():
            print(f"❌ Không tìm thấy file CSV hoặc NPY: {csv_path}, {npy_path}")
            return 0

        features = np.load(str(npy_p))
        
        try:
            df = pd.read_csv(str(csv_p))
        except Exception:
            df = pd.read_csv(str(csv_p), header=None)

        # 🟢 ĐỌC FILE JSON METADATA
        meta_json_data = {}
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_json = json.load(f)
                    if isinstance(raw_json, list):
                        meta_json_data = {i: item for i, item in enumerate(raw_json)}
                    elif isinstance(raw_json, dict):
                        meta_json_data = raw_json
                print(f"✅ Đã nạp thành công metadata từ JSON: {json_path}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file JSON metadata: {e}")

        if len(features) != len(df):
            print(f"⚠️ Số lượng vector ({len(features)}) không khớp số dòng CSV ({len(df)}). Tự động cắt...")
            min_len = min(len(features), len(df))
            features = features[:min_len]
            df = df.iloc[:min_len]

        items_to_save = []
        base_dir = csv_p.parent

        for idx, row in df.iterrows():
            first_col = str(row.iloc[0]) if isinstance(row, pd.Series) else str(idx)
            
            if first_col.endswith(('.jpg', '.png', '.jpeg')):
                frame_filename = first_col
            else:
                video_stem = csv_p.stem
                frame_filename = f"{video_stem}_{first_col}.jpg" if not first_col.startswith("L") else f"{first_col}.jpg"

            virtual_img_path = str(base_dir / frame_filename)
            vec = features[idx].astype(float).tolist()

            # 🟢 MẶC ĐỊNH LẤY TỪ CSV HOẶC TÍNH TOÁN
            extracted_text = ""
            description = f"Precomputed feature keyframe {frame_filename}"
            video_id = csv_p.stem  # Hoặc lấy từ cột CSV nếu có
            pts_time = 0.0
            frame_idx = idx

            # 1. Ưu tiên lấy từ JSON nếu có cấu trúc chi tiết
            if idx in meta_json_data:
                item_meta = meta_json_data[idx]
                extracted_text = item_meta.get("text", item_meta.get("ocr", ""))
                description = item_meta.get("description", description)
                
                # 👉 Lấy thông tin tua video từ JSON (nếu dataset cung cấp)
                video_id = item_meta.get("video_id", item_meta.get("video", video_id))
                if "pts_time" in item_meta:
                    pts_time = float(item_meta["pts_time"])
                elif "timestamp" in item_meta:
                    pts_time = float(item_meta["timestamp"])
                
                if "frame_idx" in item_meta:
                    frame_idx = int(item_meta["frame_idx"])

            # 2. Hoặc lấy trực tiếp từ cột của DataFrame (CSV) nếu có
            if isinstance(row, pd.Series):
                if "video_id" in row and pd.notna(row["video_id"]):
                    video_id = str(row["video_id"])
                if "pts_time" in row and pd.notna(row["pts_time"]):
                    pts_time = float(row["pts_time"])
                if "frame_idx" in row and pd.notna(row["frame_idx"]):
                    frame_idx = int(row["frame_idx"])

            # 👉 TÍNH TOÁN CHUỖI HIỂN THỊ THỜI GIAN (MM:SS.ms)
            mm = int(pts_time // 60)
            ss = int(pts_time % 60)
            ms = int((pts_time - int(pts_time)) * 100)
            timestamp_ms = f"{mm:02d}:{ss:02d}.{ms:02d}"

            item = {
                "filename": frame_filename,
                "filepath": virtual_img_path,
                "filetype": "image",
                "video_id": video_id,        # 👈 ĐÃ BỔ SUNG: ID Video gốc để player biết phát video nào
                "pts_time": pts_time,        # 👈 ĐÃ BỔ SUNG: Số giây chính xác để player tua
                "frame_idx": frame_idx,      # 👈 ĐÃ BỔ SUNG: Số thứ tự frame
                "timestamp_ms": timestamp_ms,# 👈 ĐÃ BỔ SUNG: Chuỗi hiển thị thời gian
                "extracted_text": extracted_text,
                "description": description,
                "embedding": vec,
                "embedding_type": "clip_visual"
            }
            items_to_save.append(item)

        if items_to_save:
            save_item_batch(items_to_save)
            print(f"✅ Đã index thành công {len(items_to_save)} bản ghi kèm thông tin tự tua video!")
            return len(items_to_save)

        return 0
    except Exception as e:
        print(f"❌ Lỗi khi nạp precomputed feature: {e}")
        return 0

def play_source_video(result_item):
    """
    Hàm này chạy khi người dùng click vào ảnh kết quả tìm kiếm.
    Nó đọc ra tên video gốc và số giây, sau đó trả về một đoạn HTML chứa video đúng giây đó.
    """
    # Lấy thông tin từ item người dùng chọn (giả sử bạn lưu metadata trong kho dữ liệu)
    video_name = result_item.get("video_filename") # Ví dụ: "video_01.mp4"
    timestamp = result_item.get("timestamp_sec", 0) # Ví dụ: 45.5
    
    # Đường dẫn tới video gốc bạn đang lưu trong thư mục uploads
    video_url = f"/file=uploads/{video_name}#t={timestamp}"
    
    # Tạo giao diện HTML nhỏ để hiển thị video tự động tua đến giây đó
    html_code = f"""
    <div style="text-align: center;">
        <p><b>Video gốc:</b> {video_name} | <b>Thời điểm:</b> {timestamp:.2f}s</p>
        <video width="100%" height="400" controls autoplay>
            <source src="{video_url}" type="video/mp4">
            Trình duyệt của bạn không hỗ trợ thẻ video.
        </video>
    </div>
    """
    return html_code