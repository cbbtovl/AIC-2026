import sqlite3
import json
import numpy as np
import faiss
import re
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN TUYỆT ĐỐI
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database.db"  # Dùng chung tên database.db với toàn hệ thống

def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA mmap_size=3000000000")  # 3GB Memory Mapping
    conn.execute("PRAGMA cache_size=-64000")      # 64MB Cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn

# Tạo alias dự phòng để file nào gọi get_connection() cũng không bị lỗi
get_connection = get_db_connection

def init_db():
    """Khởi tạo SQLite, đảm bảo bảng items tồn tại và tự động cập nhật cột nếu thiếu"""
    with get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL") 
        cursor = conn.cursor()
        
        # 1. Tạo bảng cơ bản
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                filetype TEXT NOT NULL,
                video_id TEXT,
                ordinal INTEGER,
                frame_idx INTEGER,
                pts_time REAL,
                timestamp_ms TEXT,
                extracted_text TEXT,
                description TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. KHÚC NÀY SẼ CỨU DATABASE CỦA BẠN NÈ: Tự động "vá" thêm cột mới
        cursor.execute("PRAGMA table_info(items)")
        columns = [column[1] for column in cursor.fetchall()]
        missing_columns = {
            "embedding_type": "TEXT DEFAULT 'text'",
            "ordinal": "INTEGER",
            "pts_time": "REAL",
        }
        for column_name, column_definition in missing_columns.items():
            if column_name not in columns:
                cursor.execute(f"ALTER TABLE items ADD COLUMN {column_name} {column_definition}")
            
        conn.commit()

def parse_keyframe_info(filename: str, pts_sec: float = None, fps: float = 30.0):
    """Bóc tách VideoID, frame_idx và tính thời gian dạng MM:SS.ss"""
    video_match = re.search(r'(L\d+_V\d+|V\d+|[\w-]+)', filename)
    video_id = video_match.group(1) if video_match else "UNKNOWN"
    
    if pts_sec is not None:
        seconds = float(pts_sec)
        frame_idx = int(seconds * fps)
    else:
        time_match = re.search(r'kf_(\d+\.\d+)s', filename)
        if time_match:
            seconds = float(time_match.group(1))
            frame_idx = int(seconds * fps)
        else:
            numbers = re.findall(r'\d+', filename.split('.')[0])
            frame_idx = int(numbers[-1]) if numbers else 0
            seconds = frame_idx / fps

    mm = int(seconds // 60)
    ss = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    timestamp_str = f"{mm:02d}:{ss:02d}.{ms:02d}"
    
    return video_id, frame_idx, timestamp_str


class FAISSStore:
    """Quản lý bộ nhớ Vector bằng FAISS"""
    def __init__(self, dim=None):
        self.dim = dim
        self.index = None
        self.id_map = []       
        self.metadata = {}    
        self.deleted_ids = set()
        self.load_from_db()

    def _init_index(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)

    def load_from_db(self):
        init_db()
        cache_index_path = BASE_DIR / "faiss_cache.index"
        cache_metadata_path = BASE_DIR / "faiss_cache.pkl"
        database_mtime = DB_PATH.stat().st_mtime_ns
        if cache_index_path.is_file() and cache_metadata_path.is_file():
            cache_mtime = min(cache_index_path.stat().st_mtime_ns, cache_metadata_path.stat().st_mtime_ns)
            if cache_mtime >= database_mtime:
                try:
                    try:
                        self.index = faiss.read_index(str(cache_index_path), faiss.IO_FLAG_MMAP)
                    except Exception:
                        self.index = faiss.read_index(str(cache_index_path))
                    with cache_metadata_path.open("rb") as cache_file:
                        cached = pickle.load(cache_file)
                    self.dim = cached["dim"]
                    self.id_map = cached["id_map"]
                    self.metadata = cached["metadata"]
                    print(f"[FAISS] Loaded cache: {self.index.ntotal:,} vectors.")
                    return
                except Exception as cache_error:
                    print(f"[FAISS] Cache invalid, rebuilding: {cache_error}")

        started_at = time.perf_counter()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                  SELECT id, filename, filepath, filetype, video_id, ordinal, frame_idx,
                      pts_time, timestamp_ms, extracted_text, description, embedding_type,
                       embedding
                FROM items
            """)
            rows = cursor.fetchall()
            
        vectors = []
        for row in rows:
            row_dict = dict(row)
            emb_str = row_dict.pop("embedding")
            self.metadata[row_dict["id"]] = row_dict
            
            if emb_str:
                try:
                    vec = np.array(json.loads(emb_str), dtype=np.float32)
                    if self.dim is None:
                        self._init_index(len(vec))
                    
                    if len(vec) == self.dim:
                        vectors.append(vec)
                        self.id_map.append(row_dict["id"])
                except Exception:
                    pass
                    
        if vectors and self.index is not None:
            vec_matrix = np.vstack(vectors)
            faiss.normalize_L2(vec_matrix)
            self.index.add(vec_matrix)
            faiss.write_index(self.index, str(cache_index_path))
            with cache_metadata_path.open("wb") as cache_file:
                pickle.dump({"dim": self.dim, "id_map": self.id_map, "metadata": self.metadata}, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[FAISS] Built cache in {time.perf_counter() - started_at:.1f}s: {self.index.ntotal:,} vectors.")

    def add_vectors(self, item_ids: List[int], vectors: List[List[float]], metadata_list: List[Dict[str, Any]]):
        if not vectors: return
        vec_matrix = np.array(vectors, dtype=np.float32)
        
        if self.index is not None and vec_matrix.shape[1] != self.dim:
            print(f"⚠️ Bỏ qua vector không tương thích: {vec_matrix.shape[1]}D, index hiện tại {self.dim}D.")
            return
        if self.index is None:
            self._init_index(vec_matrix.shape[1])
            
        faiss.normalize_L2(vec_matrix) 
        self.index.add(vec_matrix)
        self.id_map.extend(item_ids)
        for i, db_id in enumerate(item_ids):
            self.metadata[db_id] = metadata_list[i]
            
    def search(self, query_vector: List[float], filetype_filter="Tất cả", embedding_type_filter=None, limit=300):
        if self.index is None or self.index.ntotal == 0 or not query_vector: 
            return []
            
        q_vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        if q_vec.shape[1] != self.dim:
            return []

        faiss.normalize_L2(q_vec)
        k = min(self.index.ntotal, 1000) 
        distances, faiss_ids = self.index.search(q_vec, k)
        
        results = []
        mapping = {"Hình ảnh": "image", "Tài liệu": "document", "Âm thanh": "audio"}
        db_type = mapping.get(filetype_filter, filetype_filter)
        
        for dist, f_id in zip(distances[0], faiss_ids[0]):
            if f_id == -1: continue
            if f_id >= len(self.id_map): continue
            
            db_id = self.id_map[f_id]
            if db_id in self.deleted_ids: continue 
            
            meta = self.metadata.get(db_id)
            if not meta: continue
            
            if db_type != "Tất cả" and meta["filetype"] != db_type: continue
            if embedding_type_filter and meta.get("embedding_type") != embedding_type_filter:
                continue
            
            res = meta.copy()
            res["similarity"] = float(dist)
            results.append(res)
            if len(results) >= limit: break
        return results


# Singleton FAISS Store
_faiss_store = None

def get_faiss_store():
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    return _faiss_store


# 3. CÁC HÀM XUẤT RA DÙNG CHUNG (API CHÍNH CỦA DATABASE)
def save_item(filename: str, filepath: str, filetype: str,
              extracted_text: str, description: str,
              embedding: List[float], embedding_type: str = "text",
              video_id: str = None, ordinal: int = None,
              frame_idx: int = None, pts_time: float = None):
    init_db()
    if isinstance(embedding, (list, np.ndarray)):
        embedding = [float(x) for x in np.array(embedding).flatten()]
    embedding_str = json.dumps(embedding) if embedding else None
    parsed_video_id, parsed_frame_idx, timestamp_ms = parse_keyframe_info(filename)
    video_id = video_id or parsed_video_id
    frame_idx = frame_idx if frame_idx is not None else parsed_frame_idx

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO items (
                filename, filepath, filetype, video_id, ordinal, frame_idx, pts_time, timestamp_ms, 
                extracted_text, description, embedding, embedding_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, filepath, filetype, video_id, ordinal, frame_idx, pts_time, timestamp_ms,
              extracted_text, description, embedding_str, embedding_type))
        inserted_id = cursor.lastrowid
        conn.commit()
        
    if embedding:
        store = get_faiss_store()
        meta = {
            "id": inserted_id,
            "filename": filename,
            "filepath": filepath,
            "filetype": filetype,
            "video_id": video_id,
            "ordinal": ordinal,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "timestamp": timestamp_ms,
            "extracted_text": extracted_text,
            "description": description,
            "embedding_type": embedding_type
        }
        store.add_vectors([inserted_id], [embedding], [meta])


def save_item_batch(items: List[Dict[str, Any]]):
    if not items: return
    init_db()
    
    rows = []
    for item in items:
        emb_str = json.dumps(item["embedding"]) if item.get("embedding") else None
        parsed_video_id, parsed_frame_idx, parsed_timestamp = parse_keyframe_info(item["filename"])
        video_id = item.get("video_id", parsed_video_id)
        frame_idx = item.get("frame_idx", parsed_frame_idx)
        timestamp_ms = item.get("timestamp_ms", item.get("pts_time", parsed_timestamp))
        
        rows.append((
            item["filename"], item["filepath"], item["filetype"],
            video_id, item.get("ordinal"), frame_idx, item.get("pts_time"), timestamp_ms,
            item.get("extracted_text", ""), item.get("description", ""),
            emb_str, item.get("embedding_type", "text")
        ))
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO items (
                filename, filepath, filetype, video_id, ordinal, frame_idx, pts_time, timestamp_ms, 
                extracted_text, description, embedding, embedding_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        cursor.execute("SELECT last_insert_rowid()")
        last_id = cursor.fetchone()[0]
        inserted_ids = list(range(last_id - len(items) + 1, last_id + 1))
        conn.commit()
        
    vectors = []
    meta_list = []
    valid_ids = []
    for i, item in enumerate(items):
        if item.get("embedding"):
            parsed_video_id, parsed_frame_idx, parsed_timestamp = parse_keyframe_info(item["filename"])
            video_id = item.get("video_id", parsed_video_id)
            frame_idx = item.get("frame_idx", parsed_frame_idx)
            timestamp_ms = item.get("timestamp_ms", item.get("pts_time", parsed_timestamp))
            vectors.append(item["embedding"])
            meta = {
                "id": inserted_ids[i],
                "filename": item["filename"],
                "filepath": item["filepath"],
                "filetype": item["filetype"],
                "video_id": video_id,
                "ordinal": item.get("ordinal"),
                "frame_idx": frame_idx,
                "pts_time": item.get("pts_time"),
                "timestamp_ms": timestamp_ms,
                "extracted_text": item.get("extracted_text", ""),
                "description": item.get("description", ""),
                "embedding_type": item.get("embedding_type", "text")
            }
            meta_list.append(meta)
            valid_ids.append(inserted_ids[i])
            
    if vectors:
        store = get_faiss_store()
        store.add_vectors(valid_ids, vectors, meta_list)


def search_items(query_vector: List[float], filetype_filter="Tất cả", embedding_type_filter=None, limit=300):
    """Hàm wrapper cho phép indexer và app thực hiện truy vấn vector"""
    store = get_faiss_store()
    return store.search(query_vector, filetype_filter, embedding_type_filter, limit)