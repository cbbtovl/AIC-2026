"""Unit tests for SQLite database and FAISS vector search."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import init_db, parse_keyframe_info, get_faiss_store, DB_PATH
import numpy as np

def test_parse_keyframe_info():
    print("1. Testing metadata parser...")
    video_id, frame_idx, timestamp = parse_keyframe_info("L21_V001_137.jpg")
    assert video_id == "L21_V001", f"Expected L21_V001, got {video_id}"
    assert frame_idx == 137, f"Expected 137, got {frame_idx}"
    print("   [PASS] Keyframe parser works properly.")

def test_faiss_store_readiness():
    print("2. Testing FAISS vector store...")
    init_db()
    assert DB_PATH.exists(), "database.db must exist."
    store = get_faiss_store()
    print(f"   [INFO] FAISS Store dimension: {store.dim}D, Total vectors in index: {store.index.ntotal if store.index else 0:,}")
    if store.index and store.index.ntotal > 0:
        assert store.dim == 512, f"Expected 512D vectors, got {store.dim}D"
        # Test a dummy query
        dummy_query = np.random.randn(512).tolist()
        results = store.search(dummy_query, limit=3)
        print(f"   [PASS] FAISS query returned {len(results)} candidate results.")
    else:
        print("   [NOTE] FAISS index is empty or not loaded.")

if __name__ == "__main__":
    test_parse_keyframe_info()
    test_faiss_store_readiness()
    print("=== All Database & FAISS tests passed! ===")
