"""Unit tests for AI Services (Embedding 512D, Object Service, Category Parser)."""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexer import get_file_category
from services.embedding_service import get_clip_text_embedding
from services.object_service import summarize_frame_objects

def test_file_categories():
    print("1. Testing file category categorization...")
    assert get_file_category(".jpg") == "image"
    assert get_file_category(".png") == "image"
    assert get_file_category(".mp4") == "video"
    assert get_file_category(".pdf") == "document"
    assert get_file_category(".mp3") == "other"
    print("   [PASS] File category classification is correct.")

def test_embedding_dimensions():
    print("2. Testing multilingual text embedding (512D standard)...")
    sample_text = "người đàn ông đang điều khiển xe máy trên đường"
    vec = get_clip_text_embedding(sample_text)
    assert len(vec) == 512, f"Expected 512D embedding vector, got {len(vec)}D"
    print(f"   [PASS] Generated 512D text embedding vector successfully.")

def test_object_service_summary():
    print("3. Testing object service summary fallback...")
    summary = summarize_frame_objects("NON_EXISTENT_VIDEO", 1)
    assert isinstance(summary, str)
    print("   [PASS] Object service summary returns valid string.")

if __name__ == "__main__":
    test_file_categories()
    test_embedding_dimensions()
    test_object_service_summary()
    print("=== All Service tests passed! ===")
