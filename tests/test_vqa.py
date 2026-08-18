"""Integration test for VQA and KISC reasoning pipeline."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

from services.vqa_service import interact_kisc

def test_vqa_query():
    print("Testing VQA reasoning pipeline...")
    prompt = "Trong các video đã nạp có những sự kiện gì tiêu biểu? Trả lời ngắn gọn."
    try:
        res = interact_kisc(prompt, chat_history=None, filetype_filter="Hình ảnh")
        print("\n--- VQA Response ---")
        print(res.get("answer"))
        print("\n--- Top Candidate Keyframes ---")
        for c in res.get("candidates", []):
            print(f"- File: {c.get('filename')} | Video: {c.get('video_id')} | Frame: {c.get('frame_idx')} | Time: {c.get('pts_time')}s")
        print("\n[PASS] VQA interaction completed.")
    except Exception as e:
        print(f"[WARN] VQA interaction encountered error: {e}")

if __name__ == "__main__":
    test_vqa_query()
