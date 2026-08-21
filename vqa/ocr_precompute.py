# ocr_precompute.py
"""
OCR TOÀN BỘ (hoặc lấy mẫu stride) keyframe database 1 LẦN, chạy OFFLINE, LOCAL
(EasyOCR) — KHÔNG tốn 1 request OpenRouter/LLM nào. Kết quả ghi ra
OCR_CACHE_PATH (mặc định ./ocr_cache.jsonl).

TẠI SAO CẦN FILE NÀY (xem identity_rescan.py, bm25_search.py):
    Trước đây, muốn OCR để xác nhận tên riêng/địa danh, pipeline phải đợi
    candidate Tầng 1 (SigLIP/BM25 — không có tín hiệu gì cho tên riêng) đã
    hội tụ về rất ít video_id rồi mới rescan OCR trên (các) video đó — một
    vòng luẩn quẩn "con gà quả trứng": hội tụ đúng lại cần chính thứ mà rescan
    mới tạo ra được. Precompute OCR 1 lần cho TOÀN BỘ database giải quyết dứt
    điểm theo 2 cách:
      (a) bm25_search.py nạp cache này vào corpus BM25 -> Tầng 1 khớp được
          "Hồng Nhung"/"Khánh Hòa" NGAY TỪ ĐẦU, không cần rescan.
      (b) identity_rescan.py đọc cache gần như MIỄN PHÍ (tra dict) thay vì
          luôn gọi EasyOCR sống -> khi cache đã "ấm" cho 1 video, rescan có
          thể quét TOÀN BỘ frame của video đó thay vì chỉ 1 tập con lấy mẫu
          theo stride như trước.

CHIẾN LƯỢC HIỆU QUẢ CHO CPU (không có GPU):
    1. STRIDE SUBSAMPLING (mặc định OCR_PRECOMPUTE_STRIDE=3 trong config.py):
       các keyframe liền kề CÙNG VIDEO thường rất giống nhau về hình ảnh
       (banner/caption tồn tại xuyên suốt vài giây) -> chỉ OCR 1/N frame MỖI
       VIDEO (không phải toàn cục, để phủ đều mọi video) vẫn bắt được hầu hết
       text, giảm tải đúng theo tỉ lệ N mà ít mất recall.
    2. MULTIPROCESSING: mỗi process con chỉ LOAD EasyOCR reader MỘT LẦN (chi
       phí load model vài giây, không trả giá lại cho từng ảnh), xử lý song
       song trên toàn bộ core CPU máy bạn.
    3. RESUMABLE: kết quả ghi APPEND vào OCR_CACHE_PATH (jsonl) kèm flush định
       kỳ. Chạy lại (sau khi bị ngắt, hoặc muốn tăng độ phủ bằng stride nhỏ
       hơn) sẽ TỰ ĐỘNG bỏ qua các id đã có, không OCR lại từ đầu.

CHẠY:
    python ocr_precompute.py                      # theo OCR_PRECOMPUTE_STRIDE trong config.py
    python ocr_precompute.py --stride 1            # OCR TẤT CẢ (chậm nhất, đầy đủ nhất)
    python ocr_precompute.py --stride 1 --video-id L21_V001   # test nhanh 1 video
    python ocr_precompute.py --workers 8           # số process song song

SAU KHI CHẠY XONG (hoặc chạy dở, resume sau cũng được):
    python bm25_search.py       # rebuild lại BM25 index để nạp cache OCR mới vào corpus
"""

import os
import json
import time
import argparse
import multiprocessing as mp

from config import METADATA_JSONL, OCR_CACHE_PATH, OCR_PRECOMPUTE_STRIDE, OCR_PRECOMPUTE_WORKERS


def _load_done_ids(path: str) -> set[str]:
    """Đọc lại các id ĐÃ OCR từ lần chạy trước — cho phép resume, không OCR lại."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                done.add(rec["id"])
    return done


def _load_records(video_id_filter: str | None) -> list[dict]:
    records = []
    with open(METADATA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if video_id_filter and r.get("video_id") != video_id_filter:
                continue
            records.append(r)
    return records


def _apply_stride(records: list[dict], stride: int) -> list[dict]:
    """Chỉ giữ 1/stride frame MỖI VIDEO (không phải toàn cục theo thứ tự file
    metadata) — để phủ đều tất cả video ngay từ đầu, thay vì OCR hết video đầu
    tiên trong danh sách rồi mới sang video kế (nếu bị ngắt giữa chừng sẽ có
    video hoàn toàn chưa có gì trong cache)."""
    if stride <= 1:
        return records
    by_video: dict[str, list[dict]] = {}
    for r in records:
        by_video.setdefault(r["video_id"], []).append(r)
    out = []
    for frames in by_video.values():
        frames.sort(key=lambda x: x.get("keyframe_n", 0))
        out.extend(frames[::stride])
    return out


# ----- Worker process: load EasyOCR reader 1 LẦN/process, tái dùng cho cả chunk -----
_worker_reader = None


def _worker_init():
    global _worker_reader
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    import easyocr
    from config import OCR_LANGS
    print(f"[worker pid={os.getpid()}] Đang tải EasyOCR reader (CPU, 2 thread/process)...")
    _worker_reader = easyocr.Reader(OCR_LANGS, gpu=False)
    


def _worker_process(record: dict) -> dict:
    from config import OCR_MIN_CONF
    img_path = record.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        return {"id": record["id"], "ocr_text": ""}
    try:
        results = _worker_reader.readtext(img_path, detail=1)
        lines = [text.strip() for (_bbox, text, conf) in results
                 if conf >= OCR_MIN_CONF and text.strip()]
        return {"id": record["id"], "ocr_text": " | ".join(lines)}
    except Exception as e:
        print(f"⚠️ Lỗi OCR {img_path}: {type(e).__name__}: {e}")
        return {"id": record["id"], "ocr_text": ""}


def precompute_ocr_cache(stride: int, workers: int, video_id_filter: str | None, flush_every: int = 200):
    if not os.path.exists(METADATA_JSONL):
        print(f"❌ Chưa có {METADATA_JSONL}. Chạy `python build_metadata.py` trước.")
        return

    all_records = _load_records(video_id_filter)
    sampled = _apply_stride(all_records, stride)

    done_ids = _load_done_ids(OCR_CACHE_PATH)
    todo = [r for r in sampled if r["id"] not in done_ids]

    print(f"📊 Tổng keyframe metadata: {len(all_records)}")
    print(f"📊 Sau stride={stride} (1/{stride} mỗi video): {len(sampled)} frame mục tiêu")
    print(f"📊 Đã OCR sẵn từ lần chạy trước (resume): {len(done_ids)}")
    print(f"📊 Còn lại cần OCR lần này: {len(todo)}")

    if not todo:
        print("✅ Không còn gì để OCR — cache đã đầy đủ theo stride hiện tại.")
        return

    t0 = time.time()
    n_done = 0
    with open(OCR_CACHE_PATH, "a", encoding="utf-8") as out_f:
        with mp.Pool(processes=workers, initializer=_worker_init) as pool:
            for result in pool.imap_unordered(_worker_process, todo, chunksize=4):
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % flush_every == 0:
                    out_f.flush()
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = len(todo) - n_done
                    eta_min = (remaining / rate / 60) if rate > 0 else 0
                    print(f"  ⏳ {n_done}/{len(todo)} | {rate:.2f} ảnh/s | "
                          f"ETA còn lại ~{eta_min:.1f} phút")
        out_f.flush()

    total_min = (time.time() - t0) / 60
    print(f"✅ Hoàn tất {n_done} ảnh trong {total_min:.1f} phút. Cache tại: {OCR_CACHE_PATH}")
    print("👉 Chạy tiếp: python bm25_search.py   (để rebuild BM25 index nạp cache OCR mới)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute OCR cache offline cho toàn bộ keyframe database")
    parser.add_argument("--stride", type=int, default=OCR_PRECOMPUTE_STRIDE,
                         help=f"Chỉ OCR 1/N frame mỗi video (mặc định {OCR_PRECOMPUTE_STRIDE} theo config.py)")
    parser.add_argument("--workers", type=int, default=OCR_PRECOMPUTE_WORKERS,
                         help=f"Số process song song (mặc định {OCR_PRECOMPUTE_WORKERS} theo config.py)")
    parser.add_argument("--video-id", type=str, default=None,
                         help="Chỉ OCR 1 video cụ thể (dùng để test nhanh)")
    parser.add_argument("--flush-every", type=int, default=200)
    args = parser.parse_args()

    precompute_ocr_cache(
        stride=args.stride,
        workers=args.workers,
        video_id_filter=args.video_id,
        flush_every=args.flush_every,
    )