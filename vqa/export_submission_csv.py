"""
Xuất file nộp bài AIC VQA từ danh sách câu hỏi.

=== THAY ĐỔI (bản này) ===
1. INPUT giờ nhận CẢ 2 định dạng (tự nhận diện theo đuôi file):
   - .txt : MỖI DÒNG 1 CÂU HỎI (đơn giản nhất, không cần escape JSON gì cả).
            Dòng trống hoặc bắt đầu bằng "#" sẽ bị bỏ qua (dùng làm comment).
   - .json: giữ nguyên định dạng mảng cũ [{"query_id": "...", "question": "..."}]
            để không phá code/pipeline cũ nếu bạn đã có sẵn file json.

2. OUTPUT (--output, mặc định submission.csv) giờ ghi ĐÚNG format nộp bài
   AIC (3 cột: <video_name>,<frame_idx>,<answer>), KHÔNG HEADER, quote/escape
   chuẩn (chỉ bọc "..." khi answer có dấu phẩy/ngoặc kép/xuống dòng) — dùng
   trực tiếp module `csv` (quoting=csv.QUOTE_MINIMAL) nên KHÔNG cần chạy thêm
   bước export CSV riêng nào nữa sau khi batch xong.

3. (Tuỳ chọn) --debug_csv: nếu muốn, vẫn có thể xuất THÊM 1 file debug riêng
   (question, confidence, latency...) để tự kiểm tra chất lượng — file này
   KHÔNG dùng để nộp bài, chỉ để bạn xem log.

4. BUGFIX: code cũ tìm frame_idx bằng cách match "cand.get('video_id') ==
   g.get('video_id')" — nếu 1 video có NHIỀU candidate (nhiều keyframe khác
   nhau) trong top_candidates, có thể lấy NHẦM frame_idx của 1 keyframe khác
   không phải keyframe mà model thực sự chọn trả lời. Sửa: match ĐÚNG theo
   "id" (frame_id) — định danh duy nhất của candidate mà model đã chọn — chứ
   không chỉ theo video_id.

Chạy:
  python batch_submission.py --input questions.txt --output submission.csv
  python batch_submission.py --input questions.json --output submission.csv
"""

import argparse
import csv
import json
import os
import time
from pipeline import run_pipeline


def load_questions(input_path: str) -> list[dict]:
    """Tự nhận diện định dạng theo đuôi file (.txt hoặc .json)."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        return _load_questions_json(input_path)
    return _load_questions_txt(input_path)


def _load_questions_txt(input_path: str) -> list[dict]:
    """MỖI DÒNG = 1 câu hỏi. Dòng trống hoặc bắt đầu bằng '#' -> bỏ qua.
    query_id tự sinh theo số thứ tự dòng THỰC SỰ được dùng (Q001, Q002, ...),
    KHÔNG tính dòng trống/comment vào số đếm."""
    questions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n").strip()
            if not line or line.startswith("#"):
                continue
            questions.append({"query_id": f"Q{len(questions) + 1:03d}", "question": line})
    return questions


def _load_questions_json(input_path: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = []
    for i, item in enumerate(data, 1):
        qid = item.get("query_id") or item.get("id") or f"Q{i:03d}"
        questions.append({"query_id": qid, "question": item["question"]})
    return questions


def _find_frame_idx(res: dict, g: dict) -> str:
    """Tra đúng frame_idx của candidate mà model ĐÃ CHỌN — match theo "id"
    (frame_id, định danh duy nhất) thay vì chỉ video_id (xem BUGFIX ở đầu
    file: 1 video có thể có nhiều candidate/nhiều keyframe khác nhau)."""
    target_id = g.get("frame_id", "")
    for cand in res.get("top_candidates", []) or []:
        if cand.get("id") == target_id:
            for key in ("frame_idx", "keyframe_n", "n"):
                if cand.get(key) not in (None, ""):
                    return cand[key]
            break
    # fallback: nếu không match được theo id, thử match theo video_id (kém
    # chính xác hơn nhưng còn hơn để trống hoàn toàn)
    for cand in res.get("top_candidates", []) or []:
        if cand.get("video_id") == g.get("video_id"):
            return cand.get("frame_idx", "")
    return ""


def run_batch(
    input_path: str,
    output_path: str,
    use_self_consistency: bool = True,
    debug_csv: str | None = None,
) -> None:
    questions = load_questions(input_path)
    print(f"📋 Đã đọc {len(questions)} câu hỏi từ {input_path}")

    submission_rows = []   # [(video_id, frame_idx, answer), ...] -> file nộp bài THẬT
    debug_rows = []        # chỉ dùng nếu --debug_csv được truyền

    for i, item in enumerate(questions, 1):
        qid = item["query_id"]
        question = item["question"]
        print(f"[{i}/{len(questions)}] {qid}: {question[:80]}")

        t0 = time.time()
        res = run_pipeline(question, use_self_consistency=use_self_consistency)
        g = res["gemini_output"]
        elapsed = time.time() - t0

        frame_idx = _find_frame_idx(res, g)

        submission_rows.append((g.get("video_id", ""), frame_idx, g.get("answer", "")))

        if debug_csv:
            debug_rows.append({
                "query_id": qid,
                "question": question,
                "video_id": g.get("video_id", ""),
                "pts_time": g.get("pts_time", 0.0),
                "frame_idx": frame_idx,
                "answer": g.get("answer", ""),
                "confidence": g.get("confidence", ""),
                "latency_sec": round(elapsed, 2),
            })

    # ---- File nộp bài THẬT: đúng 3 cột, KHÔNG header, quote/escape chuẩn ----
    # csv.writer mặc định (quoting=QUOTE_MINIMAL) đã tự làm đúng mọi quy tắc:
    # chỉ bọc "..." khi answer chứa dấu phẩy/ngoặc kép/xuống dòng, escape "
    # bằng cách lặp đôi, KHÔNG tự trim khoảng trắng, CRLF line ending.
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(submission_rows)
    print(f"✅ Đã ghi {len(submission_rows)} dòng nộp bài → {output_path}")

    # ---- (Tuỳ chọn) file debug riêng, có header + cột phụ để bạn tự kiểm tra ----
    if debug_csv:
        with open(debug_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_id", "question", "video_id", "pts_time",
                    "frame_idx", "answer", "confidence", "latency_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(debug_rows)
        print(f"🐞 Đã ghi debug log ({len(debug_rows)} dòng) → {debug_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                         help="File .txt (mỗi dòng 1 câu hỏi) hoặc .json (mảng {query_id, question})")
    parser.add_argument("--output", default="submission.csv",
                         help="File CSV NỘP BÀI cuối cùng (3 cột, không header)")
    parser.add_argument("--debug_csv", default=None,
                         help="(Tuỳ chọn) file CSV debug riêng có thêm cột question/confidence/latency, có header")
    parser.add_argument("--no-sc", action="store_true")
    args = parser.parse_args()
    run_batch(args.input, args.output, use_self_consistency=not args.no_sc, debug_csv=args.debug_csv)