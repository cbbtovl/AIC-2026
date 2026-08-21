"""
CLI gộp toàn bộ pipeline. Chạy theo đúng thứ tự:

    python main.py build-metadata
    python main.py ingest
    python main.py bm25-build
    python main.py query "người đàn ông mặc áo đỏ đang đi trên cầu"
"""

import argparse

from config import RETRIEVAL_TOP_K, OBJECT_RERANK_TOP_K, FINAL_TOP_K
from build_metadata import build_metadata
from store import ingest
from bm25_search import bm25_engine
from pipeline import run_pipeline
from export_submission_csv import append_submission_row, result_row_from_pipeline_output


def main():
    parser = argparse.ArgumentParser(description="AIC2026 Hybrid VQA/RAG Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-metadata", help="Đọc CSV + media-info, tạo metadata_all.jsonl")
    sub.add_parser("ingest", help="Embed ảnh bằng SigLIP và lưu vào ChromaDB")
    sub.add_parser("bm25-build", help="Build lại BM25 index từ metadata_all.jsonl")

    query_parser = sub.add_parser("query", help="Truy vấn bằng câu hỏi tiếng Việt/Anh (full pipeline)")
    query_parser.add_argument("text", type=str, help="Câu hỏi / mô tả cần tìm")
    query_parser.add_argument("--top_k_retrieval", type=int, default=RETRIEVAL_TOP_K)
    query_parser.add_argument("--top_k_object_rerank", type=int, default=OBJECT_RERANK_TOP_K)
    query_parser.add_argument("--top_k_final", type=int, default=FINAL_TOP_K)
    query_parser.add_argument("--no_self_consistency", action="store_true",
                               help="Tắt self-consistency, chỉ chạy 1 lượt Nemotron")
    query_parser.add_argument("--csv_out", type=str, default="submission.csv",
                               help="Đường dẫn file CSV để TỰ ĐỘNG append kết quả sau mỗi lần query "
                                    "(mặc định: submission.csv trong thư mục hiện tại). "
                                    "Dùng --csv_out '' để tắt xuất CSV.")

    args = parser.parse_args()

    if args.command == "build-metadata":
        build_metadata()
    elif args.command == "ingest":
        ingest()
    elif args.command == "bm25-build":
        bm25_engine.build_or_load(force_rebuild=True)
    elif args.command == "query":
        r = run_pipeline(
            args.text,
            use_self_consistency=not args.no_self_consistency,
            top_k_retrieval=args.top_k_retrieval,
            top_k_object_rerank=args.top_k_object_rerank,
            top_k_final=args.top_k_final,
        )

        print(f"\n🌐 Bản dịch tiếng Anh dùng cho tìm kiếm: {r.get('search_query_en')}")
        print(f"\n🔎 Candidate cuối cùng đưa cho Nemotron:")
        for i, m in enumerate(r.get("top_candidates", [])):
            group_note = f" (gộp {m['group_size']} frame)" if m.get("group_size", 1) > 1 else ""
            ocr_note = f" | OCR: {m['ocr_text'][:40]}..." if m.get("ocr_text") else ""
            print(f"  {i+1}. id={m.get('id')} | video={m.get('video_id')} | "
                  f"t={m.get('pts_time', 0):.2f}s | object_score={m.get('object_rerank_score', 0):.2f}"
                  f"{group_note}{ocr_note}")

        print("\n🤖 KẾT QUẢ NỘP BÀI:")
        print("=" * 60)
        print(f"frame_id : {r.get('frame_id')}")
        print(f"video_id : {r.get('video_id')}")
        print(f"pts_time : {r.get('pts_time')}")
        print(f"answer   : {r.get('answer')}")
        print(f"confidence: {r.get('confidence')}")
        print("=" * 60)
        print(f"\nThời gian từng bước (giây): { {k: round(v, 2) for k, v in r['timings'].items()} }")

        # ---- Tự động xuất CSV theo đúng format nộp bài (xem export_submission_csv.py) ----
        if args.csv_out:
            row = result_row_from_pipeline_output(r)
            append_submission_row(*row, csv_path=args.csv_out)
            print(f"\n📄 Đã append kết quả vào: {args.csv_out}  →  {row}")


if __name__ == "__main__":
    main()