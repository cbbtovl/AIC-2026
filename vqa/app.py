import os
import time
import streamlit as st
from pipeline import run_pipeline
from export_submission_csv import result_row_from_pipeline_output, build_csv_bytes
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

st.set_page_config(
    page_title="AIC2026 · VQA Video Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CSS — giao diện tối, thẻ (card) bo góc, badge màu theo độ tin cậy
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1a2138 0%, #0e1117 55%);
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(99,102,241,0.25);
    }
    .hero h1 { color: white; margin: 0; font-size: 1.7rem; }
    .hero p { color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .answer-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(236,72,153,0.10));
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .answer-text { font-size: 1.25rem; font-weight: 600; color: #f5f5ff; line-height: 1.5; }
    .badge {
        display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; margin-bottom: 0.3rem;
    }
    .badge-id { background: rgba(99,102,241,0.18); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4); }
    .badge-time { background: rgba(236,72,153,0.15); color: #f9a8d4; border: 1px solid rgba(236,72,153,0.4); }
    .badge-hi { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
    .badge-mid { background: rgba(234,179,8,0.18); color: #fde047; border: 1px solid rgba(234,179,8,0.4); }
    .badge-lo { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }

    .chip {
        display:inline-block; padding: 0.35rem 0.8rem; margin: 0.15rem;
        border-radius: 999px; border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.03); font-size: 0.82rem; color: #cbd5e1;
    }
    section[data-testid="stSidebar"] { background: #11141d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar — cấu hình pipeline
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Tùy chọn Pipeline")
    use_sc = st.checkbox("Bật Self-Consistency (bầu chọn nhiều lượt)", value=True,
                          help="Chạy nhiều lượt VQA và lấy đáp án đa số — tăng độ ổn định "
                               "nhưng tốn thêm quota OpenRouter cho mỗi lượt.")

    with st.expander("🔧 Nâng cao (Top-K từng tầng)"):
        from config import RETRIEVAL_TOP_K, OBJECT_RERANK_TOP_K, FINAL_TOP_K
        top_k_retrieval = st.slider("Tầng 1 — SigLIP + BM25", 10, 60, RETRIEVAL_TOP_K, 5)
        top_k_object = st.slider("Tầng 2 — Object Rerank", 4, 20, OBJECT_RERANK_TOP_K, 2)
        top_k_final = st.slider("Tầng 3 — Sau Grouping (đưa cho VLM)", 1, 10, FINAL_TOP_K, 1)

    st.markdown("---")
    st.markdown(
        "**Kiến trúc 4 tầng**\n"
        "1. Dịch câu hỏi + SigLIP + BM25 Retrieval\n"
        "2. Object Rerank (khớp vật thể) + Florence-2\n"
        "3. Temporal Grouping + OCR trên keyframe thật\n"
        "4. VLM CoT + Self-Consistency"
    )
    st.caption("Câu hỏi càng mô tả cụ thể vật thể/màu sắc/hành động → Tầng 2 & 3 càng chính xác.")

    st.markdown("---")
    st.markdown(f"**📄 CSV phiên hiện tại:** {len(st.session_state.get('csv_rows', []))} dòng")
    if st.button("🗑️ Xoá lịch sử CSV", use_container_width=True):
        st.session_state.csv_rows = []
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 AIC2026 — Tìm Kiếm & Hỏi Đáp Video</h1>
        <p>Pipeline lai SigLIP + BM25 + Object Rerank + OCR + VLM Self-Consistency</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "question" not in st.session_state:
    st.session_state.question = ""
if "csv_rows" not in st.session_state:
    st.session_state.csv_rows = []  # tích luỹ (video_name, frame_idx, answer) mọi câu hỏi trong phiên này

EXAMPLES = [
    "Người đeo kính đang nói chuyện trước tòa nhà cao tầng",
    "Người đàn ông mặc áo đỏ đang đi trên cầu",
    "Xe cứu hỏa màu đỏ trước một đám cháy",
]

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Câu hỏi",
        value=st.session_state.question,
        placeholder="Ví dụ: Người đeo kính đang nói chuyện trước tòa nhà cao tầng",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)

chip_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if chip_cols[i].button(f"💡 {ex[:38]}{'…' if len(ex) > 38 else ''}", key=f"ex_{i}", use_container_width=True):
        st.session_state.question = ex
        st.rerun()

run_now = search_clicked and question.strip()

if run_now:
    st.session_state.question = question

if run_now:
    t_ui_start = time.time()
    with st.spinner("Đang thực thi Pipeline — Dịch → Retrieval → Rerank → OCR → VLM..."):
        try:
            res = run_pipeline(
                question,
                use_self_consistency=use_sc,
                top_k_retrieval=top_k_retrieval,
                top_k_object_rerank=top_k_object,
                top_k_final=top_k_final,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Xuất CSV: build sẵn 1 dòng (video_name, frame_idx, answer) đúng
    # format nộp bài (xem export_submission_csv.py), và tích luỹ vào lịch sử
    # phiên làm việc để có thể tải tổng hợp nhiều câu hỏi cùng lúc. ----
    csv_row = result_row_from_pipeline_output(res)
    st.session_state.csv_rows.append(csv_row)

    # ---- Metrics tầng thời gian ----
    timings = res.get("timings", {})
    m_cols = st.columns(5)
    m_cols[0].metric("⚡ Tổng", f"{timings.get('total', 0):.2f}s")
    m_cols[1].metric("🔎 Retrieval", f"{timings.get('retrieval', 0):.2f}s")
    m_cols[2].metric("🎯 Object Rerank", f"{timings.get('object_rerank', 0) + timings.get('florence_rerank', 0):.2f}s")
    m_cols[3].metric("📝 OCR", f"{timings.get('ocr', 0):.2f}s")
    m_cols[4].metric("🤖 VLM", f"{timings.get('gemini', 0):.2f}s")

    if res.get("search_query_en") and res.get("search_query_en") != question:
        st.info(f"🌐 Bản dịch tiếng Anh dùng cho tìm kiếm (SigLIP + Object Rerank): *{res['search_query_en']}*")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ---- Thẻ đáp án chính ----
    confidence = res.get("confidence", 0) or 0
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        conf_val = 0.0
    if conf_val >= 0.66:
        conf_badge = "badge-hi"
    elif conf_val >= 0.33:
        conf_badge = "badge-mid"
    else:
        conf_badge = "badge-lo"

    answer_text = res.get("answer", "Không có câu trả lời")
    st.markdown(
        f"""
        <div class="answer-card">
            <div class="answer-text">💬 {answer_text}</div>
            <div style="margin-top:0.8rem;">
                <span class="badge badge-id">🆔 frame: {res.get('frame_id', 'N/A')}</span>
                <span class="badge badge-id">🎞️ video: {res.get('video_id', 'N/A')}</span>
                <span class="badge badge-time">⏱️ {res.get('pts_time', 0)}s ({res.get('timestamp_str', '00:00')})</span>
                <span class="badge {conf_badge}">📊 Độ tin cậy: {conf_val:.0%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Nút tải CSV — đúng format nộp bài, không cần ghi file ra đĩa server ----
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 Tải CSV (câu hỏi này)",
            data=build_csv_bytes([csv_row]),
            file_name="submission_1_dong.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            f"📦 Tải CSV (cả phiên — {len(st.session_state.csv_rows)} dòng)",
            data=build_csv_bytes(st.session_state.csv_rows),
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("🧠 Chain-of-Thought / Lý do suy luận"):
        st.write(res.get("reasoning", "(không có)"))

    # ---- Video gốc ----
    video_path = res.get("video_path", "")
    if video_path and os.path.exists(video_path):
        st.markdown("#### 🎬 Video gốc — tua tới đúng thời điểm")
        try:
            start_sec = int(round(float(res.get("pts_time", 0))))
        except (TypeError, ValueError):
            start_sec = 0
        st.video(video_path, start_time=start_sec)
    elif video_path:
        st.warning(f"⚠️ Không tìm thấy file video tại: `{video_path}` (kiểm tra VIDEOS_ROOT_DIR trong config.py)")
    else:
        st.caption("ℹ️ Chưa có video_path cho candidate này — chạy lại `build_metadata.py` sau khi đã có VIDEOS_ROOT_DIR đúng.")

    st.divider()

    # ---- Gallery candidate — hiển thị ĐÚNG ảnh keyframe thật (không còn lưới 2x2) ----
    st.markdown("#### 🖼️ Candidate sau Object Rerank + Grouping")
    top_candidates = res.get("top_candidates", [])

    if top_candidates:
        tab_labels = [
            f"#{i+1} · {c.get('video_id', '?')} · {c.get('pts_time', 0):.1f}s"
            for i, c in enumerate(top_candidates)
        ]
        tabs = st.tabs(tab_labels)
        for tab, cand in zip(tabs, top_candidates):
            with tab:
                img_col, info_col = st.columns([3, 2])
                with img_col:
                    img_path = cand.get("image_path", "")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=f"Keyframe thật tại t={cand.get('pts_time', 0):.2f}s", use_container_width=True)
                    else:
                        st.warning("Không tìm thấy ảnh cho candidate này.")
                with info_col:
                    st.markdown(
                        f"<span class='badge badge-id'>🆔 {cand.get('id', 'N/A')}</span>"
                        f"<span class='badge badge-id'>🎞️ {cand.get('video_id', 'N/A')}</span>"
                        f"<span class='badge badge-time'>⏱️ {cand.get('pts_time', 0):.2f}s</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Object Rerank score:** `{cand.get('object_rerank_score', 0):.2f}`")
                    matched = cand.get("object_rerank_matched") or []
                    if matched:
                        st.write("**Từ khóa khớp:** " + " ".join(f"<span class='chip'>{m}</span>" for m in matched), unsafe_allow_html=True)
                    if cand.get("florence_score") is not None:
                        st.write(f"**Florence-2 score:** `{cand.get('florence_score', 0):.2f}`")
                    if cand.get("group_size", 1) > 1:
                        st.caption(f"↳ Gộp {cand['group_size']} frame liền kề (cùng cảnh)")
                    if cand.get("identity_match"):
                        st.success("✅ Xác nhận bằng Identity Rescan (OCR khớp tên)")
                    ocr_text = cand.get("ocr_text", "")
                    if ocr_text:
                        st.write("**📝 OCR (trên chính ảnh này):**")
                        st.code(ocr_text, language=None)
    else:
        st.warning("⚠️ Không tìm thấy candidate nào. Kiểm tra đã build_metadata / ingest / bm25 index chưa.")

    with st.expander("🐞 DEBUG: xem nguyên res trả về"):
        st.json({k: v for k, v in res.items() if k not in ("top_candidates", "gemini_output")})

elif not run_now:
    st.markdown(
        """
        <div class="card">
            <b>👋 Bắt đầu</b> — nhập câu hỏi mô tả video (người, vật thể, hành động, màu sắc...)
            hoặc bấm 1 trong các gợi ý phía trên, rồi bấm <b>🔍 Tìm kiếm</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )