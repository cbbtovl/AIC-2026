import os
import re
import zipfile
import traceback
import time
import numpy as np
import pandas as pd
from pathlib import Path
import gradio as gr
from PIL import Image
import json
import urllib.parse

from config import UPLOAD_DIR
from database import init_db, save_item_batch
from indexer import index_file, query_search_text
from services.vqa_service import interact_kisc
from services.trake_service import perform_trake_search


# 1. Định nghĩa và chuẩn hóa đường dẫn gốc
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Đảm bảo UPLOAD_DIR_PATH luôn là đối tượng Path và lấy chuẩn tuyệt đối
UPLOAD_DIR_PATH = UPLOAD_DIR.resolve()
UPLOAD_DIR_PATH.mkdir(parents=True, exist_ok=True)
upload_path_str = str(UPLOAD_DIR_PATH.resolve())
init_db()

# 2. Tạo danh sách "chống mù" bao trọn mọi kiểu đường dẫn trên Windows
allowed_paths_list = [
    PROJECT_ROOT,
    upload_path_str,
    upload_path_str.replace("\\", "/"),
    upload_path_str.lower(),
]

os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(allowed_paths_list)

# ==========================================
# HÀM BỔ TRỢ & TIỆN ÍCH DÙNG CHUNG (UTILS)
# ==========================================
def get_file_path(file_obj):
    """Trích xuất đường dẫn file an toàn bất kể Gradio trả về str, Dict hay File Object"""
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path") or file_obj.get("filename")
    return str(file_obj)

def extract_vid_info(filename):
    """Bóc tách chính xác VideoID và Thời gian (giây) từ tên file hoặc metadata"""
    vid_match = re.search(r'(L\d+_V\d+|V\d+|[\w-]+)', filename)
    video_id = vid_match.group(1) if vid_match else "Unknown"

    time_sec = 0.0
    frame_id = "N/A"
    
    time_match = re.search(r'kf_([\d\.]+)s', filename)
    frame_match = re.search(r'(?: - |_|)(\d+)\.(?:jpg|png|jpeg)$', filename, re.IGNORECASE)

    if time_match:
        time_sec = float(time_match.group(1))
        frame_id = str(int(time_sec * 25))
    elif frame_match:
        frame_id = frame_match.group(1)
        time_sec = int(frame_id) / 25.0

    return video_id, frame_id, time_sec

def find_file_in_roots(filename, search_roots):
    """(MỚI) Hàm dùng chung để quét tìm file trong danh sách các thư mục, giảm lặp code rglob"""
    for search_root in search_roots:
        matches = list(Path(search_root).rglob(filename))
        if matches:
            return str(matches[0].resolve()).replace('\\', '/')
    return None

def resolve_image_path(filepath, filename=None):
    """Hàm phân giải đường dẫn tinh gọn, chỉ tin tưởng DB đã chuẩn hóa."""
    if not filepath:
        return ""
    
    clean_path = str(filepath).replace("\\", "/")
    
    # 1. Trường hợp chuẩn nhất: Ghép đường dẫn tương đối từ DB với thư mục gốc dự án
    candidate_rel = Path(PROJECT_ROOT) / clean_path
    if candidate_rel.is_file():
        return candidate_rel.resolve().as_posix()

    # 2. Trường hợp dự phòng: Nếu DB lưu sẵn đường dẫn tuyệt đối hợp lệ
    if os.path.exists(clean_path):
        return clean_path

    # Không thấy thì trả về rỗng, tuyệt đối không tìm dự phòng để tránh bốc nhầm ảnh
    return ""




# ==========================================
# CÁC HÀM XỬ LÝ LOGIC CHÍNH (HANDLERS)
# ==========================================
def perform_search(query, file_filter, limit):
    if not query or not query.strip():
        return [], "⚠️ Vui lòng nhập từ khóa tìm kiếm!", "❌ Chưa nhập từ khóa.", []
        
    try:
        started_at = time.perf_counter()
        target_limit = int(limit)
        fetch_limit = max(300, target_limit * 15) 
        
        print(f"🔍 Đang tìm kiếm từ khóa: '{query}' với limit={fetch_limit}...")
        results = query_search_text(query, filetype_filter=file_filter, limit=fetch_limit)
        print(f"📊 Tìm thấy tổng cộng: {len(results)} kết quả thô từ DB/FAISS.")
        
        image_outputs = []
        text_outputs = ""
        valid_image_results = []
        missing_image_count = 0
        seen_scenes = {}
        
        for r in results:
            if len(image_outputs) >= target_limit:
                break
                
            similarity = r.get('similarity', 0)
            relevance_percent = max(0.0, min(100.0, similarity * 100))
            filename = r.get('filename', '')
            rel_path = r.get('filepath', '')
            
            if r.get('filetype') == 'image':
                # ĐÃ TỐI ƯU: Chỉ gọi extract_vid_info 1 lần
                parsed_vid, parsed_frame, parsed_time = extract_vid_info(filename)
                
                video_id = r.get('video_id') or parsed_vid
                ordinal = r.get('ordinal')
                frame_idx = r.get('frame_idx')
                display_name = f"{video_id} / keyframe {int(ordinal):03d}" if ordinal is not None else filename
                
                abs_path = resolve_image_path(rel_path, filename)
                if not abs_path:
                    missing_image_count += 1
                    continue

                stored_pts_time = r.get('pts_time')
                t_sec = float(stored_pts_time) if stored_pts_time is not None else parsed_time
                
                is_duplicate = False
                if video_id != "Unknown" and video_id in seen_scenes:
                    for seen_t in seen_scenes[video_id]:
                        if abs(seen_t - t_sec) <= 2.5:
                            is_duplicate = True
                            break
                            
                if is_duplicate:
                    continue
                
                if video_id not in seen_scenes:
                    seen_scenes[video_id] = []
                seen_scenes[video_id].append(t_sec)
                
                r['filepath'] = abs_path
                try:
                    with Image.open(abs_path) as opened_image:
                        pil_img = opened_image.convert("RGB").copy()
                    image_outputs.append((pil_img, f"{display_name} | frame {frame_idx} ({relevance_percent:.1f}%)"))
                    valid_image_results.append(r)
                except (OSError, ValueError) as image_error:
                    print(f"⚠️ Không mở được ảnh {abs_path}: {image_error}")
            else:
                text_outputs += f"### 📄 {filename} (Độ tương đồng: {relevance_percent:.1f}%)\n"
                text_outputs += f"> {r.get('extracted_text', '')}...\n\n---\n"
                
        if not text_outputs and file_filter in ["Tất cả", "Tài liệu"]:
            text_outputs = "*Không tìm thấy tài liệu văn bản nào phù hợp.*"
            
        status_finish = f"✅ Đã tìm thấy {len(image_outputs)} CẢNH ĐỘC LẬP."
        if missing_image_count:
            status_finish += f" Bỏ qua {missing_image_count} kết quả vì không có file ảnh tương ứng."
        status_finish += f" Thời gian: {time.perf_counter() - started_at:.2f}s."
        return image_outputs, text_outputs, status_finish, valid_image_results

    except Exception as e:
        print(f"❌ LỖI TÌM KIẾM: {e}")
        traceback.print_exc()
        return [], f"❌ Lỗi xử lý: {str(e)}", f"❌ Lỗi: {str(e)}", []


def get_exact_timestamp_from_csv(video_id, frame_idx, ordinal=None):
    """Tra cứu chính xác số giây (pts_time) từ file CSV mapkeyframe"""
    # ĐÃ TỐI ƯU: Dùng chung hàm find_file_in_roots thay vì viết lại vòng rglob
    csv_path_str = find_file_in_roots(f"{video_id}.csv", [Path("mapkeyframe"), Path(".")])
    
    if csv_path_str:
        try:
            df = pd.read_csv(csv_path_str)
            row = df[df['n'] == int(ordinal)] if ordinal is not None and 'n' in df.columns else df[df['frame_idx'] == int(frame_idx)]
            if not row.empty:
                return float(row.iloc[0]['pts_time'])
        except Exception as e:
            print(f"Lỗi đọc CSV: {e}")
            
    return int(frame_idx) / 30.0

def on_image_select(evt: gr.SelectData, search_results):
    try:
        if not search_results or evt.index >= len(search_results):
            return "<p>Chưa chọn kết quả</p>", "N/A", "N/A", "N/A"
            
        selected_item = search_results[evt.index]
        filepath = selected_item.get('filepath', '')
        filename = selected_item.get('filename', '')
        
        parsed_vid, parsed_frame_id, parsed_time = extract_vid_info(filename)
        
        video_id = selected_item.get('video_id') or parsed_vid
        source_frame_idx = selected_item.get('frame_idx', parsed_frame_id)
        ordinal = selected_item.get('ordinal')
        frame_id = source_frame_idx if source_frame_idx is not None else parsed_frame_id
        
        stored_pts_time = selected_item.get('pts_time')
        time_sec = float(stored_pts_time) if stored_pts_time is not None else get_exact_timestamp_from_csv(video_id, source_frame_idx, ordinal)
        
        mm = int(time_sec // 60)
        ss = int(time_sec % 60)
        timestamp_str = f"{mm:02d}:{ss:02d} (giây thứ {time_sec:.2f}s)"
        
        target_video_name = f"{video_id}.mp4"
        video_search_roots = [Path(PROJECT_ROOT) / "data" / "videos", UPLOAD_DIR_PATH]
        
        video_path = find_file_in_roots(target_video_name, video_search_roots)
        if not video_path:
            video_path = str((Path(PROJECT_ROOT) / "data" / "videos" / target_video_name).resolve())
        
        # IN LOG DEBUG RA TERMINAL ĐỂ KIỂM TRA
        print(f"🔍 [DEBUG VIDEO] Tim video: {target_video_name} -> Duong dan: {video_path}")
        
        if os.path.exists(video_path):
            # 🚀 CHUYỂN ĐƯỜNG DẪN TUYỆT ĐỐ THÀNH TƯƠNG ĐỐI SO VỚI PROJECT_ROOT
            # Loại bỏ hoàn toàn "C:/VS Code/..." khỏi URL
            try:
                rel_path = os.path.relpath(video_path, PROJECT_ROOT).replace("\\", "/")
            except ValueError:
                rel_path = os.path.abspath(video_path).replace("\\", "/")
            # Bắt buộc có MUTED và PLAYSINLINE để trình duyệt không chặn Autoplay
            video_html = f"""
            <div style="width: 100%; background: #000; border-radius: 8px; overflow: hidden; text-align: center;">
                <video 
                    width="100%" 
                    controls 
                    autoplay 
                    muted 
                    playsinline
                    style="max-height: 450px;"
                    onloadedmetadata="this.currentTime = {time_sec};"
                >
                    <source src="/file={rel_path}" type="video/mp4">
                    Trình duyệt không hỗ trợ phát video.
                </video>
            </div>
            """
            return video_html, frame_id, timestamp_str, filepath
        else:
            print(f"❌ [DEBUG VIDEO] KHÔNG TIM THAY FILE: {video_path}")
            return f"<p style='color: red; text-align: center;'>⚠️ Không tìm thấy file video: {target_video_name}</p>", frame_id, timestamp_str, filepath
            
    except Exception as e:
        traceback.print_exc()
        return f"<p style='color: red;'>Lỗi: {str(e)}</p>", "Lỗi", "Lỗi", str(e)
def handle_vqa_search(event_description, question):
    if not event_description or not event_description.strip():
        return "⚠️ Hãy nhập mô tả sự kiện trước.", ""
    if not question or not question.strip():
        return "⚠️ Hãy nhập câu hỏi cần trả lời.", ""

    combined_query = f"Mô tả sự kiện: {event_description.strip()}\nCâu hỏi: {question.strip()}"
    
    try:
        result = interact_kisc(combined_query, filetype_filter="Hình ảnh")
    except Exception as e:
        return f"⚠️ Lỗi kết nối KISC: {str(e)}", ""

    # Kiểm tra xem result có đúng kiểu dict không
    if not isinstance(result, dict):
        return "⚠️ Dữ liệu trả về từ KISC không đúng định dạng.", ""

    candidates = result.get("candidates", []) or []
    candidate_lines = []
    
    for index, item in enumerate(candidates, 1):
        filename = item.get("filename", "")
        frame_idx = item.get("frame_idx", "N/A")
        pts_time = item.get("pts_time", "N/A")
        
        # An toàn với giá trị similarity bị None
        sim_raw = item.get("similarity")
        sim_val = float(sim_raw) if sim_raw is not None else 0.0
        
        candidate_lines.append(
            f"{index}. {filename} | frame {frame_idx} | {pts_time}s | {sim_val:.3f}"
        )
        
    candidate_text = "\n".join(candidate_lines)
    return result.get("answer", "Không có câu trả lời."), candidate_text


def handle_trake_search(event_lines, limit):
    events = [line.strip() for line in (event_lines or "").splitlines() if line.strip()]
    if len(events) < 2:
        return "⚠️ Nhập ít nhất 2 giai đoạn, mỗi giai đoạn một dòng."
        
    try:
        sequences = perform_trake_search(events, limit=int(limit))
    except Exception as e:
        return f"⚠️ Lỗi thực thi TRAKE: {str(e)}"
        
    if not sequences:
        return "Không tìm thấy chuỗi sự kiện theo đúng thứ tự thời gian."

    output = []
    for sequence_index, sequence in enumerate(sequences, 1):
        if not sequence:
            continue
        first = sequence[0]
        output.append(f"### Chuỗi {sequence_index}: {first.get('video_id', 'N/A')}")
        for event_index, result in enumerate(sequence, 1):
            sim_raw = result.get("similarity")
            sim_val = float(sim_raw) if sim_raw is not None else 0.0
            
            output.append(
                f"- Giai đoạn {event_index}: frame {result.get('frame_idx', 'N/A')}, "
                f"{result.get('pts_time', 'N/A')}s, similarity {sim_val:.3f}"
            )
            
    # Đã sửa: Đưa return ra ngoài vòng lặp chính
    return "\n".join(output)

def _process_single_pair(csv_p: Path, npy_p: Path) -> int:
    try:
        features = np.load(str(npy_p))
        try:
            df = pd.read_csv(str(csv_p))
        except Exception:
            df = pd.read_csv(str(csv_p), header=None)

        if len(features) != len(df):
            min_len = min(len(features), len(df))
            features = features[:min_len]
            df = df.iloc[:min_len]

        base_dir = csv_p.parent
        video_stem = csv_p.stem

        # --- BỔ SUNG 1: Tìm và đọc file JSON chứa Objects/OCR tương ứng ---
        json_data = {}
        # Tìm file json cùng tên video trong thư mục hiện tại hoặc thư mục objects
        candidate_jsons = [
            base_dir / f"{video_stem}.json",
            PROJECT_ROOT / "objects-aic25-b1" / f"{video_stem}.json",
            UPLOAD_DIR_PATH / f"{video_stem}.json"
        ]
        
        for j_path in candidate_jsons:
            if j_path.is_file():
                try:
                    with open(j_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    print(f"✅ Đã nạp dữ liệu Object từ JSON: {j_path.name}")
                    break
                except Exception as e:
                    print(f"⚠️ Không thể đọc file JSON {j_path}: {e}")

        items_to_save = []

        for idx, row in df.iterrows():
            # 1. Bóc tách tên file & ordinal
            first_col = str(row.iloc[0]) if isinstance(row, pd.Series) else str(idx)
            if first_col.lower().endswith(('.jpg', '.png', '.jpeg')):
                frame_filename = first_col
                ordinal = idx + 1
            else:
                ordinal = int(float(first_col))
                frame_filename = f"{video_stem}_{ordinal:03d}.jpg"

            # 2. Tìm đường dẫn thực tế của ảnh
            candidate_paths = [
                UPLOAD_DIR_PATH / video_stem / f"{ordinal:03d}.jpg",
                UPLOAD_DIR_PATH / video_stem / f"{ordinal}.jpg",
                UPLOAD_DIR_PATH / video_stem / frame_filename,
                base_dir / frame_filename
            ]

            virtual_img_path = None
            for candidate in candidate_paths:
                if candidate.is_file():
                    try:
                        virtual_img_path = candidate.relative_to(PROJECT_ROOT).as_posix()
                    except ValueError:
                        virtual_img_path = candidate.as_posix()
                    break

            if not virtual_img_path:
                virtual_img_path = str(base_dir / frame_filename).replace('\\', '/')

            # --- BỔ SUNG 2: Bóc tách danh sách Object/OCR từ JSON theo Keyframe ---
            # Thường key trong JSON sẽ là ordinal (vd: "1"), frame_filename (vd: "001.jpg"), hoặc index
            obj_info = (
                json_data.get(str(ordinal)) or 
                json_data.get(frame_filename) or 
                json_data.get(f"{ordinal:03d}.jpg") or 
                ""
            )
            
            # Nếu thông tin object trong JSON là dạng danh sách [ "person", "car" ], chuyển thành chuỗi "person, car"
            if isinstance(obj_info, list):
                extracted_text = ", ".join([str(x) for x in obj_info])
            elif isinstance(obj_info, dict):
                extracted_text = json.dumps(obj_info, ensure_ascii=False)
            else:
                extracted_text = str(obj_info)

            # 3. Lấy vector embedding & metadata
            vec = [float(x) for x in np.array(features[idx]).flatten()]
            frame_idx = int(row.get('frame_idx', ordinal - 1))
            pts_time = row.get('pts_time')
            pts_time = float(pts_time) if pd.notna(pts_time) else None

            # 4. Đóng gói item (đã có thông tin từ JSON)
            item = {
                "filename": frame_filename,
                "filepath": virtual_img_path,
                "ordinal": ordinal,
                "frame_idx": frame_idx,
                "video_id": video_stem,
                "pts_time": pts_time,
                "filetype": "image",
                "extracted_text": extracted_text,  # Đã gán thông tin Object/OCR vào đây
                "description": f"Precomputed keyframe {frame_filename}. Objects: {extracted_text}",
                "embedding": vec,
                "embedding_type": "clip_visual"
            }
            items_to_save.append(item)

        # 5. Lưu vào DB
        if items_to_save:
            save_item_batch(items_to_save)
            return len(items_to_save)
        return 0

    except Exception as e:
        print(f"❌ Lỗi nạp cặp {csv_p.name}: {e}")
        traceback.print_exc()
        return 0

def handle_precomputed_smart(zip_or_csv_file, npy_file=None, json_file=None):
    zip_or_csv_path_str = get_file_path(zip_or_csv_file)
    if not zip_or_csv_path_str or not os.path.exists(zip_or_csv_path_str):
        return "⚠️ Vui lòng tải lên file ZIP tổng hoặc chọn file CSV hợp lệ!"

    in_path = Path(zip_or_csv_path_str)

    # ================= TRƯỜNG HỢP 1: NẠP TỪ FILE ZIP TỔNG =================
    if in_path.suffix.lower() == '.zip':
        try:
            extract_dir = UPLOAD_DIR_PATH / f"_zip_extract_{in_path.stem}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(str(in_path), 'r') as zip_ref:
                zip_ref.extractall(str(extract_dir))

            all_csvs = list(extract_dir.rglob("*.csv")) + list(extract_dir.rglob("*.CSV"))
            if not all_csvs:
                return "⚠️ Không tìm thấy file .csv nào trong file ZIP!"

            total_loaded = 0
            processed_pairs = 0
            
            for csv_f in all_csvs:
                # 1. Tìm file .npy tương ứng
                possible_npys = [
                    f for f in extract_dir.rglob("*")
                    if f.suffix.lower() == '.npy' and f.stem.lower() == csv_f.stem.lower()
                ]

                # 2. BỔ SUNG: Tìm file .json tương ứng trong ZIP
                possible_jsons = [
                    f for f in extract_dir.rglob("*")
                    if f.suffix.lower() == '.json' and f.stem.lower() == csv_f.stem.lower()
                ]
                json_p = possible_jsons[0] if possible_jsons else None

                # 3. Nạp dữ liệu
                if possible_npys:
                    # Gọi hàm xử lý (đã cập nhật đọc JSON tự động hoặc truyền json_p)
                    count = _process_single_pair(csv_f, possible_npys[0])
                    total_loaded += count
                    processed_pairs += 1

            if processed_pairs == 0:
                return "⚠️ Tìm thấy file CSV nhưng không tìm thấy file .npy tương ứng trong file ZIP!"

            return f"🎉 ĐÃ NẠP THÀNH CÔNG HÀNG LOẠT: {total_loaded} keyframe features từ {processed_pairs} bộ file!"
        except Exception as e:
            traceback.print_exc()
            return f"❌ Lỗi khi giải nén/nạp ZIP: {str(e)}"

    # ================= TRƯỜNG HỢP 2: NẠP FILE LẺ (CSV + NPY + JSON) =================
    npy_path_str = get_file_path(npy_file)
    if not npy_path_str or not os.path.exists(npy_path_str):
        return "⚠️ Nếu không dùng ZIP tổng, bạn phải nạp ít nhất 2 file: CSV và NPY tương ứng!"

    npy_path = Path(npy_path_str)

    # BỔ SUNG: Lấy đường dẫn file JSON lẻ nếu người dùng chọn
    json_path_str = get_file_path(json_file)
    json_path = Path(json_path_str) if (json_path_str and os.path.exists(json_path_str)) else None

    # Gọi hàm nạp đơn lẻ
    count = _process_single_pair(in_path, npy_path)
    
    msg_json = f" (kèm JSON: {json_path.name})" if json_path else ""
    return f"🎉 Đã nạp thành công {count} keyframes từ bộ file {in_path.stem}{msg_json}!"

# ==========================================
# GIAO DIỆN GRADIO (UI)
# ==========================================
with gr.Blocks(title="AI Multimodal Retrieval System", theme=gr.themes.Soft()) as demo:
    raw_search_state = gr.State([])
    
    gr.Markdown("# 🚀 AI MULTIMODAL SEARCH & INSPECTOR")
    
    with gr.Tabs():
        with gr.Tab("📁 1. Nạp Dữ Liệu"):
            gr.Markdown("### ⚙️ CHỌN PHƯƠNG THỨC NẠP DỮ LIỆU")
            
            with gr.Tabs():
                with gr.Tab("📸 Nạp File / Ảnh Keyframe / Video Thủ Công"):
                    gr.Markdown("Dành cho file ảnh keyframe lẻ, video MP4, file tài liệu hoặc file ZIP/RAR. AI sẽ tự động phân tích và tạo vector.")
                    
                    with gr.Row():
                        file_upload = gr.File(
                            label="Chọn File (Ảnh keyframe, Video, Tài liệu, ZIP)", 
                            file_count="single"
                        )
                    
                    btn_upload = gr.Button("⚡ Bắt Đầu Phân Tích & Nạp Dữ Liệu", variant="primary")
                    upload_status = gr.Textbox(label="Trạng thái Nạp", interactive=False)
                    
                    def handle_index(file):
                        filepath = get_file_path(file)
                        if not filepath or not os.path.exists(filepath):
                            return "⚠️ Chưa chọn file hoặc file không tồn tại!"
                        res = index_file(filepath)
                        return f"✅ Nạp thành công: {res.get('filename', os.path.basename(filepath))}"
                        
                    btn_upload.click(fn=handle_index, inputs=[file_upload], outputs=[upload_status])

                with gr.Tab("🚀 Nạp Feature Trích Xuất Sẵn (TỰ ĐỘNG ZIP HÀNG LOẠT)"):
                    gr.Markdown("🔥 **MỚI:** Bạn có thể nén tất cả file .csv và .npy vào **1 FILE ZIP TỔNG** rồi kéo thả vào ô bên dưới. Hệ thống sẽ tự nạp hàng loạt trong vài giây!")
                    
                    with gr.Row():
                        csv_in = gr.File(label="1. Chọn FILE ZIP TỔNG (hoặc File .csv lẻ)")
                        npy_in = gr.File(label="2. Chọn file .npy (Bỏ qua nếu đã up File ZIP ở ô 1)")
                        json_in = gr.File(label="3. Chọn file .json (Tùy chọn)")
                    
                    btn_index_pre = gr.Button("⚡ Bắt Đầu Nạp Nhanh Vector", variant="primary")
                    pre_status = gr.Textbox(label="Trạng thái Nạp", interactive=False)

                    btn_index_pre.click(
                        fn=handle_precomputed_smart, 
                        inputs=[csv_in, npy_in, json_in], 
                        outputs=[pre_status]
                    )

        with gr.Tab("🔍 2. Textual KIS (Tìm kiếm)"):
            with gr.Row():
                with gr.Column(scale=7):
                    query_input = gr.Textbox(
                        label="Nhập từ khóa mô tả sự kiện",
                        placeholder="Gõ từ khóa tìm kiếm tại đây...",
                        lines=2
                    )
                with gr.Column(scale=3):
                    filter_radio = gr.Radio(choices=["Tất cả", "Hình ảnh", "Tài liệu"], value="Tất cả", label="Bộ lọc file")
                    limit_slider = gr.Slider(minimum=1, maximum=50, value=15, step=1, label="Số lượng kết quả trả về")
            
            search_btn = gr.Button("🚀 TÌM KIẾM KIS", variant="primary")
            status_output = gr.Markdown("❌ Trạng thái: Sẵn sàng tìm kiếm.")
            
            with gr.Row():
                with gr.Column(scale=6):
                    gr.Markdown("### 🖼️ KẾT QUẢ HÌNH ẢNH")
                    gallery_output = gr.Gallery(label="Kết quả", show_label=False, columns=3, object_fit="contain")
                    text_output = gr.Markdown(label="Kết quả văn bản")

                with gr.Column(scale=4):
                    gr.Markdown("### 🎬 VIDEO KEYFRAME INSPECTOR")
                    video_player = gr.HTML(label="Video gốc")
                    frame_id_out = gr.Textbox(label="Frame ID", interactive=False)
                    timestamp_out = gr.Textbox(label="Thời gian xuất hiện (Phút:Giây)", interactive=False)
                    filepath_out = gr.Textbox(label="File Path", interactive=False)

            search_btn.click(
                fn=perform_search, inputs=[query_input, filter_radio, limit_slider],
                outputs=[gallery_output, text_output, status_output, raw_search_state]
            )
            query_input.submit(
                fn=perform_search, inputs=[query_input, filter_radio, limit_slider],
                outputs=[gallery_output, text_output, status_output, raw_search_state]
            )
            gallery_output.select(
                fn=on_image_select, inputs=[raw_search_state],
                outputs=[video_player, frame_id_out, timestamp_out, filepath_out]
            )

        with gr.Tab("💬 3. VQA (Hỏi đáp ảnh)"):
            with gr.Row():
                with gr.Column():
                    vqa_event_input = gr.Textbox(
                        label="Mô tả sự kiện",
                        placeholder="Ví dụ: hai người đang đứng thuyết trình trên sân khấu",
                        lines=3,
                    )
                    vqa_question_input = gr.Textbox(
                        label="Câu hỏi",
                        placeholder="Ví dụ: Có bao nhiêu người đang thuyết trình?",
                        lines=2,
                    )
                    vqa_button = gr.Button("🚀 TÌM VÀ TRẢ LỜI", variant="primary")
                with gr.Column():
                    vqa_answer_output = gr.Markdown(label="Câu trả lời")
                    vqa_evidence_output = gr.Markdown(label="Frame bằng chứng")

            vqa_button.click(
                fn=handle_vqa_search,
                inputs=[vqa_event_input, vqa_question_input],
                outputs=[vqa_answer_output, vqa_evidence_output],
            )

        with gr.Tab("⏱️ 4. TRAKE (Chuỗi sự kiện)"):
            trake_events_input = gr.Textbox(
                label="Các giai đoạn theo thứ tự thời gian",
                placeholder="Mỗi giai đoạn một dòng\nVí dụ: Chạy đà\nGiậm nhảy\nBay qua xà\nTiếp đất",
                lines=6,
            )
            trake_limit_input = gr.Slider(
                minimum=1, maximum=10, value=5, step=1, label="Số chuỗi trả về"
            )
            trake_button = gr.Button("🚀 TÌM CHUỖI TRAKE", variant="primary")
            trake_output = gr.Markdown()
            trake_button.click(
                fn=handle_trake_search,
                inputs=[trake_events_input, trake_limit_input],
                outputs=[trake_output],
            )

if __name__ == "__main__":
    print("👉 Thư mục Project:", PROJECT_ROOT)
    print("👉 Thư mục Uploads (Tuyệt đối):", upload_path_str)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=allowed_paths_list
    )