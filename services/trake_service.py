from typing import List, Dict, Any
from collections import defaultdict
from indexer import query_search_text

def parse_result_position(result: Dict[str, Any]):
    video_id = result.get("video_id")
    timestamp = result.get("pts_time")
    if video_id is None or timestamp is None:
        return None, None
    return str(video_id), float(timestamp)

def perform_trake_search(events: List[str], limit: int = 5) -> List[List[Dict[str, Any]]]:
    """Tìm kiếm chuỗi sự kiện theo đúng trình tự dòng thời gian (Đã xử lý chặn nổ đệ quy)"""
    if not events or len(events) < 2:
        return []

    # 1. Tìm kiếm thô diện rộng cho từng sự kiện độc lập
    event_raw_results = [query_search_text(evt, "Tất cả", limit=40) if evt.strip() else [] for evt in events]

    # 2. Phân loại và cấu trúc lại dữ liệu theo từng Video để cô lập không gian tìm kiếm
    # Cấu trúc: video_groups[tên_video][vị_trí_sự_kiện] = [danh sách khung hình]
    video_groups = defaultdict(lambda: defaultdict(list))
    
    for event_idx, step_results in enumerate(event_raw_results):
        for r in step_results:
            v_name, timestamp = parse_result_position(r)
            if v_name is not None and timestamp is not None:
                video_groups[v_name][event_idx].append({'time': timestamp, 'data': r})

    valid_sequences = []

    # 3. Quét trình tự thời gian trên từng Video độc lập
    for v_name, steps in video_groups.items():
        # Nếu video đó không chứa đầy đủ tất cả các sự kiện yêu cầu -> Bỏ qua luôn
        if len(steps) < len(events):
            continue
            
        # Cắt tỉa (Pruning): Sắp xếp các khung hình trong từng sự kiện theo điểm tương đồng, 
        # chỉ giữ lại tối đa 5 khung hình đỉnh nhất để chặn đứng nguy cơ bùng nổ tổ hợp (Combinatorial Explosion)
        for idx in range(len(events)):
            steps[idx].sort(key=lambda x: x['data'].get('similarity', 0), reverse=True)
            steps[idx] = steps[idx][:5]

        # Thuật toán duyệt tìm chuỗi tăng dần trên tập dữ liệu đã thu gọn
        def dfs_video(current_seq: List[Dict], current_step_idx: int):
            if current_step_idx == len(events):
                valid_sequences.append([item['data'] for item in current_seq])
                return
                
            prev_time = current_seq[-1]['time']
            for item in steps[current_step_idx]:
                if item['time'] > prev_time: # Điều kiện sống còn: Thời gian phải tăng dần
                    dfs_video(current_seq + [item], current_step_idx + 1)

        # Bắt đầu kích hoạt DFS cho riêng video này từ sự kiện đầu tiên
        for first_item in steps[0]:
            dfs_video([first_item], 1)

    # 4. Xếp hạng toàn bộ chuỗi tìm được dựa trên điểm trung bình tổng thể
    scored_sequences = []
    for seq in valid_sequences:
        avg_score = sum(r.get('similarity', 0) for r in seq) / len(seq)
        scored_sequences.append((avg_score, seq))
        
    scored_sequences.sort(key=lambda x: x[0], reverse=True)
    return [seq for score, seq in scored_sequences[:limit]]