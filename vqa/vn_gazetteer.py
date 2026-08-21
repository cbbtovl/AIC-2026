# vn_gazetteer.py
"""
Danh sách tĩnh các tỉnh/thành Việt Nam — dùng làm "anchor entity" khi câu hỏi
nhắc tới địa danh cụ thể (vd "một xã thuộc tỉnh Khánh Hòa").

ĐỘNG LỰC: BM25/SigLIP ở Tầng 1 không có tín hiệu ưu tiên nào cho tên tỉnh/
thành — nếu media-info (title/description) của video không tình cờ chứa
đúng chữ "Khánh Hòa", pipeline coi như "mù" với anchor quan trọng nhất của
câu hỏi. Module này chỉ làm 1 việc: match câu hỏi với gazetteer, trả về tên
tỉnh chuẩn hoá để boost retrieval + làm rõ hơn context cho VLM/OCR.

LƯU Ý: đây chỉ là danh sách CẤP TỈNH. Nếu bạn có sẵn danh sách xã/huyện theo
tỉnh (thường AIC không cung cấp), có thể mở rộng thêm VN_DISTRICTS/VN_WARDS
theo cùng cách để tăng độ chính xác nhận diện, nhưng với hầu hết câu hỏi thi
chỉ cần match ĐÚNG TỈNH là đủ để thu hẹp video_id nghi vấn.

Cập nhật danh sách nếu có sáp nhập/đổi tên hành chính mới hơn.
"""

VN_PROVINCES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bạc Liêu", "Bắc Giang", "Bắc Kạn",
    "Bắc Ninh", "Bến Tre", "Bình Dương", "Bình Định", "Bình Phước",
    "Bình Thuận", "Cà Mau", "Cao Bằng", "Cần Thơ", "Đà Nẵng", "Đắk Lắk",
    "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang",
    "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang",
    "Hòa Bình", "Hồ Chí Minh", "Hưng Yên", "Khánh Hòa", "Kiên Giang",
    "Kon Tum", "Lai Châu", "Lạng Sơn", "Lào Cai", "Lâm Đồng", "Long An",
    "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên",
    "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh", "Quảng Trị",
    "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên",
    "Thanh Hóa", "Thừa Thiên Huế", "Huế", "Tiền Giang", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
]

# Alias/viết tắt phổ biến hay gặp trong câu hỏi/media-info (không dấu, viết
# tắt TP., hoặc cách gọi khác) -> map về tên chuẩn trong VN_PROVINCES.
_PROVINCE_ALIASES = {
    "tp hcm": "Hồ Chí Minh",
    "sài gòn": "Hồ Chí Minh",
    "saigon": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh",
    "brvt": "Bà Rịa - Vũng Tàu",
    "vũng tàu": "Bà Rịa - Vũng Tàu",
    "hue": "Huế",
    "thanh hoa": "Thanh Hóa",
    "khanh hoa": "Khánh Hòa",
}


def find_province(text: str) -> str | None:
    """Tìm tên tỉnh/thành xuất hiện trong `text` (so khớp substring, ưu tiên
    tên DÀI HƠN trước để tránh match nhầm 1 phần của tên dài hơn, vd
    "Thừa Thiên Huế" phải được ưu tiên trước "Huế").
    Trả về tên tỉnh CHUẨN HOÁ (đúng chính tả trong VN_PROVINCES), hoặc None."""
    if not text:
        return None
    lowered = text.lower()

    for alias, canon in _PROVINCE_ALIASES.items():
        if alias in lowered:
            return canon

    for prov in sorted(VN_PROVINCES, key=len, reverse=True):
        if prov.lower() in lowered:
            return prov
    return None