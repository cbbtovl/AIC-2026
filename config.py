import os
import json
from pathlib import Path

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent

# Thư mục lưu trữ tệp tải lên
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn cơ sở dữ liệu SQLite và File cấu hình JSON
DB_PATH = BASE_DIR / "database.db"
CONFIG_FILE = BASE_DIR / "config.json"

# --- Các cấu hình mặc định ---
DEFAULT_PROVIDER = "local"  
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_TEXT_EMBEDDING = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_VISION_MODEL = "ViT-B-32"
DEFAULT_LOCAL_WHISPER = "small"
DEFAULT_GEMINI_API_KEY = ""

class ConfigManager:
    def __init__(self):
        # Dùng toán tử 'or': Nếu os.environ.get() không lấy được từ .env, nó sẽ dùng DEFAULT ngay bên cạnh
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        self.gemini_model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.provider = DEFAULT_PROVIDER
        self.text_embedding_model = DEFAULT_TEXT_EMBEDDING
        self.vision_model = DEFAULT_VISION_MODEL
        self.local_whisper_model = DEFAULT_LOCAL_WHISPER
        
        # Tự động nạp cấu hình đã lưu từ config.json
        self.load_config()

    def load_config(self):
        """Đọc cấu hình từ file config.json cục bộ"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.gemini_api_key = data.get("gemini_api_key", self.gemini_api_key)
                    self.gemini_model = data.get("gemini_model", self.gemini_model) # 👈 Nạp gemini_model
                    self.provider = data.get("provider", self.provider)
                    self.text_embedding_model = data.get("text_embedding_model", self.text_embedding_model) # 👈 Đã sửa lỗi tên biến
                    self.vision_model = data.get("vision_model", self.vision_model)
                    self.local_whisper_model = data.get("local_whisper_model", self.local_whisper_model)
                print("[Config] Đã tải cấu hình lưu trữ thành công.")
            except Exception as e:
                print(f"[Config] Không thể đọc config.json: {e}")

    def save_config(self):
        """Ghi cấu hình hiện tại xuống file config.json"""
        try:
            data = {
                "gemini_api_key": self.gemini_api_key,
                "gemini_model": self.gemini_model, # 👈 Lưu gemini_model
                "provider": self.provider,
                "text_embedding_model": self.text_embedding_model, # 👈 Đã sửa lỗi tên biến
                "vision_model": self.vision_model,
                "local_whisper_model": self.local_whisper_model
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("[Config] Đã lưu cấu hình mới xuống config.json.")
        except Exception as e:
            print(f"[Config] Lỗi khi ghi file config.json: {e}")

    def set_api_key(self, api_key: str):
        self.gemini_api_key = api_key.strip()
        self.provider = "gemini" if self.gemini_api_key else "local"
        self.save_config()

    def set_provider(self, provider: str):
        if provider in ["gemini", "local"]:
            self.provider = provider
            self.save_config()

config = ConfigManager()