
from pathlib import Path
from dotenv import load_dotenv
import os

# Lokasi utama proyek
BASE_DIR = Path(__file__).resolve().parent

# Membaca file .env
load_dotenv(BASE_DIR / ".env")

# Konfigurasi database
DATABASE_URL = os.getenv("DATABASE_URL")

# Konfigurasi sumber data
TICKER = "IDR=X"
TANGGAL_MULAI = "2016-01-01"

# Validasi harga (PRD bagian 5.4)
HARGA_MIN = 5000
HARGA_MAX = 50000

# Lokasi penyimpanan model
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
