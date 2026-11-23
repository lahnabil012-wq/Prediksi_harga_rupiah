"""
Titik masuk program prediksi USD/IDR.

Saat ini (Fase 3) main.py menjalankan pelatihan model SVR:
ambil data dari PostgreSQL -> fitur -> tuning -> simpan model -> contoh prediksi.
Pada Fase 5, file ini akan diganti dengan aplikasi desktop Tkinter.

Jalankan dari folder utama proyek:  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.latih_model import main as latih_model


def main():
    try:
        latih_model()
    except RuntimeError as e:
        # Biasanya DATABASE_URL belum diisi di file .env
        print("Gagal:", e)
        sys.exit(1)
    except Exception as e:
        print(f"Terjadi kesalahan: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()