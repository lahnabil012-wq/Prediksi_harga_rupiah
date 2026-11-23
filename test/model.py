"""
Pengujian Fase 3 tanpa database (data sintetis).
Jalankan dari folder utama proyek:  python test/model.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.model as m
from src.features import hitung_fitur, KOLOM_FITUR


def buat_dataset_sintetis(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 15000 * np.cumprod(1 + rng.normal(0, 0.005, n))
    tanggal = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
    mingguan = pd.DataFrame({"tanggal": tanggal, "close": close})
    return hitung_fitur(mingguan)


def main():
    dataset = buat_dataset_sintetis()

    # 1. Pipeline terdiri dari scaler lalu SVR
    model = m.buat_model_svr()
    assert list(model.named_steps) == ["scaler", "svr"]

    # 2. Pelatihan tanpa tuning
    model, paket = m.latih_model(dataset, tuning=False)
    assert paket["parameter"]["kernel"] == "rbf"

    # 3. Pelatihan dengan tuning (Grid Search + TimeSeriesSplit)
    model, paket = m.latih_model(dataset, tuning=True)
    assert paket["parameter"]["kernel"] in ("linear", "rbf")
    ringkasan = m.ringkas_per_kernel(paket["hasil_cv"])
    assert set(ringkasan["kernel"]) == {"linear", "rbf"}

    # 4. Prediksi dan konversi ke harga
    r = m.prediksi_return(model, paket["X_prediksi"])
    harga = m.konversi_ke_harga(r, paket["harga_acuan"])
    assert abs(harga - paket["harga_acuan"] * (1 + r / 100)) < 1e-6
    assert m.konversi_ke_harga(0, 15000) == 15000

    # 5. Simpan lalu muat kembali: hasil prediksi harus identik
    with tempfile.TemporaryDirectory() as tmp:
        m.MODEL_DIR = Path(tmp)
        nama = m.simpan_model(model, paket, nama="uji")
        isi = m.muat_model(nama)
        r2 = m.prediksi_return(isi["pipeline"], paket["X_prediksi"])
        assert abs(r - r2) < 1e-12
        assert isi["fitur"] == KOLOM_FITUR

        try:
            m.muat_model("tidak_ada.joblib")
            raise AssertionError("seharusnya FileNotFoundError")
        except FileNotFoundError:
            pass

    print("Semua pengujian Fase 3 lulus.")
    print(ringkasan.to_string(index=False))


if __name__ == "__main__":
    main()
