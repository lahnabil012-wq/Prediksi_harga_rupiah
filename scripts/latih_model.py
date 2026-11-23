import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.db import get_engine, ambil_harian_tersimpan, buat_tabel, catat_log
from src.ingestion import ke_mingguan
from src.features import hitung_fitur
from src.model import (
    latih_model,
    ringkas_per_kernel,
    prediksi_return,
    konversi_ke_harga,
    simpan_model,
    daftarkan_model,
)


def main():
    engine = get_engine()
    buat_tabel(engine)

    # Data harian -> mingguan -> fitur dan target
    harian = ambil_harian_tersimpan(engine)
    mingguan = ke_mingguan(harian)
    dataset = hitung_fitur(mingguan)

    print("=== FASE 3: PELATIHAN MODEL SVR ===")
    print("Mencari parameter (Grid Search + TimeSeriesSplit) ...")

    model, paket = latih_model(dataset, tuning=True)

    print("\nPerbandingan kernel (MAE validasi silang, satuan % return):")
    print(ringkas_per_kernel(paket["hasil_cv"]).to_string(index=False))

    param = paket["parameter"]
    print("\nParameter terpilih:")
    for kunci, nilai in param.items():
        print(f"  {kunci}: {nilai}")

    print("\nJumlah data latih:", len(paket["X_train"]))

    # Simpan file model dan catat versinya di database
    nama_file = simpan_model(model, paket)
    model_id = daftarkan_model(engine, model, paket, dataset, nama_file)

    catat_log(
        engine,
        status="OK",
        baris_diproses=len(paket["X_train"]),
        pesan=f"Model {nama_file} dilatih (id={model_id}, kernel={param['kernel']})",
        sumber="latih_model",
    )

    print(f"\nModel disimpan: models/{nama_file}")
    print(f"Terdaftar di model_versi dengan id = {model_id} (aktif)")

    # Prediksi contoh untuk minggu berikutnya
    r = prediksi_return(model, paket["X_prediksi"])
    harga = konversi_ke_harga(r, paket["harga_acuan"])
    target = pd.to_datetime(paket["tanggal_acuan"]) + pd.Timedelta(days=7)

    print("\nContoh prediksi:")
    print(f"  Tanggal acuan : {pd.to_datetime(paket['tanggal_acuan']).date()}")
    print(f"  Tanggal target: {target.date()}")
    print(f"  Harga acuan   : Rp {paket['harga_acuan']:,.2f}")
    print(f"  Return        : {r:.6f}%")
    print(f"  Harga prediksi: Rp {harga:,.2f}")


if __name__ == "__main__":
    main()
