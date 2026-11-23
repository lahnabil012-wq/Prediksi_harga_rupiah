
import pandas as pd

from src.db import get_engine, ambil_harian_tersimpan
from src.ingestion import ke_mingguan
from src.features import hitung_fitur
from src.model import latih_model, prediksi_return, konversi_ke_harga


def main():
    engine = get_engine()

    # Mengambil seluruh data historis
    harian = ambil_harian_tersimpan(engine)

    # Membentuk data mingguan
    mingguan = ke_mingguan(harian)

    # Menghitung fitur dan target
    dataset = hitung_fitur(mingguan)

    # Melatih SVR
    model, paket = latih_model(dataset)

    # Memprediksi return minggu berikutnya
    return_prediksi = prediksi_return(
        model,
        paket["X_prediksi"]
    )

    # Mengubah return menjadi harga kurs
    harga_prediksi = konversi_ke_harga(
        return_prediksi,
        paket["harga_acuan"]
    )

    tanggal_target = (
        pd.to_datetime(paket["tanggal_acuan"])
        + pd.Timedelta(days=7)
    )

    if return_prediksi > 0:
        arah = "Rupiah melemah"
    elif return_prediksi < 0:
        arah = "Rupiah menguat"
    else:
        arah = "Relatif stabil"

    print("=== HASIL TRAINING DAN PREDIKSI SVR ===")

    print("\nJumlah data training:", len(paket["X_train"]))
    print("Jumlah fitur:", paket["X_train"].shape[1])

    print("\nTanggal acuan:", paket["tanggal_acuan"])
    print("Tanggal target:", tanggal_target.date())

    print(f"Harga acuan: Rp {paket['harga_acuan']:,.2f}")
    print(f"Prediksi return: {return_prediksi:.6f}%")
    print(f"Prediksi harga: Rp {harga_prediksi:,.2f}")
    print("Arah prediksi:", arah)

    print("\nTraining dan prediksi berhasil.")


if __name__ == "__main__":
    main()
