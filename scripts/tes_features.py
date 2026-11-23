
from src.db import get_engine, ambil_harian_tersimpan
from src.ingestion import ke_mingguan
from src.features import hitung_fitur
from src.features import siapkan_dataset


def main():
    engine = get_engine()

    # Mengambil seluruh data harian dari PostgreSQL
    harian = ambil_harian_tersimpan(engine)

    # Membentuk data mingguan historis
    mingguan = ke_mingguan(harian)

    # Menghitung lima fitur dan target
    dataset = hitung_fitur(mingguan)

    paket = siapkan_dataset(dataset)

    print("\n=== PEMISAHAN DATASET ===")
    print("X_train:", paket["X_train"].shape)
    print("y_train:", paket["y_train"].shape)
    print("X_prediksi:", paket["X_prediksi"].shape)
    print("Harga acuan:", paket["harga_acuan"])
    print("Tanggal acuan:", paket["tanggal_acuan"])

    kolom_fitur = [
        "ret_1",
        "ret_2",
        "gap_sma4",
        "rsi_14",
        "volatilitas_4",
    ]

    # Data yang sudah memiliki lima fitur dan target lengkap
    data_training = dataset.dropna(
        subset=kolom_fitur + ["target"]
    )

    # Baris terbaru yang memiliki lima fitur lengkap
    data_prediksi = dataset.dropna(
        subset=kolom_fitur
    ).tail(1)

    print("=== PENGUJIAN FEATURE ENGINEERING ===")

    print("\nJumlah data mingguan:", len(mingguan))
    print("Jumlah hasil perhitungan fitur:", len(dataset))
    print("Jumlah data siap training:", len(data_training))

    print("\nJumlah data fitur lengkap untuk prediksi:",
          len(data_prediksi))

    print("\n5 Data Terakhir:")
    print(dataset.tail().to_string(index=False))

    print("\nData Terbaru untuk Prediksi:")
    print(data_prediksi.to_string(index=False))

    print("\nPengujian selesai.")


if __name__ == "__main__":
    main()
