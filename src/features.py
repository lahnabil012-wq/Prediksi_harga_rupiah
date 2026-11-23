import pandas as pd


def hitung_fitur(mingguan):
    """
    Menghitung lima fitur dan target return minggu berikutnya
    dari data kurs mingguan.
    """

    kolom_hasil = [
        "tanggal",
        "close",
        "ret_1",
        "ret_2",
        "gap_sma4",
        "rsi_14",
        "volatilitas_4",
        "target",
    ]

    if mingguan is None or mingguan.empty:
        return pd.DataFrame(columns=kolom_hasil)

    # Mengambil tanggal dan Close saja
    data = mingguan[["tanggal", "close"]].copy()

    # Memastikan tipe data benar dan urutan tanggal
    data["tanggal"] = pd.to_datetime(data["tanggal"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")

    data = data.dropna(subset=["tanggal", "close"])
    data = data.sort_values("tanggal").reset_index(drop=True)

    close = data["close"]

    # Menghitung return mingguan dalam persen
    ret = close.pct_change(fill_method=None) * 100

    # 1. Return minggu berjalan
    ret_1 = ret

    # 2. Return satu minggu sebelumnya
    ret_2 = ret.shift(1)

    # 3. Selisih Close terhadap SMA 4 minggu
    sma_4 = close.rolling(window=4).mean()
    gap_sma4 = (close / sma_4 - 1) * 100

    # 4. RSI 14 dengan metode Wilder
    perubahan = close.diff()

    kenaikan = perubahan.clip(lower=0)
    penurunan = -perubahan.clip(upper=0)

    rata_kenaikan = kenaikan.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rata_penurunan = penurunan.ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = rata_kenaikan / rata_penurunan
    rsi_14 = 100 - (100 / (1 + rs))

    # 5. Volatilitas dari empat return mingguan terakhir
    volatilitas_4 = ret.rolling(window=4).std()

    # Target: return pada minggu berikutnya
    target = ret.shift(-1)

    hasil = pd.DataFrame({
        "tanggal": data["tanggal"],
        "close": close,
        "ret_1": ret_1,
        "ret_2": ret_2,
        "gap_sma4": gap_sma4,
        "rsi_14": rsi_14,
        "volatilitas_4": volatilitas_4,
        "target": target,
    })

    return hasil



KOLOM_FITUR = [
    "ret_1",
    "ret_2",
    "gap_sma4",
    "rsi_14",
    "volatilitas_4",
]


def siapkan_dataset(dataset):
    """
    Memisahkan dataset untuk training dan prediksi terbaru.
    """

    if dataset is None or dataset.empty:
        raise ValueError("Dataset fitur kosong.")

    # Data yang memiliki seluruh fitur dan target
    data_training = dataset.dropna(
        subset=KOLOM_FITUR + ["target"]
    ).copy()

    # Data terbaru yang memiliki fitur lengkap,
    # tetapi belum memiliki target
    data_prediksi = dataset[
        dataset["target"].isna()
    ].dropna(
        subset=KOLOM_FITUR
    ).tail(1).copy()

    if data_training.empty:
        raise ValueError("Data training tidak tersedia.")

    if data_prediksi.empty:
        raise ValueError("Data untuk prediksi tidak tersedia.")

    # Memisahkan input dan target training
    X_train = data_training[KOLOM_FITUR]
    y_train = data_training["target"]

    # Mengambil fitur terbaru untuk prediksi
    X_prediksi = data_prediksi[KOLOM_FITUR]

    # Mengambil harga dan tanggal acuan
    harga_acuan = float(data_prediksi.iloc[0]["close"])
    tanggal_acuan = data_prediksi.iloc[0]["tanggal"]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_prediksi": X_prediksi,
        "harga_acuan": harga_acuan,
        "tanggal_acuan": tanggal_acuan,
    }

