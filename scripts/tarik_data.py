import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TANGGAL_MULAI, TICKER
from src.db import (
    buat_tabel,
    catat_log,
    get_engine,
    simpan_harian,
    tanggal_terakhir,
    ambil_harian_tersimpan,
)
from src.ingestion import ambil_data_harian, ke_mingguan, validasi


def main():
    engine = get_engine()
    buat_tabel(engine)

    terakhir = tanggal_terakhir(engine)
    if terakhir is None:
        mulai = TANGGAL_MULAI
    else:
        mulai = terakhir - timedelta(days=14)

    print(f"Mengambil data {TICKER} sejak {mulai} ...")

    try:
        df = ambil_data_harian(TICKER, mulai=mulai)
        df_bersih, catatan = validasi(df)

        if df_bersih.empty:
            catat_log(
                engine,
                "GAGAL",
                0,
                "Tidak ada baris valid setelah validasi.",
            )
            print("Gagal: tidak ada baris valid.")
            return

        jumlah_diproses = simpan_harian(df_bersih, engine)

        # Mengambil seluruh data harian dari PostgreSQL
        df_historis = ambil_harian_tersimpan(engine)

        # Membentuk data mingguan dari seluruh data historis
        mingguan = ke_mingguan(df_historis)

        print(f"Baris harian diproses: {jumlah_diproses}")
        print(f"Total data harian di database: {len(df_historis)}")
        print(f"Total data mingguan historis: {len(mingguan)}")

        if not mingguan.empty:
            print(
                f"Rentang mingguan: {mingguan['tanggal'].min().date()} "
                f"sampai {mingguan['tanggal'].max().date()}"
            )
        print(
            f"Rentang: {df_bersih['tanggal'].min().date()} sampai "
            f"{df_bersih['tanggal'].max().date()}"
        )

        pesan = "; ".join(catatan) if catatan else None
        catat_log(engine, "SUKSES", jumlah_diproses, pesan)
        if catatan:
            print(f"Catatan PERIKSA: {len(catatan)}")
            for c in catatan:
                print(f"  - {c}")

    except Exception as e:
        catat_log(engine, "GAGAL", 0, str(e))
        print(f"Gagal: {e}")
        raise


if __name__ == "__main__":
    main()
