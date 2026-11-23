CREATE TABLE IF NOT EXISTS harga_harian (
    tanggal     DATE PRIMARY KEY,
    open        NUMERIC(14,4),
    high        NUMERIC(14,4),
    low         NUMERIC(14,4),
    close       NUMERIC(14,4) NOT NULL,
    sumber      VARCHAR(30) NOT NULL,
    status      VARCHAR(10) NOT NULL DEFAULT 'VALID',
    diperbarui  TIMESTAMP DEFAULT NOW()
);


-- Menambahkan kolom untuk database yang sudah terlanjur dibuat
ALTER TABLE harga_harian
ADD COLUMN IF NOT EXISTS status VARCHAR(10) NOT NULL DEFAULT 'VALID';

-- Memperbarui status seluruh data historis berdasarkan aturan validasi
WITH data_berurutan AS (
    SELECT
        tanggal,
        close,
        LAG(tanggal) OVER (ORDER BY tanggal) AS tanggal_sebelumnya,
        LAG(close) OVER (ORDER BY tanggal) AS close_sebelumnya
    FROM harga_harian
),
status_hasil AS (
    SELECT
        tanggal,
        CASE
            WHEN (
                close_sebelumnya IS NOT NULL
                AND ABS(
                    (close - close_sebelumnya)
                    / NULLIF(close_sebelumnya, 0) * 100
                ) > 3
            )
            OR (
                tanggal_sebelumnya IS NOT NULL
                AND tanggal - tanggal_sebelumnya > 7
            )
            THEN 'PERIKSA'
            ELSE 'VALID'
        END AS status
    FROM data_berurutan
)
UPDATE harga_harian AS h
SET status = s.status
FROM status_hasil AS s
WHERE h.tanggal = s.tanggal;




CREATE TABLE IF NOT EXISTS model_versi (
    id                   SERIAL PRIMARY KEY,
    nama                 VARCHAR(60),
    dibuat_pada          TIMESTAMP,
    fitur                TEXT,
    kernel               VARCHAR(10),
    c                    NUMERIC,
    epsilon              NUMERIC,
    gamma                VARCHAR(20),
    tanggal_data_awal    DATE,
    tanggal_data_akhir   DATE,
    mae_persen           NUMERIC,
    rmse_persen          NUMERIC,
    akurasi_arah         NUMERIC,
    mae_baseline_persen  NUMERIC,
    file_model           VARCHAR(200),
    aktif                BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS prediksi (
    id             SERIAL PRIMARY KEY,
    tanggal_acuan  DATE,
    tanggal_target DATE,
    harga_acuan    NUMERIC(14,4),
    return_persen  NUMERIC(10,6),
    harga_prediksi NUMERIC(14,4),
    model_id       INTEGER REFERENCES model_versi(id),
    dibuat_pada    TIMESTAMP DEFAULT NOW(),
    harga_aktual   NUMERIC(14,4),
    selisih        NUMERIC(14,4),
    galat_persen   NUMERIC(10,6),
    arah_benar     BOOLEAN,
    CONSTRAINT prediksi_unik UNIQUE (tanggal_acuan, model_id)
);


CREATE TABLE IF NOT EXISTS log_update (
    id              SERIAL PRIMARY KEY,
    waktu           TIMESTAMP DEFAULT NOW(),
    sumber          VARCHAR(30),
    baris_diproses  INTEGER,
    status          VARCHAR(10),
    pesan           TEXT
);

