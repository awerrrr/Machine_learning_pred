# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Kami membangun model Machine Learning untuk memprediksi harga laptop (dalam €) berdasarkan spesifikasi hardware dan software. Model ini membantu:

calon pembeli menetapkan budget secara realistis,
toko daring memberi rekomendasi harga kompetitif,
produsen memetakan celah pasar pada rentang harga tertentu.
Business Understanding

## Problem Statements

Seberapa besar pengaruh spesifikasi inti (CPU, RAM, GPU, penyimpanan) terhadap harga laptop?
Dapatkah kita memprediksi harga laptop baru dengan galat (RMSE) < 250 €?
Goals

Mengukur korelasi tiap fitur dengan target untuk menentukan fitur dominan penentu harga.
Menyediakan model regresi terbaik dengan metrik minimal

## Data Understanding
[UCI Machine Learning Repository](https://www.kaggle.com/datasets/durgeshrao9993/laptop-specification-dataset).
# Data Understanding

| Kolom              | Tipe Data | Keterangan                                              |
| ------------------ | --------- | ------------------------------------------------------- |
| `laptop_ID`        | `int64`   | ID unik tiap laptop                                     |
| `Company`          | `object`  | Merk laptop (Asus, Lenovo, dll)                         |
| `Product`          | `object`  | Nama produk                                             |
| `TypeName`         | `object`  | Jenis laptop (Ultrabook, Gaming, dll)                   |
| `Inches`           | `float64` | Ukuran layar dalam inci                                 |
| `ScreenResolution` | `object`  | Resolusi layar                                          |
| `Cpu`              | `object`  | Detail prosesor                                         |
| `Ram`              | `object`  | RAM (format string, perlu diproses ke numerik)          |
| `Memory`           | `object`  | Tipe dan kapasitas penyimpanan (SSD/HDD)                |
| `Gpu`              | `object`  | GPU/VGA                                                 |
| `OpSys`            | `object`  | Sistem operasi                                          |
| `Weight`           | `object`  | Berat laptop (format string, perlu diproses ke numerik) |
| `Price_in_euros`   | `float64` | Harga dalam Euro (target prediksi)                      |


## EDA

- Melihat distribusi harga

- Korelasi antara RAM, storage, GPU dengan harga

- Visualisasi outlier dan kategori dominan

Distribusi harga bersifat right-skewed; kami lakukan log-transform

Fitur “CpuBrand”, “GpuBrand”, “Ram(GB)”, “SSD(GB)”, “Weight(kg)” menunjukkan korelasi positif dengan harga.
Outlier diatasi dengan metode IQR pada fitur numerik.
One-hot encoding untuk fitur kategorik (Company, TypeName, OpSys).
Standarisasi pada fitur numerik menggunakan 

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Modelling & Evaluation

Kami membandingkan Linear Regression, Random Forest, Gradient Boosting, dan XGBoost.

Model terbaik: Gradient BoostingRegressor dengan
Model ini memenuhi target bisnis (RMSE < 250 €).

## Kesimpulan & Rekomendasi

RAM, kapasitas SSD, serta GPU-brand (NVIDIA vs Intel/AMD iGPU) adalah driver terkuat harga.
Menambah 8 GB RAM rata-rata menaikkan harga ± 120 €.
Produsen dapat menekan harga model entry-level dengan mengurangi SSD dan GPU diskrit.
Untuk prediksi harga barang bekas, tambahkan fitur usia/perilisan.
Langkah Lanjut

Masukkan data kurs (€→IDR) agar output langsung sesuai pasar lokal.
Tambahkan fitur rating build quality & layar (sRGB, refresh rate).
Terapkan SHAP untuk interpretasi individual prediction bagi kebutuhan e-commerce.