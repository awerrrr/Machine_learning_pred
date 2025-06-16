# Laporan Proyek Machine Learning - Najwar Putra Kusumah Wardana

## Domain Proyek
### Latar Belakang
Di era digital ini, pasar laptop berkembang sangat cepat dengan beberapa model, spesifikasi, dan harga yang berbeda. Karena jumlah informasi yang diberikan terlalu banyak, beragam serta tidak terpadu, pembeli, ahli serta perusahaan seringkali menemui kesulitan dalam memilih model laptop yang memerlukan untuk kebutuhan spesifik yang mereka butuhkan. Tidak adanya sistem klasifikasi dan perangkingan otomatis dan berbasis data untuk mengetahui laptop-laptop merk apa yang tepat untuk Anda mengakibatkan proses pengambilan keputusan tidak efisien dan penuh kesalahan.

Maka dari itu terciptalah keinginan saya untuk membangun model Machine Learning untuk memprediksi harga laptop (dalam Rupiah) berdasarkan spesifikasi hardware dan software untuk memudahkan para masyarakat dalam hal memprediksi harga laptop yang mereka inginkan.

Model ini membantu:
calon pembeli menetapkan budget secara realistis,
toko daring memberi rekomendasi harga kompetitif,
produsen memetakan celah pasar pada rentang harga tertentu.

## Business Understanding
### Problem Statements
1. Bagaimana cara kita untuk memprediksi harga laptop jika sistem tersebut berbayar dan rumit?
Di era ini tidak banyak tersedia sistem untuk aplikasi machine learning yang mampu mengkelompokkan laptop ke dalam berbagai kategori tertentu dan memberikan prediksi tentang laptop yang mereka inginkan.
2. Apakah seseorang ingin melakukan analisis terlebih dahulu untuk mengetahui rekomendasi laptop yang mereka inginkan?
Tentu tidak, hal tersebut membuat user akan melakukan analisis manual yang memakan waktu yang bayak, melakukan kesalahan dalm pembelian, membutuhkan efort yang lebih dan harus melakuakn riset yang mendalam agar laptop bisa sesuai dengan kebutuhan penggunanya.
3. Bagaimana hal ini bisa berdampak pada semua kalangan?
Banyak orang yang akan terdampak kebaikan dari aplikasi ini sendiri seperti Pengguna individu (pekerja profesional, gamer, pelajar, desainer), berbagai macam industri seperti (sekolah, perusahaan, pabrik, dll) dan pihak yang hendak membeli laptop dalam jumlah yang sangat besar.

### Goals
1. Mengukur korelasi tiap fitur dengan target untuk menentukan fitur dominan penentu harga.
2. Menyediakan model regresi terbaik dengan metrik minimal

## Data Understanding
### Data source
[UCI Machine Learning Repository](https://www.kaggle.com/datasets/durgeshrao9993/laptop-specification-dataset).
### Kolom Data
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
* df.shape *
(1303, 13)
dari data diatas menunjukkan bahwa data memiliki 1303 kolom dan 13 baris sebelum dilakukannya prores data preparation.

## Keadaan Data
- Missing value
company              0
product              0
type_name            0
inches               0
screen_resolution    0
cpu                  0
ram                  0
memory               0
gpu                  0
opsys                0
weight               0
price_in_euros       0
price_in_idr         0
dtype: int64

Tidak ada missing value pada data yang saya pakai.

- Duplikasi data
* df.duplicated().sum() *
28
Ada 28 duplikasi pada data dan akan dilakukan drop duplikasi pada data preparation.

### EDA

- Melihat distribusi harga

- Korelasi antara RAM, storage, GPU dengan harga

- Visualisasi outlier dan kategori dominan

Distribusi harga bersifat right-skewed; saya lakukan log-transform

- Fitur “CpuBrand”, “GpuBrand”, “Ram(GB)”, “SSD(GB)”, “Weight(kg)” menunjukkan korelasi positif dengan harga.

## Data Preparation
### Feature Engineering
Pada cell pertama Feature Engineering yang saya lakukan adalah menginisialisasi storage column (ssd, hdd, flash, hybrid), membuat column 'memory' menjadi lowercase untuk mempertahankan konsistensi pada data, dan terakhir melakukan parsing pada data.

Pada cell kedua saya melakukan pengambilan kata pertama pada kolom yang dipakai untuk machine learning atau kolom utama yaitu 'cpu_brand' dan 'gpu_brand', lalu menambahkan kolom total storage untuk menyimpan nilai dari kolom ssd, hdd, flash, dan hybrid.

Pada cell ketiga dilakukan penghapusan kolom mentah seperti kolom memory, cpu, dan gpu. Hal ini dilakukan agar dataset lebih rapih, mengurangi noise dan memudahkan pada tahap modeling nantinya.

Pada cell keempat melakukan penargetan fitur untuk y(price_in_idr) dan X(df.drop(columns=['price_in_euros', 'price_in_idr', 'product'])) dilakukan drop pada data frame untuk X hal ini dilakukan karena terlalu spesifik dan dapat meningkatkan kompleksitas data.

Pada cell kelima dilakukan teknik one-hot encoding yang dimana ia mengubah kolom kategorikal menjadi fitur numerik biner (0/1) agar bisa digunakan oleh model machine learning.

### Standardization
Dalam tahap ini saya melakukan standardisasi (scaling) terhadap fitur numerik
Membuat data berada dalam skala yang seimbang (mean = 0, std = 1). Hal ini penting untuk model seperti regresi linier, KNN, SVM, PCA, dan lain-lain yang sensitif terhadap skala fitur.

### IQR
Pada cell pertama saya menyimpan hasil dataframe sebelumnya menjadi df_clean yang akan digunakan untuk tahap IQR dan selanjutnya.

Pada cell kedua saya melakukan pengecekan outliers dengan teknik IQR yang akan mendeteksi data dalam kolom mana saja yang menghasilkan outlier.

Pada cell ketiga saya melakukan penghapusan untuk outliers dengan teknik IQR yang kan menghapus outlier memakai rumus lower and upper.

### Missing Value Replace
Saya melakukan replace untuk nilai nilai yang hilang walaupun tak terdeteksi dengan 'df.isnan()', hal ini dilakukan dengan tujuan untuk meningkatkan akurasi pada pemodelan nanti.

### Train, Test dan Split
saya melakukan:
- Pemisahan data fitur dan target
- Split training dan test set secara acak (80:20)
- Menyiapkan data agar bisa digunakan untuk pelatihan dan evaluasi model regresi harga laptop

## Modelling
Pada cell pertama saya melakukan persiapan untuk pemodelan dengan menginisialisasi fungsi.

Pada cell kedua saya melakukan pembandingan ataara Linear Regression, Random Forest, Gradient Boosting, dan XGBoost untuk mengetahui model regresi mana yang paling cocok untuk data yang saya pakai.

Adapun parameter utama yang saya gunakan pada tahap ini adalah:
| Model                 | Parameter Kunci                                                           |
| --------------------- | ------------------------------------------------------------------------- |
| **Linear Regression** | Default (`fit_intercept=True`)                                            |
| **Ridge**             | `alpha=1.0`                                                               |
| **Lasso**             | `alpha=0.1`                                                               |
| **ElasticNet**        | `alpha=1.0`, `l1_ratio=0.5`                                               |
| **XGBoost**           | `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`, `random_state=42` |
## Evaluation
1. Pada cell pertama saya lakukan visualisasi pada model yang sudah dilatih untuk melihat hasil Test R^2 dan Test RMSE pada model model tersebut.

2. Pada cell kedua saya menerapkan model yang paling bagus untuk dimasukkan pada modelling fix.

3. Pada cell ketiga saya melakukan pengecekan ulang untuk beberapa nilai yang telah dimasukkan model terbaik. Adapun outputnya:
--- Checks before creating plot_df ---
Shape of y_test_ts: (255,)
Shape of y_pred_test_ts: (255,)
Are there NaNs in y_test_ts? False
Are there NaNs in y_pred_test_ts? False
y_test_ts head:
 1179    11375000.0
342     12530000.0
649     27720000.0
772     17850000.0
803     30607500.0
Name: price_in_idr, dtype: float64
y_pred_test_ts head:
 0    12305960.0
1    16946688.0
2    28033976.0
3    10439736.0
4    23099148.0
dtype: float32
y_test_ts tail:
 701      6982500.0
1105    24729250.0
424     48982500.0
944     22732500.0
65      34702500.0
Name: price_in_idr, dtype: float64
y_pred_test_ts tail:
 250     7031393.5
251    29050778.0
252    48750168.0
253    21459524.0
254    30336740.0
dtype: float32
Error: 'date' column not found in clean_df!
Are test indices in clean_df index? True
- Penjelasan
1. Data yang digunakan untuk evaluasi model sudah memenuhi semua syarat validitas. Panjang data prediksi dan data aktual sama, yaitu 255 baris, sehingga aman untuk dilakukan visualisasi dan evaluasi lebih lanjut.
2. Tidak ditemukan nilai kosong (NaN) pada kedua data tersebut, baik pada nilai aktual maupun hasil prediksi.
3. Nilai-nilai harga laptop, baik yang aktual maupun yang diprediksi, terlihat masuk akal saat dilakukan pengecekan awal (head) dan akhir (tail), sehingga tidak menunjukkan anomali mencurigakan. Selain itu, kolom target 'price_in_idr' tersedia di dalam DataFrame clean_df, artinya model dapat mengakses dan menggunakan kolom tersebut tanpa error.
4. Terakhir, index dari data uji (X_test_ts) cocok dengan index pada clean_df, sehingga proses pengambilan data berdasarkan index berjalan dengan lancar dan valid.

- Dengan metric evaluasi:
| Metrik       | Fungsi                                                            |
| ------------ | ----------------------------------------------------------------- |
| **R² Score** | Menilai *goodness of fit* (seberapa baik model menjelaskan data)  |
| **RMSE**     | Mengukur rata-rata besar kesalahan prediksi (dalam satuan Rupiah) |

4. Pada cell terakhir

![Prediksi Harga](/img/v1.png)
| Hal yang Terlihat                              | Penjelasan                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| Pola prediksi **mirip dengan aktual**          | Artinya model XGBoost cukup berhasil menangkap tren harga laptop.                 |
| Beberapa lonjakan besar di `Actual Price`      | Ini bisa jadi laptop dengan spesifikasi ekstrem atau harga tidak wajar.           |
| Model masih mengalami **over/underprediction** | Kadang prediksi lebih tinggi/rendah dari aktual, terutama di puncak-puncak harga. |
| **Fluktuasi harga cukup besar**                | Harga laptop sangat bervariasi tergantung spesifikasi.                            |



## Create Plot
Kode ini digunakan untuk membuat sebuah DataFrame baru bernama plot_df yang berisi hasil perbandingan antara harga aktual dan harga prediksi dari model terbaik untuk keperluan visualisasi atau evaluasi lebih lanjut.

1. Pada cell pertama:
plot_df head:
    Index  Actual Price  Predicted Price
0      0    11375000.0       12305960.0
1      1    12530000.0       16946688.0
2      2    27720000.0       28033976.0
3      3    17850000.0       10439736.0
4      4    30607500.0       23099148.0
plot_df shape: (255, 3)
Untuk beberapa sampel, model melakukan prediksi cukup dekat dengan nilai aktual.

2. Pada cell kedua:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 255 entries, 0 to 254
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Index            255 non-null    int64  
 1   Actual Price     255 non-null    float64
 2   Predicted Price  255 non-null    float32
dtypes: float32(1), float64(1), int64(1)
memory usage: 5.1 KB
Penjelasan:
| Kolom             | Tipe Data | Penjelasan                                                       |
| ----------------- | --------- | ---------------------------------------------------------------- |
| `Index`           | `int64`   | Nomor urut baris (hanya untuk keperluan visualisasi di sumbu X). |
| `Actual Price`    | `float64` | Harga aktual dari laptop (dalam IDR), sumber dari `y_test_ts`.   |
| `Predicted Price` | `float32` | Harga prediksi dari model XGBoost, dari `y_pred_test_ts`.        |

3. Pada cell terakhir analisis ini diperoleh
- Model bekerja cukup stabil di harga menengah (10 juta - 30 juta).
- Model cukup baik dalam memprediksi.
- Mayoritas titik prediksi mengikuti tren titik aktual.
- Prediksi mendekati aktual di sebagian besar data.
![Gambar Actual vs Predict](/img/newplot.png)
