# Laporan Proyek Machine Learning - Najwar Putra Kusumah Wardana

## Domain Proyek
### Latar Belakang
Di era digital ini, pasar laptop berkembang sangat cepat dengan beberapa model, spesifikasi, dan harga yang berbeda. Karena jumlah informasi yang diberikan terlalu banyak, beragam serta tidak terpadu, pembeli, ahli serta perusahaan seringkali menemui kesulitan dalam memilih model laptop yang memerlukan untuk kebutuhan spesifik yang mereka butuhkan. Tidak adanya sistem klasifikasi dan perangkingan otomatis dan berbasis data untuk mengetahui laptop-laptop merk apa yang tepat untuk Anda mengakibatkan proses pengambilan keputusan tidak efisien dan penuh kesalahan.

Maka dari itu terciptalah keinginan saya untuk membangun model Machine Learning untuk memprediksi harga laptop (dalam Rupiah) berdasarkan spesifikasi hardware dan software untuk memudahkan para masyarakat dalam hal memprediksi harga laptop yang mereka inginkan.

Model ini membantu:
1. Calon pembeli menetapkan budget secara realistis.
2. Toko daring memberi rekomendasi harga kompetitif.
3. Produsen memetakan celah pasar pada rentang harga tertentu.

## Business Understanding
### Problem Statements
1. Bagaimana cara kita untuk memprediksi harga laptop jika sistem tersebut berbayar dan rumit?
2. Apakah seseorang ingin melakukan analisis terlebih dahulu untuk mengetahui rekomendasi laptop yang mereka inginkan?
3. Bagaimana hal ini bisa berdampak pada semua kalangan?

### Goals
1. Di era ini tidak banyak tersedia sistem untuk aplikasi machine learning yang mampu mengkelompokkan laptop ke dalam berbagai kategori tertentu dan memberikan prediksi tentang laptop yang mereka inginkan.
2. Tentu tidak, hal tersebut membuat user akan melakukan analisis manual yang memakan waktu yang bayak, melakukan kesalahan dalm pembelian, membutuhkan efort yang lebih dan harus melakuakn riset yang mendalam agar laptop bisa sesuai dengan kebutuhan penggunanya.
3. Banyak orang yang akan terdampak kebaikan dari aplikasi ini sendiri seperti Pengguna individu (pekerja profesional, gamer, pelajar, desainer), berbagai macam industri seperti (sekolah, perusahaan, pabrik, dll) dan pihak yang hendak membeli laptop dalam jumlah yang sangat besar.

## Data Understanding
### Data source
[UCI Machine Learning Repository](https://www.kaggle.com/datasets/durgeshrao9993/laptop-specification-dataset).
### Kolom Data
| Kolom              | Arti                                                               |
| ------------------ | ------------------------------------------------------------------ |
| `laptop_ID`        | ID unik untuk setiap laptop                                        |
| `Company`          | Merek pembuat laptop (misalnya Apple, HP, Dell, dsb)               |
| `Product`          | Nama produk atau model laptop                                      |
| `TypeName`         | Tipe laptop (Ultrabook, Notebook, Gaming, dll)                     |
| `Inches`           | Ukuran layar dalam inci                                            |
| `ScreenResolution` | Resolusi layar (dan kadang jenis panelnya, seperti IPS)            |
| `Cpu`              | Jenis prosesor (misalnya Intel Core i5, i7, dll)                   |
| `Ram`              | Kapasitas RAM (biasanya dalam GB)                                  |
| `Memory`           | Jenis dan kapasitas penyimpanan (misalnya SSD, HDD, atau gabungan) |
| `Gpu`              | Jenis kartu grafis (misalnya Intel HD, AMD Radeon, dll)            |
| `OpSys`            | Sistem operasi bawaan laptop (misalnya Windows, macOS, No OS, dll) |
| `Weight`           | Berat laptop (dalam kilogram, disertai "kg")                       |
| `Price_in_euros`   | Harga laptop dalam Euro                                            |

### Info Pada Data
| No | Kolom              | Non-Null Count | Tipe Data | Penjelasan Singkat                          |
| -- | ------------------ | -------------- | --------- | ------------------------------------------- |
| 0  | `laptop_ID`        | 1303           | `int64`   | ID unik laptop, berupa angka bulat          |
| 1  | `Company`          | 1303           | `object`  | Nama merek laptop (Apple, HP, dll)          |
| 2  | `Product`          | 1303           | `object`  | Nama produk / model                         |
| 3  | `TypeName`         | 1303           | `object`  | Jenis laptop (Ultrabook, Notebook, dll)     |
| 4  | `Inches`           | 1303           | `float64` | Ukuran layar dalam inci                     |
| 5  | `ScreenResolution` | 1303           | `object`  | Resolusi layar dan jenis panel              |
| 6  | `Cpu`              | 1303           | `object`  | Nama dan model CPU                          |
| 7  | `Ram`              | 1303           | `object`  | RAM (dalam format string, misalnya '8GB')   |
| 8  | `Memory`           | 1303           | `object`  | Kapasitas dan tipe penyimpanan              |
| 9  | `Gpu`              | 1303           | `object`  | Nama GPU / kartu grafis                     |
| 10 | `OpSys`            | 1303           | `object`  | Sistem operasi (Windows, macOS, dll)        |
| 11 | `Weight`           | 1303           | `object`  | Berat laptop (dalam string, misal '1.37kg') |
| 12 | `Price_in_euros`   | 1303           | `float64` | Harga laptop dalam Euro                     |

##### Dari sini kita ketahui bahwa data mempunyai 1303 baris dan 13 kolom dengan tiga tipe data yaitu int, float, dan object.
##### Dalam Output ini pun kita bisa melakukan pengecekan untuk missing value dengan melihat output yang tersedia pada kolom output "Non-Null Count".

### Manipulasi Kolom
#### Melakukan Rename Pada Kolom
# masukkan gambar ss manipulasidata

Melakukan rename pada kolom data yang akan kita analisis untuk menghasilkan data yang lebih konsisten, pythonic, dan untuk mencegah error pada analisis kedepannya dan yang terpenting adalah menghindari typo.
- Lebih konsisten: semua nama pakai huruf kecil.
- Lebih Pythonic: penamaan kolom seperti price_in_euros lebih mudah digunakan dalam analisis atau visualisasi.
- Mencegah error: nama seperti ScreenResolution rentan typo, screen_resolution lebih mudah diketik dan dibaca, typo pengguna.

#### Menghapus Kolom Yang Kurang Berguna
# masukkan gambar ss manipulasi_data_hapuskolom

#### Mengkonversi Kolom `price_in_euros`
# masukkan gambar ss manipulasi price

#### Mengkonversi Tipe Data Kolom `ram` & `weight`
# masukkan gambar ss manipulasi tipe data

Tujuan:
1. Mengubah tipe data kolom ram dari string ke integer dan menghapus 'GB' pada isi data kolomnya.
2. Mengubah tipe data kolom weight dari string ke float dan menghapus 'kg' pada isi data kolomnya.
3. Hal ini dilakukan karena nanti dalam melakukan pencarian untuk implementasi machine learningnya hanya akan memakai angka saja.

## Keadaan Data
#### Pengecekan Missing Value
Missing value
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

#### Pengecekan Duplikat
- Duplikasi data
* df.duplicated().sum() *
28
Ada 28 duplikasi pada data dan akan dilakukan drop duplikasi pada data preparation.

## EDA
### Distribusi Harga Laptop (IDR)
# masukkin visualisasi_1

🧾 Jenis Visualisasi:
- Menggunakan histogram dengan Kernel Density Estimation (KDE).
- Tools: seaborn.histplot() — histogram menunjukkan frekuensi harga, sedangkan KDE (garis lengkung hijau) memperkirakan bentuk distribusi harga secara halus.

| Temuan                                    | Implikasi                                                      |
| ----------------------------------------- | -------------------------------------------------------------- |
| Harga didominasi laptop kelas menengah    | Mayoritas laptop di dataset ini berada di segmen konsumen umum |
| Distribusi miring ke kanan (right-skewed) | Perlu pertimbangan log transformasi jika ingin modeling harga  |
| Ada outlier harga tinggi                  | Perlu hati-hati agar tidak bias model prediksi harga           |

### Distribusi Untuk Kolom RAM, Ukuran Layar dan Berat
# masukkan visualisasi_2

### 📊 Visualisasi Distribusi Fitur Numerik

#### 1. 📊 Distribusi RAM (GB)
- Mayoritas laptop memiliki **RAM 8 GB**, terlihat dari batang histogram tertinggi.
- Ada kelompok signifikan pada **4 GB** dan **16 GB**.
- RAM ekstrem seperti **32 GB** atau **64 GB** sangat jarang (outlier).
- Distribusi **tidak normal** dan **skewed ke kanan** (ada nilai tinggi yang jarang muncul).

**💡 Kesimpulan**:  
Laptop dengan RAM **8 GB** paling umum dan cocok untuk penggunaan sehari-hari.  
RAM di atas **16 GB** merupakan outlier, umumnya ditemukan pada laptop gaming atau workstation.

---

#### 2. 📺 Distribusi Ukuran Layar (Inches)
- Ukuran layar paling umum adalah **15.6 inci**.
- Diikuti oleh **14 inci** dan **13.3 inci**.
- Ukuran sangat kecil (**<12"**) dan sangat besar (**>17"**) jarang ditemukan.

**💡 Kesimpulan**:  
Sebagian besar laptop memiliki ukuran layar **mainstream**, cocok untuk kebutuhan umum.  
Ukuran ekstrem adalah **niche market** (misalnya laptop mini atau gaming ekstrem).

---

#### 3. ⚖️ Distribusi Berat Laptop (Kg)
- Berat paling umum adalah sekitar **2.0 kg**.
- Laptop ringan di bawah **1.5 kg** (ultrabook) dan berat di atas **3 kg** (laptop gaming) jarang ditemukan.
- Distribusi mendekati **normal (simetris)**, namun terdapat beberapa outlier hingga **4.7 kg**.

**💡 Kesimpulan**:  
Sebagian besar laptop cukup **portabel (1.5–2.5 kg)**.  
Laptop yang sangat ringan atau berat merupakan kasus khusus.

---

### 📌 Kesimpulan Keseluruhan Visualisasi:

| **Fitur**        | **Umum**              | **Jarang / Outlier**     |
|------------------|------------------------|---------------------------|
| **RAM**          | 8 GB                   | 32 GB, 64 GB              |
| **Ukuran layar** | 15.6", 14", 13.3"      | <12" atau >17"            |
| **Berat**        | 2.0–2.5 kg             | >3.5 kg atau <1.2 kg      |

### Distribusi Untuk Kolom Produk per Merek & OS
# masukkan visualisasi_3_1 dan 3_2.

1. **Visualisasi Pertama**
- 📊 Jenis Visualisasi:
    - Countplot dari seaborn: Menampilkan frekuensi kemunculan tiap nilai unik pada kolom company.
    - Disusun berdasarkan urutan terbanyak ke paling sedikit.
🧾 Insight Utama dari Grafik:
- 🥇 5 Merek Teratas Paling Banyak Produk:
    - Dell, Lenovo, dan HP mendominasi jumlah produk, masing-masing hampir 300-an unit.
    - Disusul oleh Asus dan Acer, masing-masing sekitar 150 dan 100 produk.
    - Kelima brand ini mencakup mayoritas total laptop di dataset, mencerminkan pangsa pasar besar mereka.
- 📉 Merek Menengah & Kecil
    - MSI, Toshiba, Apple memiliki jumlah produk jauh lebih sedikit (di bawah 60).
    - Merek seperti Samsung, Razer, Mediacom, Microsoft, Xiaomi, Vero, dan lainnya memiliki jumlah sangat sedikit (kurang dari 20).
- 🧭 Dominasi Pasar Laptop
    - Grafik ini bisa mencerminkan dominasi brand-brand mainstream & enterprise di pasar.
    - Brand seperti Apple walaupun populer, hanya punya sedikit produk dalam dataset, kemungkinan karena model mereka lebih sedikit tapi spesifik.

---

2. **Visualisasi Kedua**
- 📊 Jenis Visualisasi
    - Menggunakan countplot dari Seaborn: menghitung frekuensi/kemunculan tiap OS.
    - Disusun berdasarkan jumlah produk terbanyak ke paling sedikit.
🧾 Insight Utama dari Grafik:
- 🥇 Windows 10 Sangat Dominan
    - Lebih dari 1000 laptop (sekitar 80%+) menggunakan Windows 10.
    - Ini menunjukkan bahwa Windows 10 adalah OS paling populer dalam dataset — wajar karena stabil, kompatibel, dan banyak digunakan secara global.
- 🥈 OS Lainnya: Minoritas
    - No OS, Linux, dan Windows 7 hanya memiliki jumlah kecil (masing-masing sekitar 50–100).
    - Chrome OS, macOS, Mac OS X, Windows 10 S, dan Android hampir tidak terlihat — kemungkinan hanya beberapa unit saja.
- 🧯 OS Non-Windows Sangat Sedikit
    - macOS & Mac OS X totalnya sangat sedikit → karena hanya digunakan di produk Apple (yang jumlahnya juga sedikit dalam dataset).
    - Linux dan Chrome OS juga minim, meskipun banyak digunakan dalam bidang tertentu (developer, pendidikan).
### Korelasi Antar Fitur Numerik
# masukkan visualisasi 3_3

**Visualisasi Korelasi Fitur Numerik**

- 🧾 Penjelasan Insight Utama dari Heatmap:

| Pasangan Fitur            | Nilai Korelasi | Interpretasi                                                                 |
| ------------------------- | -------------- | ---------------------------------------------------------------------------- |
| `price_in_idr` & `ram`    | **0.74**       | Korelasi **positif kuat** → Semakin besar RAM, semakin mahal harga laptop.   |
| `price_in_idr` & `weight` | 0.21           | Korelasi **lemah positif** → Berat punya sedikit pengaruh ke harga.          |
| `price_in_idr` & `inches` | 0.067          | Korelasi **sangat lemah** → Ukuran layar hampir tidak memengaruhi harga.     |
| `ram` & `weight`          | 0.39           | Korelasi **sedang** → RAM lebih besar cenderung sedikit menambah berat.      |
| `ram` & `inches`          | 0.24           | Korelasi **lemah positif** → RAM sedikit lebih besar jika layar lebih besar. |
| `weight` & `inches`       | **0.83**       | Korelasi **positif sangat kuat** → Layar besar = laptop lebih berat.         |

- 🧠 Kesimpulan Analisis Korelasi:
    - Untuk memprediksi harga, prioritaskan fitur ram.
    - Fitur inches dan weight lebih cocok untuk klasifikasi berdasarkan portabilitas atau ukuran daripada prediksi harga.
    - Korelasi tinggi antar weight dan inches menunjukkan kemungkinan multikolinearitas, yang perlu diperhatikan dalam model regresi.
📎 **Catatan**:  
Visualisasi ini membantu kita memahami **karakteristik umum** dari data dan mengidentifikasi **outlier** potensial, yang penting untuk pengambilan keputusan dalam preprocessing dan modeling.

### Insight EDA
📊 EDA Untuk Dataset Laptop
1. Distribusi Fitur Numerik:
- RAM (GB):
    - Mayoritas laptop memiliki RAM 4GB dan 8GB, disusul oleh 16GB.
    - Hanya sedikit laptop yang punya RAM di atas 32GB.
    - Distribusi ini menunjukkan bahwa pasar didominasi oleh laptop kelas menengah.
- Ukuran Layar (Inch):
    - Didominasi oleh ukuran 15.6 inci, kemudian 14 inci dan 13.3 inci.
    - Ukuran layar besar seperti 17 inci cukup jarang.
- Berat (Weight):
    - Distribusi berat laptop berbentuk normal dengan puncak pada kisaran 2 – 2.5 kg.
    - Hanya sedikit laptop yang sangat ringan (< 1.5 kg) atau sangat berat (> 3 kg).

---

2. Distribusi Kategorikal:
- Merek (Company):
    - 3 merek terbesar: Dell, Lenovo, dan HP – masing-masing dengan lebih dari 250 produk.
    - Brand seperti Huawei, LG, Chuwi, Vero sangat jarang muncul.
    - Dell menjadi pemain dominan dalam data.
- Sistem Operasi (OS):
    - Windows 10 mendominasi secara absolut.
    - OS lain seperti Linux, Chrome OS, atau macOS hanya sebagian kecil dari total data.
    - Beberapa laptop dijual tanpa OS.

---

3. Korelasi Antar Fitur Numerik:
- RAM memiliki korelasi tinggi dengan harga (0.74) → semakin besar RAM, semakin mahal harga laptop.
- Ukuran layar dan berat sangat berkorelasi (0.83) → layar besar biasanya menambah bobot.
- Bobot laptop tidak terlalu memengaruhi harga.

---

✅ Insight Utama:
1. Laptop dengan RAM besar cenderung lebih mahal, jadi RAM adalah fitur penting dalam prediksi harga.
2. Mayoritas laptop di pasaran berada pada kelas menengah (RAM 4–8GB, ukuran 14–15.6 inci, berat sekitar 2 kg).
3. Windows 10 adalah sistem operasi yang paling umum – penting untuk strategi distribusi OS.
4. Ukuran layar dan berat sangat terkait, tetapi tidak terlalu penting untuk memprediksi harga.


## Data Preparation
### Feature Engineering
1. Membuat Data storage
# masukkan screenshot FE_1
🎯 Tujuan Feature Engineering Ini:
Membuat data storage bisa dianalisis secara kuantitatif (numerik), bukan sekadar teks.

Memungkinkan agar dapat melakukan:
- Membandingkan kapasitas SSD vs HDD
- Melakukan visualisasi atau korelasi dengan harga (price_in_idr)
- Memasukkan ssd, hdd, flash, hybrid sebagai fitur numerik ke model machine learning

2. Mengubah Nama Kolom
# masukkan ss FE_2
🎯 Tujuan:
- Mengubah teks panjang CPU dan GPU menjadi merek yang lebih ringkas dan bisa digunakan untuk kategorisasi.
- Menyediakan angka total storage agar bisa digunakan dalam analisis numerik.

---

💡 Manfaat pada data:
- Lebih bersih
- Lebih informatif
- Siap untuk visualisasi, analisis statistik, atau model machine learning.

3. Menghapus Kolom
# masukkan ss FE_3

🎯 Tujuan:
- Kolom-kolom ini dihapus karena sudah dipecah dan disederhanakan ke dalam fitur baru yang:
- Lebih mudah dianalisis
- Lebih berguna untuk visualisasi dan modeling
- Mengurangi redundansi dan kebingungan saat eksplorasi data

4. Memisahkan Kolom
# masukkan ss_fe_4

| Komponen | Fungsi                                                     |
| -------- | ---------------------------------------------------------- |
| `y`      | Target: harga laptop dalam IDR                             |
| `X`      | Fitur: atribut-atribut laptop selain harga dan nama produk |

Langkah ini adalah standar dalam supervised learning untuk memisahkan input dan output, sehingga data siap digunakan untuk pelatihan model.

5. Melakukan Encoding pada Fitur Kategorikal
# masukkan ss_fe_5

🎯 Tujuan:
- Kolom-kolom kategorikal diubah menjadi angka (biner)
- Dataset X sekarang sepenuhnya numerik, dan bisa langsung digunakan untuk modeling
- Teknik ini disebut One-Hot Encoding, dan sangat umum dipakai sebelum melatih model regresi, decision tree, random forest, dll.

#### Insight Feature Engineering
🧠 Tahap Data Preparation – Kesimpulan
Setelah melalui tahap data preparation, dataset kini telah:

##### ✅ Bersih dan Konsisten:
    - Nilai teks diseragamkan (lowercase).
    - Kolom yang tidak relevan atau redundan (cpu, gpu, memory, product) telah dihapus.
---

##### 🦾 Mengandung Fitur Baru (Feature Engineering):
    - Storage dipecah menjadi: ssd, hdd, flash, hybrid, dan total_storage.
    - Merek CPU dan GPU diambil dari string mentah menjadi kolom cpu_brand dan gpu_brand.
---

##### 🔢 Berisi Fitur Numerik Siap Pakai
    - Kolom kategorikal seperti company, opsys, dll sudah diubah menjadi kolom numerik melalui One-Hot Encoding.
    - Dataset X sekarang hanya terdiri dari angka, cocok untuk digunakan oleh algoritma machine learning.
---

##### 🎯 Target Sudah Ditentukan
    - Target prediksi (price_in_idr) telah dipisahkan dalam variabel y.

### Standardization
1. Melakukan Standarisasi pada Kolom Numerik
# Masukkan ss_standar_1

##### 🔍 Tujuan Scaling / Standardisasi:
    - Agar semua fitur numerik memiliki skala yang seragam, yaitu:
      - Mean = 0
      - Standard Deviation = 1
---

##### 🚨 Ini penting karena:
Beberapa algoritma machine learning (seperti Linear Regression, KNN, SVM) sangat sensitif terhadap perbedaan skala antar fitur.

Misalnya: fitur weight bisa punya nilai 1–4, sedangkan total_storage bisa ratusan hingga ribuan. Tanpa scaling, fitur dengan angka besar akan dominan dan menyesatkan model.

### IQR
**IQR** adalah konsep statistik yang terkait dengan distribusi data, dan penggunaannya untuk outlier adalah salah satu aplikasi utamanya. IQR mewakili rentang nilai yang mencakup 50% bagian tengah data Anda ketika diurutkan. Ini adalah ukuran penyebaran data yang "tahan" terhadap nilai-nilai ekstrem.

- Kuartil Pertama (Q1): Nilai di bawahnya terletak 25% data.
- Kuartil Ketiga (Q3): Nilai di bawahnya terletak 75% data (atau 25% data terletak di atasnya).
- IQR: Adalah perbedaan antara Kuartil Ketiga (Q3) dan Kuartil Pertama (Q1). IQR = Q3 - Q1

1. Melakukan Pengecekan Outlier pada Dataset X
# masukkan ss_iqr_1

##### 📌 Pada kode cell ini dilakukan pengecekan outlier pada data X dengan teknik *IQR Outlier*.

Berhasil menjalankan dan menghasilkan insight berupa:
- Outlier (ram): 219 outliers
- Outlier (inches): 37 outliers
- Outlier (weight): 45 outliers

Ini menunjukkan adanya outlier pada kolom-kolom tersebut.

2. Melakukan Penghapusan Outlier pada Dataset X
# masukkan ss_iqr_2

##### 🧼 Melakukan penghapusan nilai outlier yang ada pada data X dengan menggunakan *IQR drop Outlier*.

Hal ini berhasil dilakukan dan menghasilkan insight seperti berikut:

After Drop Outliers
- Outlier (ram): 0 outliers
- Outlier (inches): 0 outliers
- Outlier (weight): 0 outliers

3. Melakukan Penggantian Outlier pada Dataset Y
# masukkan ss_iqr_3

##### 🧼 Melakukan penggantian nilai outlier dengan menggunakan rata-rata yang ada pada data y_clean dengan menggunakan *zscore*.

Hal ini berhasil dilakukan dan menghasilkan insight seperti berikut:
- Jumlah outlier yang diganti: `12`
- Rata-rata yang digunakan: `19358061.46`

### Missing Value
# masukkan ss_dropmiss

##### 🧼 Penanganan Data Kosong
Untuk memastikan dataset `X` yang berisikan atribut pada laptop dan `y_clean` yang berisi `target` tidak memiliki nilai kosong yang dapat mengganggu proses training model, digunakan metode `dropna()` untuk menghapus seluruh baris yang mengandung nilai `NaN`. Ini menjamin bahwa dataset bersih dan siap digunakan untuk proses machine learning selanjutnya.

### Train, Test dan Split
1. Melakukan Split Data pada X dan y_clean
# masukkan ss_splt_1

##### 🎯 Tujuan Train-Test Split:
    - Untuk melatih model pada sebagian data (train) dan mengujinya pada data yang belum pernah dilihat (test).
    - Membantu menilai kinerja generalisasi model, apakah model bekerja baik di data nyata yang tidak pernah dilatih sebelumnya.

---

##### 🔀 Train-Test Split
    Dataset telah dibagi menjadi:
    - **Training Set** (80%) sebanyak 1.020 data
    - **Testing Set** (20%) sebanyak 255 data
---
Pembagian dilakukan secara acak menggunakan `train_test_split()` dari `scikit-learn`, dengan `random_state=42` untuk memastikan hasil yang konsisten. Ini memungkinkan evaluasi performa model secara adil pada data yang tidak pernah dilatih sebelumnya.

2. Melakukan Visualisasi pada Split Data
# Masukkan ss_split_2

#### 📈 Distribusi Harga Laptop pada Train dan Test Set

Grafik menunjukkan distribusi harga laptop untuk data latih (`Train Set`) dan data uji (`Test Set`) setelah dilakukan pembagian dengan `train_test_split`.

---

##### 📊 Penjelasan Grafik:

1. **Histogram dan KDE (Kernel Density Estimation)**
   - **Histogram biru**: menunjukkan distribusi harga laptop dari data latih (`y_train_ts`).
   - **Histogram merah**: menunjukkan distribusi harga laptop dari data uji (`y_test_ts`).
   - **Garis lengkung KDE**: menggambarkan estimasi kepadatan data untuk menunjukkan tren distribusi secara halus.


2. **Distribusi Positif (Right-skewed)**
   - Kedua dataset (train dan test) menunjukkan distribusi yang **miring ke kanan**.
   - Artinya, sebagian besar harga laptop berada di kisaran rendah hingga menengah, dan hanya sebagian kecil yang sangat mahal.


3. **Pola Distribusi Mirip**
   - Distribusi pada train dan test terlihat **konsisten dan serupa**, yang menunjukkan:
     - Data uji mewakili karakteristik data latih secara proporsional.
     - Pembagian data dilakukan dengan baik dan **tidak bias**, serta tidak terjadi **data leakage**.
---

##### 🎯 Tujuan Visualisasi Ini:
    - **Memvalidasi pembagian data train dan test** secara acak.
    - Memastikan bahwa model nantinya akan diuji pada data yang memiliki distribusi mirip dengan data latih.
    - Menghindari **overfitting** atau **underfitting** akibat ketidakseimbangan distribusi antara train dan test set.

---

##### ✅ **Kesimpulan**:
Distribusi harga laptop pada data latih dan data uji menunjukkan kemiripan pola yang baik. Ini mengindikasikan bahwa proses pembagian data telah dilakukan secara representatif, yang menjadi dasar penting sebelum melatih model regresi.

## Modelling: Regression
**🚨 Karena kita akan membuat model *Predictive Modeling* `Price Laptop` yang berhubungan dengan mengetahui data perkiraan terkait apa yang mungkin terjadi pada data yang telah saya dapatkan.**

1. Pelatihan Model
# masukkan ss_model_1

##### 🧠 Tujuan Umum
Kode ini digunakan untuk **melatih**, **menguji**, dan **mengevaluasi kinerja model machine learning** dalam memprediksi harga laptop (`price_in_idr`).

---

##### ✅ Kesimpulan
Fungsi `evaluate_model()` penting karena:
- 📏 Melakukan **evaluasi model regresi** menggunakan metrik yang umum dipakai, yaitu:
  - **R² (R-squared)**: menunjukkan seberapa baik model menjelaskan variansi data.
  - **RMSE (Root Mean Squared Error)**: menunjukkan seberapa jauh prediksi dari nilai aktual.

- 📊 Memungkinkan untuk **membandingkan beberapa model** dengan cara yang sistematis dan otomatis.

- 📈 Menyimpan hasil evaluasi ke dalam list `results`, sehingga dapat digunakan untuk **analisis lanjutan atau visualisasi perbandingan model**.

---

2. Pencarian Model Terbaik
# masukkan ss_model_2

##### 🧠 Tujuan Kode
Kode ini bertujuan untuk **menginisialisasi beberapa model regresi** yang akan digunakan dalam proses pelatihan dan evaluasi terhadap data fitur laptop untuk memprediksi `price_in_idr`.

---

| Model               | Penjelasan Singkat                                                                 |
|---------------------|------------------------------------------------------------------------------------|
| `LinearRegression()` | Regresi linear biasa, digunakan sebagai baseline.                                 |
| `Ridge()`            | Regresi linear dengan regularisasi **L2**, menghindari overfitting.               |
| `Lasso()`            | Regresi linear dengan regularisasi **L1**, bisa menyusutkan beberapa koefisien ke nol (fitur seleksi). |
| `ElasticNet()`       | Kombinasi dari Lasso dan Ridge (menggabungkan penalti **L1** dan **L2**).         |

---

##### 🧠 Kesimpulan

Kamu sedang menyiapkan berbagai jenis **model regresi dengan karakteristik yang berbeda**, untuk:
- Membandingkan performanya terhadap **prediksi harga laptop**.
- Menentukan model **mana yang paling akurat dan generalizable**.
- Memahami perbedaan antara **model linear, tree-based, dan ensemble-based** terhadap dataset ini.

### Evaluation
1. Review Model
# masukkan ss_model_3

##### ✅ Tujuan Kode

Kode ini digunakan untuk:

- Melatih dan mengevaluasi semua model regresi yang telah disiapkan.
- Menghitung performa setiap model menggunakan dua metrik utama:
  - **R² (R-squared)**: Mengukur seberapa baik model menjelaskan varians dalam data.
  - **RMSE (Root Mean Squared Error)**: Mengukur rata-rata kesalahan prediksi dalam satuan harga (Rupiah).

---

##### 🧪 Hasil Evaluasi Model

| Model               | R² Score | RMSE (Rp)      | Interpretasi                                                                 |
|---------------------|----------|----------------|------------------------------------------------------------------------------|
| Linear Regression    | 0.65     | 6.38 juta       | Baseline yang cukup baik. Model linear mampu menjelaskan 74% varians harga. |
| Ridge Regression     | 0.65     | 6.38 juta       | Sedikit lebih baik, regularisasi L2 membantu menghindari overfitting.       |
| Lasso Regression     | 0.65     | 6.38 juta       | Mirip Linear, tetapi dengan penalti L1 yang juga bisa melakukan seleksi fitur. |
| ElasticNet Regression| 0.47     | 7.81 juta       | Performa kurang baik, penalti gabungan L1 dan L2 kurang cocok di sini.      |
| XGBoost Regressor    | **0.74** | **5.47 juta**   | ✅ Model terbaik. Menangkap hubungan non-linear dan hasil prediksi paling akurat. |

---

##### 🎯 Interpretasi Metrik

- **R² (Koefisien Determinasi):**
  - Nilai 1.0 berarti prediksi sempurna.
  - Semakin mendekati 1, semakin baik.
  - Urutan performa RMSE: **XGBoost (0.74) > Ridge/Linear/Lasso (0.65) > ElasticNet (0.47)**

- **RMSE (Root Mean Squared Error):**
  - Mengukur rata-rata deviasi prediksi dari nilai sebenarnya.
  - Semakin kecil nilainya, semakin baik performa model.
  - RMSE terbaik dimiliki oleh **XGBoost** dengan **5.47 juta Rupiah**.

---

##### 🧠 Kesimpulan

- 🔧 **XGBoost adalah model terbaik** untuk prediksi harga laptop dalam dataset ini.
- 📉 **ElasticNet underperform**, kemungkinan karena parameter default tidak optimal.
- 🧪 **Linear, Ridge, dan Lasso** masih kuat sebagai baseline, ringan dan mudah diinterpretasi.

2. Pemodelan `XGBoost`
# masukkan ss_model_4

##### 🚀 XGBoost Regressor dengan Parameter Tuning
Model XGBoost digunakan dengan parameter yang telah disesuaikan untuk meningkatkan performa prediksi harga laptop.

---

##### ⚙️ Konfigurasi Parameter
| Parameter          | Nilai | Fungsi                                                     |
| ------------------ | ----- | ---------------------------------------------------------- |
| `n_estimators`     | 500   | Jumlah pohon dalam model boosting                          |
| `max_depth`        | 7     | Kedalaman maksimum tiap pohon                              |
| `learning_rate`    | 0.05  | Kecepatan pembelajaran model boosting                      |
| `subsample`        | 0.8   | Persentase data yang digunakan untuk setiap pohon          |
| `colsample_bytree` | 0.9   | Persentase fitur yang digunakan untuk membangun tiap pohon |
| `gamma`            | 0.1   | Minimum loss reduction untuk membuat split                 |
| `reg_alpha`        | 0.01  | Regularisasi L1 (menghindari overfitting)                  |
| `reg_lambda`       | 0.1   | Regularisasi L2 (menghindari overfitting)                  |
| `random_state`     | 42    | Reproducibility (hasil konsisten)                          |
| `n_jobs`           | -1    | Menggunakan semua core CPU                                 |

---

##### 📊 Evaluasi Performa Model
| Metrik | Nilai      | Interpretasi                                                         |
| ------ | ---------- | -------------------------------------------------------------------- |
| R²     | 0.7619     | Model mampu menjelaskan **76.19%** variasi harga laptop di data uji. |
| RMSE   | Rp 5.25 jt | Rata-rata kesalahan prediksi adalah sekitar **5.25 juta**.           |

---

##### ✅ Kesimpulan
1. Model XGBoost yang telah dituning menunjukkan performa cukup baik, dan berhasil meningkatkan akurasinya dari **0.74** menjadi **0.7619**.
2. RMSE berhasil ditekan dengan nilai awal sebelum tuning **5.47 juta** menjadi **5.25 juta**.

3. Sanity Check
# masukkan ss_model_5
##### ✅ Tujuan Kode

Kode ini digunakan sebagai **langkah validasi (sanity check)** sebelum membuat DataFrame `plot_df` yang akan digunakan untuk visualisasi hasil **prediksi vs. harga aktual**.  
Tujuannya adalah memastikan bahwa semua data yang akan dipakai **benar, konsisten, dan bebas dari error** seperti:

- Missing values (`NaN`)
- Ketidaksesuaian bentuk array
- Ketidaksesuaian indeks antara data prediksi dan data asli (`clean_df`)

---

##### 📌 Kesimpulan

Semua pengecekan telah **lolos**, sehingga:

- ✅ Data prediksi dan data aktual memiliki **jumlah yang sama**  
- ✅ Tidak ada **missing value** dalam `y_test_ts` maupun `y_pred_test_ts`  
- ✅ Nilai-nilai berada dalam **kisaran harga wajar**  
- ✅ **Indeks data uji cocok** dengan indeks `clean_df`, artinya aman untuk digabung

> 💡 Artinya, data siap digunakan untuk membuat visualisasi hasil prediksi harga laptop secara akurat.

4. Visualisasi Prediksi Harga Laptop
# masukkan ss_model_6

### 📈 Grafik: Prediksi Harga Laptop vs Nilai Aktual (Test Set)

Grafik ini menampilkan perbandingan visual antara **harga laptop sebenarnya** dan **hasil prediksi dari model XGBoost** pada data uji (Test Set).

---

##### 🧾 Detail Grafik:

- **Garis Biru** (`Actual Price`):  
  Menunjukkan harga laptop sebenarnya dari dataset.

- **Garis Oranye** (`Predicted Price (XGB)`):  
  Menunjukkan hasil prediksi harga dari model XGBoost.

- **Sumbu X (Index)**:  
  Merupakan indeks dari data dalam test set — urutan data, bukan waktu.

- **Sumbu Y (Harga - Rupiah)**:  
  Mewakili nilai harga laptop (dalam satuan Rupiah).

---

##### 🔍 Interpretasi Visual:

- ✅ **Polanya mirip**  
  Prediksi mengikuti pola umum dari data aktual dengan cukup baik, menunjukkan bahwa model berhasil mempelajari tren data.

- ✅ **Akurasi keseluruhan baik**  
  Sebagian besar prediksi berada dekat dengan nilai aktual.

---

##### 🎯 Kesimpulan:

Grafik ini menunjukkan bahwa **model XGBoost cukup efektif dalam memprediksi harga laptop**, dengan pola prediksi yang mengikuti harga aktual secara keseluruhan.

> 💡 Visualisasi seperti ini sangat membantu untuk **mengevaluasi stabilitas dan generalisasi model** sebelum digunakan dalam sistem nyata.


## Create Plot
1. Membuat `plot_df`
# masukkan ss_cp_1

##### ✅ Tujuan Kode
Kode ini digunakan untuk **menyiapkan DataFrame baru (`plot_df`)** yang berisi:

- **Harga aktual** (`Actual Price`)
- **Harga hasil prediksi** dari model XGBoost (`Predicted Price`)
- **Indeks data** untuk keperluan visualisasi (plot)

---

##### 🧱 Penjelasan Struktur `plot_df`

DataFrame `plot_df` dibuat dengan 3 kolom utama:

| Kolom             | Deskripsi                                                                 |
|-------------------|---------------------------------------------------------------------------|
| `Index`           | Urutan angka dari 0 sampai jumlah data test (sebagai sumbu X pada grafik) |
| `Actual Price`    | Nilai harga sesungguhnya dari `y_test_ts`                                 |
| `Predicted Price` | Nilai hasil prediksi dari model XGBoost (`y_pred_test_ts`)                |

---

##### 🖨️ Menampilkan `ukuran DataFrame`:


Menampilkan **5 baris pertama** dari DataFrame `plot_df shape: (255, 3)`:
- **255 baris** = jumlah sampel dalam data test.
- **3 kolom** = `Index`, `Actual Price`, dan `Predicted Price`.

---

##### 🧠 Kesimpulan

Data `plot_df` ini **sudah siap digunakan** untuk:

- Membuat **visualisasi** perbandingan antara harga aktual dan harga prediksi.
- Mempermudah **analisis kesalahan prediksi** (under/over prediction) oleh model.
- Meningkatkan **interpretabilitas hasil model** secara visual.

2. Memahami `plot_df`
# masukkan ss_cp_2

##### 🧾 Rincian:

| Informasi           | Keterangan                                                                                                |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `RangeIndex`        | DataFrame memiliki 255 baris, dengan indeks dari 0 hingga 254                                             |
| Kolom total         | Ada 3 kolom: `Index`, `Actual Price`, dan `Predicted Price`                                               |
| Non-Null Count      | Semua kolom memiliki **255 data** (tidak ada nilai yang hilang / NaN)                                     |
| Tipe Data (`dtype`) | - `Index`: `int64` (bilangan bulat)  <br> - `Actual Price`: `float64` <br> - `Predicted Price`: `float32` |
| Penggunaan Memori   | Total memori yang digunakan adalah sekitar **5.1 KB**                                                     |


3. Visualisasi Laptop Price Predicted
# masukkan ss_cp_3

##### 🔧 Kode Visualisasi:
Visualisasi ini dibuat menggunakan plotly.graph_objects.

##### 📌 Detail Grafik:

| Elemen                      | Penjelasan                                                            |
| --------------------------- | --------------------------------------------------------------------- |
| **Sumbu X (Index)**         | Indeks dari data dalam test set (urutan data, bukan tanggal/waktu).   |
| **Sumbu Y (Price - IDR)**   | Harga laptop dalam satuan Rupiah.                                     |
| **Garis Biru**              | Harga aktual dari data uji (`Actual Price`).                          |
| **Garis Merah Putus-putus** | Prediksi harga dari model XGBoost (`Predicted Price`).                |
| **Titik Marker**            | Menambahkan detail per titik, berguna saat hover (interaktif).        |
| **Hovermode: x unified**    | Saat hover, semua nilai pada titik X yang sama akan muncul bersamaan. |

---

##### 🔍 Interpretasi Visual:
- ✅ Polanya konsisten
Garis merah (prediksi) mengikuti pola dari garis biru (aktual), menandakan model menangkap tren harga cukup baik.

- ✅ Detail interaktif
Hovermode memudahkan pengguna membandingkan harga aktual dan prediksi secara langsung untuk setiap data.
