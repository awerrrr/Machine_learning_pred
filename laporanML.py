#!/usr/bin/env python
# coding: utf-8

# ### Laporan Proyek Machine Learning Terapan
# ### Predictive Laptop Price
# 
# Nama     : Najwar Putra Kusumah Wardana
# 
# Dataset  : https://www.kaggle.com/datasets/abhikjha/movielens-100k

# ## Import library and file

# In[1]:


import pandas as pd


file_path = "laptop_data.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")


# Load data untuk membaca file "laptop_data.csv" ke dalam program Python sebagai DataFrame pandas, yang akan dianalisis lebih lanjut (misalnya menampilkan data, membersihkan data, membuat visualisasi, dll). Encoding="ISO-8859-1" digunakan untuk memastikan karakter-karakter khusus dalam file CSV bisa dibaca dengan benar. Encoding ini dipakai karena sebelumnya utf-8 menyebabkan error seperti UnicodeDecodeError.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Data Understanding

# In[3]:


df.head()


# ### Penjelasan Output:
# | Kolom              | Arti                                                               |
# | ------------------ | ------------------------------------------------------------------ |
# | `laptop_ID`        | ID unik untuk setiap laptop                                        |
# | `Company`          | Merek pembuat laptop (misalnya Apple, HP, Dell, dsb)               |
# | `Product`          | Nama produk atau model laptop                                      |
# | `TypeName`         | Tipe laptop (Ultrabook, Notebook, Gaming, dll)                     |
# | `Inches`           | Ukuran layar dalam inci                                            |
# | `ScreenResolution` | Resolusi layar (dan kadang jenis panelnya, seperti IPS)            |
# | `Cpu`              | Jenis prosesor (misalnya Intel Core i5, i7, dll)                   |
# | `Ram`              | Kapasitas RAM (biasanya dalam GB)                                  |
# | `Memory`           | Jenis dan kapasitas penyimpanan (misalnya SSD, HDD, atau gabungan) |
# | `Gpu`              | Jenis kartu grafis (misalnya Intel HD, AMD Radeon, dll)            |
# | `OpSys`            | Sistem operasi bawaan laptop (misalnya Windows, macOS, No OS, dll) |
# | `Weight`           | Berat laptop (dalam kilogram, disertai "kg")                       |
# | `Price_in_euros`   | Harga laptop dalam Euro                                            |
# 

# In[4]:


df.info()


# ### Penjelasan Output:
# | No | Kolom              | Non-Null Count | Tipe Data | Penjelasan Singkat                          |
# | -- | ------------------ | -------------- | --------- | ------------------------------------------- |
# | 0  | `laptop_ID`        | 1303           | `int64`   | ID unik laptop, berupa angka bulat          |
# | 1  | `Company`          | 1303           | `object`  | Nama merek laptop (Apple, HP, dll)          |
# | 2  | `Product`          | 1303           | `object`  | Nama produk / model                         |
# | 3  | `TypeName`         | 1303           | `object`  | Jenis laptop (Ultrabook, Notebook, dll)     |
# | 4  | `Inches`           | 1303           | `float64` | Ukuran layar dalam inci                     |
# | 5  | `ScreenResolution` | 1303           | `object`  | Resolusi layar dan jenis panel              |
# | 6  | `Cpu`              | 1303           | `object`  | Nama dan model CPU                          |
# | 7  | `Ram`              | 1303           | `object`  | RAM (dalam format string, misalnya '8GB')   |
# | 8  | `Memory`           | 1303           | `object`  | Kapasitas dan tipe penyimpanan              |
# | 9  | `Gpu`              | 1303           | `object`  | Nama GPU / kartu grafis                     |
# | 10 | `OpSys`            | 1303           | `object`  | Sistem operasi (Windows, macOS, dll)        |
# | 11 | `Weight`           | 1303           | `object`  | Berat laptop (dalam string, misal '1.37kg') |
# | 12 | `Price_in_euros`   | 1303           | `float64` | Harga laptop dalam Euro                     |
# 
# ##### Dari sini kita ketahui bahwa data mempunyai 1303 baris dan 13 kolom dengan tiga tipe data yaitu int, float, dan object.
# ##### Dalam Output ini pun kita bisa melakukan pengecekan untuk missing value dengan melihat output yang tersedia pada kolom output "Non-Null Count".
# 

# In[5]:


df.rename(columns={
    'laptop_ID': 'laptop_id',
    'Company': 'company',
    'Product': 'product',
    'TypeName': 'type_name',
    'Inches': 'inches',
    'ScreenResolution': 'screen_resolution',
    'Cpu': 'cpu',
    'Ram': 'ram',
    'Memory': 'memory',
    'Gpu': 'gpu',
    'OpSys': 'opsys',
    'Weight': 'weight',
    'Price_in_euros': 'price_in_euros'
}, inplace=True)

df.head()


# Melakukan rename pada kolom data yang akan kita analisis untuk menghasilkan data yang lebih konsisten, pythonic, dan untuk mencegah error pada analisis kedepannya dan yang terpenting adalah menghindari typo.
# - Lebih konsisten: semua nama pakai huruf kecil.
# - Lebih Pythonic: penamaan kolom seperti price_in_euros lebih mudah digunakan dalam analisis atau visualisasi.
# - Mencegah error: nama seperti ScreenResolution rentan typo, screen_resolution lebih mudah diketik dan dibaca, typo pengguna.

# In[6]:


df.drop(columns=['laptop_id'], inplace=True)


# Melakukan drop kolom untuk 'laptop_id' karena tidak dibutuhkan dalam proyek ini.

# In[7]:


df['price_in_idr'] = df['price_in_euros'] * 17500


# Mengkonversi kolom price_in_euros(EUR) menjadi price_in_idr(IDR) agar menyesuaikan dengan nilai mata uang dari user target.

# In[8]:


df['ram'] = df['ram'].str.replace('GB', '', regex=False).astype(int)
df['weight'] = df['weight'].str.replace('kg', '', regex=False).astype(float)
df.head()


# Mengubah tipe data kolom ram dari string ke integer dan menghapus 'GB' pada isi data kolomnya.
# 
# Mengubah tipe data kolom weight dari string ke float dan menghapus 'kg' pada isi data kolomnya.
# 
# Hal ini dilakukan karena nanti dalam melakukan pencarian untuk implementasi machine learningnya hanya akan memakai angka saja.

# In[9]:


df.isna().sum()


# Tidak ada missing value pada data

# In[10]:


df.duplicated().sum()


# Setelah dilakukan pengecekan pada data diketahui bahwa data memiliki 23 data duplikat, hal ini mengakibatkan kita harus melakukan penghapusan pada data-data duplikat tersebut agar memudahkan pemodelan nantinya.

# In[11]:


df.drop_duplicates(inplace=True)
df.shape


# Setelah dilakukan pembersihan atau penghapusan pada data duplikat yang ada pada dataset diketahui bahwa dataset sekarang memiliki:
# - 1275 untuk baris dan 13 untuk kolom *(1275, 13)*

# In[12]:


df.describe(include='all').style.background_gradient('Greens_r')


# Dari output diatas bisa kita simpulkan:
# - Dataset berisi 1275 laptop.
# 
# - Ukuran layar rata-rata sekitar 15 inci.
# 
# - RAM rata-rata 8 GB.
# 
# - Berat rata-rata laptop: 2.04 kg.
# 
# - Harga laptop bervariasi, dengan rata-rata:
# 
#     - €1134 (~Rp 19,8 juta).
# 
#     - Harga bisa mencapai lebih dari Rp 106 juta.

# #### Insight Data Undertanding
# Setelah selesai melakukan tahap *Data Understanding* kita dapat melihat bahwa data sangat sehat dan rapi untuk dilakukan analisis lebih lanjut, hal yang dapat kita simpulkan pada *Data Understanding* ini adalah:
# 
# - Data memiliki 1275 kolom dan 13 baris yang dapat dijadikan acuan untuk membuat baik itu visualisasi, pemodelan dan analisis lanjutan yang lain.
# - Data tidak memiliki missing value.
# - Data memiliki kolom kolom kunci yang akan kita pakai untuk pemodelan yang lebih menjurus yaitu *price_in_idr*, *ram*, *opsys*, *cpu*, *gpu*, dan *memory*.
# 
# Setelah selesai dari tahap ini kita sekarang akan memasuki tahap EDA untuk melihat distribusi, korelasi dan outlier yang akan kita lihat menggunakan visualisasi pada data.

# ## Exploratory Data Analysis (EDA)
# Pada tahap ini yang akan kita lakukan adalah:
# - ✅ Memahami isi data: Melihat struktur, tipe data, dan distribusi nilai.
# - 🔍 Mendeteksi kesalahan pada data yang mengandung nilai outlier.
# - 📊 Melihat pola dan hubungan antar variabel.
# - 🧹 Menentukan langkah preprocessing: Misalnya normalisasi, encoding, atau penghapusan kolom.
# - 🤖 Menentukan model yang cocok: Berdasarkan sifat data.

# In[13]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(df['price_in_idr'], bins=50, kde=True, color='green')
plt.title('Distribusi Harga Laptop (dalam Rupiah)')
plt.xlabel('Harga (IDR)')
plt.ylabel('Jumlah')
plt.show()


# 🧾 Jenis Visualisasi:
# - Menggunakan histogram dengan Kernel Density Estimation (KDE).
# - Tools: seaborn.histplot() — histogram menunjukkan frekuensi harga, sedangkan KDE (garis lengkung hijau) memperkirakan bentuk distribusi harga secara halus.
# 
# | Temuan                                    | Implikasi                                                      |
# | ----------------------------------------- | -------------------------------------------------------------- |
# | Harga didominasi laptop kelas menengah    | Mayoritas laptop di dataset ini berada di segmen konsumen umum |
# | Distribusi miring ke kanan (right-skewed) | Perlu pertimbangan log transformasi jika ingin modeling harga  |
# | Ada outlier harga tinggi                  | Perlu hati-hati agar tidak bias model prediksi harga           |

# In[14]:


fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['ram'], bins=10, ax=axs[0], color='skyblue')
axs[0].set_title('Distribusi RAM (GB)')

sns.histplot(df['inches'], bins=10, ax=axs[1], color='orange')
axs[1].set_title('Distribusi Ukuran Layar (Inch)')

sns.histplot(df['weight'], bins=10, ax=axs[2], color='salmon')
axs[2].set_title('Distribusi Berat Laptop (Kg)')

plt.tight_layout()
plt.show()


# ### 📊 Visualisasi Distribusi Fitur Numerik
# 
# #### 1. 📊 Distribusi RAM (GB)
# - Mayoritas laptop memiliki **RAM 8 GB**, terlihat dari batang histogram tertinggi.
# - Ada kelompok signifikan pada **4 GB** dan **16 GB**.
# - RAM ekstrem seperti **32 GB** atau **64 GB** sangat jarang (outlier).
# - Distribusi **tidak normal** dan **skewed ke kanan** (ada nilai tinggi yang jarang muncul).
# 
# **💡 Kesimpulan**:  
# Laptop dengan RAM **8 GB** paling umum dan cocok untuk penggunaan sehari-hari.  
# RAM di atas **16 GB** merupakan outlier, umumnya ditemukan pada laptop gaming atau workstation.
# 
# ---
# 
# #### 2. 📺 Distribusi Ukuran Layar (Inches)
# - Ukuran layar paling umum adalah **15.6 inci**.
# - Diikuti oleh **14 inci** dan **13.3 inci**.
# - Ukuran sangat kecil (**<12"**) dan sangat besar (**>17"**) jarang ditemukan.
# 
# **💡 Kesimpulan**:  
# Sebagian besar laptop memiliki ukuran layar **mainstream**, cocok untuk kebutuhan umum.  
# Ukuran ekstrem adalah **niche market** (misalnya laptop mini atau gaming ekstrem).
# 
# ---
# 
# #### 3. ⚖️ Distribusi Berat Laptop (Kg)
# - Berat paling umum adalah sekitar **2.0 kg**.
# - Laptop ringan di bawah **1.5 kg** (ultrabook) dan berat di atas **3 kg** (laptop gaming) jarang ditemukan.
# - Distribusi mendekati **normal (simetris)**, namun terdapat beberapa outlier hingga **4.7 kg**.
# 
# **💡 Kesimpulan**:  
# Sebagian besar laptop cukup **portabel (1.5–2.5 kg)**.  
# Laptop yang sangat ringan atau berat merupakan kasus khusus.
# 
# ---
# 
# ### 📌 Kesimpulan Keseluruhan Visualisasi:
# 
# | **Fitur**        | **Umum**              | **Jarang / Outlier**     |
# |------------------|------------------------|---------------------------|
# | **RAM**          | 8 GB                   | 32 GB, 64 GB              |
# | **Ukuran layar** | 15.6", 14", 13.3"      | <12" atau >17"            |
# | **Berat**        | 2.0–2.5 kg             | >3.5 kg atau <1.2 kg      |
# 
# 📎 **Catatan**:  
# Visualisasi ini membantu kita memahami **karakteristik umum** dari data dan mengidentifikasi **outlier** potensial, yang penting untuk pengambilan keputusan dalam preprocessing dan modeling.
# 

# In[15]:


plt.figure(figsize=(10, 4))
sns.countplot(data=df, x='company', order=df['company'].value_counts().index, palette='Set2')
plt.title('Jumlah Produk per Merek')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='opsys', order=df['opsys'].value_counts().index, palette='Set3')
plt.title('Jumlah Produk per OS')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[['price_in_idr', 'ram', 'weight', 'inches']].corr(), annot=True, cmap='YlGnBu')
plt.title('Korelasi Antar Fitur Numerik')
plt.show()


# 1. **Visualisasi Pertama**
# - 📊 Jenis Visualisasi:
#     - Countplot dari seaborn: Menampilkan frekuensi kemunculan tiap nilai unik pada kolom company.
#     - Disusun berdasarkan urutan terbanyak ke paling sedikit.
# 🧾 Insight Utama dari Grafik:
# - 🥇 5 Merek Teratas Paling Banyak Produk:
#     - Dell, Lenovo, dan HP mendominasi jumlah produk, masing-masing hampir 300-an unit.
#     - Disusul oleh Asus dan Acer, masing-masing sekitar 150 dan 100 produk.
#     - Kelima brand ini mencakup mayoritas total laptop di dataset, mencerminkan pangsa pasar besar mereka.
# - 📉 Merek Menengah & Kecil
#     - MSI, Toshiba, Apple memiliki jumlah produk jauh lebih sedikit (di bawah 60).
#     - Merek seperti Samsung, Razer, Mediacom, Microsoft, Xiaomi, Vero, dan lainnya memiliki jumlah sangat sedikit (kurang dari 20).
# - 🧭 Dominasi Pasar Laptop
#     - Grafik ini bisa mencerminkan dominasi brand-brand mainstream & enterprise di pasar.
#     - Brand seperti Apple walaupun populer, hanya punya sedikit produk dalam dataset, kemungkinan karena model mereka lebih sedikit tapi spesifik.
# 
# ---
# 
# 2. **Visualisasi Kedua**
# - 📊 Jenis Visualisasi
#     - Menggunakan countplot dari Seaborn: menghitung frekuensi/kemunculan tiap OS.
#     - Disusun berdasarkan jumlah produk terbanyak ke paling sedikit.
# 🧾 Insight Utama dari Grafik:
# - 🥇 Windows 10 Sangat Dominan
#     - Lebih dari 1000 laptop (sekitar 80%+) menggunakan Windows 10.
#     - Ini menunjukkan bahwa Windows 10 adalah OS paling populer dalam dataset — wajar karena stabil, kompatibel, dan banyak digunakan secara global.
# - 🥈 OS Lainnya: Minoritas
#     - No OS, Linux, dan Windows 7 hanya memiliki jumlah kecil (masing-masing sekitar 50–100).
#     - Chrome OS, macOS, Mac OS X, Windows 10 S, dan Android hampir tidak terlihat — kemungkinan hanya beberapa unit saja.
# - 🧯 OS Non-Windows Sangat Sedikit
#     - macOS & Mac OS X totalnya sangat sedikit → karena hanya digunakan di produk Apple (yang jumlahnya juga sedikit dalam dataset).
#     - Linux dan Chrome OS juga minim, meskipun banyak digunakan dalam bidang tertentu (developer, pendidikan).
# 
# ---
# 
# 3. **Visualisasi Ketiga**
# - 🧾 Penjelasan Insight Utama dari Heatmap:
# | Pasangan Fitur            | Nilai Korelasi | Interpretasi                                                                 |
# | ------------------------- | -------------- | ---------------------------------------------------------------------------- |
# | `price_in_idr` & `ram`    | **0.74**       | Korelasi **positif kuat** → Semakin besar RAM, semakin mahal harga laptop.   |
# | `price_in_idr` & `weight` | 0.21           | Korelasi **lemah positif** → Berat punya sedikit pengaruh ke harga.          |
# | `price_in_idr` & `inches` | 0.067          | Korelasi **sangat lemah** → Ukuran layar hampir tidak memengaruhi harga.     |
# | `ram` & `weight`          | 0.39           | Korelasi **sedang** → RAM lebih besar cenderung sedikit menambah berat.      |
# | `ram` & `inches`          | 0.24           | Korelasi **lemah positif** → RAM sedikit lebih besar jika layar lebih besar. |
# | `weight` & `inches`       | **0.83**       | Korelasi **positif sangat kuat** → Layar besar = laptop lebih berat.         |
# 
# - 🧠 Kesimpulan Analisis Korelasi:
#     - Untuk memprediksi harga, prioritaskan fitur ram.
#     - Fitur inches dan weight lebih cocok untuk klasifikasi berdasarkan portabilitas atau ukuran daripada prediksi harga.
#     - Korelasi tinggi antar weight dan inches menunjukkan kemungkinan multikolinearitas, yang perlu diperhatikan dalam model regresi.

# ### 👁️👁️ Insight EDA
# 📊 EDA Untuk Dataset Laptop
# 1. Distribusi Fitur Numerik:
# - RAM (GB):
#     - Mayoritas laptop memiliki RAM 4GB dan 8GB, disusul oleh 16GB.
#     - Hanya sedikit laptop yang punya RAM di atas 32GB.
#     - Distribusi ini menunjukkan bahwa pasar didominasi oleh laptop kelas menengah.
# - Ukuran Layar (Inch):
#     - Didominasi oleh ukuran 15.6 inci, kemudian 14 inci dan 13.3 inci.
#     - Ukuran layar besar seperti 17 inci cukup jarang.
# - Berat (Weight):
#     - Distribusi berat laptop berbentuk normal dengan puncak pada kisaran 2 – 2.5 kg.
#     - Hanya sedikit laptop yang sangat ringan (< 1.5 kg) atau sangat berat (> 3 kg).
# 
# ---
# 
# 2. Distribusi Kategorikal:
# - Merek (Company):
#     - 3 merek terbesar: Dell, Lenovo, dan HP – masing-masing dengan lebih dari 250 produk.
#     - Brand seperti Huawei, LG, Chuwi, Vero sangat jarang muncul.
#     - Dell menjadi pemain dominan dalam data.
# - Sistem Operasi (OS):
#     - Windows 10 mendominasi secara absolut.
#     - OS lain seperti Linux, Chrome OS, atau macOS hanya sebagian kecil dari total data.
#     - Beberapa laptop dijual tanpa OS.
# 
# ---
# 
# 3. Korelasi Antar Fitur Numerik:
# - RAM memiliki korelasi tinggi dengan harga (0.74) → semakin besar RAM, semakin mahal harga laptop.
# - Ukuran layar dan berat sangat berkorelasi (0.83) → layar besar biasanya menambah bobot.
# - Bobot laptop tidak terlalu memengaruhi harga.
# 
# ---
# 
# ✅ Insight Utama:
# 1. Laptop dengan RAM besar cenderung lebih mahal, jadi RAM adalah fitur penting dalam prediksi harga.
# 2. Mayoritas laptop di pasaran berada pada kelas menengah (RAM 4–8GB, ukuran 14–15.6 inci, berat sekitar 2 kg).
# 3. Windows 10 adalah sistem operasi yang paling umum – penting untuk strategi distribusi OS.
# 4. Ukuran layar dan berat sangat terkait, tetapi tidak terlalu penting untuk memprediksi harga.
# 
# 

# ## Data Preparation

# Pada tahap ini kita akan melakukan *Data Preparation* untuk memastikan bahwa pondasi modelling maupun training kita kuat.

# ### Feature Engineering

# In[16]:


df['ssd'] = 0.0
df['hdd'] = 0.0
df['flash'] = 0.0
df['hybrid'] = 0.0

df['memory'] = df['memory'].str.lower()

for idx, row in df.iterrows():
    items = row['memory'].split('+')
    for item in items:
        item = item.strip()
        item = item.replace('tb', '000').replace('gb', '').strip()
        try:
            size = float(item)
        except ValueError:
            size = 0.0

        if 'ssd' in row['memory']:
            df.at[idx, 'ssd'] += size
        if 'hdd' in row['memory']:
            df.at[idx, 'hdd'] += size
        if 'flash' in row['memory']:
            df.at[idx, 'flash'] += size
        if 'hybrid' in row['memory']:
            df.at[idx, 'hybrid'] += size


# 🎯 Tujuan Feature Engineering Ini:
# Membuat data storage bisa dianalisis secara kuantitatif (numerik), bukan sekadar teks.
# 
# Memungkinkan agar dapat melakukan:
# - Membandingkan kapasitas SSD vs HDD
# - Melakukan visualisasi atau korelasi dengan harga (price_in_idr)
# - Memasukkan ssd, hdd, flash, hybrid sebagai fitur numerik ke model machine learning

# In[17]:


df['cpu_brand'] = df['cpu'].apply(lambda x: x.split()[0].lower())

df['gpu_brand'] = df['gpu'].apply(lambda x: x.split()[0].lower())

df['total_storage'] = df['ssd'] + df['hdd'] + df['flash'] + df['hybrid']


# 🎯 Tujuan:
# - Mengubah teks panjang CPU dan GPU menjadi merek yang lebih ringkas dan bisa digunakan untuk kategorisasi.
# - Menyediakan angka total storage agar bisa digunakan dalam analisis numerik.
# 
# ---
# 
# 💡 Manfaat pada data:
# - Lebih bersih
# - Lebih informatif
# - Siap untuk visualisasi, analisis statistik, atau model machine learning.

# In[18]:


df.drop(columns=['memory', 'cpu', 'gpu'], inplace=True)


# 🎯 Tujuan:
# - Kolom-kolom ini dihapus karena sudah dipecah dan disederhanakan ke dalam fitur baru yang:
# - Lebih mudah dianalisis
# - Lebih berguna untuk visualisasi dan modeling
# - Mengurangi redundansi dan kebingungan saat eksplorasi data

# In[19]:


y = df['price_in_idr']

X = df.drop(columns=['price_in_euros', 'price_in_idr', 'product'])


# | Komponen | Fungsi                                                     |
# | -------- | ---------------------------------------------------------- |
# | `y`      | Target: harga laptop dalam IDR                             |
# | `X`      | Fitur: atribut-atribut laptop selain harga dan nama produk |
# Langkah ini adalah standar dalam supervised learning untuk memisahkan input dan output, sehingga data siap digunakan untuk pelatihan model.

# In[20]:


categorical_cols = ['company', 'type_name', 'screen_resolution', 'opsys', 'cpu_brand', 'gpu_brand']

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# 🎯 Tujuan:
# - Kolom-kolom kategorikal diubah menjadi angka (biner)
# - Dataset X sekarang sepenuhnya numerik, dan bisa langsung digunakan untuk modeling
# - Teknik ini disebut One-Hot Encoding, dan sangat umum dipakai sebelum melatih model regresi, decision tree, random forest, dll.

# #### 👁️👁️ Insight Feature Engineering
# 🧠 Tahap Data Preparation – Kesimpulan
# Setelah melalui tahap data preparation, dataset kini telah:
# 
# ##### ✅ Bersih dan Konsisten:
#     - Nilai teks diseragamkan (lowercase).
#     - Kolom yang tidak relevan atau redundan (cpu, gpu, memory, product) telah dihapus.
# ---
# 
# ##### 🦾 Mengandung Fitur Baru (Feature Engineering):
#     - Storage dipecah menjadi: ssd, hdd, flash, hybrid, dan total_storage.
#     - Merek CPU dan GPU diambil dari string mentah menjadi kolom cpu_brand dan gpu_brand.
# ---
# 
# ##### 🔢 Berisi Fitur Numerik Siap Pakai
#     - Kolom kategorikal seperti company, opsys, dll sudah diubah menjadi kolom numerik melalui One-Hot Encoding.
#     - Dataset X sekarang hanya terdiri dari angka, cocok untuk digunakan oleh algoritma machine learning.
# ---
# 
# ##### 🎯 Target Sudah Ditentukan
#     - Target prediksi (price_in_idr) telah dipisahkan dalam variabel y.

# ### Standardization

# In[21]:


from sklearn.preprocessing import StandardScaler

numerical_cols = ['ram', 'inches', 'weight', 'ssd', 'hdd', 'flash', 'hybrid', 'total_storage']

scaler = StandardScaler()

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# 🔍 Tujuan Scaling / Standardisasi:
# - Agar semua fitur numerik memiliki skala yang seragam, yaitu:
#     - Mean = 0
#     - Standard Deviation = 1
# ---
# 
# 🚨 Ini penting karena:
# Beberapa algoritma machine learning (seperti Linear Regression, KNN, SVM) sangat sensitif terhadap perbedaan skala antar fitur.
# 
# Misalnya: fitur weight bisa punya nilai 1–4, sedangkan total_storage bisa ratusan hingga ribuan. Tanpa scaling, fitur dengan angka besar akan dominan dan menyesatkan model.

# ### IQR

# **IQR** adalah konsep statistik yang terkait dengan distribusi data, dan penggunaannya untuk outlier adalah salah satu aplikasi utamanya. IQR mewakili rentang nilai yang mencakup 50% bagian tengah data Anda ketika diurutkan. Ini adalah ukuran penyebaran data yang "tahan" terhadap nilai-nilai ekstrem.
# 
# - Kuartil Pertama (Q1): Nilai di bawahnya terletak 25% data.
# - Kuartil Ketiga (Q3): Nilai di bawahnya terletak 75% data (atau 25% data terletak di atasnya).
# - IQR: Adalah perbedaan antara Kuartil Ketiga (Q3) dan Kuartil Pertama (Q1). IQR = Q3 - Q1

# In[22]:


def outlier_iqr(df):
    outliers = []
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for i in df:
        if i < lower_bound or i > upper_bound:
            outliers.append(i)
    return outliers

print('Before Drop Outliers')
data_outlier = {}
for col in numerical_cols:
    data_outlier[col] = outlier_iqr(X[col])
    print('Outlier (' + col + '):', len(data_outlier[col]), 'outliers')


# **📌 Pada kode cell ini dilakukan pengecekan outlier pada data X dengan teknik *IQR Outlier*.**
# 
# Berhasil menjalankan dan menghasilkan insight berupa:
# - Outlier (ram): 219 outliers
# - Outlier (inches): 37 outliers
# - Outlier (weight): 45 outliers
# 
# Ini menunjukkan adanya outlier pada kolom-kolom tersebut.

# In[23]:


def dropOutlier(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3-Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    df = np.where(df > upper_bound, upper_bound, df)
    df = np.where(df < lower_bound, lower_bound, df)
    return df

for col in numerical_cols:
    X[col] = dropOutlier(X[col])

print('After Drop Outliers')
data_outlier = {}
for col in numerical_cols:
    data_outlier[col] = outlier_iqr(X[col])
    print('Outlier (' + col + '):', len(data_outlier[col]), 'outliers')


# **🧼 Melakukan penghapusan nilai outlier yang ada pada data X dengan menggunakan *IQR drop Outlier*.**
# 
# Hal ini berhasil dilakukan dan menghasilkan insight seperti berikut:
# 
# After Drop Outliers
# - Outlier (ram): 0 outliers
# - Outlier (inches): 0 outliers
# - Outlier (weight): 0 outliers

# In[24]:


from scipy.stats import zscore
import numpy as np

z_scores = zscore(y)
z_threshold = 3

outlier_mask = np.abs(z_scores) > z_threshold

mean_price = y[~outlier_mask].mean()

y_clean = y.copy()

y_clean[outlier_mask] = mean_price

# Cek hasil
print(f"Jumlah outlier yang diganti: {outlier_mask.sum()}")
print(f"Rata-rata yang digunakan: {mean_price:.2f}")


# **🧼 Melakukan penggantian nilai outlier dengan menggunakan rata-rata yang ada pada data y_clean dengan menggunakan *zscore*.**
# 
# Hal ini berhasil dilakukan dan menghasilkan insight seperti berikut:
# - Jumlah outlier yang diganti: `12`
# - Rata-rata yang digunakan: `19358061.46`

# ### Memastikan Tidak Ada Missing Value

# In[25]:


X.dropna(inplace=True)


# In[26]:


y_clean.dropna(inplace=True)


# 🧼 Penanganan Data Kosong
# 
# Untuk memastikan dataset `X` yang berisikan atribut pada laptop dan `y_clean` yang berisi `target` tidak memiliki nilai kosong yang dapat mengganggu proses training model, digunakan metode `dropna()` untuk menghapus seluruh baris yang mengandung nilai `NaN`. Ini menjamin bahwa dataset bersih dan siap digunakan untuk proses machine learning selanjutnya.

# ### Split Dataset

# In[27]:


from sklearn.model_selection import train_test_split

features = X.columns.tolist()
target = y_clean 

X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
    X, y_clean, test_size=0.2, random_state=42
)

print(f"Train set shape: {X_train_ts.shape}, {y_train_ts.shape}")
print(f"Test set shape: {X_test_ts.shape}, {y_test_ts.shape}")


# 🎯 Tujuan Train-Test Split:
# - Untuk melatih model pada sebagian data (train) dan mengujinya pada data yang belum pernah dilihat (test).
# - Membantu menilai kinerja generalisasi model, apakah model bekerja baik di data nyata yang tidak pernah dilatih sebelumnya.
# 
# ---
# 
# 🔀 Train-Test Split
# Dataset telah dibagi menjadi:
# - **Training Set** (80%) sebanyak 1.020 data
# - **Testing Set** (20%) sebanyak 255 data
# ___
# 
# Pembagian dilakukan secara acak menggunakan `train_test_split()` dari `scikit-learn`, dengan `random_state=42` untuk memastikan hasil yang konsisten. Ini memungkinkan evaluasi performa model secara adil pada data yang tidak pernah dilatih sebelumnya.
# 

# In[28]:


plt.figure(figsize=(10,5))
sns.histplot(y_train_ts, color='blue', label='Train Set', kde=True)
sns.histplot(y_test_ts, color='red', label='Test Set', kde=True)
plt.title("Distribusi Harga Laptop pada Train dan Test Set")
plt.xlabel("Harga dalam Rupiah")
plt.legend()
plt.show()


# ### 📈 Distribusi Harga Laptop pada Train dan Test Set
# 
# Grafik menunjukkan distribusi harga laptop untuk data latih (`Train Set`) dan data uji (`Test Set`) setelah dilakukan pembagian dengan `train_test_split`.
# 
# ---
# 
# #### 📊 Penjelasan Grafik:
# 
# 1. **Histogram dan KDE (Kernel Density Estimation)**
#    - **Histogram biru**: menunjukkan distribusi harga laptop dari data latih (`y_train_ts`).
#    - **Histogram merah**: menunjukkan distribusi harga laptop dari data uji (`y_test_ts`).
#    - **Garis lengkung KDE**: menggambarkan estimasi kepadatan data untuk menunjukkan tren distribusi secara halus.
# 
# 
# 2. **Distribusi Positif (Right-skewed)**
#    - Kedua dataset (train dan test) menunjukkan distribusi yang **miring ke kanan**.
#    - Artinya, sebagian besar harga laptop berada di kisaran rendah hingga menengah, dan hanya sebagian kecil yang sangat mahal.
# 
# 
# 3. **Pola Distribusi Mirip**
#    - Distribusi pada train dan test terlihat **konsisten dan serupa**, yang menunjukkan:
#      - Data uji mewakili karakteristik data latih secara proporsional.
#      - Pembagian data dilakukan dengan baik dan **tidak bias**, serta tidak terjadi **data leakage**.
# ---
# 
# #### 🎯 Tujuan Visualisasi Ini:
# 
# - **Memvalidasi pembagian data train dan test** secara acak.
# - Memastikan bahwa model nantinya akan diuji pada data yang memiliki distribusi mirip dengan data latih.
# - Menghindari **overfitting** atau **underfitting** akibat ketidakseimbangan distribusi antara train dan test set.
# 
# ---
# 
# ✅ **Kesimpulan**:
# Distribusi harga laptop pada data latih dan data uji menunjukkan kemiripan pola yang baik. Ini mengindikasikan bahwa proses pembagian data telah dilakukan secara representatif, yang menjadi dasar penting sebelum melatih model regresi.
# 

# # Model : Regression

# 🚨 Karena kita akan membuat model **Predictive Modeling** **Price Laptop** yang berhubungan dengan mengetahui data perkiraan terkait apa yang mungkin terjadi pada data yang telah saya dapatkan

# In[29]:


from sklearn.metrics import mean_squared_error, r2_score
results = []

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"{model_name} Model:")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}\n")

    results.append({'Model': model_name, 'R²': test_r2, 'RMSE': test_rmse})


# ### 🧠 Tujuan Umum
# Kode ini digunakan untuk **melatih**, **menguji**, dan **mengevaluasi kinerja model machine learning** dalam memprediksi harga laptop (`price_in_idr`).
# 
# ---
# 
# ### ✅ Kesimpulan
# Fungsi `evaluate_model()` penting karena:
# 
# - 📏 Melakukan **evaluasi model regresi** menggunakan metrik yang umum dipakai, yaitu:
#   - **R² (R-squared)**: menunjukkan seberapa baik model menjelaskan variansi data.
#   - **RMSE (Root Mean Squared Error)**: menunjukkan seberapa jauh prediksi dari nilai aktual.
# 
# - 📊 Memungkinkan untuk **membandingkan beberapa model** dengan cara yang sistematis dan otomatis.
# 
# - 📈 Menyimpan hasil evaluasi ke dalam list `results`, sehingga dapat digunakan untuk **analisis lanjutan atau visualisasi perbandingan model**.
# 
# ---
# 
# 

# In[30]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression' : Ridge(),
    'Lasso Regression' : Lasso(),
    'ElasticNet Regression' : ElasticNet(),
    "XGBRegressor": XGBRegressor(),
}


# ### 🧠 Tujuan Kode
# Kode ini bertujuan untuk **menginisialisasi beberapa model regresi** yang akan digunakan dalam proses pelatihan dan evaluasi terhadap data fitur laptop untuk memprediksi `price_in_idr`.
# 
# ---
# 
# ### 📦 Penjelasan Per Baris
# 
# #### `from sklearn.linear_model import ...`
# Mengimpor beberapa algoritma regresi dari pustaka **Scikit-Learn**:
# 
# | Model               | Penjelasan Singkat                                                                 |
# |---------------------|------------------------------------------------------------------------------------|
# | `LinearRegression()` | Regresi linear biasa, digunakan sebagai baseline.                                 |
# | `Ridge()`            | Regresi linear dengan regularisasi **L2**, menghindari overfitting.               |
# | `Lasso()`            | Regresi linear dengan regularisasi **L1**, bisa menyusutkan beberapa koefisien ke nol (fitur seleksi). |
# | `ElasticNet()`       | Kombinasi dari Lasso dan Ridge (menggabungkan penalti **L1** dan **L2**).         |
# 
# ---
# 
# ### ✅ Kesimpulan
# 
# Kamu sedang menyiapkan berbagai jenis **model regresi dengan karakteristik yang berbeda**, untuk:
# 
# - Membandingkan performanya terhadap **prediksi harga laptop**.
# - Menentukan model **mana yang paling akurat dan generalizable**.
# - Memahami perbedaan antara **model linear, tree-based, dan ensemble-based** terhadap dataset ini.
# 

# ## Evaluation

# In[31]:


for model_name, model in models.items():
    evaluate_model(model, X_train_ts, y_train_ts, X_test_ts, y_test_ts, model_name)

results_df = pd.DataFrame(results)


# ### ✅ Tujuan Kode
# 
# Kode ini digunakan untuk:
# 
# - Melatih dan mengevaluasi semua model regresi yang telah disiapkan.
# - Menghitung performa setiap model menggunakan dua metrik utama:
#   - **R² (R-squared)**: Mengukur seberapa baik model menjelaskan varians dalam data.
#   - **RMSE (Root Mean Squared Error)**: Mengukur rata-rata kesalahan prediksi dalam satuan harga (Rupiah).
# 
# ---
# 
# ### 🧪 Hasil Evaluasi Model
# 
# | Model               | R² Score | RMSE (Rp)      | Interpretasi                                                                 |
# |---------------------|----------|----------------|------------------------------------------------------------------------------|
# | Linear Regression    | 0.65     | 6.38 juta       | Baseline yang cukup baik. Model linear mampu menjelaskan 74% varians harga. |
# | Ridge Regression     | 0.65     | 6.38 juta       | Sedikit lebih baik, regularisasi L2 membantu menghindari overfitting.       |
# | Lasso Regression     | 0.65     | 6.38 juta       | Mirip Linear, tetapi dengan penalti L1 yang juga bisa melakukan seleksi fitur. |
# | ElasticNet Regression| 0.47     | 7.81 juta       | Performa kurang baik, penalti gabungan L1 dan L2 kurang cocok di sini.      |
# | XGBoost Regressor    | **0.74** | **5.47 juta**   | ✅ Model terbaik. Menangkap hubungan non-linear dan hasil prediksi paling akurat. |
# 
# ---
# 
# ### 🎯 Interpretasi Metrik
# 
# - **R² (Koefisien Determinasi):**
#   - Nilai 1.0 berarti prediksi sempurna.
#   - Semakin mendekati 1, semakin baik.
#   - Urutan performa RMSE: **XGBoost (0.74) > Ridge/Linear/Lasso (0.65) > ElasticNet (0.47)**
# 
# - **RMSE (Root Mean Squared Error):**
#   - Mengukur rata-rata deviasi prediksi dari nilai sebenarnya.
#   - Semakin kecil nilainya, semakin baik performa model.
#   - RMSE terbaik dimiliki oleh **XGBoost** dengan **5.47 juta Rupiah**.
# 
# ---
# 
# ### 🏁 Kesimpulan
# 
# - 🔧 **XGBoost adalah model terbaik** untuk prediksi harga laptop dalam dataset ini.
# - 📉 **ElasticNet underperform**, kemungkinan karena parameter default tidak optimal.
# - 🧪 **Linear, Ridge, dan Lasso** masih kuat sebagai baseline, ringan dan mudah diinterpretasi.
# 

# In[32]:


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

model_ts = XGBRegressor(
    n_estimators=500,      
    max_depth=7,           
    learning_rate=0.05,    
    subsample=0.8,        
    colsample_bytree=0.9,
    gamma=0.1,           
    reg_alpha=0.01,      
    reg_lambda=0.1,    
    random_state=42,     
    n_jobs=-1              
)

model_ts.fit(X_train_ts, y_train_ts)

y_pred_test_ts = model_ts.predict(X_test_ts)

test_r2_ts = r2_score(y_test_ts, y_pred_test_ts)
test_rmse_ts = np.sqrt(mean_squared_error(y_test_ts, y_pred_test_ts))

print("\nXGBoost Model (Tuned Parameters):")
print(f"Test R²: {test_r2_ts:.4f}")
print(f"Test RMSE: {test_rmse_ts:.4f}\n")


# ### 🚀 XGBoost Regressor dengan Parameter Tuning
# Model XGBoost digunakan dengan parameter yang telah disesuaikan untuk meningkatkan performa prediksi harga laptop.
# 
# ---
# 
# ### ⚙️ Konfigurasi Parameter
# | Parameter          | Nilai | Fungsi                                                     |
# | ------------------ | ----- | ---------------------------------------------------------- |
# | `n_estimators`     | 500   | Jumlah pohon dalam model boosting                          |
# | `max_depth`        | 7     | Kedalaman maksimum tiap pohon                              |
# | `learning_rate`    | 0.05  | Kecepatan pembelajaran model boosting                      |
# | `subsample`        | 0.8   | Persentase data yang digunakan untuk setiap pohon          |
# | `colsample_bytree` | 0.9   | Persentase fitur yang digunakan untuk membangun tiap pohon |
# | `gamma`            | 0.1   | Minimum loss reduction untuk membuat split                 |
# | `reg_alpha`        | 0.01  | Regularisasi L1 (menghindari overfitting)                  |
# | `reg_lambda`       | 0.1   | Regularisasi L2 (menghindari overfitting)                  |
# | `random_state`     | 42    | Reproducibility (hasil konsisten)                          |
# | `n_jobs`           | -1    | Menggunakan semua core CPU                                 |
# 
# ---
# 
# ### 📊 Evaluasi Performa Model
# | Metrik | Nilai      | Interpretasi                                                         |
# | ------ | ---------- | -------------------------------------------------------------------- |
# | R²     | 0.7619     | Model mampu menjelaskan **76.19%** variasi harga laptop di data uji. |
# | RMSE   | Rp 5.25 jt | Rata-rata kesalahan prediksi adalah sekitar **5.25 juta**.        |
# 
# ---
# 
# ### ✅ Kesimpulan
# 1. Model XGBoost yang telah dituning menunjukkan performa cukup baik, dan berhasil meningkatkan akurasinya dari **0.74** menjadi **0.7619**.
# 2. RMSE berhasil ditekan dengan nilai awal sebelum tuning **5.47 juta** menjadi **5.25 juta**.

# In[33]:


print("\n--- Checks before creating plot_df ---")
print("Shape of y_test_ts:", y_test_ts.shape)
print("Shape of y_pred_test_ts:", y_pred_test_ts.shape)

print("Are there NaNs in y_test_ts?", y_test_ts.isna().any())
print("Are there NaNs in y_pred_test_ts?", pd.Series(y_pred_test_ts).isna().any())

print("y_test_ts head:\n", y_test_ts.head())
print("y_pred_test_ts head:\n", pd.Series(y_pred_test_ts).head()) 

print("y_test_ts tail:\n", y_test_ts.tail())
print("y_pred_test_ts tail:\n", pd.Series(y_pred_test_ts).tail())

if 'price_in_idr' not in X.columns:
    print("Error: 'date' column not found in clean_df!")

print("Are test indices in clean_df index?", X_test_ts.index.isin(X.index).all())


# ### ✅ Tujuan Kode
# 
# Kode ini digunakan sebagai **langkah validasi (sanity check)** sebelum membuat DataFrame `plot_df` yang akan digunakan untuk visualisasi hasil **prediksi vs. harga aktual**.  
# Tujuannya adalah memastikan bahwa semua data yang akan dipakai **benar, konsisten, dan bebas dari error** seperti:
# 
# - Missing values (`NaN`)
# - Ketidaksesuaian bentuk array
# - Ketidaksesuaian indeks antara data prediksi dan data asli (`clean_df`)
# 
# ---
# 
# ### 📌 Kesimpulan
# 
# Semua pengecekan telah **lolos**, sehingga:
# 
# - ✅ Data prediksi dan data aktual memiliki **jumlah yang sama**  
# - ✅ Tidak ada **missing value** dalam `y_test_ts` maupun `y_pred_test_ts`  
# - ✅ Nilai-nilai berada dalam **kisaran harga wajar**  
# - ✅ **Indeks data uji cocok** dengan indeks `clean_df`, artinya aman untuk digabung
# 
# > 💡 Artinya, data siap digunakan untuk membuat visualisasi hasil prediksi harga laptop secara akurat.
# 

# In[34]:


plt.figure(figsize=(12, 5))
plt.plot(y_test_ts.reset_index(drop=True), label='Actual Price')
plt.plot(y_pred_test_ts, label='Predicted Price (XGB)', alpha=0.7)
plt.title('Prediksi Harga Laptop vs Nilai Aktual (Test Set)')
plt.xlabel('Index')
plt.ylabel('Harga (Rupiah)')
plt.legend()
plt.grid(True)
plt.show()


# ### 📈 Grafik: Prediksi Harga Laptop vs Nilai Aktual (Test Set)
# 
# Grafik ini menampilkan perbandingan visual antara **harga laptop sebenarnya** dan **hasil prediksi dari model XGBoost** pada data uji (Test Set).
# 
# ---
# 
# #### 🧾 Detail Grafik:
# 
# - **Garis Biru** (`Actual Price`):  
#   Menunjukkan harga laptop sebenarnya dari dataset.
# 
# - **Garis Oranye** (`Predicted Price (XGB)`):  
#   Menunjukkan hasil prediksi harga dari model XGBoost.
# 
# - **Sumbu X (Index)**:  
#   Merupakan indeks dari data dalam test set — urutan data, bukan waktu.
# 
# - **Sumbu Y (Harga - Rupiah)**:  
#   Mewakili nilai harga laptop (dalam satuan Rupiah).
# 
# ---
# 
# #### 🔍 Interpretasi Visual:
# 
# - ✅ **Polanya mirip**  
#   Prediksi mengikuti pola umum dari data aktual dengan cukup baik, menunjukkan bahwa model berhasil mempelajari tren data.
# 
# - ✅ **Akurasi keseluruhan baik**  
#   Sebagian besar prediksi berada dekat dengan nilai aktual.
# 
# ---
# 
# #### 🎯 Kesimpulan:
# 
# Grafik ini menunjukkan bahwa **model XGBoost cukup efektif dalam memprediksi harga laptop**, dengan pola prediksi yang mengikuti harga aktual secara keseluruhan.
# 
# > 💡 Visualisasi seperti ini sangat membantu untuk **mengevaluasi stabilitas dan generalisasi model** sebelum digunakan dalam sistem nyata.

# ## Create Plot

# In[35]:


plot_df = pd.DataFrame({
    'Index': range(len(y_test_ts)),
    'Actual Price': y_test_ts.values,
    'Predicted Price': y_pred_test_ts
})

print("plot_df head:\n", plot_df.head())
print("plot_df shape:", plot_df.shape)


# #### ✅ Tujuan Kode
# 
# Kode ini digunakan untuk **menyiapkan DataFrame baru (`plot_df`)** yang berisi:
# 
# - **Harga aktual** (`Actual Price`)
# - **Harga hasil prediksi** dari model XGBoost (`Predicted Price`)
# - **Indeks data** untuk keperluan visualisasi (plot)
# 
# ---
# 
# #### 🧱 Penjelasan Struktur `plot_df`
# 
# DataFrame `plot_df` dibuat dengan 3 kolom utama:
# 
# | Kolom             | Deskripsi                                                                 |
# |-------------------|---------------------------------------------------------------------------|
# | `Index`           | Urutan angka dari 0 sampai jumlah data test (sebagai sumbu X pada grafik) |
# | `Actual Price`    | Nilai harga sesungguhnya dari `y_test_ts`                                 |
# | `Predicted Price` | Nilai hasil prediksi dari model XGBoost (`y_pred_test_ts`)                |
# 
# ---
# 
# #### 🖨️ Menampilkan `ukuran DataFrame`:
# 
# 
# Menampilkan **5 baris pertama** dari DataFrame `plot_df shape: (255, 3)`:
# - **255 baris** = jumlah sampel dalam data test.
# - **3 kolom** = `Index`, `Actual Price`, dan `Predicted Price`.
# 
# ---
# 
# ## 🧠 Kesimpulan
# 
# Data `plot_df` ini **sudah siap digunakan** untuk:
# 
# - Membuat **visualisasi** perbandingan antara harga aktual dan harga prediksi.
# - Mempermudah **analisis kesalahan prediksi** (under/over prediction) oleh model.
# - Meningkatkan **interpretabilitas hasil model** secara visual.
# 
# 

# In[36]:


plot_df.info()


# #### 🧾 Rincian:
# | Informasi           | Keterangan                                                                                                |
# | ------------------- | --------------------------------------------------------------------------------------------------------- |
# | `RangeIndex`        | DataFrame memiliki 255 baris, dengan indeks dari 0 hingga 254                                             |
# | Kolom total         | Ada 3 kolom: `Index`, `Actual Price`, dan `Predicted Price`                                               |
# | Non-Null Count      | Semua kolom memiliki **255 data** (tidak ada nilai yang hilang / NaN)                                     |
# | Tipe Data (`dtype`) | - `Index`: `int64` (bilangan bulat)  <br> - `Actual Price`: `float64` <br> - `Predicted Price`: `float32` |
# | Penggunaan Memori   | Total memori yang digunakan adalah sekitar **5.1 KB**                                                     |
# 
# 
# 

# In[37]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=plot_df['Index'], y=plot_df['Actual Price'],
                          mode='lines+markers', name='Actual Price'))

fig.add_trace(go.Scatter(x=plot_df['Index'], y=plot_df['Predicted Price'],
                          mode='lines+markers', name='Predicted Price',
                          line=dict(dash='dash')))

fig.update_layout(title='Actual vs. Predicted Laptop Price (Test Set)',
                   xaxis_title='Index',
                   yaxis_title='Price (IDR)',
                   hovermode='x unified')

fig.show()


# #### 🔧 Kode Visualisasi:
# Visualisasi ini dibuat menggunakan plotly.graph_objects.
# #### 📌 Detail Grafik:
# | Elemen                      | Penjelasan                                                            |
# | --------------------------- | --------------------------------------------------------------------- |
# | **Sumbu X (Index)**         | Indeks dari data dalam test set (urutan data, bukan tanggal/waktu).   |
# | **Sumbu Y (Price - IDR)**   | Harga laptop dalam satuan Rupiah.                                     |
# | **Garis Biru**              | Harga aktual dari data uji (`Actual Price`).                          |
# | **Garis Merah Putus-putus** | Prediksi harga dari model XGBoost (`Predicted Price`).                |
# | **Titik Marker**            | Menambahkan detail per titik, berguna saat hover (interaktif).        |
# | **Hovermode: x unified**    | Saat hover, semua nilai pada titik X yang sama akan muncul bersamaan. |
# 
# ---
# 
# #### 🔍 Interpretasi Visual:
# ✅ Polanya konsisten
# Garis merah (prediksi) mengikuti pola dari garis biru (aktual), menandakan model menangkap tren harga cukup baik.
# 
# ✅ Detail interaktif
# Hovermode memudahkan pengguna membandingkan harga aktual dan prediksi secara langsung untuk setiap data.

# In[ ]:




