#!/usr/bin/env python
# coding: utf-8

# # Proyek Predictive Analysis
# # Laporan Proyek Machine Learning - Najwar Putra Kusumah Wardana
# # Predictive Laptop Price

# ## Import library and file

# In[1]:


# Import library yang dibutuhkan
import pandas as pd


file_path = "laptop_data.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# # Data Understanding

# In[3]:


df.head()


# In[4]:


df.info()


# | Kolom              | Tipe Data | Keterangan                                              |
# | ------------------ | --------- | ------------------------------------------------------- |
# | `laptop_ID`        | `int64`   | ID unik tiap laptop                                     |
# | `Company`          | `object`  | Merk laptop (Asus, Lenovo, dll)                         |
# | `Product`          | `object`  | Nama produk                                             |
# | `TypeName`         | `object`  | Jenis laptop (Ultrabook, Gaming, dll)                   |
# | `Inches`           | `float64` | Ukuran layar dalam inci                                 |
# | `ScreenResolution` | `object`  | Resolusi layar                                          |
# | `Cpu`              | `object`  | Detail prosesor                                         |
# | `Ram`              | `object`  | RAM (format string, perlu diproses ke numerik)          |
# | `Memory`           | `object`  | Tipe dan kapasitas penyimpanan (SSD/HDD)                |
# | `Gpu`              | `object`  | GPU/VGA                                                 |
# | `OpSys`            | `object`  | Sistem operasi                                          |
# | `Weight`           | `object`  | Berat laptop (format string, perlu diproses ke numerik) |
# | `Price_in_euros`   | `float64` | Harga dalam Euro (target prediksi)                      |
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


# In[6]:


# Hapus kolom ID karena tidak dibutuhkan dalam modeling
df.drop(columns=['laptop_id'], inplace=True)


# In[7]:


# Konversi harga ke Rupiah (IDR) dengan kurs 17.500
df['price_in_idr'] = df['price_in_euros'] * 17500


# In[8]:


# Ubah kolom ram dari string ke integer (hapus 'GB')
df['ram'] = df['ram'].str.replace('GB', '', regex=False).astype(int)

# Ubah kolom weight dari string ke float (hapus 'kg')
df['weight'] = df['weight'].str.replace('kg', '', regex=False).astype(float)

# Tampilkan 5 baris awal data setelah dibersihkan
df.head()


# In[9]:


df.describe(include='all').style.background_gradient('Greens_r')


# In[10]:


df.isna().sum()


# In[11]:


df.duplicated().sum()


# ## EDA

# - Melihat distribusi harga
# 
# - Korelasi antara RAM, storage, GPU dengan harga
# 
# - Visualisasi outlier dan kategori dominan

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(df['price_in_idr'], bins=50, kde=True, color='green')
plt.title('Distribusi Harga Laptop (dalam Rupiah)')
plt.xlabel('Harga (IDR)')
plt.ylabel('Jumlah')
plt.show()


# In[13]:


fig, axs = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['ram'], bins=10, ax=axs[0], color='skyblue')
axs[0].set_title('Distribusi RAM (GB)')

sns.histplot(df['inches'], bins=10, ax=axs[1], color='orange')
axs[1].set_title('Distribusi Ukuran Layar (Inch)')

sns.histplot(df['weight'], bins=10, ax=axs[2], color='salmon')
axs[2].set_title('Distribusi Berat Laptop (Kg)')

plt.tight_layout()
plt.show()


# In[14]:


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


# ## Feature Engineering

# In[15]:


# Inisialisasi storage columns
df['ssd'] = 0.0
df['hdd'] = 0.0
df['flash'] = 0.0
df['hybrid'] = 0.0

# Lowercase untuk konsistensi
df['memory'] = df['memory'].str.lower()

# Parsing isi kolom memory
for idx, row in df.iterrows():
    items = row['memory'].split('+')
    for item in items:
        item = item.strip()
        # Ganti tb → 1000 dan hilangkan gb
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


# In[16]:


# CPU brand (ambil kata pertama)
df['cpu_brand'] = df['cpu'].apply(lambda x: x.split()[0].lower())

# GPU brand (ambil kata pertama)
df['gpu_brand'] = df['gpu'].apply(lambda x: x.split()[0].lower())

df['total_storage'] = df['ssd'] + df['hdd'] + df['flash'] + df['hybrid']


# In[17]:


df.drop(columns=['memory', 'cpu', 'gpu'], inplace=True)


# In[18]:


# Fitur target
y = df['price_in_idr']

# Drop kolom target dari fitur
X = df.drop(columns=['price_in_euros', 'price_in_idr', 'product'])  # 'product' di-drop karena terlalu spesifik


# In[19]:


# Kolom kategorikal yang ingin di-encode
categorical_cols = ['company', 'type_name', 'screen_resolution', 'opsys', 'cpu_brand', 'gpu_brand']

# One-hot encoding + join ke X
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# ## Standardization

# In[20]:


from sklearn.preprocessing import StandardScaler

# Tentukan kolom numerik yang akan diskalakan
numerical_cols = ['ram', 'inches', 'weight', 'ssd', 'hdd', 'flash', 'hybrid', 'total_storage']

# Inisialisasi scaler
scaler = StandardScaler()

# Terapkan scaling
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# ## IQR

# **IQR** adalah konsep statistik yang terkait dengan distribusi data, dan penggunaannya untuk outlier adalah salah satu aplikasi utamanya. IQR mewakili rentang nilai yang mencakup 50% bagian tengah data Anda ketika diurutkan. Ini adalah ukuran penyebaran data yang "tahan" terhadap nilai-nilai ekstrem.
# 
# - Kuartil Pertama (Q1): Nilai di bawahnya terletak 25% data.
# - Kuartil Ketiga (Q3): Nilai di bawahnya terletak 75% data (atau 25% data terletak di atasnya).
# - IQR: Adalah perbedaan antara Kuartil Ketiga (Q3) dan Kuartil Pertama (Q1). IQR = Q3 - Q1

# In[21]:


# Simpan dataframe hasil data preparation sebagai clean_df
clean_df = df.copy()
cols = clean_df.columns


# In[22]:


# Cek Outlier dengan IQR Outlier
def outlier_iqr(data):
    outliers = []
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for i in data:
        if i < lower_bound or i > upper_bound:
            outliers.append(i)
    return outliers

print('Before Drop Outliers')
data_outlier = {}
for col in numerical_cols:
    data_outlier[col] = outlier_iqr(clean_df[col])
    print('Outlier (' + col + '):', len(data_outlier[col]), 'outliers')


# In[23]:


#Drop Outliers
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
    clean_df[col] = dropOutlier(clean_df[col])

print('After Drop Outliers')
data_outlier = {}
for col in numerical_cols:
    data_outlier[col] = outlier_iqr(clean_df[col])
    print('Outlier (' + col + '):', len(data_outlier[col]), 'outliers')


# In[24]:


cols


# In[25]:


clean_df.dropna(inplace=True) # Add this line to drop rows with NaN


# In[26]:


from sklearn.model_selection import train_test_split

# Fitur dan target untuk prediksi harga
features = X.columns.tolist()  # X sebelumnya sudah dibentuk dan di-encode
target = y  # Target sudah ditetapkan sebagai price_in_idr

# Train-test split secara acak (bukan berdasarkan waktu karena bukan time series)
X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set shape: {X_train_ts.shape}, {y_train_ts.shape}")
print(f"Test set shape: {X_test_ts.shape}, {y_test_ts.shape}")


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,5))
sns.histplot(y_train_ts, color='blue', label='Train Set', kde=True)
sns.histplot(y_test_ts, color='red', label='Test Set', kde=True)
plt.title("Distribusi Harga Laptop pada Train dan Test Set")
plt.xlabel("Harga dalam Rupiah")
plt.legend()
plt.show()


# # Model : Regression

# Karena kita akan membuat model **Predictive Modeling** **Price Laptop** yang berhubungan dengan mengetahui data perkiraan terkait apa yang mungkin terjadi pada data yang telah saya dapatkan

# In[28]:


from sklearn.metrics import mean_squared_error, r2_score
# Initialize list to collect results
results = []

# Function to evaluate models and store results for visualization
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Calculate metrics: R² and RMSE
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Print results
    print(f"{model_name} Model:")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}\n")

    # Append results to the list for visualization
    results.append({'Model': model_name, 'R²': test_r2, 'RMSE': test_rmse})


# In[29]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# Initialize models
models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression' : Ridge(),
    'Lasso Regression' : Lasso(),
    'ElasticNet Regression' : ElasticNet(),
    "XGBRegressor": XGBRegressor(),
}


# ## Evaluation

# In[30]:


# Iterate over the models to evaluate each
for model_name, model in models.items():
    evaluate_model(model, X_train_ts, y_train_ts, X_test_ts, y_test_ts, model_name)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)


# XGBRegressor menghasilkan skor **R Square** yang paling tinggi, maka diputuskan untuk menggunakan model XGBRegressor untuk **Predictive Model Price Laptop**.

# In[31]:


# Initialize and train a model (using XGBoost as it was the last one)
model_ts = XGBRegressor()

# Train the model using the time series split data
model_ts.fit(X_train_ts, y_train_ts)

# Make predictions on the test set (future data)
y_pred_test_ts = model_ts.predict(X_test_ts)

# Evaluate the model
test_r2_ts = r2_score(y_test_ts, y_pred_test_ts)
test_rmse_ts = np.sqrt(mean_squared_error(y_test_ts, y_pred_test_ts))

print("\nXGBoost Model with Time Series Features:")
print(f"Test R²: {test_r2_ts:.2f}")
print(f"Test RMSE: {test_rmse_ts:.2f}\n")


# In[32]:


print("\n--- Checks before creating plot_df ---")
print("Shape of y_test_ts:", y_test_ts.shape)
print("Shape of y_pred_test_ts:", y_pred_test_ts.shape) # This is a numpy array

print("Are there NaNs in y_test_ts?", y_test_ts.isna().any())
print("Are there NaNs in y_pred_test_ts?", pd.Series(y_pred_test_ts).isna().any())

# Check the first few values of y_test_ts and y_pred_test_ts
print("y_test_ts head:\n", y_test_ts.head())
print("y_pred_test_ts head:\n", pd.Series(y_pred_test_ts).head()) # Convert to Series for head()

# Check the last few values
print("y_test_ts tail:\n", y_test_ts.tail())
print("y_pred_test_ts tail:\n", pd.Series(y_pred_test_ts).tail())

# Check if 'date' column is in clean_df
if 'date' not in clean_df.columns:
    print("Error: 'date' column not found in clean_df!")

# Check if any of the test indices are present in clean_df's index
print("Are test indices in clean_df index?", X_test_ts.index.isin(clean_df.index).all())


# In[33]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(y_test_ts.reset_index(drop=True), label='Actual Price')
plt.plot(y_pred_test_ts, label='Predicted Price (XGB)', alpha=0.7)
plt.title('Prediksi Harga Laptop vs Nilai Aktual (Test Set)')
plt.xlabel('Index')
plt.ylabel('Harga (Rupiah)')
plt.legend()
plt.grid(True)
plt.show()


# ## Create Plot

# In[34]:


# Pastikan y_test_ts dan y_pred_xgb sudah tersedia
plot_df = pd.DataFrame({
    'Index': range(len(y_test_ts)),
    'Actual Price': y_test_ts.values,
    'Predicted Price': y_pred_test_ts
})

# Tampilkan beberapa data awal dan bentuk data
print("plot_df head:\n", plot_df.head())
print("plot_df shape:", plot_df.shape)


# In[35]:


plot_df.info()


# In[36]:


# Tambahkan kolom pseudo-date (index waktu buatan)
plot_df['index'] = plot_df.index

# Agregasi (meski tidak ada artinya besar untuk harga, hanya contoh)
plotting = plot_df.groupby('index').agg({
    'Actual Price': 'sum',
    'Predicted Price': 'sum'
})
print(plotting.head())


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

