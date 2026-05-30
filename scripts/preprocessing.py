import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print("Ham veri okunuyor...")
df = pd.read_csv('../data/raw/hamdata.csv', encoding='utf-8')
print(f"Toplam satır: {len(df)}")

# 1. TARİH PARÇALAMA & DÖNGÜSEL KODLAMA (Cyclical Encoding)
# Format: "18/04/2026 10:47:02"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

# 2. AYKIRI DEĞER TEMİZLİĞİ (Sadece Derinlik Sınırı)
initial_count = len(df)
df = df[df['Depth'] <= 150].copy()
print(f"Depth > 150 km olan {initial_count - len(df)} satır çıkarıldı.")

# 3. GEREKSİZ VE SIZINTI (Leakage) YAPAN KOLONLARIN ÇIKARILMASI
# Sadece anlık sismik veriler (Tarih, Konum, Derinlik) bırakıldı. Type ve Rms eklendiğinde leakage oluşur.
df_clean = df[['Year', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
             'Latitude', 'Longitude', 'Depth', 'Magnitude']].copy()

print(f"\nİşlenecek net satır sayısı: {len(df_clean)}")

# 4. EĞİTİM / TEST AYRIMI (%80 / %20)
# Herhangi bir veri silme (undersampling) yapılmıyor. Gerçek dünya dağılımı korunuyor.
X = df_clean.drop(columns=['Magnitude'])
y = df_clean['Magnitude']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Eğitim seti: {len(X_train)} satır")
print(f"Test seti  : {len(X_test)} satır")

# 5. SCALER (YSA ve XGBoost İçin Kritik Adım)
# Scaler, kırpılmış veriye değil, TÜM X_train verisine fit ediliyor.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 6. KAYDETME
os.makedirs('../data/processed', exist_ok=True)

# Artık original/bal gibi karmaşık isimlendirmelere gerek yok. Standart dosyalar:
X_train_scaled_df.to_csv('../data/processed/X_train.csv', index=False)
pd.DataFrame(y_train, columns=['Magnitude']).to_csv('../data/processed/y_train.csv', index=False)

X_test_scaled_df.to_csv('../data/processed/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Magnitude']).to_csv('../data/processed/y_test.csv', index=False)

# Scaler kaydediliyor
joblib.dump(scaler, '../data/processed/deprem_scaler.pkl')
with open('../data/processed/scaler_features.txt', 'w', encoding='utf-8') as f:
    f.write(','.join(X_train.columns.tolist()))

print("\n[BAŞARILI] Veri Fabrikası işlemini tamamladı.")
print("Tüm veriler temiz, sızıntısız ve YSA/XGBoost/RF yarışına hazır şekilde kaydedildi.")