import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
import joblib
import os

print("==================================================")
print(" SEISNEURAL - SIZINTISIZ (LEAKAGE-FREE) VERİ FABRİKASI ")
print("==================================================")

print("Ham veri okunuyor...")
df = pd.read_csv('../data/raw/hamdata.csv', encoding='utf-8')
print(f"Toplam satır: {len(df)}")

# 1. TARİH PARÇALAMA & DÖNGÜSEL KODLAMA
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

# 2. AYKIRI DEĞER TEMİZLİĞİ
initial_count = len(df)
df = df[df['Depth'] <= 150].copy()
print(f"Depth > 150 km olan {initial_count - len(df)} satır çıkarıldı.")

# 3. KATEGORİ OLUŞTURMA (Test ve Analizler İçin)
def risk_siniflandir(buyukluk):
    if buyukluk < 4.5: return "Hafif"
    elif buyukluk < 6.0: return "Orta"
    else: return "Yıkıcı"

df['Magnitude_Category'] = df['Magnitude'].apply(risk_siniflandir)

# 4. GEREKSİZ KOLONLARIN ÇIKARILMASI (Sıfır Sızıntı)
# NOT: Zamansal (Rolling) özellikler Data Leakage yarattığı için tamamen KALDIRILDI!
df_clean = df[['Year', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
             'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Magnitude_Category']].copy()

print(f"\nİşlenecek net satır sayısı: {len(df_clean)}")

# =====================================================================
# 5. EĞİTİM / TEST AYRIMI (K-Means'ten ÖNCE YAPILMALI!)
# =====================================================================
X = df_clean.drop(columns=['Magnitude', 'Magnitude_Category'])
y = df_clean['Magnitude']
y_cat = df_clean['Magnitude_Category']

# Rastgele ayrım yapıyoruz. Veri silme (undersampling) YOK.
X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
    X, y, y_cat, test_size=0.20, random_state=42
)

print(f"Eğitim seti: {len(X_train)} satır")
print(f"Test seti  : {len(X_test)} satır")

# =====================================================================
# 6. MEKANSAL ZEKA (Sadece Eğitim Setinde Eğitilen K-Means)
# =====================================================================
print("\n[ZEKA EKLENİYOR] Sızıntısız K-Means Fay Hattı Kümeleri Oluşturuluyor...")

# KMeans SADECE X_train ile öğrenir (Test verisi kesinlikle gösterilmez!)
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
train_clusters = kmeans.fit_predict(X_train[['Latitude', 'Longitude']])
test_clusters = kmeans.predict(X_test[['Latitude', 'Longitude']]) # Test verisi sadece tahmin edilir

# One-Hot Encoding ile bu 6 kümeyi kolonlara ayırıyoruz
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_clusters_ohe = ohe.fit_transform(train_clusters.reshape(-1, 1))
test_clusters_ohe = ohe.transform(test_clusters.reshape(-1, 1))

# OHE kolonlarını DataFrame'e çevirip ana veriye ekliyoruz
cluster_cols = [f'Bolge_{i}' for i in range(train_clusters_ohe.shape[1])]
X_train_clusters = pd.DataFrame(train_clusters_ohe, columns=cluster_cols, index=X_train.index)
X_test_clusters = pd.DataFrame(test_clusters_ohe, columns=cluster_cols, index=X_test.index)

X_train = pd.concat([X_train, X_train_clusters], axis=1)
X_test = pd.concat([X_test, X_test_clusters], axis=1)

print(f"  - 6 Sismik Küme başarıyla modellere entegre edildi.")

# =====================================================================
# 7. SCALER (YSA ve XGBoost İçin)
# =====================================================================
# Scaler, sadece X_train'e fit edilir (Bu kural zaten doğruydu)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 8. KAYDETME İŞLEMLERİ
os.makedirs('../data/processed', exist_ok=True)

X_train_scaled_df.to_csv('../data/processed/X_train.csv', index=False)
pd.DataFrame(y_train, columns=['Magnitude']).to_csv('../data/processed/y_train.csv', index=False)

X_test_scaled_df.to_csv('../data/processed/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Magnitude']).to_csv('../data/processed/y_test.csv', index=False)

pd.DataFrame(y_train_cat, columns=['Magnitude_Category']).to_csv('../data/processed/y_train_category.csv', index=False)
pd.DataFrame(y_test_cat, columns=['Magnitude_Category']).to_csv('../data/processed/y_test_category.csv', index=False)

joblib.dump(scaler, '../data/processed/deprem_scaler.pkl')
joblib.dump(kmeans, '../data/processed/deprem_kmeans.pkl') # Canlı sistemde arayüzden gelen enlem/boylamı kümelemek için Enes'e lazım olacak!

with open('../data/processed/scaler_features.txt', 'w', encoding='utf-8') as f:
    f.write(','.join(X_train.columns.tolist()))

print("\n[BAŞARILI] Veri Fabrikası işlemini tamamladı.")
print("Veriler tamamen sızıntısız, jüri onaylı ve ML yarışına hazır!")