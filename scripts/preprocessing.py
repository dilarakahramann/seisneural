import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


print("Ham veri okunuyor...")
df = pd.read_csv('../data/raw/hamdata.csv', encoding='utf-8')
print(f"Toplam satır: {len(df)}")

# 1. TARİH PARÇALAMA
# Format: "18/04/2026 10:47:02"
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['Date'])  # Tarih parse edilemeyen satırları çıkar

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 2. CYCLICAL ENCODING (Tarih döngüsel yapısı)
# Ay: 1-12 → sin/cos
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
# Gün: 1-31 → sin/cos (gün sayısı ayda değişir ama 31 normalize edici)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

# Yıl için cyclical encoding anlamlı değil (aralık çok geniş değil),
# sadece Min-Max scaling uygulanacak.

# 3. AYKIRI DEĞER TEMİZLİĞİ
# Künyede derinlik slider'ı 0-150 km. Türkiye için 150 km üstü outlier kabul edilir.
initial_count = len(df)
df = df[df['Depth'] <= 150].copy()
removed_depth = initial_count - len(df)
print(f"Depth > 150 km olan {removed_depth} satır çıkarıldı.")

# Magnitude < 3.0 olanlar da künyede belirtildiği üzere filtrelenmiş zaten (min 3.0)

# 4. MAGNITUDE KATEGORİZASYONU (Karmaşıklık Matrisi için)
# Model Magnitude tahmin eder. Confusion matrix hedef değişken üzerinden kurulur.
# Derinlik değil, Magnitude kategorileri kullanılır.
# Bu kolon model eğitimine dahil edilmez, sadece test/metrik amaçlıdır.
def magnitude_category(mag):
    if mag < 4.5:
        return "Hafif"
    elif mag < 6.0:
        return "Orta"
    else:
        return "Yikici"

df['Magnitude_Category'] = df['Magnitude'].apply(magnitude_category)

# 5. GEREKSİZ KOLONLARI ÇIKAR
# Künyede sadece Enlem, Boylam, Derinlik, Tarih ve Magnitude var.
# 'Type' (ML/Mw/Md) kolonunu atmamızın sebebi: data leakage engellemek.
# Ayrıca farklı ölçüm tipleri aynı 'Magnitude' altında birleştirilmiştir.
df_clean = df[['Year', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
             'Latitude', 'Longitude', 'Depth', 'Magnitude', 'Magnitude_Category']].copy()

print(f"İşlenecek satır sayısı: {len(df_clean)}")
print(f"Magnitude dağılımı:\n{df_clean['Magnitude'].describe()}")
print(f"Depth dağılımı:\n{df_clean['Depth'].describe()}")

# 6. EĞİTİM / TEST AYRIMI (%80 / %20)
X = df_clean.drop(columns=['Magnitude', 'Magnitude_Category'])
y = df_clean['Magnitude']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nEğitim seti: {len(X_train)} satır")
print(f"Test seti  : {len(X_test)} satır")


# Eğitim setini birleştir (X + y)
train_combined = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

# Magnitude'u kategorilere ayır (dengesizlik analizi için)
def mag_category(mag):
    if mag < 3.3:
        return "Dusuk"
    elif mag < 3.6:
        return "Dusuk_Orta"
    elif mag < 4.0:
        return "Orta"
    elif mag < 4.5:
        return "Orta_Yuksek"
    else:
        return "Yuksek"

train_combined['Mag_Cat'] = train_combined['Magnitude'].apply(mag_category)

cat_counts = train_combined['Mag_Cat'].value_counts().sort_index()
print("\nOrijinal eğitim seti kategori dağılımı:")
print(cat_counts)

# Hedef: En az sayıdaki kategorinin 3 katını hedef al
# Böylece az temsil edilen kategoriler korunur, fazla temsil edilenler azaltılır
min_cat_count = cat_counts.min()
target_per_cat = min_cat_count * 3
print(f"\nHedef kategori başına örnek sayısı: {target_per_cat}")

balanced_list = []
for cat in train_combined['Mag_Cat'].unique():
    cat_df = train_combined[train_combined['Mag_Cat'] == cat]
    if len(cat_df) > target_per_cat:
        balanced_list.append(cat_df.sample(n=target_per_cat, random_state=42))
    else:
        balanced_list.append(cat_df)

balanced_train = pd.concat(balanced_list).drop(columns=['Mag_Cat'])

# Dengeli eğitim setini X ve y olarak ayır
X_train_bal = balanced_train.drop(columns=['Magnitude'])
y_train_bal = balanced_train['Magnitude']

print(f"\nDengeli eğitim seti: {len(X_train_bal)} satır")
print(f"Dengeli magnitude dağılımı:\n{y_train_bal.describe()}")


# 8. SAMPLE WEIGHTS (RF Eğitiminde Kullanılmak Üzere)
# Her magnitude değerinin frekansına göre ağırlıklandırma.
# Nadir magnitude değerlerine (büyük depremler) daha yüksek ağırlık.

hist, bin_edges = np.histogram(y_train, bins=20)
bin_indices = np.digitize(y_train, bin_edges[:-1])
bin_indices = np.clip(bin_indices - 1, 0, len(hist) - 1)  # Sınır kontrolü
bin_freq = hist[bin_indices]
sample_weights = 1.0 / (bin_freq + 1e-6)
sample_weights = sample_weights / sample_weights.max()  # 0-1 arası normalize

print(f"\nSample weights hesaplandı. Aralık: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

# 9. MIN-MAX SCALING
# Scaler'ı sadece DENGELİ eğitim verisiyle fit ediyoruz.
# Test setine dokunulmaz.

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_bal.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# 10. KAYDETME
os.makedirs('../data/processed', exist_ok=True)

# Ham işlenmiş veri (scaler öncesi, referans için)
df_clean.to_csv('../data/processed/yeni_ysadata.csv', index=False)

# Scaled veriler — DENGELİ eğitim seti
X_train_scaled_df.to_csv('../data/processed/X_train.csv', index=False)
pd.DataFrame(y_train_bal, columns=['Magnitude']).to_csv('../data/processed/y_train.csv', index=False)

# Orijinal eğitim seti (opsiyonel, karşılaştırma için)
X_train_scaled_orig = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_train_scaled_orig.to_csv('../data/processed/X_train_original.csv', index=False)
pd.DataFrame(y_train, columns=['Magnitude']).to_csv('../data/processed/y_train_original.csv', index=False)

# Test seti
X_test_scaled_df.to_csv('../data/processed/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Magnitude']).to_csv('../data/processed/y_test.csv', index=False)

# Sample weights
pd.DataFrame(sample_weights, columns=['sample_weight']).to_csv('../data/processed/sample_weights.csv', index=False)

# Scaler ve feature isimleri
joblib.dump(scaler, '../data/processed/deprem_scaler.pkl')
with open('../data/processed/scaler_features.txt', 'w', encoding='utf-8') as f:
    f.write(','.join(X_train_bal.columns.tolist()))

print("\n[BAŞARILI] Yeni veriler kaydedildi:")
print("  - data/processed/yeni_ysadata.csv")
print("  - data/processed/X_train.csv, data/processed/y_train.csv  (DENGELİ)")
print("  - data/processed/X_train_original.csv, data/processed/y_train_original.csv  (ORIJINAL)")
print("  - data/processed/X_test.csv, data/processed/y_test.csv")
print("  - data/processed/sample_weights.csv")
print("  - data/processed/deprem_scaler.pkl")
print("  - data/processed/scaler_features.txt")

# Özet
print("\n=== ÖZET ===")
print(f"Scaler kolon sırası: {list(X_train_bal.columns)}")
print(f"Scaler data_min: {scaler.data_min_}")
print(f"Scaler data_max: {scaler.data_max_}")
print(f"Dengeli eğitim: {len(X_train_bal)} satır | Test: {len(X_test)} satır")
print(f"Orijinal eğitim: {len(X_train)} satır (opsiyonel)")
