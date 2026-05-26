import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# CSV DOSYALARINI OKU
# =========================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("═" * 60)
print("         FEATURE ENGINEERING BAŞLADI")
print("═" * 60)

# ---------------------------------------------------------
# 1. Latitude * Longitude Feature
# ---------------------------------------------------------

X_train["Lat_Long"] = X_train["Latitude"] * X_train["Longitude"]
X_test["Lat_Long"] = X_test["Latitude"] * X_test["Longitude"]

# ---------------------------------------------------------
# 2. Month Sin/Cos Features (Cyclical Encoding)
# ---------------------------------------------------------

X_train["Month_sin"] = np.sin(2 * np.pi * X_train["Month"] / 12)
X_train["Month_cos"] = np.cos(2 * np.pi * X_train["Month"] / 12)

X_test["Month_sin"] = np.sin(2 * np.pi * X_test["Month"] / 12)
X_test["Month_cos"] = np.cos(2 * np.pi * X_test["Month"] / 12)

# ---------------------------------------------------------
# 3. Depth Category Feature
# ---------------------------------------------------------

def depth_category(depth):

    if depth < 0.33:
        return 0   # Sığ Deprem

    elif depth < 0.66:
        return 1   # Orta Derinlik

    else:
        return 2   # Derin Deprem

X_train["Depth_Category"] = X_train["Depth"].apply(depth_category)
X_test["Depth_Category"] = X_test["Depth"].apply(depth_category)

# =========================================================
# FEATURE ENGINEERING SONUÇLARI
# =========================================================

print("\nYeni Özellikler Eklendi:")
print(X_train.columns)

# =========================================================
# NORMALIZATION (Scaling)
# =========================================================

print("\n" + "═" * 60)
print("              NORMALIZATION")
print("═" * 60)

# Min-Max Scaler oluştur
scaler = MinMaxScaler()

# X_train üzerinde fit + transform
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

# X_test üzerinde sadece transform
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

print("\nNormalization Tamamlandı.")

# =========================================================
# YENİ DATASETLERİ KAYDET
# =========================================================

X_train_scaled.to_csv("X_train_featured.csv", index=False)
X_test_scaled.to_csv("X_test_featured.csv", index=False)

print("\nYeni Dataset Dosyaları Kaydedildi:")
print("- X_train_featured.csv")
print("- X_test_featured.csv")

# =========================================================
# İLK 5 SATIRI GÖSTER
# =========================================================

print("\n" + "═" * 60)
print("     FEATURE ENGINEERING SONRASI İLK 5 SATIR")
print("═" * 60)

print(X_train_scaled.head())