import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

print("==================================================")
print(" SEISNEURAL - YSA (MLP) EĞİTİMİ (V2 - ENERJİ DÖNÜŞÜMLÜ) ")
print("==================================================")

# 1. VERİLERİ YÜKLE
print("Veri Fabrikasından temizlenen X_train ve y_train yükleniyor...")
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print("NOT: sklearn MLPRegressor sample_weight desteklemez.")
print("      Hedef (Mw) -> Sismik Enerji'ye dönüştürülerek dar dağılım genişletilecek.")

# =====================================================================
# 2. HEDEF DÖNÜŞÜMÜ (TARGET TRANSFORMATION)
# =====================================================================
print("\n[HEDEF DÖNÜŞÜMÜ] Büyüklük (Mw) -> Sismik Enerji -> MinMaxScaler")

# 2a. Büyüklüğü (Mw) Sismik Enerjiye Çevir
# Fiziksel formül: Energy ∝ 10^(1.5 * Mw)
# Bu dönüşüm, dar Mw dağılımını (std: 0.40) logaritmik-üstel olarak genişletir.
y_train_energy = 10 ** (1.5 * y_train)
print(f"  Enerji aralığı (ham): [{y_train_energy.min():.2e}, {y_train_energy.max():.2e}]")

# 2b. Enerjiyi [0, 1] arasına sıkıştır (YSA stabilitesi için)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train_energy.reshape(-1, 1)).ravel()

print(f"  Enerji aralığı (scaled): [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")

# 2c. Scaler'ı TEST ve canlı sistemde kullanmak üzere KAYDET
# DİKKAT: Bu scaler SADECE y (hedef) değişken içindir.
# X değişkeni zaten preprocessing.py'de scale edilmiştir.
scaler_path = './mlp_y_scaler.pkl'
joblib.dump(y_scaler, scaler_path)
print(f"[BAŞARILI] Hedef dönüştürücü kaydedildi: {scaler_path}")
print("           (Bu dosya, test aşamasında enerjiyi geri Mw'ye çevirmek için şart)")
# =====================================================================

# 3. MLP HİPERPARAMETRE HAVUZU (RandomizedSearch)
print("\nHiperparametre Uzayı (RandomizedSearch) oluşturuluyor...")
param_distributions = {
    'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128, 64)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.005],
    'max_iter': [1000],
}

mlp_model = MLPRegressor(
    activation="relu",
    solver="adam",
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

# 4. RANDOMIZED SEARCH KURULUMU
random_search = RandomizedSearchCV(
    estimator=mlp_model,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=2,
    pre_dispatch='2*n_jobs',
    verbose=2,
    return_train_score=True,
    random_state=42,
)

# 5. EĞİTİMİ BAŞLAT
print("\n[EĞİTİM BAŞLIYOR] MLP (YSA) yapay sinir ağı eğitimi devrede...")
print("Bu işlem 10 farklı kombinasyon deneyecek. 10-30 dakika sürebilir.\n")

# DİKKAT: Artık orijinal y_train değil, sıkıştırılmış y_train_scaled veriyoruz!
# Modelin öğrendiği uzay: Sismik Enerji (0-1 arası)
random_search.fit(X_train, y_train_scaled)

# 6. EĞİTİM SONUÇLARI VE METRİKLER (Enerji Ölçeğinde)
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# CV RMSE Hesabı
cv_mse = -random_search.best_score_
cv_rmse = np.sqrt(cv_mse)

# Eğitim Seti Üzerindeki Performans (Enerji ölçeğinde)
train_predictions_scaled = best_model.predict(X_train)
train_mse = mean_squared_error(y_train_scaled, train_predictions_scaled)
train_rmse = np.sqrt(train_mse)

overfitting_gap = cv_rmse - train_rmse

# R² skorları (Enerji ölçeğinde)
train_r2 = r2_score(y_train_scaled, train_predictions_scaled)

print("\n==================================================")
print("AŞAMA 1: EĞİTİM RAPORU - MLP_MODEL_V2 (ENERJİ DÖNÜŞÜMLÜ)")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ ENERJİ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE (Enerji): {cv_rmse:.4f}")
print(f"  Eğitim R² Skoru   (Enerji): {train_r2:.4f}")
print(f"  CV Eğitim RMSE    (Enerji): {train_rmse:.4f}")
print(f"  Overfitting Gap   (Enerji): {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 7. ÖZNİTELİK ÖNEMİ (Permutation Importance)
print("\nPermutation Importance hesaplanıyor (MLP için)...")
perm_importance = permutation_importance(
    best_model, X_train, y_train_scaled, n_repeats=10, random_state=42, n_jobs=2
)

features = X_train.columns
importances = perm_importance.importances_mean

feature_importance_df = pd.DataFrame({
    'Öznitelik': features,
    'Etki Oranı (%)': importances * 100
}).sort_values(by='Etki Oranı (%)', ascending=False)

print("\n=== RİSKİ TETİKLEYEN FAKTÖRLER ===")
print(feature_importance_df.to_string(index=False))

# 8. MODELİ, GRAFİĞİ VE LOGU KAYDET
mevcut_dosyalar = os.listdir('.')
versiyonlar = [int(d.replace('mlp_model_v', '').replace('.pkl', ''))
               for d in mevcut_dosyalar if d.startswith('mlp_model_v') and d.endswith('.pkl')]
yeni_versiyon_no = max(versiyonlar) + 1 if versiyonlar else 1

model_adi = f'mlp_model_v{yeni_versiyon_no}.pkl'
model_path = f'./{model_adi}'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

oznitelik_adi = f'./oznitelik_onemi_v{yeni_versiyon_no}.png'
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Etki Oranı (%)', y='Öznitelik', data=feature_importance_df,
    hue='Öznitelik', palette='viridis', legend=False
)
plt.title(f"Riski En Çok Etkileyen Faktörler (v{yeni_versiyon_no} - Enerji)", fontsize=14)
plt.xlabel("Etki Oranı (%)", fontsize=12)
plt.ylabel("Öznitelikler", fontsize=12)
plt.tight_layout()
plt.savefig(oznitelik_adi)
plt.close()
print(f"[BAŞARILI] Öznitelik grafiği kaydedildi: {oznitelik_adi}")

# 9. LOG KAYDET
log_path = './mlp_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n==================================================\n")
    f.write(f"AŞAMA 1: EĞİTİM RAPORU - {model_adi.replace('.pkl', '').upper()} (ENERJİ DÖNÜŞÜMLÜ)\n")
    f.write(f"==================================================\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE (Enerji): {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru   (Enerji): {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE    (Enerji): {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap   (Enerji): {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("\nEğitim süreci başarıyla tamamlandı. Artık Test (mlp_test.py) aşamasına geçebilirsin!")
print("UYARI: Test kodunda ./mlp_y_scaler.pkl dosyasını load edip geri dönüşüm yapmayı UNUTMA!")
