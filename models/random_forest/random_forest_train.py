import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

print("==================================================")
print(" SEISNEURAL - RANDOM FOREST EĞİTİMİ (V4) BAŞLIYOR ")
print("==================================================")

# 1. VERİLERİ YÜKLE
print("Veri Fabrikasından temizlenen X_train ve y_train yükleniyor...")
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

print(f"Eğitim Seti Boyutu: {X_train.shape}")
print("NOT: Regresyonda Sample Weighting zararlı çıktığı için kaldırıldı.")
print("     Model doğal dağılımda öğrenecek.")

# 2. RANDOM FOREST - RandomizedSearchCV (20 iterasyon)
print("\nHiperparametre Uzayı (RandomizedSearch) oluşturuluyor...")
print("20 Kombinasyon × 5-Fold = 100 fit. Tahmini süre: 10-20 dakika.\n")

param_distributions = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'max_features': ['sqrt']
}

# 3. MODEL VE RANDOMIZED SEARCH KURULUMU
rf_model = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# 4. EĞİTİMİ BAŞLAT
print("[EĞİTİM BAŞLIYOR] RandomizedSearchCV 20 kombinasyon deniyor...\n")

# DİKKAT: Ağırlıklandırma tamamen kaldırıldı.
random_search.fit(X_train, y_train)

# 5. EĞİTİM SONUÇLARI VE METRİKLER
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# CV RMSE Hesabı
cv_mse = -random_search.best_score_
cv_rmse = np.sqrt(cv_mse)

# Eğitim Seti Üzerindeki Performans (Overfitting kontrolü için)
train_predictions = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

overfitting_gap = cv_rmse - train_rmse

# Eğitim Seti Üzerindeki R² (Overfitting kontrolü için)
train_r2 = r2_score(y_train, train_predictions)

print("\n==================================================")
print("AŞAMA 1: EĞİTİM RAPORU - RF_MODEL_V4")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE: {cv_rmse:.4f}")
print(f"  Eğitim R² Skoru  : {train_r2:.4f}")
print(f"  CV Eğitim RMSE   : {train_rmse:.4f}")
print(f"  Overfitting Gap  : {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 6. MODELİ KAYDET
model_path = './rf_model_v4.pkl'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

# Deney geçmişini dosyaya yazdır
log_path = './rf_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n==================================================\n")
    f.write(f"AŞAMA 1: EĞİTİM RAPORU - RF_MODEL_V4 (RandomizedSearch, Ağırlıksız)\n")
    f.write(f"==================================================\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE: {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru  : {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE: {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("Eğitim süreci başarıyla tamamlandı. Artık Test (random_forest_test.py) aşamasına geçebilirsin!")
