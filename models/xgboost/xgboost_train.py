import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import joblib
import os

print("==================================================")
print(" SEISNEURAL - XGBOOST EĞİTİMİ (V2) BAŞLIYOR ")
print("==================================================")

# 1. VERİLERİ YÜKLE
print("Veri Fabrikasından temizlenen X_train ve y_train yükleniyor...")
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

print(f"Eğitim Seti Boyutu: {X_train.shape}")

# 2. AĞIRLIK HESAPLAMA FONKSİYONU (On-the-fly Smoothed Weights)
def calculate_smoothed_weights(y_data, bins=20):
    """Deprem büyüklük frekanslarına göre karekök ile yumuşatılmış ağırlık hesaplar."""
    print("Sınıf dengesizliği için 'Smoothed Sample Weights' hesaplanıyor...")
    hist, bin_edges = np.histogram(y_data, bins=bins)
    bin_indices = np.digitize(y_data, bin_edges[:-1])
    bin_indices = np.clip(bin_indices - 1, 0, len(hist) - 1)
    bin_freq = hist[bin_indices]

    # Ekstrem cezaları engellemek için karekök (sqrt) ile yumuşatma
    weights = 1.0 / np.sqrt(bin_freq + 1e-6)
    normalized_weights = weights / weights.max()

    print(f"Ağırlık Aralığı: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")
    return normalized_weights

sample_weights = calculate_smoothed_weights(y_train)

# 3. XGBOOST HİPERPARAMETRE HAVUZU (RandomizedSearch)
print("\nHiperparametre Uzayı (RandomizedSearch) oluşturuluyor...")
param_distributions = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
)

# 4. RANDOMIZED SEARCH KURULUMU
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=2,
    pre_dispatch='2*n_jobs',
    return_train_score=True
)

# 5. EĞİTİMİ BAŞLAT
print("\n[EĞİTİM BAŞLIYOR] XGBoost hiperparametre araması devrede...")
print("Bu işlem 15 farklı kombinasyon deneyecek. 10-25 dakika sürebilir.\n")

random_search.fit(X_train, y_train, sample_weight=sample_weights)

# 6. EĞİTİM SONUÇLARI VE METRİKLER
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# CV RMSE Hesabı
cv_mse = -random_search.best_score_
cv_rmse = np.sqrt(cv_mse)

# Eğitim Seti Üzerindeki Performans
train_predictions = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

overfitting_gap = cv_rmse - train_rmse

# R² skorları (Sadece Eğitim Seti Üzerinde - Gerçek R2 Test Dosyasında Çıkacak)
from sklearn.metrics import r2_score
train_r2 = r2_score(y_train, train_predictions)

print("\n==================================================")
print("AŞAMA 1: EĞİTİM RAPORU - XGB_MODEL_V2")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE: {cv_rmse:.4f}")
print(f"  CV Eğitim RMSE   : {train_rmse:.4f}")
print(f"  Eğitim R² Skoru  : {train_r2:.4f}")
print(f"  Overfitting Gap  : {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 7. MODELİ KAYDET
model_path = './xgb_model_v2.pkl'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

# Deney geçmişini dosyaya yazdır
log_path = './xgb_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n==================================================\n")
    f.write(f"AŞAMA 1: EĞİTİM RAPORU - XGB_MODEL_V2\n")
    f.write(f"==================================================\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE: {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru  : {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE: {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("Eğitim süreci başarıyla tamamlandı. Artık Test (xgboost_test.py) aşamasına geçebilirsin!")
