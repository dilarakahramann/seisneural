import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("==================================================")
print(" SEISNEURAL - XGBOOST EĞİTİMİ (V3 - AKILLI VERİ / AĞIRLIKSIZ) ")
print("==================================================")

# 1. VERİLERİ YÜKLE
print("Veri Fabrikasından temizlenen ve K-Means zekası eklenen X_train yükleniyor...")
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

print(f"Eğitim Seti Boyutu: {X_train.shape}")
print("NOT: Yapay ağırlık cezaları (Sample Weights) modelin psikolojisini bozduğu için İPTAL EDİLDİ.")
print("     Model, K-Means 'Sismik Bölge' kolonlarını kullanarak doğal bir öğrenme yapacak.")

# 2. XGBOOST HİPERPARAMETRE HAVUZU (RandomizedSearch)
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

# 3. RANDOMIZED SEARCH KURULUMU
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

# 4. EĞİTİMİ BAŞLAT
print("\n[EĞİTİM BAŞLIYOR] XGBoost hiperparametre araması devrede...")
print("Bu işlem 15 farklı kombinasyon deneyecek. Lütfen bekleyin...\n")

# DİKKAT: sample_weight tamamen kaldırıldı! Model doğal akışında öğrenecek.
random_search.fit(X_train, y_train)

# 5. EĞİTİM SONUÇLARI VE METRİKLER
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# CV RMSE Hesabı
cv_mse = -random_search.best_score_
cv_rmse = np.sqrt(cv_mse)

# Eğitim Seti Üzerindeki Performans
train_predictions = best_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

overfitting_gap = cv_rmse - train_rmse

print("\n==================================================")
print("AŞAMA 1: EĞİTİM RAPORU - XGB_MODEL_V3 (AĞIRLIKSIZ)")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE: {cv_rmse:.4f}")
print(f"  CV Eğitim RMSE   : {train_rmse:.4f}")
print(f"  Eğitim R² Skoru  : {train_r2:.4f}")
print(f"  Overfitting Gap  : {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 6. MODELİ KAYDET
mevcut_dosyalar = os.listdir('.')
versiyonlar = [int(d.replace('xgb_model_v', '').replace('.pkl', '')) for d in mevcut_dosyalar if d.startswith('xgb_model_v') and d.endswith('.pkl')]
yeni_versiyon_no = max(versiyonlar) + 1 if versiyonlar else 1

model_adi = f'xgb_model_v{yeni_versiyon_no}.pkl'
model_path = f'./{model_adi}'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

# Deney geçmişini dosyaya yazdır
log_path = './xgb_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n==================================================\n")
    f.write(f"AŞAMA 1: EĞİTİM RAPORU - {model_adi.replace('.pkl', '').upper()}\n")
    f.write(f"==================================================\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE: {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru  : {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE: {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("\nEğitim süreci başarıyla tamamlandı. Artık Test (xgboost_test.py) aşamasına geçebilirsin!")