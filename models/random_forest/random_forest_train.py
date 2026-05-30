import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

print("==================================================")
print(" SEISNEURAL - RANDOM FOREST EĞİTİMİ (V2) BAŞLIYOR ")
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

# 3. RANDOM FOREST (GridSearch)
print("\nHiperparametre Uzayı (Grid) oluşturuluyor...")
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', None],
    'bootstrap': [True]
}

# 4. MODEL VE GRID SEARCH KURULUMU
rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # 5-Fold Cross Validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # Tüm işlemci çekirdeklerini kullan
    verbose=2
)

# 5. EĞİTİMİ BAŞLAT
print("\n[EĞİTİM BAŞLIYOR]...")
print("Bu işlem hiperparametre sayısına ve veri büyüklüğüne göre 15-40 dakika sürebilir.\n")

# DİKKAT: Ağırlıkları sadece fit aşamasında sisteme veriyoruz
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# 6. EĞİTİM SONUÇLARI VE METRİKLER
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# CV RMSE Hesabı
cv_mse = -grid_search.best_score_
cv_rmse = np.sqrt(cv_mse)

# Eğitim Seti Üzerindeki Performans (Overfitting kontrolü için)
train_predictions = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

overfitting_gap = cv_rmse - train_rmse

# Eğitim Seti Üzerindeki R² (Overfitting kontrolü için)
train_r2 = r2_score(y_train, train_predictions)

print("\n==================================================")
print("AŞAMA 1: EĞİTİM RAPORU - RF_MODEL_V2")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE: {cv_rmse:.4f}")
print(f"  Eğitim R² Skoru  : {train_r2:.4f}")
print(f"  CV Eğitim RMSE   : {train_rmse:.4f}")
print(f"  Overfitting Gap  : {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 7. MODELİ KAYDET
model_path = './rf_model_v2.pkl'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

# Deney geçmişini dosyaya yazdır
log_path = './rf_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\nVersiyon: V2 (Smoothed Weights & Tam Veri)\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE: {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru  : {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE: {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("Eğitim süreci başarıyla tamamlandı. Artık Test (random_forest_test.py) aşamasına geçebilirsin!")
