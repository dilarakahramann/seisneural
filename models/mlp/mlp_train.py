import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

print("==================================================")
print(" SEISNEURAL - YSA (MLP) EĞİTİMİ (V4) BAŞLIYOR ")
print("==================================================")

# 1. VERİLERİ YÜKLE
print("Veri Fabrikasından temizlenen X_train ve y_train yükleniyor...")
X_train = pd.read_csv('../../data/processed/X_train.csv')
y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

print(f"Eğitim Seti Boyutu: {X_train.shape}")
print("NOT: sklearn MLPRegressor sample_weight desteklemez. Model ağırlıksız eğitilecek.")

# 2. MLP HİPERPARAMETRE HAVUZU (RandomizedSearch)
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

# 3. RANDOMIZED SEARCH KURULUMU
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

# 4. EĞİTİMİ BAŞLAT
print("\n[EĞİTİM BAŞLIYOR] MLP (YSA) yapay sinir ağı eğitimi devrede...")
print("Bu işlem 10 farklı kombinasyon deneyecek. 10-30 dakika sürebilir.\n")

random_search.fit(X_train, y_train)

# 5. EĞİTİM SONUÇLARI VE METRİKLER
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
print("AŞAMA 1: EĞİTİM RAPORU - MLP_MODEL_V4")
print("==================================================")
print(f"En İyi Parametreler : {best_params}")
print("\n[ DOĞRULAMA METRİKLERİ ]")
print(f"  CV Doğrulama RMSE: {cv_rmse:.4f}")
print(f"  Eğitim R² Skoru  : {train_r2:.4f}")
print(f"  CV Eğitim RMSE   : {train_rmse:.4f}")
print(f"  Overfitting Gap  : {overfitting_gap:.4f}")
print("--------------------------------------------------")

# 6. ÖZNİTELİK ÖNEMİ (Permutation Importance)
print("\nPermutation Importance hesaplanıyor (MLP için)...")
perm_importance = permutation_importance(
    best_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

features = X_train.columns
importances = perm_importance.importances_mean

feature_importance_df = pd.DataFrame({
    'Öznitelik': features,
    'Etki Oranı (%)': importances * 100
}).sort_values(by='Etki Oranı (%)', ascending=False)

print("\n=== RİSKİ TETİKLEYEN FAKTÖRLER ===")
print(feature_importance_df.to_string(index=False))

# 7. MODELİ, GRAFİĞİ VE LOGU KAYDET
model_path = './mlp_model_v4.pkl'
joblib.dump(best_model, model_path)
print(f"\n[BAŞARILI] Model kaydedildi: {model_path}")

oznitelik_adi = './oznitelik_onemi_v4.png'
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Etki Oranı (%)', y='Öznitelik', data=feature_importance_df,
    hue='Öznitelik', palette='viridis', legend=False
)
plt.title("Riski En Çok Etkileyen Faktörler (v4)", fontsize=14)
plt.xlabel("Etki Oranı (%)", fontsize=12)
plt.ylabel("Öznitelikler", fontsize=12)
plt.tight_layout()
plt.savefig(oznitelik_adi)
plt.close()
print(f"[BAŞARILI] Öznitelik grafiği kaydedildi: {oznitelik_adi}")

# 8. LOG KAYDET
log_path = './mlp_deney_gecmisi.txt'
with open(log_path, 'a', encoding='utf-8') as f:
    f.write(f"\n==================================================\n")
    f.write(f"AŞAMA 1: EĞİTİM RAPORU - MLP_MODEL_V2\n")
    f.write(f"==================================================\n")
    f.write(f"Parametreler: {best_params}\n")
    f.write(f"CV Doğrulama RMSE: {cv_rmse:.4f}\n")
    f.write(f"Eğitim R² Skoru  : {train_r2:.4f}\n")
    f.write(f"CV Eğitim RMSE: {train_rmse:.4f}\n")
    f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
    f.write("-" * 50 + "\n")

print("Eğitim süreci başarıyla tamamlandı. Artık Test (mlp_test.py) aşamasına geçebilirsin!")
