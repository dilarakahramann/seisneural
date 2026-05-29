import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# macOS/Windows'ta çoklu işlem hatalarını önlemek için ana blok
if __name__ == '__main__':

    # =========================================================
    # 1. ORİJİNAL EĞİTİM VERİLERİNİ OKU (Sample Weight ile)
    # =========================================================
    print("Eğitim verileri yükleniyor...")
    X_train = pd.read_csv('../../data/processed/X_train_original.csv')
    y_train = pd.read_csv('../../data/processed/y_train_original.csv').values.ravel()
    sample_weights = pd.read_csv('../../data/processed/sample_weights.csv').values.ravel()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Öznitelikler: {list(X_train.columns)}")
    print(f"Sample weights aralığı: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # =========================================================
    # 2. MODEL VE PARAMETRE HAVUZU
    # =========================================================
    mlp_model = MLPRegressor(
        activation="relu",
        solver="adam",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    # RandomizedSearchCV: Rastgele kombinasyonlar dener (Hızlı arama)
    param_distributions = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128, 64)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.005],
        'max_iter': [1000],
    }

    # =========================================================
    # 3. EĞİTİM (RandomizedSearchCV)
    # =========================================================
    print("Hiperparametre optimizasyonu (RandomizedSearchCV) başlatılıyor. Bu işlem birkaç dakika sürebilir...")
    random_search = RandomizedSearchCV(
        estimator=mlp_model,
        param_distributions=param_distributions,
        n_iter=10,               # 10 farklı rastgele kombinasyon dene
        cv=5,                    # 5-Fold Cross Validation
        scoring='neg_mean_squared_error',
        n_jobs=2,                # Hafıza sızıntısını önlemek için çekirdek kısıtlaması
        pre_dispatch='2*n_jobs',
        verbose=2,
        return_train_score=True,
        random_state=42,
    )

    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    # =========================================================
    # 4. CROSS-VALIDATION (DOĞRULAMA) SONUÇLARI
    # =========================================================
    cv_sonuclari = random_search.cv_results_
    en_iyi_index = random_search.best_index_

    # ÖNCE validation RMSE tanımlanmalı (NameError düzeltmesi)
    cv_val_rmse = np.sqrt(-cv_sonuclari['mean_test_score'][en_iyi_index])
    cv_train_rmse = np.sqrt(-cv_sonuclari['mean_train_score'][en_iyi_index])
    overfit_gap = cv_train_rmse - cv_val_rmse

    # R² skorları da hesaplayalım (cross_val_score ile ayrıca)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_r2_scores = cross_val_score(
        random_search.best_estimator_, X_train, y_train, cv=kfold, scoring='r2'
    )
    cv_val_r2 = cv_r2_scores.mean()

    print("\n=== CROSS-VALIDATION (DOĞRULAMA) KONTROLÜ ===")
    print(f"En İyi Parametreler             : {random_search.best_params_}")
    print(f"K-Fold Ortalama Doğrulama RMSE  : {cv_val_rmse:.4f}")
    print(f"K-Fold Ortalama Doğrulama R²    : {cv_val_r2:.4f}")
    print(f"K-Fold Ortalama Eğitim RMSE     : {cv_train_rmse:.4f}")
    print(f"Overfitting Gap                 : {overfit_gap:.4f}")

    # =========================================================
    # 5. ÖZNİTELİK ÖNEMİ (Permutation Importance)
    # =========================================================
    best_mlp = random_search.best_estimator_
    print("\nPermutation Importance hesaplanıyor (MLP için)...")
    perm_importance = permutation_importance(
        best_mlp, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )

    features = X_train.columns
    importances = perm_importance.importances_mean

    feature_importance_df = pd.DataFrame({
        'Öznitelik': features,
        'Etki Oranı (%)': importances * 100
    }).sort_values(by='Etki Oranı (%)', ascending=False)

    print("\n=== RİSKİ TETİKLEYEN FAKTÖRLER ===")
    print(feature_importance_df.to_string(index=False))

    # =========================================================
    # 6. MODELİ, GRAFİĞİ VE LOGU VERSİYONLU KAYDETME
    # =========================================================
    mevcut_dosyalar = os.listdir('.')
    versiyonlar = [int(d.replace('mlp_model_v', '').replace('.pkl', '')) for d in mevcut_dosyalar if d.startswith('mlp_model_v') and d.endswith('.pkl')]
    yeni_versiyon_no = max(versiyonlar) + 1 if versiyonlar else 1

    model_adi = f'mlp_model_v{yeni_versiyon_no}.pkl'
    oznitelik_adi = f'oznitelik_onemi_v{yeni_versiyon_no}.png'

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Etki Oranı (%)',
        y='Öznitelik',
        data=feature_importance_df,
        hue='Öznitelik',
        palette='viridis',
        legend=False
    )
    plt.title(f"Riski En Çok Etkileyen Faktörler (v{yeni_versiyon_no})", fontsize=14)
    plt.xlabel("Etki Oranı (%)", fontsize=12)
    plt.ylabel("Öznitelikler", fontsize=12)
    plt.tight_layout()
    plt.savefig(oznitelik_adi)
    plt.close()

    joblib.dump(best_mlp, model_adi)
    print(f"\n[BAŞARILI] Model '{model_adi}' ve Öznitelik grafiği '{oznitelik_adi}' kaydedildi.")

    # =========================================================
    # 7. EĞİTİM SONUÇLARINI LOGLAMA
    # =========================================================
    log_dosyasi = "mlp_deney_gecmisi.txt"
    zaman = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    log_metni = f"""
==================================================
AŞAMA 1: EĞİTİM RAPORU - {model_adi.replace('.pkl', '').upper()}
==================================================
Tarih        : {zaman}
Parametreler : {random_search.best_params_}

[ DOĞRULAMA METRİKLERİ ]
  CV Doğrulama RMSE: {cv_val_rmse:.4f}
  CV Doğrulama R²  : {cv_val_r2:.4f}
  CV Eğitim RMSE   : {cv_train_rmse:.4f}
  Overfitting Gap  : {overfit_gap:.4f}
--------------------------------------------------\n"""

    with open(log_dosyasi, "a", encoding="utf-8") as dosya:
        dosya.write(log_metni)
