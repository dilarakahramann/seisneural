import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# macOS/Windows'ta çoklu işlem hatalarını önlemek için ana blok
if __name__ == '__main__':

    # 1. VERİLERİ YÜKLEME (Sadece Eğitim Verisi)
    print("Eğitim verileri yükleniyor...")
    X_train = pd.read_csv('../../data/processed/X_train.csv')
    y_train = pd.read_csv('../../data/processed/y_train.csv').values.ravel()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Öznitelikler: {list(X_train.columns)}")

    # 2. MODEL VE PARAMETRE HAZIRLIĞI
    rf_model = RandomForestRegressor(random_state=42)

    # GridSearchCV: Tüm kombinasyonları dener (En kapsamlı arama)
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # 3. EĞİTİM (GridSearchCV)
    print("Hiperparametre optimizasyonu (GridSearchCV) başlatılıyor. Bu işlem uzun sürebilir...")
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=2,               # Hafıza sızıntısını (memory leak) önlemek için çekirdek kısıtlaması
        pre_dispatch='2*n_jobs',
        verbose=2,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    # 4. CROSS-VALIDATION (DOĞRULAMA) SONUÇLARI
    cv_sonuclari = grid_search.cv_results_
    en_iyi_index = grid_search.best_index_

    # ÖNCE validation RMSE tanımlanmalı (NameError düzeltmesi)
    cv_val_rmse = np.sqrt(-cv_sonuclari['mean_test_score'][en_iyi_index])
    cv_train_rmse = np.sqrt(-cv_sonuclari['mean_train_score'][en_iyi_index])
    overfit_gap = cv_train_rmse - cv_val_rmse

    print("\n=== CROSS-VALIDATION (DOĞRULAMA) KONTROLÜ ===")
    print(f"En İyi Parametreler             : {grid_search.best_params_}")
    print(f"K-Fold Ortalama Doğrulama RMSE  : {cv_val_rmse:.4f}")
    print(f"K-Fold Ortalama Eğitim RMSE     : {cv_train_rmse:.4f}")
    print(f"Overfitting Gap                 : {overfit_gap:.4f}")

    # 5. ÖZNİTELİK ÖNEMİ (Feature Importance)
    best_rf = grid_search.best_estimator_
    importances = best_rf.feature_importances_
    features = X_train.columns

    feature_importance_df = pd.DataFrame({
        'Öznitelik': features,
        'Etki Oranı (%)': importances * 100
    }).sort_values(by='Etki Oranı (%)', ascending=False)

    print("\n=== RİSKİ TETİKLEYEN FAKTÖRLER ===")
    print(feature_importance_df.to_string(index=False))

    # 6. MODELİ VE GRAFİĞİ VERSİYONLU KAYDETME
    mevcut_dosyalar = os.listdir('.')
    versiyonlar = [int(d.replace('rf_model_v', '').replace('.pkl', '')) for d in mevcut_dosyalar if d.startswith('rf_model_v') and d.endswith('.pkl')]
    yeni_versiyon_no = max(versiyonlar) + 1 if versiyonlar else 1

    model_adi = f'rf_model_v{yeni_versiyon_no}.pkl'
    oznitelik_adi = f'oznitelik_onemi_v{yeni_versiyon_no}.png'

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Etki Oranı (%)', y='Öznitelik', data=feature_importance_df, hue='Öznitelik', palette='viridis', legend=False)
    plt.title(f"Riski En Çok Etkileyen Faktörler (v{yeni_versiyon_no})", fontsize=14)
    plt.xlabel("Etki Oranı (%)", fontsize=12)
    plt.ylabel("Öznitelikler", fontsize=12)
    plt.tight_layout()
    plt.savefig(oznitelik_adi)
    plt.close()

    joblib.dump(best_rf, model_adi)
    print(f"\n[BAŞARILI] Model '{model_adi}' ve Öznitelik grafiği '{oznitelik_adi}' kaydedildi.")

    # 7. EĞİTİM SONUÇLARINI LOGLAMA
    log_dosyasi = "rf_deney_gecmisi.txt"
    zaman = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    log_metni = f"""
==================================================
AŞAMA 1: EĞİTİM RAPORU - {model_adi.replace('.pkl', '').upper()}
==================================================
Tarih        : {zaman}
Parametreler : {grid_search.best_params_}

[ DOĞRULAMA METRİKLERİ ]
  CV Doğrulama RMSE: {cv_val_rmse:.4f}
  CV Eğitim RMSE   : {cv_train_rmse:.4f}
  Overfitting Gap  : {overfit_gap:.4f}
--------------------------------------------------\n"""

    with open(log_dosyasi, "a", encoding="utf-8") as dosya:
        dosya.write(log_metni)
