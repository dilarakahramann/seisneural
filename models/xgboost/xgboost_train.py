import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# macOS/Windows'ta çoklu işlem hatalarını önlemek için ana blok
if __name__ == '__main__':

    # 1. VERİLERİ YÜKLEME (ORİJİNAL Eğitim Verisi + Sample Weights)
    print("Eğitim verileri yükleniyor...")
    X_train = pd.read_csv('../../data/processed/X_train_original.csv')
    y_train = pd.read_csv('../../data/processed/y_train_original.csv').values.ravel()
    sample_weights = pd.read_csv('../../data/processed/sample_weights.csv').values.ravel()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Öznitelikler: {list(X_train.columns)}")
    print(f"Sample weights aralığı: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

    # 2. DENENECEK PARAMETRE HAVUZU
    param_distributions = {
        'max_depth': [3, 5, 7, 9],                 # Karar ağaçlarının derinliği
        'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Öğrenme hızı
        'n_estimators': [100, 200, 300, 500],        # Kurulacak ağaç sayısı
        'subsample': [0.7, 0.8, 1.0],                # Aşırı öğrenmeyi engellemek için örneklem oranı
        'colsample_bytree': [0.7, 0.8, 1.0]          # Ağaç başına kullanılacak kolon oranı
    }

    # Temel model iskeleti
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 3. RANDOMIZED SEARCH İLE EN İYİ PARAMETRELERİ ARAMA
    print("Hiperparametre optimizasyonu başlatılıyor, bu birkaç dakika sürebilir...\n")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=15,          # 15 farklı rastgele kombinasyon deneyecek
        scoring='neg_mean_squared_error',
        cv=5,               # 5-Fold Cross Validation (Künyede k-Katlı Çapraz Doğrulama)
        verbose=2,          # Konsola sürecin ilerleyişini yazdırır
        random_state=42,
        n_jobs=2,           # Hafıza sızıntısını önlemek için çekirdek kısıtlaması
        pre_dispatch='2*n_jobs',
        return_train_score=True,
    )

    random_search.fit(X_train, y_train, sample_weight=sample_weights)

    # 4. CROSS-VALIDATION SONUÇLARI
    cv_sonuclari = random_search.cv_results_
    en_iyi_index = random_search.best_index_

    cv_val_rmse = np.sqrt(-cv_sonuclari['mean_test_score'][en_iyi_index])
    cv_train_rmse = np.sqrt(-cv_sonuclari['mean_train_score'][en_iyi_index])
    overfit_gap = cv_train_rmse - cv_val_rmse

    # R² skorları
    from sklearn.model_selection import cross_val_score, KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
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

    # 5. ÖZNİTELİK ÖNEMİ (Feature Importance)
    best_xgb = random_search.best_estimator_
    importances = best_xgb.feature_importances_
    features = X_train.columns

    feature_importance_df = pd.DataFrame({
        'Öznitelik': features,
        'Etki Oranı (%)': importances * 100
    }).sort_values(by='Etki Oranı (%)', ascending=False)

    print("\n=== RİSKİ TETİKLEYEN FAKTÖRLER ===")
    print(feature_importance_df.to_string(index=False))

    # 6. MODELİ, GRAFİĞİ VE LOGU VERSİYONLU KAYDETME
    mevcut_dosyalar = os.listdir('.')
    versiyonlar = [int(d.replace('xgb_model_v', '').replace('.pkl', '')) for d in mevcut_dosyalar if d.startswith('xgb_model_v') and d.endswith('.pkl')]
    yeni_versiyon_no = max(versiyonlar) + 1 if versiyonlar else 1

    model_adi = f'xgb_model_v{yeni_versiyon_no}.pkl'
    oznitelik_adi = f'oznitelik_onemi_v{yeni_versiyon_no}.png'

    import matplotlib.pyplot as plt
    import seaborn as sns

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

    joblib.dump(best_xgb, model_adi)
    print(f"\n[BAŞARILI] Model '{model_adi}' ve Öznitelik grafiği '{oznitelik_adi}' kaydedildi.")

    # 7. EĞİTİM SONUÇLARINI LOGLAMA
    log_dosyasi = "xgb_deney_gecmisi.txt"
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
