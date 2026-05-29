import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. KİLİTLİ KASADAKİ TEST VERİLERİNİ YÜKLEME
print("Test verileri yükleniyor...")
X_test = pd.read_csv('../../data/processed/X_test.csv')
y_test = pd.read_csv('../../data/processed/y_test.csv').values.ravel()

# 2. EN SON KAYDEDİLEN MODELİ YÜKLEME
mevcut_dosyalar = os.listdir('.')
versiyonlar = [int(d.replace('rf_model_v', '').replace('.pkl', '')) for d in mevcut_dosyalar if d.startswith('rf_model_v') and d.endswith('.pkl')]

if not versiyonlar:
    raise FileNotFoundError("Kaydedilmiş bir model bulunamadı! Önce Eğitim kodunu çalıştırın.")

son_versiyon_no = max(versiyonlar)
model_adi = f'rf_model_v{son_versiyon_no}.pkl'

print(f"'{model_adi}' yükleniyor ve nihai test başlatılıyor...")
best_rf = joblib.load(model_adi)

# 3. NİHAİ TEST TAHMİNLERİ
y_pred = best_rf.predict(X_test)

# Görev Tanımındaki Metriklerin Hesaplanması
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred)

print("\n=== NİHAİ TEST BAŞARI KARNESİ ===")
print(f"Test MSE  : {test_mse:.4f}")
print(f"Test RMSE : {test_rmse:.4f}")
print(f"Test R²   : {test_r2:.4f}")

# 4. KARMAŞIKLIK MATRİSİ (Risk Sınıflandırması)
def risk_siniflandir(buyukluk):
    if buyukluk < 4.5: return "Hafif"
    elif buyukluk < 6.0: return "Orta"
    else: return "Yıkıcı"

y_test_kategorik = [risk_siniflandir(deger) for deger in y_test]
y_pred_kategorik = [risk_siniflandir(deger) for deger in y_pred]

sinif_isimleri = ["Hafif", "Orta", "Yıkıcı"]
cm = confusion_matrix(y_test_kategorik, y_pred_kategorik, labels=sinif_isimleri)

matris_adi = f'karmasiklik_matrisi_v{son_versiyon_no}.png'

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sinif_isimleri, yticklabels=sinif_isimleri)
plt.title(f"Deprem Risk Sınıflandırması - Nihai Test (v{son_versiyon_no})\nRMSE: {test_rmse:.4f}")
plt.ylabel("Gerçek Risk Durumu")
plt.xlabel("Modelin Tahmin Ettiği Risk")
plt.tight_layout()
plt.savefig(matris_adi)
plt.close()

print(f"\n[BAŞARILI] Karmaşıklık Matrisi grafiği '{matris_adi}' olarak kaydedildi.")

# 5. GERÇEK vs TAHMİN SCATTER PLOT
scatter_adi = f'gercek_vs_tahmin_v{son_versiyon_no}.png'

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, s=25, edgecolors='k', linewidth=0.3, c='steelblue')
# 45° referans çizgisi (mükemmel tahmin)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Tahmin (45°)')
plt.xlabel("Gerçek Büyüklük (Mw)", fontsize=12)
plt.ylabel("Tahmin Edilen Büyüklük (Mw)", fontsize=12)
plt.title(f"Gerçek vs Tahmin — Random Forest (v{son_versiyon_no})\nRMSE: {test_rmse:.4f} | R²: {test_r2:.4f}", fontsize=13)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(scatter_adi, dpi=150)
plt.close()

print(f"[BAŞARILI] Gerçek vs Tahmin grafiği '{scatter_adi}' olarak kaydedildi.")

# 6. RESIDUAL (HATA) HİSTOGRAMI
residual_adi = f'residual_histogram_v{son_versiyon_no}.png'
residuals = y_pred - y_test
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(mean_residual, color='red', linestyle='--', linewidth=2, label=f'Ortalama Hata: {mean_residual:.3f}')
plt.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, label='Sıfır Hata')
plt.xlabel("Tahmin Hatası (Mw)", fontsize=12)
plt.ylabel("Frekans", fontsize=12)
plt.title(f"Residual Dağılımı — Random Forest (v{son_versiyon_no})\nStd: {std_residual:.4f}", fontsize=13)
plt.legend(loc='best')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(residual_adi, dpi=150)
plt.close()

print(f"[BAŞARILI] Residual histogram grafiği '{residual_adi}' olarak kaydedildi.")

# 7. TEST SONUÇLARINI LOGLAMA
log_dosyasi = "rf_deney_gecmisi.txt"
zaman = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

log_metni = f"""
==================================================
AŞAMA 2: NİHAİ TEST RAPORU - {model_adi.replace('.pkl', '').upper()}
==================================================
Tarih        : {zaman}

[ NİHAİ TEST METRİKLERİ (YSA İLE KIYASLANACAK) ]
Test MSE   : {test_mse:.4f}
Test RMSE  : {test_rmse:.4f}
Test R²    : {test_r2:.4f}
--------------------------------------------------\n"""

with open(log_dosyasi, "a", encoding="utf-8") as dosya:
    dosya.write(log_metni)

print("Test sonuçları rapora eklendi.")
