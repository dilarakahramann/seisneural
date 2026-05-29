import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# 1. Verileri Yükleme (GitHub için Göreceli Dosya Yolları)
# Not: Bu kodun çalışması için .csv dosyalarının bu Python dosyasıyla aynı klasörde olması gerekir.
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 2. Denenecek Parametre Havuzu
param_grid = {
    'max_depth': [3, 5, 7, 9],                 # Karar ağaçlarının derinliği
    'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Öğrenme hızı
    'n_estimators': [100, 200, 300, 500],      # Kurulacak ağaç sayısı
    'subsample': [0.7, 0.8, 1.0],              # Aşırı öğrenmeyi engellemek için örneklem oranı
    'colsample_bytree': [0.7, 0.8, 1.0]        # Ağaç başına kullanılacak kolon oranı
}

# Temel model iskeleti
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 3. Randomized Search ile En İyi Parametreleri Arama
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=15,          # 15 farklı rastgele kombinasyon deneyecek
    scoring='neg_mean_squared_error',
    cv=3,               # 3-Fold Cross Validation (Çapraz Doğrulama)
    verbose=2,          # Konsola sürecin ilerleyişini yazdırır
    random_state=42,
    n_jobs=-1           # Bilgisayarının tüm işlemci çekirdeklerini kullanır (Hızlandırır)
)

print("Hiperparametre optimizasyonu başlıyor, bu birkaç dakika sürebilir...\n")
random_search.fit(X_train, y_train)

# 4. En İyi Sonuçları Gösterme
print("\n--- Bulunan En İyi Parametreler ---")
print(random_search.best_params_)

# En iyi modeli seçip test verisiyle tahmin yapıyoruz
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Optimize Edilmiş Model Sonuçları ---")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-Square: {r2:.4f}")

# 5. Modeli Arayüz (Streamlit) İçin Kaydetme
joblib.dump(best_model, 'seis_neural_xgboost_optimized.pkl')
print("\nHarika! Model 'seis_neural_xgboost_optimized.pkl' olarak proje klasörüne kaydedildi.")