import pandas as pd
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    KFold,
    cross_val_score
)

# =========================================================
# FEATURE ENGINEERING SONRASI VERİLERİ OKU
# =========================================================

X_train = pd.read_csv("X_train_featured.csv")
X_test = pd.read_csv("X_test_featured.csv")

y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# =========================================================
# y DEĞERLERİNİ DÜZLEŞTİR
# =========================================================

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# =========================================================
# MLP MODEL OLUŞTURMA
# =========================================================

print("═" * 60)
print("            MLP MODEL EĞİTİMİ")
print("═" * 60)

mlp_model = MLPRegressor(

    hidden_layer_sizes=(128, 64),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=1000,

    random_state=42
)

# =========================================================
# K-FOLD CROSS VALIDATION
# =========================================================

print("\n" + "═" * 60)
print("          K-FOLD CROSS VALIDATION")
print("═" * 60)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nMLP Cross Validation Yapılıyor...")

mlp_cv_scores = cross_val_score(
    mlp_model,
    X_train,
    y_train,
    cv=kfold,
    scoring="r2"
)

print(f"MLP Ortalama R² Skoru: {mlp_cv_scores.mean():.4f}")

# =========================================================
# MODEL EĞİTİMİ
# =========================================================

print("\nMLP Eğitiliyor...")

mlp_model.fit(X_train, y_train)

# =========================================================
# TAHMİN YAP
# =========================================================

mlp_predictions = mlp_model.predict(X_test)

# =========================================================
# PERFORMANS HESAPLAMA
# =========================================================

mae = mean_absolute_error(
    y_test,
    mlp_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        mlp_predictions
    )
)

r2 = r2_score(
    y_test,
    mlp_predictions
)

# =========================================================
# SONUÇLARI GÖSTER
# =========================================================

print("\n" + "═" * 60)
print("         MLP PERFORMANS SONUÇLARI")
print("═" * 60)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =========================================================
# ÖRNEK TAHMİNLER
# =========================================================

print("\n" + "═" * 60)
print("             ÖRNEK TAHMİNLER")
print("═" * 60)

for i in range(5):

    print(f"\nGerçek Değer : {y_test[i]}")
    print(f"MLP Tahmin   : {mlp_predictions[i]:.2f}")