from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import json
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()

# CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── YOLLAR (Proje root'a göre göreceli) ───
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RF_MODELS_DIR = os.path.join(MODELS_DIR, "random_forest")
MLP_MODELS_DIR = os.path.join(MODELS_DIR, "mlp")
XGB_MODELS_DIR = os.path.join(MODELS_DIR, "xgboost")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, "deprem_scaler.pkl")
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "scaler_features.txt")
COMPARISON_PATH = os.path.join(MODELS_DIR, "comparison_metrics.json")


def load_latest_model(models_dir, prefix):
    """Belirtilen klasördeki en son {prefix}_model_v*.pkl dosyasını yükler."""
    if not os.path.exists(models_dir):
        return None
    files = os.listdir(models_dir)
    versions = [
        int(f.replace(f"{prefix}_model_v", "").replace(".pkl", ""))
        for f in files
        if f.startswith(f"{prefix}_model_v") and f.endswith(".pkl")
    ]
    if not versions:
        return None
    latest = max(versions)
    model_path = os.path.join(models_dir, f"{prefix}_model_v{latest}.pkl")
    print(f"Yüklenen model: {prefix}_model_v{latest}.pkl")
    return joblib.load(model_path)


def load_scaler():
    """deprem_scaler.pkl dosyasını yükler."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler bulunamadı: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


def load_scaler_features():
    """scaler_features.txt dosyasından feature sırasını okur."""
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return f.read().strip().split(",")


def load_comparison_metrics():
    """comparison_metrics.json'dan metrikleri okur."""
    if not os.path.exists(COMPARISON_PATH):
        return {}
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_best_model_from_metrics(metrics):
    """En düşük test_rmse'ye sahip modeli döndürür."""
    candidates = []
    for model_name, data in metrics.items():
        if data and data.get("test_rmse") is not None:
            candidates.append((model_name, data["test_rmse"]))
    if not candidates:
        return "random_forest"
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ─── Uygulama başlangıcında yükle ───
print("Yapay Zeka modelleri belleğe yükleniyor...")
models = {
    "random_forest": None,
    "mlp": None,
    "xgboost": None,
}
model_status = {}

try:
    models["random_forest"] = load_latest_model(RF_MODELS_DIR, "rf")
    model_status["random_forest"] = "loaded"
except Exception as e:
    print(f"Random Forest yüklenemedi: {e}")
    model_status["random_forest"] = "error"

try:
    models["mlp"] = load_latest_model(MLP_MODELS_DIR, "mlp")
    model_status["mlp"] = "loaded"
except Exception as e:
    print(f"MLP yüklenemedi: {e}")
    model_status["mlp"] = "not_trained"

try:
    models["xgboost"] = load_latest_model(XGB_MODELS_DIR, "xgb")
    model_status["xgboost"] = "loaded"
except Exception as e:
    print(f"XGBoost yüklenemedi: {e}")
    model_status["xgboost"] = "not_trained"

scaler = load_scaler()
scaler_features = load_scaler_features()
print(f"Scaler feature sırası: {scaler_features}")
print("Sistem hazır!")


# ─── Cyclical Encoding ───
def cyclical_encode(value: int, period: int) -> tuple[float, float]:
    """Cyclical sin/cos encoding hesaplar.

    Args:
        value: Encode edilecek değer (örn: ay=1-12, gün=1-31).
        period: Döngü periyodu (ay için 12, gün için 31).

    Returns:
        (sin_value, cos_value) tuple'ı.
    """
    sin_val = np.sin(2 * np.pi * value / period)
    cos_val = np.cos(2 * np.pi * value / period)
    return float(sin_val), float(cos_val)


# ─── Risk Kategorisi ───
def get_risk_level(magnitude: float) -> str:
    if magnitude < 4.5:
        return "Hafif"
    if magnitude < 6.0:
        return "Orta"
    return "Yikici"


# ─── Frontend'den gelecek verinin şablonu ───
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    depth: float
    year: int
    month: int
    day: int


# ─── Analiz Rotası (Köprü) ───
@app.post("/predict")
def predict_earthquake(req: PredictionRequest):
    # 1. Cyclical encoding
    month_sin, month_cos = cyclical_encode(req.month, 12)
    day_sin, day_cos = cyclical_encode(req.day, 31)

    # 2. Ham feature sözlüğü
    raw_features = {
        "Year": float(req.year),
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Day_sin": day_sin,
        "Day_cos": day_cos,
        "Latitude": req.latitude,
        "Longitude": req.longitude,
        "Depth": req.depth,
    }

    # 3. Scaler'ın beklediği sıraya göre diz
    try:
        feature_vector = np.array([[raw_features[f] for f in scaler_features]])
    except KeyError as e:
        return {"error": f"Eksik özellik: {e}. Beklenen: {scaler_features}"}

    # 4. Normalize et
    scaled_features = scaler.transform(feature_vector)

    # 5. Her modelden tahmin al
    predictions = {}

    if models["random_forest"] is not None:
        predictions["rfVal"] = float(models["random_forest"].predict(scaled_features)[0])
    else:
        predictions["rfVal"] = None

    if models["mlp"] is not None:
        predictions["mlpVal"] = float(models["mlp"].predict(scaled_features)[0])
    else:
        predictions["mlpVal"] = None

    if models["xgboost"] is not None:
        predictions["xgbVal"] = float(models["xgboost"].predict(scaled_features)[0])
    else:
        predictions["xgbVal"] = None

    # En az bir model çalışmalı
    valid_preds = {k: v for k, v in predictions.items() if v is not None}
    if not valid_preds:
        return {"error": "Hiçbir model yüklenmemiş. Lütfen önce modelleri eğitin."}

    # 6. En iyi modeli belirle
    metrics = load_comparison_metrics()
    best_model_name = get_best_model_from_metrics(metrics)

    # Eğer comparison_metrics.json'da veri yoksa, çalışan modeller arasından RF varsay
    if best_model_name not in metrics or metrics.get(best_model_name, {}).get("test_rmse") is None:
        if models["random_forest"] is not None:
            best_model_name = "random_forest"
        elif models["xgboost"] is not None:
            best_model_name = "xgboost"
        elif models["mlp"] is not None:
            best_model_name = "mlp"

    best_magnitude = valid_preds.get(f"{best_model_name.replace('_', '')}Val".replace("randomforest", "rf").replace("xgboost", "xgb").replace("random_forest", "rf").replace("mlp", "mlp"))
    # Yukarıdaki mapping karmaşık oldu, basitleştirelim:
    key_map = {
        "random_forest": "rfVal",
        "mlp": "mlpVal",
        "xgboost": "xgbVal",
    }
    best_magnitude = valid_preds.get(key_map.get(best_model_name, "rfVal"))

    # Fallback: eğer best model yoksa ilk çalışan modeli al
    if best_magnitude is None:
        best_magnitude = list(valid_preds.values())[0]
        best_model_name = list(valid_preds.keys())[0].replace("Val", "").replace("rf", "random_forest").replace("xgb", "xgboost").replace("mlp", "mlp")

    # 7. Feature importance (en iyi modelden, yoksa RF'den)
    feature_importance = {}
    fi_model = models.get(best_model_name) if models.get(best_model_name) is not None else models.get("random_forest")
    if fi_model is not None and hasattr(fi_model, "feature_importances_"):
        importances = fi_model.feature_importances_
        feature_importance = {
            name: round(float(imp), 4)
            for name, imp in zip(scaler_features, importances)
        }

    # 8. Risk seviyesi
    risk_level = get_risk_level(best_magnitude)

    # 9. Güven skoru (en iyi modelden)
    confidence = 0.0
    best_model_obj = models.get(best_model_name)
    if best_model_obj is not None and hasattr(best_model_obj, "estimators_"):
        # Random Forest: tüm ağaçların tahmin std'si
        tree_predictions = np.array([tree.predict(scaled_features)[0] for tree in best_model_obj.estimators_])
        std_dev = float(np.std(tree_predictions))
        confidence = max(0.0, min(100.0, round(100 - (std_dev * 50), 1)))
    elif best_model_obj is not None and hasattr(best_model_obj, "loss_curve_"):
        # MLP: basitçe sabit yüksek güven (MLP varyans hesabı farklı)
        confidence = 85.0
    elif best_model_obj is not None:
        # XGBoost: sabit yüksek güven (detaylı varyans için ayrı kod gerekir)
        confidence = 88.0
    else:
        confidence = 75.0

    # 10. Sonucu JSON olarak döndür
    result = {
        "magnitude": round(best_magnitude, 1),
        "best_model_name": best_model_name.replace("_", " ").title(),
        "rfVal": round(predictions["rfVal"], 1) if predictions["rfVal"] is not None else None,
        "mlpVal": round(predictions["mlpVal"], 1) if predictions["mlpVal"] is not None else None,
        "xgbVal": round(predictions["xgbVal"], 1) if predictions["xgbVal"] is not None else None,
        "confidence": confidence,
        "risk_level": risk_level,
        "feature_importance": feature_importance,
        "model_status": model_status,
    }

    # Küçük düzeltme: best_model_name display formatı
    display_names = {
        "random_forest": "Random Forest",
        "mlp": "MLP",
        "xgboost": "XGBoost",
    }
    result["best_model_name"] = display_names.get(best_model_name, best_model_name)

    return result
