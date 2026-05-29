from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
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
RF_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "random_forest")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, "deprem_scaler.pkl")
FEATURES_PATH = os.path.join(DATA_PROCESSED_DIR, "scaler_features.txt")


def load_latest_model():
    """models/random_forest/ klasöründeki en son rf_model_v*.pkl dosyasını yükler."""
    files = os.listdir(RF_MODELS_DIR)
    versions = [
        int(f.replace("rf_model_v", "").replace(".pkl", ""))
        for f in files
        if f.startswith("rf_model_v") and f.endswith(".pkl")
    ]
    if not versions:
        raise FileNotFoundError("rf_model_v*.pkl dosyası bulunamadı! Önce eğitim yapın.")
    latest = max(versions)
    model_path = os.path.join(RF_MODELS_DIR, f"rf_model_v{latest}.pkl")
    print(f"Yüklenen model: rf_model_v{latest}.pkl")
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


# ─── Uygulama başlangıcında yükle ───
print("Yapay Zeka modelleri belleğe yükleniyor...")
model = load_latest_model()
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

    # 5. Tahmin yap
    rf_prediction = model.predict(scaled_features)[0]
    rf_prediction = float(rf_prediction)

    # 6. Feature importance (modelden)
    importances = model.feature_importances_
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(scaler_features, importances)
    }

    # 7. Risk seviyesi
    risk_level = get_risk_level(rf_prediction)

    # 8. Güven skoru (model varyansından — tüm ağaçların tahmin std'si)
    tree_predictions = np.array([tree.predict(scaled_features)[0] for tree in model.estimators_])
    std_dev = float(np.std(tree_predictions))
    # Düşük std → yüksek güven (basit dönüşüm)
    confidence = max(0.0, min(100.0, round(100 - (std_dev * 50), 1)))

    # 9. Sonucu JSON olarak döndür
    return {
        "magnitude": round(rf_prediction, 1),
        "rfVal": round(rf_prediction, 1),
        # Diğer modeller (MLP ve XGBoost) eklenene kadar mock değerler
        "mlpVal": round(rf_prediction - 0.2, 1),
        "xgbVal": round(rf_prediction + 0.1, 1),
        "confidence": confidence,
        "risk_level": risk_level,
        "feature_importance": feature_importance,
    }
