# 🌍 SEIS-NEURAL: AI Tabanlı Sismik Büyüklük Tahmini

## 📌 Proje Özeti

SEIS-NEURAL, Türkiye'deki deprem verilerini kullanarak, belirli bir koordinat ve derinlikte meydana gelebilecek depremin **moment büyüklüğünü (Mw)** yapay zeka modelleriyle tahmin etmeye çalışan bir **karar destek sistemi**dir. Proje; makine öğrenmesi, sismik veri analizi ve etkileşimli web arayüzü teknolojilerini birleştirmektedir.

> **Not:** Bu sistem depremi önceden haber vermek amacıyla tasarlanmamıştır. Amaç; sismik bölge, derinlik ve mekânsal parametrelere dayalı olarak olası deprem şiddetinin istatistiksel olarak modellenmesidir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/dilarakahramann/seisneural.git
cd seisneural
```

### 2. Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

> **Not:** `requirements.txt` dosyası yoksa, aşağıdaki paketleri manuel yükleyin:
>
> ```bash
> pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn plotly geopandas shapely folium streamlit streamlit-folium fastapi uvicorn requests
> ```

## 📦 Model Dosyaları ve Büyük Veri Kümeleri

GitHub depolama sınırları nedeniyle büyük model dosyaları (`.pkl`) ve ham veri kümeleri Google Drive üzerinden paylaşılmaktadır.
(Proje Sunum Dosyasına (PDF) ve proje raporuna da aynı linkten ulaşabilirsiniz.)

🔗 **Google Drive:** [https://drive.google.com/drive/folders/1Q5FzxWP50nZd41P-ewssdBlX_Vf8Glhl?usp=sharing]

**Drive içeriği:**

- `models/random_forest/rf_model_v4.pkl` — Random Forest final modeli (~65 MB)
- `models/xgboost/xgb_model_v4.pkl` — XGBoost final modeli (~8 MB)
- `models/mlp/mlp_model_v4.pkl` — MLP final modeli (~1 MB)
- `data/processed/deprem_scaler.pkl/`
- ysarapor.docx
- sunum_dosyası.pdf

---

### 4. Backend Sunucusunu Başlatın

```bash
cd backend
python api.py
# veya
uvicorn api:app --reload --port 8000
```

Sunucu `http://localhost:8000` adresinde çalışacaktır.

### 5. Frontend Arayüzünü Başlatın

Yeni bir terminal penceresi açın:

```bash
cd frontend
streamlit run app.py
```

Arayüz tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

---

## 🛠️ Tech Stack

| Katman               | Teknoloji                           |
| -------------------- | ----------------------------------- |
| **Backend**          | FastAPI, Uvicorn                    |
| **Frontend**         | Streamlit, Folium, Plotly           |
| **Makine Öğrenmesi** | scikit-learn, XGBoost, MLPRegressor |
| **Veri İşleme**      | Pandas, NumPy, GeoPandas, Shapely   |
| **Görselleştirme**   | Matplotlib, Seaborn                 |
| **Uzamsal Analiz**   | GeoPandas (MTA shapefile işleme)    |
| **Veri Kaynağı**     | AFAD & Kandilli Rasathanesi         |

---

## 📜 Lisans

Bu proje, Trakya Üniversitesi Bilgisayar Mühendisliği Bölümü Yapay Sinir Ağları dersi kapsamında geliştirilmiştir. Akademik ve eğitim amaçlı kullanılmaktadır.

---

- **AFAD** ve **Kandilli Rasathanesi** — Deprem katalog verileri

---

**SEIS-NEURAL** — _Yapay zeka ile sismik analiz._
