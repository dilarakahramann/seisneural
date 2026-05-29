import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
import requests

# ============================================================
# SEIS-NEURAL: Deprem Büyüklüğü Tahmini — Karar Destek Sistemi
# ============================================================
# Frontend (Streamlit) — Dashboard görünümü, wide layout,
# hiçbir pop-up/modal kullanılmaz. Tüm bilgiler tek ekranda.
# ============================================================


# ─── 1. SAYFA AYARLARI ───
PAGE_TITLE = "SEIS-NEURAL — Sismik Şiddet Tahmini"
PAGE_ICON = "🌍"
LAYOUT = "wide"

TURKEY_CENTER_LAT = 39.0
TURKEY_CENTER_LNG = 35.0
MAP_ZOOM_START = 6

DEPTH_MIN = 0
DEPTH_MAX = 150
DEPTH_DEFAULT = 10

RISK_THRESHOLDS = {
    "low": {"max": 4.5, "label": "HAFİF RİSK", "bg": "#d1fae5", "fg": "#047857", "cls": "risk-low"},
    "mid": {"max": 6.0, "label": "ORTA RİSK", "bg": "#fef3c7", "fg": "#b45309", "cls": "risk-mid"},
    "high": {"max": float("inf"), "label": "YIKICI RİSK", "bg": "#ffe4e6", "fg": "#be123c", "cls": "risk-high"},
}

def configure_page() -> None:
    """Streamlit sayfa yapılandırmasını ayarlar."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="collapsed",
    )


def inject_styles() -> None:
    """Global CSS stillerini enjekte eder.
    Streamlit'in default header butonlarını (Deploy, Share, vb.) gizler."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main > div { padding-top: 1.2rem; padding-bottom: 1.2rem; }
        .block-container { padding: 1.5rem 2.5rem 1rem 2.5rem; max-width: 1400px; }
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            height: 100%;
        }
        .hero-big {
            font-size: 3.2rem;
            font-weight: 800;
            font-family: 'Inter', sans-serif;
            line-height: 1;
            letter-spacing: -0.02em;
            color: #1e293b;
        }
        .hero-unit {
            font-size: 1.25rem;
            font-weight: 600;
            color: #94a3b8;
            margin-left: 0.5rem;
        }
        .risk-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.35rem 0.9rem;
            border-radius: 6px;
            border: 1px solid;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .risk-low { background: #ecfdf5; color: #059669; border-color: #059669; }
        .risk-mid { background: #fffbeb; color: #d97706; border-color: #d97706; }
        .risk-high { background: #fef2f2; color: #dc2626; border-color: #dc2626; }
        .analysis-box {
            background: #f8fafc;
            border-left: 3px solid #0f766e;
            border-radius: 8px;
            padding: 1rem;
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.7;
        }
        .analysis-box strong { color: #0f766e; }
        .section-title { font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
        .section-sub { font-size: 0.78rem; color: #94a3b8; margin-bottom: 1rem; }
        .info-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem;
            font-size: 0.78rem;
            color: #64748b;
            line-height: 1.5;
        }
        /* Streamlit default header butonlarını gizle */
        header { visibility: hidden; }
        .stDeployButton, .st-emotion-cache-1wbqy5l, .st-emotion-cache-zq5wmm {
            display: none !important;
        }
        /* Number input stepper butonlarını gizle */
        button[kind="stepDown"], button[kind="stepUp"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── 2. SESSION STATE ───
def init_session_state() -> None:
    """Uygulama durumunu (session state) başlatır veya korur."""
    defaults = {
        "latitude": TURKEY_CENTER_LAT,
        "longitude": TURKEY_CENTER_LNG,
        "depth": DEPTH_DEFAULT,
        "analysis_date": date.today(),
        "analyzed": False,
        "results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─── 3. YARDIMCI FONKSİYONLAR ───
def get_risk_level(magnitude: float) -> dict:
    """Büyüklüğe göre risk seviyesi meta-verisi döndürür."""
    if magnitude < RISK_THRESHOLDS["low"]["max"]:
        return RISK_THRESHOLDS["low"]
    if magnitude < RISK_THRESHOLDS["mid"]["max"]:
        return RISK_THRESHOLDS["mid"]
    return RISK_THRESHOLDS["high"]


def run_api_analysis(lat: float, lng: float, depth: float, analysis_date: date) -> dict:
    """Backend API'ye POST isteği göndererek gerçek analiz verisi alır."""
    try:
        payload = {
            "latitude": float(lat),
            "longitude": float(lng),
            "depth": float(depth),
            "year": int(analysis_date.year),
            "month": int(analysis_date.month),
            "day": int(analysis_date.day),
        }
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        # API hata mesajı döndürdüyse
        if "error" in data:
            st.error(f"API Hatası: {data['error']}")
            return None

        # Feature importance sözlüğünü DataFrame'e dönüştür
        fi_dict = data.get("feature_importance", {})
        fi_df = pd.DataFrame({
            "Öznitelik": list(fi_dict.keys()),
            "Etki (%)": [v * 100 for v in fi_dict.values()],
        }).sort_values("Etki (%)", ascending=False)

        return {
            "best_magnitude": data.get("magnitude", 0.0),
            "best_model_name": data.get("best_model_name", "Random Forest"),
            "confidence": data.get("confidence", 0.0),
            "risk_level": data.get("risk_level", "Hafif"),
            "rfVal": data.get("rfVal"),
            "mlpVal": data.get("mlpVal"),
            "xgbVal": data.get("xgbVal"),
            "feature_importance": fi_df,
        }
    except requests.exceptions.ConnectionError:
        st.error("Backend API'ye bağlanılamadı. Lütfen `python backend/api.py` komutu ile sunucuyu başlatın.")
        return None
    except Exception as e:
        st.error(f"API isteği sırasında hata oluştu: {e}")
        return None


# ─── 4. BİLEŞEN RENDER FONKSİYONLARI ───
def render_input_panel() -> None:
    """Sol paneli (girdi alanlarını) çizer."""
    st.markdown(
        '<div class="section-title">Deprem Bölgesi Seçimi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">İnteraktif coğrafi hedefleme haritası</div>',
        unsafe_allow_html=True,
    )

    # ── 4.1 Harita ──
    m = folium.Map(
        location=[TURKEY_CENTER_LAT, TURKEY_CENTER_LNG],
        zoom_start=MAP_ZOOM_START,
        tiles="OpenStreetMap",
    )
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        tooltip="Hedef Nokta",
        icon=folium.DivIcon(
            html="""
            <div style="width:16px;height:16px;margin-left:-8px;margin-top:-8px;">
                <div style="width:16px;height:16px;background:rgba(15,118,110,0.25);border-radius:50%;"></div>
                <div style="width:10px;height:10px;background:#0f766e;border:2px solid white;border-radius:50%;position:absolute;top:3px;left:3px;"></div>
            </div>
            """
        ),
    ).add_to(m)
    m.add_child(folium.LatLngPopup())

    map_data = st_folium(
        m,
        width="100%",
        height=340,
        returned_objects=["last_clicked"],
        key="main_map",
    )

    if map_data is not None:
        last_clicked = map_data.get("last_clicked")
        if last_clicked is not None:
            lat_clicked = round(float(last_clicked.get("lat", st.session_state.latitude)), 4)
            lng_clicked = round(float(last_clicked.get("lng", st.session_state.longitude)), 4)
            if lat_clicked != st.session_state.latitude or lng_clicked != st.session_state.longitude:
                st.session_state.latitude = lat_clicked
                st.session_state.longitude = lng_clicked
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4.2 Koordinat Formu ──
    col_lat, col_lng = st.columns([3, 1])
    with col_lat:
        st.number_input(
            "Enlem",
            min_value=36.0,
            max_value=42.0,
            value=float(st.session_state.latitude),
            step=0.01,
            format="%.4f",
            key="lat_input",
            label_visibility="collapsed",
        )
    with col_lng:
        st.markdown(f"<div style='text-align:right;font-size:0.8rem;color:#94a3b8;padding-top:0.5rem;'>{st.session_state.latitude:.4f}°N</div>", unsafe_allow_html=True)

    col_lat2, col_lng2 = st.columns([3, 1])
    with col_lat2:
        st.number_input(
            "Boylam",
            min_value=26.0,
            max_value=45.0,
            value=float(st.session_state.longitude),
            step=0.01,
            format="%.4f",
            key="lng_input",
            label_visibility="collapsed",
        )
    with col_lng2:
        st.markdown(f"<div style='text-align:right;font-size:0.8rem;color:#94a3b8;padding-top:0.5rem;'>{st.session_state.longitude:.4f}°E</div>", unsafe_allow_html=True)

    # ── 4.3 Tarih ──
    selected_date = st.date_input(
        "Tarih",
        value=st.session_state.analysis_date,
        key="date_input",
        label_visibility="collapsed",
    )
    st.session_state.analysis_date = selected_date

    # ── 4.4 Derinlik ──
    depth = st.slider(
        "Odak Derinliği (km)",
        min_value=DEPTH_MIN,
        max_value=DEPTH_MAX,
        value=int(st.session_state.depth),
        step=1,
        key="depth_slider",
        label_visibility="collapsed",
    )
    st.session_state.depth = depth

    st.caption(
        "Türkiye sismik rejimi gereği yıkıcı fay sığlık ortalaması (10 km) "
        "varsayılan olarak atanmıştır. Simülasyon için değiştirebilirsiniz."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4.5 Analiz Butonu ──
    if st.button("Analizi Başlat", type="primary", use_container_width=True):
        with st.spinner("Modeller analiz ediyor..."):
            time.sleep(0.5)
            results = run_api_analysis(
                st.session_state.latitude,
                st.session_state.longitude,
                st.session_state.depth,
                st.session_state.analysis_date,
            )
            if results is not None:
                st.session_state.results = results
                st.session_state.analyzed = True
        if st.session_state.analyzed:
            st.rerun()

    # ── 4.6 Bilgilendirme Kutusu ──
    st.markdown(
        '<div class="info-box" style="margin-top:1rem;">'
        "Hesaplamalar; AFAD ve Kandilli Rasathanesi sismik katalog "
        "eğitim kümelerine (M3.0+) dayalı olarak sismik makaslama "
        "etkileşimlerini regresyon yöntemiyle modeller."
        "</div>",
        unsafe_allow_html=True,
    )


def render_hero_card(results: dict) -> None:
    """Ana sonuç kartını (Hero Section) çizer. Native border container ile."""
    mag = results["best_magnitude"]
    risk = get_risk_level(mag)
    confidence = results.get("confidence", 0.0)

    with st.container(border=True):
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown(
                '<div style="font-size:0.75rem;font-weight:600;color:#94a3b8;letter-spacing:0.05em;text-transform:uppercase;">'
                'Tahmin Edilen Büyüklük</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="display:flex;align-items:baseline;margin-top:0.3rem;">'
                f'<span class="hero-big">{mag:.1f}</span>'
                f'<span class="hero-unit">Mw</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:0.8rem;color:#94a3b8;margin-top:0.5rem;">'
                'Moment Büyüklüğü Ölçeği (Mw) Hesabı</div>',
                unsafe_allow_html=True,
            )
        with col_right:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align:right;"><span class="risk-badge {risk["cls"]}">{risk["label"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:#94a3b8;margin-top:0.5rem;">'
                f'Seçilen Model: <strong style="color:#475569;">{results["best_model_name"]}</strong></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:right;font-size:0.75rem;color:#94a3b8;margin-top:0.3rem;">'
                f'Güven Skoru: <strong style="color:#475569;">{confidence:.1f}%</strong></div>',
                unsafe_allow_html=True,
            )


def render_analysis_summary(results: dict) -> None:
    """Seçilen modelin tahminine dair analiz metnini çizer."""
    mag = results["best_magnitude"]
    risk = get_risk_level(mag)
    fi = results["feature_importance"]
    top_factor = fi.iloc[0]["Öznitelik"] if not fi.empty else "Bilinmiyor"
    top_pct = fi.iloc[0]["Etki (%)"] if not fi.empty else 0.0

    st.markdown(
        '<div class="section-title">Yapay Zeka Deprem Tahmini</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Sismik parametrelerin makine öğrenmesi yorumlaması</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="analysis-box">
            <strong>Koordinat:</strong> {st.session_state.latitude:.4f}°N, {st.session_state.longitude:.4f}°E &nbsp;|&nbsp;
            <strong>Tarih:</strong> {st.session_state.analysis_date.strftime("%d.%m.%Y")} &nbsp;|&nbsp;
            <strong>Derinlik:</strong> {st.session_state.depth} km<br><br>
            {results['best_model_name']} modeli, bu parametrelere göre
            <strong>{mag:.1f} Mw</strong> büyüklüğünde bir deprem tahmin ediyor.
            Bu tahmin <strong>{risk['label']}</strong> kategorisindedir.<br><br>
            En belirleyici faktör: <strong>{top_factor}</strong> ({top_pct:.1f}%).
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(features_df: pd.DataFrame) -> None:
    """Öznitelik etki ağırlığı kartını çizer. Kart + grafik tek blokta."""
    if features_df is None or features_df.empty:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:0.2rem;">'
                'Değişkenlerin Etki Ağırlığı</div>',
                unsafe_allow_html=True,
            )
            st.info("Öznitelik önemi verisi mevcut değil. Model eğitimi tamamlandıktan sonra burada görüntülenecektir.")
        return

    # Ince bar chart
    fig = px.bar(
        features_df,
        x="Etki (%)",
        y="Öznitelik",
        orientation="h",
        color="Etki (%)",
        color_continuous_scale=["#0f766e", "#14b8a6", "#2dd4bf"],
        text="Etki (%)",
    )
    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        marker_line_width=0,
        width=0.35,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=60, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        xaxis=dict(title="", showgrid=False, showticklabels=False, range=[0, 70]),
        yaxis=dict(title="", tickfont=dict(size=13, family="Inter", color="#475569")),
        font=dict(family="Inter, sans-serif"),
        coloraxis_showscale=False,
    )

    # Streamlit'in native kart konteyneri — tüm içerik aynı border içinde
    with st.container(border=True):
        st.markdown(
            '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:0.2rem;">Değişkenlerin Etki Ağırlığı</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.75rem;">Girdi değişkenlerinin sismik etki derecesi</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_leaderboard(results: dict) -> None:
    """3 modelin (MLP, RF, XGBoost) tahmin sonuçlarını karşılaştırmalı olarak gösterir."""
    best_model_name = results.get("best_model_name", "")
    rf_val = results.get("rfVal")
    mlp_val = results.get("mlpVal")
    xgb_val = results.get("xgbVal")

    rows = []
    if rf_val is not None:
        rows.append({
            "Model": "Random Forest",
            "Tahmin (Mw)": f"{rf_val:.2f}",
            "Durum": "🏆 Seçilen Model" if best_model_name == "Random Forest" else "",
        })
    if mlp_val is not None:
        rows.append({
            "Model": "MLP",
            "Tahmin (Mw)": f"{mlp_val:.2f}",
            "Durum": "🏆 Seçilen Model" if best_model_name == "MLP" else "",
        })
    if xgb_val is not None:
        rows.append({
            "Model": "XGBoost",
            "Tahmin (Mw)": f"{xgb_val:.2f}",
            "Durum": "🏆 Seçilen Model" if best_model_name == "XGBoost" else "",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:0.2rem;">'
            'Algoritma Karşılaştırması</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.75rem;">'
            'Eş zamanlı çalışan modellerin tahmin sonuçları</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Tahmin (Mw)": st.column_config.TextColumn("Tahmin", width="small"),
                "Durum": st.column_config.TextColumn("", width="medium"),
            },
        )


def render_empty_dashboard() -> None:
    """Henüz analiz yapılmamışsa gösterilecek boş durum ekranını çizer."""
    st.info(
        "Sol panelden bir koordinat seçip parametreleri belirledikten sonra "
        "**'Analizi Başlat'** butonuna basarak sismik simülasyon sonuçlarını "
        "görüntüleyebilirsiniz.",
        icon="👈",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="card" style="opacity:0.6;">'
            '<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;">Tahmini Maksimum Şiddet</div>'
            '<div style="color:#cbd5e1;font-size:2.5rem;font-weight:800;">—.— Mw</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="card" style="opacity:0.6;">'
            '<div style="color:#94a3b8;font-size:0.8rem;font-weight:600;">Güven Skoru</div>'
            '<div style="color:#cbd5e1;font-size:2.5rem;font-weight:800;">—%</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_dashboard(results: dict) -> None:
    """Analiz sonuçlarını çizen ana dashboard paneli."""
    st.markdown(
        '<div class="section-title" style="font-size:1.1rem;">Deprem Büyüklüğü Tahmini Akıllı Karar Destek Sistemi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Denetimli Öğrenme (Supervised Learning) algoritmalarını yarıştırarak, '
        'Türkiye sınırları içerisinde seçilen koordinat ve derinlikte meydana gelebilecek olası deprem '
        'moment büyüklüklerini (Mw) hesaplayın.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---", unsafe_allow_html=True)

    # BÖLÜM A: Hero Section (kart içinde)
    render_hero_card(results)
    st.markdown("<br>", unsafe_allow_html=True)

    # BÖLÜM B: Algoritma Karşılaştırması (Leaderboard)
    render_leaderboard(results)
    st.markdown("<br>", unsafe_allow_html=True)

    # BÖLÜM C: Sismik Analiz Raporu
    render_analysis_summary(results)
    st.markdown("<br>", unsafe_allow_html=True)

    # BÖLÜM D: Feature Importance (kart içinde)
    render_feature_card(results["feature_importance"])

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:0.7rem;color:#94a3b8;'
        'font-family:monospace;border-top:1px solid #e2e8f0;padding-top:0.8rem;margin-top:1.5rem;">'
        "SEIS-NEURAL — AFAD & KANDILLI VERI BAGLANTISI"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── 5. ANA UYGULAMA ───
def main() -> None:
    """Uygulama giriş noktası."""
    configure_page()
    inject_styles()
    init_session_state()

    left_col, right_col = st.columns([5, 7], gap="large")

    with left_col:
        render_input_panel()

    with right_col:
        if not st.session_state.analyzed or st.session_state.results is None:
            render_empty_dashboard()
        else:
            render_dashboard(st.session_state.results)


if __name__ == "__main__":
    main()
