import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================
# SETUP
# =========================

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

MODEL_PATH = Path(__file__).parent / "house_price_model.pkl"
HERO_IMAGE_PATH = Path(__file__).parent / "hero.jpg"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# All 44 cities as they actually appear in the training data (Project_HousePricePred.ipynb, cell 91)
CITIES = [
    "Algona", "Auburn", "Beaux Arts Village", "Bellevue", "Black Diamond", "Bothell",
    "Burien", "Carnation", "Clyde Hill", "Covington", "Des Moines", "Duvall",
    "Enumclaw", "Fall City", "Federal Way", "Inglewood-Finn Hill", "Issaquah", "Kenmore",
    "Kent", "Kirkland", "Lake Forest Park", "Maple Valley", "Medina", "Mercer Island",
    "Milton", "Newcastle", "Normandy Park", "North Bend", "Pacific", "Preston",
    "Ravensdale", "Redmond", "Renton", "Sammamish", "SeaTac", "Seattle",
    "Shoreline", "Skykomish", "Snoqualmie", "Snoqualmie Pass", "Tukwila", "Vashon",
    "Woodinville", "Yarrow Point",
]

# Real accuracy numbers from the notebook (cell 184 — Gradient Boosting is the saved model)
MODEL_METRICS = {
    "r2_score": 0.694,
    "mae": 82291,
    "rmse": 121927,
    "training_samples": 3448,
    "features_used": 14,
    "algorithm": "Gradient Boosting Regressor",
}


def get_hero_background_css():
    if HERO_IMAGE_PATH.exists():
        b64 = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode()
        ext = HERO_IMAGE_PATH.suffix.lstrip(".")
        return f"url(data:image/{ext};base64,{b64})"
    return "linear-gradient(135deg, #1A130D 0%, #2C2013 60%, #1A130D 100%)"


# =========================
# CSS — warm gold / dark-brown theme, matching the hero photo
# =========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #16110C;
    --ink-2: #221A11;
    --ink-3: #2E2416;
    --paper: #F5EFE3;
    --brass: #D9B466;
    --brass-dark: #A8823F;
    --slate: #BBAE95;
    --line: rgba(217, 180, 102, 0.2);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif; }
#MainMenu, footer { visibility: hidden; }

.stApp {
    background-color: var(--ink);
    background-image:
        radial-gradient(circle at 15% 0%, rgba(217,180,102,0.10) 0%, transparent 45%),
        radial-gradient(circle at 100% 30%, rgba(168,130,63,0.08) 0%, transparent 40%);
    background-attachment: fixed;
}

/* Tabs restyled as a nav bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(34,26,17,0.6);
    padding: 8px;
    border-radius: 999px;
    border: 1px solid var(--line);
    width: fit-content;
    margin: 0 auto 30px auto;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 999px;
    padding: 8px 26px;
    color: var(--slate);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.06em;
}
.stTabs [aria-selected="true"] {
    background: var(--brass) !important;
    color: var(--ink) !important;
}

/* PHOTO HERO — image used as-is, no overlay text (the image already has its own title) */
.photo-hero {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 50px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.5);
    border: 1px solid var(--brass-dark);
}
.photo-hero img { width: 100%; display: block; }

/* SECTIONS */
.section-title { font-family: 'Fraunces', serif; font-size: 27px; font-weight: 600; color: var(--paper); margin: 6px 0 6px 0; }
.section-sub { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--slate); margin-bottom: 24px; border-bottom: 1px solid var(--line); padding-bottom: 14px; }

.feature-card {
    position: relative;
    background: linear-gradient(160deg, rgba(217,180,102,0.08), rgba(255,255,255,0.02));
    padding: 26px 22px; border: 1px solid var(--line); min-height: 150px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.feature-card:hover { border-color: var(--brass); transform: translateY(-3px); }
.feature-card::before, .feature-card::after { content: ""; position: absolute; width: 10px; height: 10px; border-color: var(--brass); border-style: solid; }
.feature-card::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.feature-card::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }
.feature-icon { font-size: 26px; margin-bottom: 10px; }
.feature-title { font-family: 'Fraunces', serif; font-size: 17px; font-weight: 600; color: var(--paper); margin-bottom: 6px; }
.feature-text { color: var(--slate); font-size: 14px; line-height: 1.5; }

.about-text { color: var(--slate); font-size: 15px; line-height: 1.75; }
.about-text strong { color: var(--paper); font-weight: 600; }

.stat-card { background: linear-gradient(160deg, var(--ink-3) 0%, var(--ink-2) 100%); border: 1px solid var(--brass-dark); padding: 22px 20px; text-align: center; }
.stat-value { font-family: 'Fraunces', serif; font-size: 30px; font-weight: 600; color: var(--brass); }
.stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--slate); margin-top: 8px; }

.spec-panel { background: linear-gradient(160deg, var(--ink-3) 0%, var(--ink-2) 100%); border: 1px solid var(--brass-dark); box-shadow: 0 16px 44px rgba(0,0,0,0.3); padding: 32px 30px 8px 30px; margin-bottom: 10px; }
.spec-group-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--brass); margin-bottom: 10px; margin-top: 4px; }

div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
    color: #EFE6D4 !important; font-size: 13.5px !important; font-weight: 500 !important;
}
div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(22, 17, 12, 0.55) !important; border: 1px solid rgba(217,180,102,0.4) !important; color: var(--paper) !important; border-radius: 2px !important;
}
div[data-testid="stSlider"] div[role="slider"], div[data-testid="stSlider"] > div > div > div > div { background-color: var(--brass) !important; }

.stButton > button {
    background-color: var(--brass) !important; color: var(--ink) !important; border: none !important; border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-weight: 600 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; font-size: 13px !important;
}
.stButton > button:hover { background-color: #E9C989 !important; color: var(--ink) !important; }

.plaque { background: linear-gradient(160deg, var(--ink-3) 0%, var(--ink-2) 100%); border: 1px solid var(--brass); box-shadow: 0 16px 50px rgba(0,0,0,0.4); padding: 40px 30px; text-align: center; margin-top: 28px; }
.plaque::before { content: "ESTIMATED MARKET VALUE"; display: block; font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: 0.2em; color: var(--brass); margin-bottom: 16px; }
.plaque-value { font-family: 'Fraunces', serif; font-size: 52px; font-weight: 600; color: var(--paper); }
.plaque-sub { font-size: 13px; color: var(--slate); margin-top: 12px; }

.site-footer { margin-top: 50px; padding: 24px 10px 10px 10px; border-top: 1px solid var(--line); text-align: center; }
.site-footer .footer-brand { font-family: 'Fraunces', serif; color: var(--paper); font-size: 15px; margin-bottom: 6px; }
.site-footer .footer-note { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em; color: var(--slate); }

</style>
""", unsafe_allow_html=True)


# =========================
# TABS = Home / About (Get Estimate removed — full form now lives on Home)
# =========================

tab_home, tab_about = st.tabs(["🏠  Home", "📊  About"])


# ---------- HOME ----------
with tab_home:

    # Hero image used as-is (it already has its own title/tagline baked in)
    if HERO_IMAGE_PATH.exists():
        st.markdown(f"""
        <div class="photo-hero">
            <img src="data:image/jpeg;base64,{base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode()}" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="photo-hero" style="min-height: 300px; display:flex; align-items:center; justify-content:center;">
            <div class="section-sub" style="border:none;">Add a hero.jpg next to this file to show your banner image here</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why This Estimate Holds Up</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Model Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🤖</div>
        <div class="feature-title">Trained Model</div>
        <div class="feature-text">A regression model fitted on real transaction records, not a rule of thumb.</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card"><div class="feature-icon">⚡</div>
        <div class="feature-title">Instant Output</div>
        <div class="feature-text">Your estimate is generated the moment you submit the form.</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📊</div>
        <div class="feature-title">Grounded in Data</div>
        <div class="feature-text">Built on 44 King County cities' worth of housing sales.</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🏡</div>
        <div class="feature-title">Decision Ready</div>
        <div class="feature-text">Use it to sanity-check a listing price or an offer.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height: 34px;"></div>', unsafe_allow_html=True)

    # ---- GET ESTIMATED — full form, now on the Home page ----
    st.markdown('<div class="section-title">Get Estimated</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in every field for the most accurate estimate</div>', unsafe_allow_html=True)

    st.markdown('<div class="spec-panel">', unsafe_allow_html=True)

    st.markdown('<div class="spec-group-label">Layout</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=9, value=3)
        bathrooms = st.number_input("🛁 Bathrooms", min_value=0.5, max_value=5.0, value=2.0, step=0.25)
        floors = st.number_input("🏢 Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    with col2:
        sqft_living = st.number_input("📐 Living Area (sqft)", min_value=100, max_value=20000, value=1500, step=50)
        sqft_lot = st.number_input("🌳 Lot Area (sqft)", min_value=100, max_value=100000, value=5000, step=100)
        sqft_above = st.number_input("🏠 Above Ground Area (sqft)", min_value=0, max_value=20000, value=1500, step=50)
    with col3:
        sqft_basement = st.number_input("⬇️ Basement Area (sqft)", min_value=0, max_value=10000, value=0, step=50)
        city = st.selectbox("🏙️ City", CITIES)

    st.markdown('<div class="spec-group-label">Condition &amp; Position</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        waterfront = st.selectbox("🌊 Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col5:
        view = st.slider("👁️ View Quality", 0, 4, 0)
    with col6:
        condition = st.slider("⭐ Condition", 1, 5, 3)

    st.markdown('<div class="spec-group-label">History</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        yr_built = st.number_input("🏗️ Year Built", min_value=1800, max_value=2014, value=1990)
    with col8:
        yr_renovated = st.number_input("🔨 Year Renovated (0 = never)", min_value=0, max_value=2014, value=0)

    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    house_Age = 2014 - yr_built
    is_renovated = 0 if yr_renovated == 0 else 1
    yr_since_renovated = 0 if yr_renovated == 0 else 2014 - yr_renovated

    if st.button("🔮  Generate Estimate", use_container_width=True):
        input_data = pd.DataFrame([{
            "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
            "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront, "view": view,
            "condition": condition, "sqft_above": sqft_above, "sqft_basement": sqft_basement,
            "house_Age": house_Age, "is_renovated": is_renovated,
            "yr_since_renovated": yr_since_renovated, "city": city,
        }])
        prediction = model.predict(input_data)[0]
        st.markdown(f"""
        <div class="plaque">
            <div class="plaque-value">${prediction:,.2f}</div>
            <div class="plaque-sub">{bedrooms} bed · {bathrooms} bath · {sqft_living:,} sqft · {city}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="site-footer"><div class="footer-brand">🏠 House Price Prediction</div><div class="footer-note">GRADIENT BOOSTING MODEL · HOUSE PRICE PREDICTION DATA</div></div>', unsafe_allow_html=True)


# ---------- ABOUT ----------
with tab_about:

    st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Methodology &amp; Performance</div>', unsafe_allow_html=True)

    col_text, col_stats = st.columns([1.1, 1.4])

    with col_text:
        st.markdown(f"""
        <div class="about-text">
            This estimate comes from a <strong>{MODEL_METRICS['algorithm']}</strong>, tuned via
            GridSearchCV, trained on <strong>{MODEL_METRICS['training_samples']:,} historical
            property sales</strong> across <strong>44 cities</strong> in King County, Washington.
            The model considers <strong>{MODEL_METRICS['features_used']} features</strong> per
            property — layout, size, condition, city, and renovation history.
            <br><br>
            It was compared against Linear Regression and Random Forest baselines and came out
            ahead on both R² and error. That said, this dataset doesn't include construction-quality
            (grade) or exact geolocation, which are normally the strongest price predictors — so
            accuracy has a practical ceiling here regardless of tuning. Treat the estimate as a
            reference point alongside comparable local listings, not a final appraisal.
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">{MODEL_METRICS['r2_score']:.2f}</div><div class="stat-label">R² Score</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">${MODEL_METRICS['mae']:,.0f}</div><div class="stat-label">Mean Absolute Error</div></div>""", unsafe_allow_html=True)
        st.markdown('<div style="height: 14px;"></div>', unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        with s3:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">${MODEL_METRICS['rmse']:,.0f}</div><div class="stat-label">RMSE</div></div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">{MODEL_METRICS['training_samples']:,}</div><div class="stat-label">Training Records</div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Tested during development</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""<div class="feature-card"><div class="feature-title">Linear Regression</div>
        <div class="feature-text">R² 0.658 · MAE ~$88,932<br>Baseline model.</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="feature-card"><div class="feature-title">Gradient Boosting ✓</div>
        <div class="feature-text">R² 0.694 · MAE ~$82,291<br>Best performer — this is the saved model.</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="feature-card"><div class="feature-title">Random Forest</div>
        <div class="feature-text">R² 0.673 · MAE ~$84,993<br>Close second, slightly behind.</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="site-footer"><div class="footer-brand">🏠 House Price Prediction</div><div class="footer-note">GRADIENT BOOSTING MODEL · HOUSE PRICE PREDICTION DATA</div></div>', unsafe_allow_html=True)