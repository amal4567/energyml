import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib as mpl
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="EnergyML · Prédiction Énergétique",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DESIGN SYSTEM — DARK TECHNICAL THEME
# Palette : fond ardoise profond + accent électrique indigo/cyan
# Typography : Inter (display) + JetBrains Mono (data/labels)
# Signature : bande de gradient animée en haut + cartes à fond verre dépoli
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root {
    /* — Couleurs de fond — */
    --bg-base:       #F4F6FB;
    --bg-surface:    #FFFFFF;
    --bg-card:       #FFFFFF;
    --bg-card-hover: #F0F3FA;
    --bg-input:      #F8F9FC;

    /* — Accents — */
    --accent:        #2563EB;
    --accent-dim:    rgba(37,99,235,0.08);
    --accent-border: rgba(37,99,235,0.25);
    --cyan:          #0891B2;
    --emerald:       #059669;
    --amber:         #D97706;
    --rose:          #E11D48;
    --violet:        #7C3AED;

    /* — Texte — */
    --text-primary:  #0F172A;
    --text-secondary:#475569;
    --text-muted:    #94A3B8;
    --text-label:    #64748B;

    /* — Bordures — */
    --border:        rgba(15,23,42,0.10);
    --border-accent: rgba(37,99,235,0.30);

    /* — Typographie — */
    --font-display:  'Inter', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;

    /* — Rayons — */
    --r-sm:  8px;
    --r-md:  12px;
    --r-lg:  16px;
    --r-xl:  20px;

    /* — Ombres — */
    --shadow-card: 0 1px 4px rgba(15,23,42,0.06), 0 4px 16px rgba(15,23,42,0.06);
    --shadow-glow: 0 0 30px rgba(37,99,235,0.10);
    --glow-emerald: 0 0 20px rgba(5,150,105,0.12);
    --glow-rose:    0 0 20px rgba(225,29,72,0.12);
}

/* ─── RESET BASE ─── */
* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
}

/* ─── GRADIENT TOP BAR ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2563EB, #0891B2, #7C3AED, #2563EB);
    background-size: 300% 100%;
    animation: shimmer 4s linear infinite;
    z-index: 9999;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-label) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--accent-border) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
    margin-top: 1.5rem !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border-radius: var(--r-sm) !important;
}

section[data-testid="stSidebar"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    color: var(--text-muted) !important;
}

/* Slider track */
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* Radio */
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.4rem;
    transition: all 0.15s;
}

/* ─── MAIN AREA ─── */
.block-container {
    padding: 1.5rem 2.5rem 2rem !important;
    max-width: 1400px !important;
}

/* ─── HEADER PRINCIPAL ─── */
.eml-header {
    display: flex;
    align-items: flex-start;
    gap: 1.6rem;
    padding: 1.8rem 0 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.eml-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #2563EB 0%, #0891B2 100%);
    border-radius: var(--r-lg);
    font-size: 1.6rem;
    flex-shrink: 0;
    box-shadow: 0 8px 24px rgba(37,99,235,0.25);
}

.eml-title {
    font-family: var(--font-display);
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.eml-title em { color: var(--accent); font-style: normal; }

.eml-subtitle {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.eml-meta {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}

.eml-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
}

.eml-chip .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 6px var(--emerald);
    animation: pulse 2.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    padding: 0.75rem 1.6rem !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px;
    transition: all 0.2s;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: var(--accent-dim) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 2rem !important;
    background: transparent !important;
}

/* ─── SECTION DIVIDER ─── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.2em;
}

.section-divider::before {
    content: '';
    width: 4px; height: 14px;
    background: linear-gradient(180deg, var(--accent), var(--cyan));
    border-radius: 2px;
    flex-shrink: 0;
}

.section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ─── CARDS MÉTRIQUES ─── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 1.4rem 1.6rem 1.2rem;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--border);
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}

.metric-card.accent::before  { background: linear-gradient(90deg, var(--accent), var(--cyan)); }
.metric-card.success::before { background: linear-gradient(90deg, var(--emerald), var(--cyan)); }
.metric-card.warning::before { background: linear-gradient(90deg, var(--amber), var(--rose)); }

.metric-card:hover {
    border-color: var(--accent-border);
    box-shadow: var(--shadow-card), var(--shadow-glow);
}

.m-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-label);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.7rem;
}

.m-value {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1;
}

.metric-card.accent .m-value  { color: var(--accent); }
.metric-card.success .m-value { color: var(--emerald); }
.metric-card.warning .m-value { color: var(--amber); }

.m-unit {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
}

/* ─── INFOBANNIÈRE MODE ─── */
.mode-info {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: var(--bg-card);
    border: 1px solid var(--accent-border);
    border-radius: var(--r-lg);
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.8rem;
}

.mode-info-icon { font-size: 1.4rem; flex-shrink: 0; line-height: 1.4; }

.mode-info-tag {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.mode-info-text {
    font-size: 0.83rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ─── ALERTE HORS PLAGE ─── */
.oor-banner {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: var(--r-lg);
    padding: 1rem 1.3rem;
    margin-bottom: 1.5rem;
}

.oor-tag {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--amber);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.oor-text { font-size: 0.81rem; color: var(--text-secondary); line-height: 1.45; margin-bottom: 0.5rem; }

.oor-item {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: #D97706;
    margin: 0.15rem 0;
}

/* ─── VERDICT BANNER ─── */
.verdict {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border-radius: var(--r-lg);
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.5rem;
    border: 1px solid transparent;
}

.verdict.over    { background: rgba(251,113,133,0.07); border-color: rgba(251,113,133,0.25); }
.verdict.under   { background: rgba(52,211,153,0.07);  border-color: rgba(52,211,153,0.25); }
.verdict.aligned { background: rgba(37,99,235,0.07);  border-color: rgba(37,99,235,0.25); }

.verdict-icon { font-size: 1.5rem; flex-shrink: 0; line-height: 1.4; }

.verdict-tag {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.2rem;
}

.verdict.over .verdict-tag    { color: var(--rose); }
.verdict.under .verdict-tag   { color: var(--emerald); }
.verdict.aligned .verdict-tag { color: var(--accent); }

.verdict-text { font-size: 0.83rem; color: var(--text-secondary); line-height: 1.5; }

/* ─── NOTE TEMPORELLE (mode projet) ─── */
.temporal-note {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: var(--r-md);
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.4rem;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: #C4B5FD;
    line-height: 1.6;
}

.temporal-note strong { color: var(--violet); font-weight: 600; }

/* ─── BOUTON PREDICT ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #3B82F6 100%) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2.5rem !important;
    border-radius: var(--r-md) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.30) !important;
    transition: all 0.2s ease !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,0.40) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ─── ALERTES STREAMLIT ─── */
.stAlert {
    background: rgba(52,211,153,0.07) !important;
    border: 1px solid rgba(52,211,153,0.25) !important;
    border-radius: var(--r-md) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.73rem !important;
    color: var(--emerald) !important;
}

/* ─── DATAFRAME ─── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.73rem !important;
    box-shadow: var(--shadow-card);
}

/* ─── GAUGE CONTAINER ─── */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-xl) !important;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}

/* ─── HISTORY HEADER ─── */
.history-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.85rem 1.2rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: var(--r-md) var(--r-md) 0 0;
    margin-top: 2rem;
}

.history-title {
    font-family: var(--font-mono);
    font-size: 0.63rem;
    color: var(--accent);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 500;
}

.history-badge {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--accent);
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ─── HEADINGS ─── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* ─── INFO / WARNING STREAMLIT ─── */
[data-testid="stInfoBox"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--accent-border) !important;
    border-radius: var(--r-md) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
}

/* ─── HIDE DEFAULT STREAMLIT DECORATIONS ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# CHARGEMENT MODÈLES (4 modèles par meter) & DONNÉES
# =====================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_PATH, "data", "model_dataset.parquet")

METER_NAMES = {
    0: "Électricité",
    1: "Eau réfrigérée",
    2: "Vapeur",
    3: "Eau chaude"
}

# Vérification dataset
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Dataset introuvable : {DATA_PATH}")
    st.stop()

# Chargement dataset
df = pd.read_parquet(DATA_PATH)

# Ajouter building_age si absent (pour VALIDITY_RANGES)
if "building_age" not in df.columns and "year_built" in df.columns:
    df["building_age"] = 2025 - df["year_built"]

# Variables features
X = df.drop(columns=["meter_reading", "log_meter_reading"])

# Chargement des 4 modèles
MODELS = {}
missing_models = []

for meter_id in METER_NAMES:
    model_path = os.path.join(
        BASE_PATH, "models", f"best_model_meter{meter_id}.pkl"
    )
    if os.path.exists(model_path):
        MODELS[meter_id] = joblib.load(model_path)
    else:
        missing_models.append(meter_id)

if missing_models:
    st.error(
        f"❌ Modèles manquants pour meter(s) : {missing_models}. "
        f"Relancez 07_train_models.py."
    )
    st.stop()

st.success(f"✅ 4 modèles chargés ({', '.join(METER_NAMES.values())})")

# Plages de validité (domaine d'entraînement)
VALIDITY_RANGES = {}
for _col in ["square_feet", "year_built", "air_temperature",
             "dew_temperature", "cloud_coverage", "wind_speed",
             "sea_level_pressure", "building_age"]:
    if _col in df.columns:
        VALIDITY_RANGES[_col] = (float(df[_col].min()), float(df[_col].max()))


def check_oor(value, col):
    if col not in VALIDITY_RANGES:
        return False, None, None
    lo, hi = VALIDITY_RANGES[col]
    return (value < lo or value > hi), lo, hi


# =====================================================
# EN-TÊTE PRINCIPAL
# =====================================================
st.markdown("""
<div class="eml-header">
    <div class="eml-badge">⚡</div>
    <div>
        <div class="eml-title">Energy<em>ML</em></div>
        <div class="eml-subtitle">Système de prédiction énergétique · ASHRAE Dataset · XGBoost</div>
        <div class="eml-meta">
            <div class="eml-chip"><span class="dot"></span>Modèle opérationnel</div>
            <div class="eml-chip">v2.2 · Master DSBD · FSBM</div>
            <div class="eml-chip">4 modèles · par type de compteur</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Mode d'utilisation")

usage_mode = st.sidebar.radio(
    "Objectif",
    ["Estimer un bâtiment en projet", "Analyser un bâtiment existant"],
    help=(
        "Projet : estimer la consommation avant construction ou rénovation.\n\n"
        "Existant : comparer la consommation réelle d'un bâtiment à la norme."
    )
)
is_existing = (usage_mode == "Analyser un bâtiment existant")

st.sidebar.markdown("---")
st.sidebar.header("Identification")

project_ref = st.sidebar.text_input(
    "Référence bâtiment",
    value="Bâtiment A",
    help="Étiquette libre — n'influence pas la prédiction."
)

meter_map = {
    "Électricité": 0,
    "Eau réfrigérée": 1,
    "Vapeur": 2,
    "Eau chaude": 3,
}
meter_label = st.sidebar.selectbox("Type de compteur", list(meter_map.keys()))
meter = meter_map[meter_label]

st.sidebar.markdown("---")
st.sidebar.header("Caractéristiques du bâtiment")

primary_use = st.sidebar.selectbox("Type d'usage", [
    "Education",
    "Office",
    "Entertainment/public assembly",
    "Healthcare",
    "Lodging/residential",
    "Manufacturing/industrial",
    "Food sales and service",
    "Other",
    "Parking",
    "Public services",
    "Religious worship",
    "Retail",
    "Services",
    "Technology/science",
    "Utility",
    "Warehouse/storage"
])

square_feet = st.sidebar.number_input(
    "Surface (sq ft)", 100, 200_000, 5_000, step=500)
if "square_feet" in VALIDITY_RANGES:
    lo, hi = VALIDITY_RANGES["square_feet"]
    st.sidebar.caption(f"Plage entraînement : {lo:,.0f} – {hi:,.0f} sq ft")

year_built = st.sidebar.number_input("Année de construction", 1900, 2025, 2000)
if "year_built" in VALIDITY_RANGES:
    lo, hi = VALIDITY_RANGES["year_built"]
    st.sidebar.caption(f"Plage entraînement : {int(lo)} – {int(hi)}")

st.sidebar.markdown("---")
st.sidebar.header("Conditions météorologiques")

air_temperature = st.sidebar.slider(
    "Température de l'air (°C)", -30.0, 50.0, 20.0, 0.5)
if "air_temperature" in VALIDITY_RANGES:
    lo, hi = VALIDITY_RANGES["air_temperature"]
    st.sidebar.caption(f"Plage entraînement : {lo:.1f} – {hi:.1f} °C")

dew_temperature = st.sidebar.slider(
    "Point de rosée (°C)", -30.0, 40.0, 10.0, 0.5)
if "dew_temperature" in VALIDITY_RANGES:
    lo, hi = VALIDITY_RANGES["dew_temperature"]
    st.sidebar.caption(f"Plage entraînement : {lo:.1f} – {hi:.1f} °C")

cloud_coverage = st.sidebar.slider(
    "Couverture nuageuse (oktas)", 0.0, 9.0, 4.0, 0.5)
precip_depth_1_hr = st.sidebar.slider(
    "Précipitations horaires (mm)", 0.0, 100.0, 0.0)
sea_level_pressure = st.sidebar.slider(
    "Pression atm. (hPa)", 900.0, 1100.0, 1013.0, 0.5)
wind_direction = st.sidebar.slider("Direction du vent (°)", 0, 360, 180)
wind_speed = st.sidebar.slider("Vitesse du vent (m/s)", 0.0, 30.0, 5.0, 0.5)

# ─── Gestion temporelle INTELLIGENTE ───────────────────────────────────────
# En mode PROJET : on n'expose PAS des inputs heure/jour/mois directs car
# un bâtiment en conception n'a pas de timestamp fixe. On demande plutôt
# "Saison typique" et "Profil journalier", puis on mappe vers les valeurs
# numériques que le modèle attend. Cela donne une prédiction représentative
# du scénario d'usage, sans que l'utilisateur ait à comprendre l'encodage
# interne du modèle.
# En mode EXISTANT : on expose les sliders exacts pour reproduire un relevé
# de compteur précis.
# ───────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")

if is_existing:
    st.sidebar.header("Horodatage du relevé")
    month = st.sidebar.slider("Mois", 1, 12, 6)
    day = st.sidebar.slider("Jour", 1, 31, 15)
    hour = st.sidebar.slider("Heure", 0, 23, 12)
    weekday = st.sidebar.slider("Jour de semaine (0 = Lundi)", 0, 6, 2)
    st.sidebar.caption(
        "Reproduisez l'horodatage exact de votre relevé de compteur.")

else:
    # Mode PROJET — paramètres sémantiques
    st.sidebar.header("Scénario d'usage typique")

    saison_map = {
        "Hiver (Déc–Fév)":    {"month": 1,  "temp_hint": "froide"},
        "Printemps (Mar–Mai)": {"month": 4,  "temp_hint": "douce"},
        "Été (Juin–Août)":     {"month": 7,  "temp_hint": "chaude"},
        "Automne (Sep–Nov)":   {"month": 10, "temp_hint": "fraîche"},
    }
    saison_label = st.sidebar.selectbox(
        "Saison représentative",
        list(saison_map.keys()),
        help="Choisissez la saison pour laquelle vous voulez estimer la consommation. "
             "Le modèle utilisera un mois représentatif de cette saison."
    )
    month = saison_map[saison_label]["month"]
    day = 15  # jour médian du mois — valeur neutre

    profil_map = {
        "Heures de bureau (9h–18h)": {"hour": 12, "weekday": 2, "is_weekend": 0},
        "Soirée / après bureau":     {"hour": 19, "weekday": 2, "is_weekend": 0},
        "Nuit (bâtiment à l'arrêt)": {"hour": 3,  "weekday": 2, "is_weekend": 0},
        "Week-end — activité réduite": {"hour": 12, "weekday": 6, "is_weekend": 1},
    }
    profil_label = st.sidebar.selectbox(
        "Profil horaire d'utilisation",
        list(profil_map.keys()),
        help="Choisissez la tranche horaire typique à simuler. "
             "Cela influence le cycle d'occupation que le modèle prend en compte."
    )
    profil = profil_map[profil_label]
    hour = profil["hour"]
    weekday = profil["weekday"]
    # is_weekend géré ci-dessous

    st.sidebar.caption(
        f"→ Saison encodée : mois {month}  ·  Heure : {hour}h  ·  "
        f"Jour semaine : {weekday}"
    )

# ─── Consommation réelle (mode existant uniquement) ───────────────────────
actual_consumption = None
if is_existing:
    st.sidebar.markdown("---")
    st.sidebar.header("Mesure réelle")
    actual_consumption = st.sidebar.number_input(
        "Consommation relevée (kWh)",
        min_value=0.0, value=0.0, step=10.0,
        help="Valeur relevée sur votre compteur pour la même période."
    )

# =====================================================
# FEATURE ENGINEERING (identique au pipeline d'entraînement)
# =====================================================
if is_existing:
    is_weekend = 1 if weekday >= 5 else 0
else:
    is_weekend = profil["is_weekend"]

building_age = max(1, 2025 - year_built)
log_square_feet = np.log1p(square_feet)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(
    ["⚡  Prédiction", "◈  Importance des variables", "▦  Données"])

# ─────────────────────────────────────────────────────
# TAB 1 — PRÉDICTION
# ─────────────────────────────────────────────────────
with tab1:

    # Bandeau de mode
    if is_existing:
        st.markdown("""
        <div class="mode-info">
            <div class="mode-info-icon">🔍</div>
            <div>
                <div class="mode-info-tag">Mode : Détection d'écart — Bâtiment existant</div>
                <div class="mode-info-text">
                    Le modèle estime la consommation <strong>attendue</strong> pour un bâtiment
                    avec ces caractéristiques. Saisissez votre consommation réelle
                    (barre latérale) pour détecter un écart et identifier un potentiel
                    problème énergétique.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="mode-info">
            <div class="mode-info-icon">🏗️</div>
            <div>
                <div class="mode-info-tag">Mode : Estimation prévisionnelle — Projet</div>
                <div class="mode-info-text">
                    Estimation pour un bâtiment <strong>en conception ou en rénovation</strong>.
                    La prédiction représente la consommation horaire typique pour
                    <em>{saison_label}</em> · <em>{profil_label}</em>.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Note explicative sur l'encodage temporel en mode projet
        st.markdown(f"""
        <div class="temporal-note">
            <strong>Comment la dimension temporelle est-elle gérée en mode projet ?</strong><br>
            Le modèle XGBoost a appris des patterns saisonniers et horaires sur des relevés réels.
            En mode projet, au lieu d'exposer des sliders bruts (heure, mois) sans signification
            opérationnelle, EnergyML mappe votre choix de saison et de profil d'usage vers les
            valeurs numériques correspondantes : <strong>mois = {month}</strong>,
            <strong>heure = {hour}h</strong>, <strong>week-end = {is_weekend}</strong>.
            La prédiction est ainsi représentative du scénario sélectionné.
        </div>
        """, unsafe_allow_html=True)

    # ─── Vérification hors plage ───────────────────────────────────────────
    oor_items = []
    checks = [
        (square_feet,    "square_feet",       "Surface"),
        (year_built,     "year_built",         "Année de construction"),
        (air_temperature, "air_temperature",    "Température de l'air"),
        (dew_temperature, "dew_temperature",    "Point de rosée"),
        (cloud_coverage, "cloud_coverage",     "Couverture nuageuse"),
        (wind_speed,     "wind_speed",         "Vitesse du vent"),
        (sea_level_pressure, "sea_level_pressure", "Pression atm."),
        (building_age,   "building_age",       "Âge du bâtiment"),
    ]
    for val, col, label in checks:
        is_oor, lo, hi = check_oor(val, col)
        if is_oor:
            oor_items.append((label, val, lo, hi))

    if oor_items:
        items_html = "".join(
            f'<div class="oor-item">→ {lbl} = {v:g}  (entraînement : {lo:g} – {hi:g})</div>'
            for lbl, v, lo, hi in oor_items
        )
        st.markdown(f"""
        <div class="oor-banner">
            <div style="font-size:1.4rem;flex-shrink:0">⚠️</div>
            <div>
                <div class="oor-tag">Valeur(s) hors domaine d'entraînement</div>
                <div class="oor-text">
                    XGBoost extrapole mal hors de sa distribution d'entraînement.
                    Les variables suivantes dépassent les plages observées — la prédiction
                    reste calculable mais sa fiabilité n'est pas garantie.
                </div>
                {items_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider">Panneau d\'inférence</div>',
                unsafe_allow_html=True)

    # ─── Construction du vecteur d'entrée ─────────────────────────────────
    input_data = {col: 0 for col in X.columns}
    input_data.update({
        "meter":             meter,
        "square_feet":       square_feet,
        "year_built":        year_built,
        "air_temperature":   air_temperature,
        "cloud_coverage":    cloud_coverage,
        "dew_temperature":   dew_temperature,
        "precip_depth_1_hr": precip_depth_1_hr,
        "sea_level_pressure": sea_level_pressure,
        "wind_direction":    wind_direction,
        "wind_speed":        wind_speed,
        "month":             month,
        "day":               day,
        "hour":              hour,
        "weekday":           weekday,
        "is_weekend":        is_weekend,
        "building_age":      building_age,
        "log_square_feet":   log_square_feet,
    })

    pu_col = f"primary_use_{primary_use}"
    if pu_col in input_data:
        input_data[pu_col] = 1

    input_df = pd.DataFrame([input_data])

    # ─── BOUTON PRÉDIRE ───────────────────────────────────────────────────
    if st.button("▶  Lancer la prédiction"):

        # Sélection du modèle selon le type de compteur choisi
        model = MODELS[meter]
        pred_log = model.predict(input_df)[0]
        pred_kwh = max(0.0, float(np.expm1(pred_log)))
        reliability = "⚠ hors plage" if oor_items else "✓ domaine valide"

        # ─── Cartes de résultat ───────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card accent">
                <div class="m-label">Consommation estimée</div>
                <div class="m-value">{pred_kwh:.1f}</div>
                <div class="m-unit">kWh · {reliability}</div>
            </div>
            <div class="metric-card">
                <div class="m-label">Surface</div>
                <div class="m-value">{square_feet:,}</div>
                <div class="m-unit">sq ft · surface brute</div>
            </div>
            <div class="metric-card">
                <div class="m-label">Âge du bâtiment</div>
                <div class="m-value">{building_age}</div>
                <div class="m-unit">ans · depuis construction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Comparaison réel vs prédit (mode existant) ───────────────────
        gap_pct = None
        if is_existing and actual_consumption and actual_consumption > 0:
            gap_pct = (actual_consumption - pred_kwh) / pred_kwh * 100

            if gap_pct > 15:
                vcls, vico = "over", "⚠️"
                vtag = "Sur-consommation détectée"
                vtxt = (
                    f"La consommation réelle ({actual_consumption:.1f} kWh) dépasse "
                    f"l'estimation de {gap_pct:.1f} %. Cela peut signaler une isolation "
                    f"déficiente, un équipement énergivore ou un usage non optimisé. "
                    f"Un audit énergétique est recommandé."
                )
            elif gap_pct < -15:
                vcls, vico = "under", "✅"
                vtag = "Consommation en dessous de la norme"
                vtxt = (
                    f"La consommation réelle ({actual_consumption:.1f} kWh) est "
                    f"{abs(gap_pct):.1f} % inférieure à l'estimation. "
                    f"Ce bâtiment présente une efficacité énergétique supérieure à la moyenne."
                )
            else:
                vcls, vico = "aligned", "ℹ️"
                vtag = "Consommation conforme aux attentes"
                vtxt = (
                    f"La consommation réelle ({actual_consumption:.1f} kWh) est alignée "
                    f"avec l'estimation (écart {gap_pct:+.1f} %). "
                    f"Le comportement énergétique est normal pour ce profil de bâtiment."
                )

            st.markdown(f"""
            <div class="verdict {vcls}">
                <div class="verdict-icon">{vico}</div>
                <div>
                    <div class="verdict-tag">{vtag}</div>
                    <div class="verdict-text">{vtxt}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            card_cls = "warning" if gap_pct > 15 else (
                "success" if gap_pct < -15 else "accent")
            st.markdown(f"""
            <div class="metric-row" style="grid-template-columns:repeat(2,1fr)">
                <div class="metric-card {card_cls}">
                    <div class="m-label">Consommation réelle</div>
                    <div class="m-value">{actual_consumption:.1f}</div>
                    <div class="m-unit">kWh · relevé compteur</div>
                </div>
                <div class="metric-card {card_cls}">
                    <div class="m-label">Écart vs modèle</div>
                    <div class="m-value">{gap_pct:+.1f}%</div>
                    <div class="m-unit">par rapport à la prédiction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif is_existing:
            st.info(
                "→ Saisissez votre consommation réelle dans la barre latérale pour afficher le diagnostic.")

        # ─── JAUGE ────────────────────────────────────────────────────────
        gauge_max = max(pred_kwh * 2, 100)
        ratio = pred_kwh / gauge_max

        if ratio < 0.4:
            bar_color = "#34D399"
        elif ratio < 0.75:
            bar_color = "#FBBF24"
        else:
            bar_color = "#FB7185"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_kwh,
            number={
                "font": {"family": "Inter", "color": "#2563EB", "size": 40},
                "suffix": " kWh"
            },
            title={
                "text": "CONSOMMATION HORAIRE PRÉDITE",
                "font": {"family": "JetBrains Mono", "color": "#94A3B8", "size": 11}
            },
            gauge={
                "axis": {
                    "range": [0, gauge_max],
                    "tickfont": {"family": "JetBrains Mono", "color": "#94A3B8", "size": 10},
                    "tickcolor": "#CBD5E1",
                },
                "bar":      {"color": bar_color, "thickness": 0.20},
                "bgcolor":  "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,              gauge_max * 0.40],
                        "color": "rgba(5,150,105,0.07)"},
                    {"range": [gauge_max*0.40,  gauge_max * 0.75],
                        "color": "rgba(217,119,6,0.07)"},
                    {"range": [gauge_max*0.75,  gauge_max],
                        "color": "rgba(225,29,72,0.07)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 2},
                    "thickness": 0.82,
                    "value": pred_kwh
                }
            }
        ))

        fig_gauge.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font_color="#64748B",
            margin=dict(t=70, b=20, l=30, r=30),
            height=290,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ─── Historique session ────────────────────────────────────────────
        if "history" not in st.session_state:
            st.session_state.history = []

        record = {
            "Référence":       project_ref,
            "Mode":            "Existant" if is_existing else "Projet",
            "Type d'usage":    primary_use,
            "Surface (sq ft)": square_feet,
            "Prédiction (kWh)": round(pred_kwh, 2),
        }
        if is_existing and gap_pct is not None:
            record["Réel (kWh)"] = round(actual_consumption, 2)
            record["Écart (%)"] = round(gap_pct, 1)
        else:
            if not is_existing:
                record["Saison"] = saison_label
                record["Profil"] = profil_label

        st.session_state.history.append(record)
        st.success(
            "✓  Inférence terminée — résultat enregistré dans le journal de session")

    # ─── Tableau historique ────────────────────────────────────────────────
    if st.session_state.get("history"):
        hist_df = pd.DataFrame(st.session_state.history)
        n = len(hist_df)
        st.markdown(f"""
        <div class="history-bar">
            <div class="history-title">◈ Journal de session</div>
            <div class="history-badge">{n} enregistrement{'s' if n > 1 else ''}</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────
# TAB 2 — IMPORTANCE DES VARIABLES
# ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-divider">Explicabilité du modèle · Poids des variables</div>',
                unsafe_allow_html=True)

    # Sélecteur de meter pour l'importance
    fi_meter_label = st.selectbox(
        "Choisir le type de compteur",
        list(METER_NAMES.values()),
        key="fi_meter"
    )
    fi_meter_id = [k for k, v in METER_NAMES.items() if v == fi_meter_label][0]
    model_fi = MODELS[fi_meter_id]

    if hasattr(model_fi, "feature_importances_"):
        importance = pd.DataFrame({
            "Variable":   X.columns,
            "Importance": model_fi.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.dataframe(importance.head(
            15), use_container_width=True, hide_index=True)

        top10 = importance.head(10).iloc[::-1].copy()

        # Libellés français pour les features clés
        label_fr = {
            "log_square_feet":   "Surface (log)",
            "air_temperature":   "Température air",
            "dew_temperature":   "Point de rosée",
            "hour":              "Heure",
            "building_age":      "Âge bâtiment",
            "month":             "Mois",
            "weekday":           "Jour semaine",
            "cloud_coverage":    "Couverture nuageuse",
            "wind_speed":        "Vitesse vent",
            "sea_level_pressure": "Pression atm.",
        }
        top10["Label"] = top10["Variable"].map(lambda x: label_fr.get(x, x))

        mpl.rcParams.update({
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor":   "#F8F9FC",
            "axes.edgecolor":   "#E2E8F0",
            "axes.labelcolor":  "#64748B",
            "xtick.color":      "#94A3B8",
            "ytick.color":      "#334155",
            "text.color":       "#475569",
            "grid.color":       "#F1F5F9",
            "font.family":      "monospace",
            "font.size":        9,
        })

        fig_fi, ax = plt.subplots(figsize=(9, 4.8))
        fig_fi.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8F9FC")

        n_bars = len(top10)
        # Dégradé bleu → cyan pour les barres
        colors = [
            f"#{int(37 + (8-37)*i/(n_bars-1)):02X}"
            f"{int(99 + (145-99)*i/(n_bars-1)):02X}"
            f"{int(235 + (213-235)*i/(n_bars-1)):02X}"
            for i in range(n_bars)
        ]

        bars = ax.barh(
            top10["Label"],
            top10["Importance"],
            color=colors,
            height=0.55,
            edgecolor="none"
        )

        ax.set_xlabel("Score d'importance", labelpad=8, fontsize=8,
                      color="#94A3B8", fontfamily="monospace")
        ax.set_title("TOP 10 VARIABLES PRÉDICTIVES", fontsize=9,
                     color="#94A3B8", fontfamily="monospace",
                     fontweight="normal", loc="left", pad=12)

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")
        ax.xaxis.grid(True, alpha=0.6, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", ha="left",
                    fontsize=7, color="#94A3B8", fontfamily="monospace")

        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

        # Interprétation textuelle
        st.markdown('<div class="section-divider">Interprétation</div>',
                    unsafe_allow_html=True)
        top1 = importance.iloc[0]
        top2 = importance.iloc[1]
        interp_map = {
            "log_square_feet": "la **surface du bâtiment** (transformée en log) — variable physiquement cohérente : un grand bâtiment consomme plus.",
            "air_temperature": "la **température extérieure** — pilote directement les besoins CVC (chauffage/climatisation).",
            "dew_temperature": "le **point de rosée** — indicateur d'humidité, influence la déshumidification et le confort thermique.",
            "hour":            "l'**heure de la journée** — capture les cycles d'occupation et d'activité.",
            "building_age":    "l'**âge du bâtiment** — proxy de la qualité d'isolation thermique.",
        }
        t1_txt = interp_map.get(top1["Variable"], f"**{top1['Variable']}**")
        t2_txt = interp_map.get(top2["Variable"], f"**{top2['Variable']}**")
        st.markdown(f"""
        Modèle sélectionné : **{type(model_fi).__name__}** · Compteur : **{fi_meter_label}**

        Le prédicteur le plus influent est {t1_txt}  
        En second rang : {t2_txt}
        """)
    else:
        st.info("ℹ️ Ce modèle ne supporte pas feature_importances_ (ex: LinearRegression).")


# ─────────────────────────────────────────────────────
# TAB 3 — DONNÉES
# ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-divider">Aperçu du jeu de données d\'entraînement</div>',
                unsafe_allow_html=True)

    r, c = df.shape
    c1, c2, c3 = st.columns(3)

    def stat_card(col_obj, label, value, unit=""):
        col_obj.markdown(f"""
        <div class="metric-card" style="border-radius:12px;border:1px solid rgba(15,23,42,0.10);
             padding:1.2rem 1.4rem;background:#FFFFFF;box-shadow:0 1px 4px rgba(15,23,42,0.06)">
            <div class="m-label">{label}</div>
            <div class="m-value" style="font-size:2rem;color:#2563EB">{value}</div>
            <div class="m-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

    stat_card(c1, "Observations", f"{r:,}", "lignes")
    stat_card(c2, "Variables",    str(c),   "colonnes")
    stat_card(c3, "Mémoire",
              f"{df.memory_usage(deep=True).sum()/1e6:.1f}", "Mo")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-divider">20 premières lignes</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-divider">Statistiques descriptives</div>',
                unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)