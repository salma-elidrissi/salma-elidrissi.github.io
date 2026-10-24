"""
=============================================================================
HCP – Haut-Commissariat au Plan
Tableau de Bord de Prévision Économique
VECM · Random Forest · XGBoost
=============================================================================
"""

import streamlit as st
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler 

# ── XGBoost ──────────────────────────────────────────────────────────────────
import xgboost as xgb

# ── Statsmodels (VECM) ───────────────────────────────────────────────────────
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCP – Prévision Économique",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  (couleurs officielles HCP : bleu #003F87 + ocre #C8A951)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Root variables (Couleurs ajustées fidèlement au logo HCP) ── */
/* ── Root variables (Palette Violet & Orange) ── */
:root {
    --hcp-blue:      #4A154B; /* Violet Profond */
    --hcp-blue-mid:  #611F69; /* Violet Intermédiaire pour les dégradés */
    --hcp-blue-light:#F4EDF6; /* Fond violet très clair pour les survol de tableaux */
    --hcp-gold:      #E67E22; /* Orange Dynamique */
    --hcp-gold-light:#FDF2E9; /* Fond orange très clair pour les alertes */
    --hcp-dark:      #300D33; /* Violet très sombre (presque noir) pour le texte de structure */
    --hcp-text:      #1A1A1A; /* Couleur principale du texte liseux */
    --hcp-muted:     #7F8C8D; 
    --hcp-border:    #E0D0E3;
    --hcp-bg:        #FAFAFB; /* Fond de page épuré */
    --hcp-white:     #FFFFFF;
    --success:       #27AE60;
    --warning:       #E67E22;
    --danger:        #C0392B;
    --radius:        10px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--hcp-text);
}
.stApp { background: var(--hcp-bg); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--hcp-dark) 0%, var(--hcp-blue) 100%);
    border-right: 3px solid var(--hcp-gold);
}

/* Texte général et labels dans la sidebar */
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown { 
    color: #F4EDF6 !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stTextInput label { 
    color: var(--hcp-gold) !important; 
    font-weight: 600; 
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--hcp-gold) !important;
    border-bottom: 1px solid rgba(230,126,34,0.4);
    padding-bottom: 6px;
}

/* ── Lisibilité des champs de saisie (Inputs) ── */
section[data-testid="stSidebar"] input {
    color: #2F0C33 !important; /* Texte saisi en violet très sombre */
    background-color: #FFFFFF !important; /* Fond blanc pur */
    border: 1px solid var(--hcp-border) !important;
}

/* Style spécifique pour le texte d'aide / placeholder (ex: 2026/06/10) */
section[data-testid="stSidebar"] input::placeholder {
    color: #7F8C8D !important; /* Gris parfaitement lisible */
    opacity: 1; 
}    

/* Forcer la visibilité du texte saisi et des dates (Fix Chrome/Safari/Edge) */
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] input[type="date"],
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] input {
    color: #2F0C33 !important;
    -webkit-text-fill-color: #2F0C33 !important;
    background-color: #FFFFFF !important;
}

/* Ajustement pour l'icône de calendrier des champs de date */
section[data-testid="stSidebar"] input::-webkit-calendar-picker-indicator {
    filter: invert(0.3); 
    cursor: pointer;
}
/* ── Header banner ── */
.hcp-header {
    background: linear-gradient(135deg, var(--hcp-blue) 0%, var(--hcp-blue-mid) 60%, var(--hcp-dark) 100%);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    border-left: 6px solid var(--hcp-gold);
    box-shadow: 0 4px 20px rgba(0,63,135,0.18);
}
.hcp-header-title { flex: 1; }
.hcp-header-title h1 {
    color: #FFFFFF;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
}
.hcp-header-title p {
    color: rgba(232,240,250,0.8);
    font-size: 0.85rem;
    margin: 0;
    font-weight: 300;
}
.hcp-badge {
    background: var(--hcp-gold);
    color: var(--hcp-dark);
    font-weight: 700;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ── KPI Cards ── */
.kpi-grid { display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi-card {
    background: var(--hcp-white);
    border: 1px solid var(--hcp-border);
    border-radius: var(--radius);
    padding: 18px 22px;
    flex: 1;
    min-width: 150px;
    border-top: 3px solid var(--hcp-blue);
    box-shadow: 0 2px 8px rgba(0,63,135,0.06);
}
.kpi-card.gold { border-top-color: var(--hcp-gold); }
.kpi-card.success { border-top-color: var(--success); }
.kpi-card.danger  { border-top-color: var(--danger); }
.kpi-label { font-size: 0.72rem; color: var(--hcp-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; }
.kpi-value { font-size: 1.55rem; font-weight: 700; color: var(--hcp-blue); font-family: 'IBM Plex Mono', monospace; }
.kpi-value.gold { color: var(--hcp-gold); }
.kpi-value.success { color: var(--success); }
.kpi-value.danger  { color: var(--danger); }
.kpi-sub { font-size: 0.72rem; color: var(--hcp-muted); margin-top: 4px; }

/* ── Section headers ── */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--hcp-blue);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-left: 4px solid var(--hcp-gold);
    padding-left: 12px;
    margin: 22px 0 14px 0;
}

/* ── Cards / panels ── */
.panel {
    background: var(--hcp-white);
    border: 1px solid var(--hcp-border);
    border-radius: var(--radius);
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,63,135,0.05);
}

/* ── Decision box ── */
.decision-box {
    border-radius: var(--radius);
    padding: 20px 24px;
    margin: 10px 0;
    border-left: 5px solid;
}
.decision-box.hausse  { background: #EAF5EE; border-color: var(--success); }
.decision-box.baisse  { background: #FDEDED; border-color: var(--danger); }
.decision-box.stable  { background: var(--hcp-gold-light); border-color: var(--hcp-gold); }
.decision-title { font-weight: 700; font-size: 1rem; margin-bottom: 6px; }
.decision-text  { font-size: 0.88rem; color: var(--hcp-text); line-height: 1.6; }

/* ── Table ── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.styled-table th { background: var(--hcp-blue); color: white; padding: 10px 14px; text-align: left; font-weight: 600; }
.styled-table td { padding: 9px 14px; border-bottom: 1px solid var(--hcp-border); }
.styled-table tr:hover td { background: var(--hcp-blue-light); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--hcp-bg); padding: 4px; border-radius: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 500;
    color: var(--hcp-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--hcp-blue) !important;
    color: white !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--hcp-blue);
    color: white;
    border: none;
    border-radius: 7px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.87rem;
    transition: all .2s;
}
.stButton > button:hover { background: var(--hcp-blue-mid); box-shadow: 0 4px 12px rgba(0,63,135,0.25); }

/* ── Metric tweaks ── */
[data-testid="stMetricValue"] { color: var(--hcp-blue) !important; font-family: 'IBM Plex Mono' !important; }

/* ── Footer ── */
.hcp-footer {
    text-align: center;
    font-size: 0.75rem;
    color: var(--hcp-muted);
    padding: 20px 0 8px 0;
    border-top: 1px solid var(--hcp-border);
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
HCP_COLORS = {
    "blue":  "#4A154B", # Violet principal
    "gold":  "#E67E22", # Orange principal
    "teal":  "#9B59B6", # Violet clair secondaire
    "red":   "#C0392B",
    "green": "#27AE60",
    "gray":  "#7F8C8D",
}

def hcp_fig(nrows=1, ncols=1, figsize=(11, 4.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in (np.array(axes).flatten() if nrows*ncols > 1 else [axes]):
        ax.set_facecolor("#F4F7FB")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#D0DCE8")
        ax.tick_params(colors="#6B7A8D", labelsize=8)
        ax.xaxis.label.set_color("#1C2B3A")
        ax.yaxis.label.set_color("#1C2B3A")
        ax.title.set_color("#003F87")
        ax.title.set_fontweight("bold")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#D0DCE8", alpha=0.8)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#   VECM  –  PARAMÈTRES MODIFIABLES   ← MODIFIEZ CES VALEURS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Rang de cointégration (nombre de relations à long terme) ───────────────
VECM_COINT_RANK = 1            # int ≥ 1

# ── 2. Nombre de retards (lags) en différences ────────────────────────────────
VECM_K_AR_DIFF  = 1           # int ≥ 1

# ── 3. Déterministe : "nc" | "co" | "ci" | "lo" | "li" ──────────────────────
VECM_DETERMINISTIC = "co"      # constante hors espace de cointégration

# ── 4. Matrice de cointégration β (vecteurs colonnes) ─────────────────────────
#      Laissez None pour l'estimer automatiquement, ou fournissez un array
#      numpy de forme (n_variables, coint_rank).
#      Ex : VECM_BETA = np.array([[1], [-0.8], [0.3]])
VECM_BETA = None               # None = estimation automatique

# ── 5. Matrice d'ajustement α (vitesses de rappel) ────────────────────────────
#      None = estimation automatique.
#      Ex : VECM_ALPHA = np.array([[-0.15], [0.08], [0.02]])
VECM_ALPHA = None              # None = estimation automatique

# ── 6. Horizons de prévision disponibles (trimestres) ────────────────────────
VECM_HORIZON_OPTIONS = [4, 8, 12, 16, 20]


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM FOREST – PARAMÈTRES MODIFIABLES
# ─────────────────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS   = 300
RF_MAX_DEPTH      = None       # None = sans limite
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF  = 2
RF_MAX_FEATURES   = "sqrt"
RF_LAGS           = 4          # nombre de retards comme features

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST – PARAMÈTRES MODIFIABLES
# ─────────────────────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS      = 500
XGB_MAX_DEPTH         = 4
XGB_LEARNING_RATE     = 0.05
XGB_SUBSAMPLE         = 0.8
XGB_COLSAMPLE_BYTREE  = 0.8
XGB_REG_ALPHA         = 0.1    # L1
XGB_REG_LAMBDA        = 1.0    # L2
XGB_LAGS              = 4


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def metrics_dict(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}


def make_lag_features(series: np.ndarray, lags: int):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def adf_test(series):
    result = adfuller(series, autolag="AIC")
    return {"stat": result[0], "p-value": result[1], "stationary": result[1] < 0.05}


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════  VECM MODEL  ══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_vecm(data: pd.DataFrame, horizon: int):
    """Ajuste un VECM et retourne prévisions + résumé."""
    try:
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:]
        model = VECM(
            endog=data,
            k_ar_diff=VECM_K_AR_DIFF,
            coint_rank=VECM_COINT_RANK,
            deterministic=VECM_DETERMINISTIC
        )
        fitted = model.fit()
        y_pred_test = fitted.predict(steps=len(test))
        target_idx = train.columns.get_loc(target_col)
        y_true = test.iloc[:, target_idx].values
        y_pred = y_pred_test[:, target_idx]
        vecm_metrics = metrics_dict(y_true, y_pred)


        forecast_obj = fitted.predict(steps=horizon)
        # forecast_obj : array (horizon, n_vars)
        forecast_df = pd.DataFrame(
            forecast_obj,
            columns=data.columns,
            index=pd.RangeIndex(1, horizon + 1),
        )
        return fitted, forecast_df, vecm_metrics, None
    except Exception as e:
        return None, None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════  RF MODEL  ════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_rf(series: np.ndarray, horizon: int, target_col: str):
    lags = RF_LAGS
    X, y = make_lag_features(series, lags)
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_depth        = RF_MAX_DEPTH,
        min_samples_split= RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        max_features     = RF_MAX_FEATURES,
        random_state     = 42,
        n_jobs           = -1,
    )
    rf.fit(X_train_s, y_train)
    y_pred_test = rf.predict(X_test_s)

    # Prévisions itératives
    last_window = series[-lags:].tolist()
    preds = []
    for _ in range(horizon):
        x_in = np.array(last_window[-lags:]).reshape(1, -1)
        x_in_s = scaler.transform(x_in)
        p = rf.predict(x_in_s)[0]
        preds.append(p)
        last_window.append(p)

    m = metrics_dict(y_test, y_pred_test) if len(y_test) > 0 else {}
    importances = pd.Series(rf.feature_importances_,
                            index=[f"lag_{i+1}" for i in range(lags)])
    return np.array(preds), m, importances


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════  XGB MODEL  ═══════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_xgb(series: np.ndarray, horizon: int, target_col: str):
    lags = XGB_LAGS
    X, y = make_lag_features(series, lags)
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators     = XGB_N_ESTIMATORS,
        max_depth        = XGB_MAX_DEPTH,
        learning_rate    = XGB_LEARNING_RATE,
        subsample        = XGB_SUBSAMPLE,
        colsample_bytree = XGB_COLSAMPLE_BYTREE,
        reg_alpha        = XGB_REG_ALPHA,
        reg_lambda       = XGB_REG_LAMBDA,
        random_state     = 42,
        verbosity        = 0,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              verbose=False)
    y_pred_test = model.predict(X_test)

    last_window = series[-lags:].tolist()
    preds = []
    for _ in range(horizon):
        x_in = np.array(last_window[-lags:]).reshape(1, -1)
        p = model.predict(x_in)[0]
        preds.append(p)
        last_window.append(p)

    m = metrics_dict(y_test, y_pred_test) if len(y_test) > 0 else {}
    importances = pd.Series(
        model.feature_importances_,
        index=[f"lag_{i+1}" for i in range(lags)]
    )
    return np.array(preds), m, importances


# ─────────────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_decision(series_hist, vecm_fc, rf_fc, xgb_fc, target_col, threshold=0.02):
    last_val = series_hist[-1]
    preds = {
        "VECM":          vecm_fc[0] if vecm_fc is not None else None,
        "Random Forest": rf_fc[0]   if rf_fc  is not None else None,
        "XGBoost":       xgb_fc[0]  if xgb_fc is not None else None,
    }
    valid = {k: v for k, v in preds.items() if v is not None}
    if not valid:
        return None, "Aucune prévision disponible.", {}

    consensus = np.mean(list(valid.values()))
    change_pct = (consensus - last_val) / (abs(last_val) + 1e-9)

    if change_pct > threshold:
        signal = "hausse"
        icon   = "📈"
        msg    = (f"**Signal HAUSSE** — Consensus des modèles : +{change_pct*100:.1f}% "
                  f"par rapport à la dernière observation ({last_val:.2f}). "
                  f"Une politique d'accompagnement à la croissance pourrait être envisagée.")
    elif change_pct < -threshold:
        signal = "baisse"
        icon   = "📉"
        msg    = (f"**Signal BAISSE** — Consensus des modèles : {change_pct*100:.1f}% "
                  f"par rapport à la dernière observation ({last_val:.2f}). "
                  f"Des mesures correctives ou de soutien peuvent être nécessaires.")
    else:
        signal = "stable"
        icon   = "➡️"
        msg    = (f"**Signal STABLE** — Variation prévue : {change_pct*100:.1f}%. "
                  f"Le contexte actuel semble se maintenir à court terme.")

    return signal, f"{icon} {msg}", valid


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════  INTERFACE  ═══════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hcp-header">
  <div class="hcp-header-title">
    <h1>Tableau de Bord de Prévision Économique</h1>
    <p>Haut-Commissariat au Plan &nbsp;·&nbsp; Modèles VECM · Random Forest · XGBoost &nbsp;·&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # ── Source de données ──────────────────────────────────────────────────
    st.markdown("#### 📥 Source des données")
    data_source = st.radio(
        "Mode de saisie",
        [" Importer un fichier", " Saisie manuelle"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("####  Modèles actifs")
    use_vecm = st.checkbox("VECM",          value=True)
    use_rf   = st.checkbox("Random Forest", value=True)
    use_xgb  = st.checkbox("XGBoost",       value=True)

    st.markdown("---")
    st.markdown("####  Période de prévision")
    date_debut = st.date_input(
        "Date de début"
    )

    date_fin = st.date_input(
        "Date de fin"
    )

    st.markdown("---")
    st.markdown("#### 🎯 Variable cible")
    target_col = st.text_input(
        "Nom de la variable cible", 
        value="",
        placeholder="Entrez le nom de la variable cible"
    )

    

    st.markdown("---")
    run_btn = st.button("▶  Lancer les prévisions", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem;color:rgba(232,240,250,0.5);text-align:center;line-height:1.8">
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT – TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_results, tab_models, tab_diagnostic = st.tabs([
    "📊 Données",
    "📈 Prévisions",
    "🔬 Évaluation des Modèles",
    "📋 Diagnostic et Orientations ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    df = None

    # ── Import fichier ─────────────────────────────────────────────────────
    if "Importer" in data_source:
        st.markdown('<div class="section-title">Importation des données</div>', unsafe_allow_html=True)
        col_up, col_info = st.columns([2, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Déposez votre fichier (CSV, Excel, ODS)",
                type=["csv", "xlsx", "xls", "ods"],
            )
        with col_info:
            st.info("**Formats acceptés**\n\nCSV · Excel (.xlsx / .xls) · ODS\n\n"
                    "La première colonne peut être une date (index).")

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    sep = st.selectbox("Séparateur CSV", [",", ";", "\\t", "|"], index=1)
                    sep = "\t" if sep == "\\t" else sep
                    df = pd.read_csv(uploaded, sep=sep, index_col=0, parse_dates=True)
                else:
                    df = pd.read_excel(uploaded, index_col=0, parse_dates=True)
                df = df.select_dtypes(include="number").dropna()
                st.success(f"✅ Fichier chargé : {df.shape[0]} observations · {df.shape[1]} variable(s)")
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")

    # ── Saisie manuelle ────────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-title">Saisie manuelle des données</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            n_obs  = st.number_input("Nombre d'observations", 10, 500, 40, 1)
        with c2:
            n_vars = st.number_input("Nombre de variables",   1, 10,  3,  1)
        with c3:
            var_names_raw = st.text_input("Noms des variables (virgule)",
                                          value=",".join([f"Var{i+1}" for i in range(int(n_vars))]))
        var_names = [v.strip() for v in var_names_raw.split(",")][:int(n_vars)]

        st.markdown("**Saisissez vos données ci-dessous :**")
        default_data = {}
        for v in var_names:
            default_data[v] = [round(100 + np.random.randn() * 5 + i * 0.3, 2)
                                for i in range(int(n_obs))]
        df_edit = pd.DataFrame(default_data)
        df_edit = st.data_editor(df_edit, use_container_width=True, num_rows="dynamic")

        if st.button("✔ Valider la saisie"):
            df = df_edit.select_dtypes(include="number").dropna()
            st.success(f"✅ Données validées : {df.shape[0]} obs · {df.shape[1]} variable(s)")
        elif df_edit is not None and not df_edit.empty:
            df = df_edit.select_dtypes(include="number").dropna()

    # ── Preview ────────────────────────────────────────────────────────────
    if df is not None and not df.empty:
        st.session_state["df"] = df

        st.markdown('<div class="section-title">Aperçu & Statistiques</div>', unsafe_allow_html=True)
        col_prev, col_stat = st.columns([3, 2])
        with col_prev:
            st.dataframe(df.head(15), use_container_width=True, height=320)
        with col_stat:
            st.dataframe(df.describe().T.round(3), use_container_width=True, height=320)

        # ADF tests
        st.markdown('<div class="section-title">Tests de Stationnarité (ADF)</div>', unsafe_allow_html=True)
        adf_results = []
        for col in df.columns:
            r = adf_test(df[col].values)
            adf_results.append({
                "Variable": col,
                "Stat ADF":  round(r["stat"], 4),
                "p-value":   round(r["p-value"], 4),
                "Stationnaire": "✅ Oui" if r["stationary"] else "❌ Non (I(1)?)",
            })
        st.dataframe(pd.DataFrame(adf_results).set_index("Variable"),
                     use_container_width=True)

        # Série temporelle
        st.markdown('<div class="section-title">Visualisation des séries</div>', unsafe_allow_html=True)
        n_cols_plot = min(len(df.columns), 3)
        fig_data, axes_data = hcp_fig(
            nrows=max(1, (len(df.columns) + n_cols_plot - 1) // n_cols_plot),
            ncols=n_cols_plot,
            figsize=(13, 3.5 * max(1, (len(df.columns) + n_cols_plot - 1) // n_cols_plot)),
        )
        ax_list = np.array(axes_data).flatten() if hasattr(axes_data, "__len__") else [axes_data]
        colors_cycle = list(HCP_COLORS.values())
        for idx, col in enumerate(df.columns):
            if idx < len(ax_list):
                ax_list[idx].plot(df[col].values, color=colors_cycle[idx % len(colors_cycle)], linewidth=1.8)
                ax_list[idx].set_title(col)
                ax_list[idx].set_xlabel("Périodes")
        for idx in range(len(df.columns), len(ax_list)):
            ax_list[idx].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_data)
    else:
        st.info("ℹ️ Aucune donnée chargée. Importez un fichier ou saisissez vos données.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – PRÉVISIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    df_loaded = st.session_state.get("df", None)

    if df_loaded is None or df_loaded.empty:
        st.warning("⚠️ Veuillez d'abord charger des données dans l'onglet **Données**.")
    else:
        # Variable cible (sidebar placeholder)
        
        if target_col == "":
            st.warning("Veuillez saisir le nom de la variable cible.")
            st.stop()

        if target_col not in df_loaded.columns:
            st.error(
            f"La variable '{target_col}' n'existe pas dans les données."
            )
            st.stop()
        target_series = df_loaded[target_col].values.astype(float)

        if run_btn or st.session_state.get("ran_once"):
            st.session_state["ran_once"] = True

            # Bug 2 — validation des dates doit être DANS le bloc conditionnel
            if date_fin <= date_debut:
                st.error("La date de fin doit être postérieure à la date de début.")
                st.stop()

            # Calcul du nombre de trimestres
            horizon = (
                (date_fin.year - date_debut.year) * 4
                + (date_fin.month - date_debut.month) // 3
            ) + 1

            dates_prevision = pd.date_range(
                start=date_debut,
                periods=horizon,
                freq="QE"
            )

            # ── Spinner + modèles ──────────────────────────────────────────
            with st.spinner("Entraînement des modèles en cours…"):
                vecm_forecast_df, rf_preds, xgb_preds = None, None, None
                vecm_fitted, vecm_err = None, None
                rf_metrics, xgb_metrics, vecm_metrics = {}, {}, {}


                if use_vecm and len(df_loaded.columns) >= 2:
                    vecm_fitted, vecm_forecast_df,vecm_metrics, vecm_err = run_vecm(
                        df_loaded, horizon)
                elif use_vecm and len(df_loaded.columns) < 2:
                    vecm_err = "VECM nécessite au moins 2 variables."

                if use_rf:
                    rf_preds, rf_metrics, _ = run_rf(
                       target_series, horizon, target_col)

                if use_xgb:
                    xgb_preds, xgb_metrics, _ = run_xgb(
                        target_series, horizon, target_col)

            # ── KPIs ──────────────────────────────────────────────────────
            # ── Horizon prévu ─────────────────────────────────────────────
            st.markdown(
                '<div class="section-title">Horizon de Prévision</div>',
                unsafe_allow_html=True
            )

            kpi_html = f"""
            <div class="kpi-card gold">
                <div class="kpi-label">Horizon prévu</div>
                <div class="kpi-value gold">{horizon}</div>
                <div class="kpi-sub">trimestres</div>
            </div>
            """

            st.markdown(kpi_html, unsafe_allow_html=True)   
            # ── Graphique prévisions ───────────────────────────────────────
            st.markdown('<div class="section-title">Prévisions – ' + target_col + '</div>',
                        unsafe_allow_html=True)
            
            n_hist = 30

            fig_fc, ax_fc = plt.subplots(figsize=(13,5))
            hist_dates = df_loaded.index[-n_hist:]
            hist_series = target_series[-n_hist:]
            ax_fc.plot(
                hist_dates, 
                hist_series,
                color="#003F87",
                linewidth=2,
                label="Historique"
            )
            ax_fc.axvline(
                x=df_loaded.index[-1],         
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5
            )
        

            if vecm_forecast_df is not None and target_col in vecm_forecast_df.columns:
                ax_fc.plot(
                    dates_prevision, 
                    vecm_forecast_df[target_col].values,
                    linewidth=2, 
                    marker="o",
                    markersize=3,
                    label="VECM"
                )

            if rf_preds is not None:
                ax_fc.plot(
                    dates_prevision, 
                    rf_preds,
                    linewidth=2,
                    marker="o", 
                    markersize=3, 
                    label="Random Forest"
                )

            if xgb_preds is not None:
                ax_fc.plot(
                    dates_prevision, 
                    xgb_preds,
                    linewidth=2,
                    marker="o", 
                    markersize=3, 
                    label="XGBoost"
                )

            ax_fc.set_xlabel("")
            ax_fc.set_ylabel(target_col)

            ax_fc.grid(axis="y", alpha=0.3)

            ax_fc.legend(
                   ncol=4,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.12),
                   frameon=False
            )
            ticks = list(hist_dates[::4]) + list(dates_prevision)

            labels = [
                f"{d.year}T{((d.month-1)//3)+1}"
                for d in ticks
            ]

            ax_fc.set_xticks(ticks)
            ax_fc.set_xticklabels(labels, rotation=45)

            plt.tight_layout()

            st.pyplot(fig_fc)

        

            # ── Tableau synthèse ───────────────────────────────────────────
            st.markdown('<div class="section-title">Tableau de synthèse des prévisions</div>',
                        unsafe_allow_html=True)
            # Bug 4 — espace parasite devant `summary` (IndentationError) + clé
            #          "Horizon (T+)" incohérente avec "Date" utilisé plus loin
            summary = {
                "Date": [str(d.date()) for d in dates_prevision]
            }
            if vecm_forecast_df is not None and target_col in vecm_forecast_df.columns:
                summary["VECM"] = vecm_forecast_df[target_col].round(3).tolist()
            if rf_preds is not None:
                summary["Random Forest"] = np.round(rf_preds, 3).tolist()
            if xgb_preds is not None:
                summary["XGBoost"] = np.round(xgb_preds, 3).tolist()

            # consensus (Bug 5 — filtre sur "Horizon (T+)" qui n'existe plus,
            #            remplacé par "Date")

            st.dataframe(pd.DataFrame(summary).set_index("Date"),
                         use_container_width=True)

            # ── VECM erreur éventuelle ─────────────────────────────────────
            if vecm_err:
                st.warning(f"⚠️ VECM : {vecm_err}")

            # ── Store for other tabs ───────────────────────────────────────
            st.session_state["results"] = {
                "vecm_fc": vecm_forecast_df[target_col].values
                           if (vecm_forecast_df is not None and target_col in vecm_forecast_df.columns)
                           else None,
                "rf_fc":   rf_preds,
                "xgb_fc":  xgb_preds,
                "rf_metrics":  rf_metrics,
                "xgb_metrics": xgb_metrics,
                "vecm_metrics": vecm_metrics,
                "vecm_fitted": vecm_fitted,
                "target_series": target_series,
                "target_col":    target_col,
                "horizon":       horizon,
            }
        # Bug 6 — `else` orphelin (sans `if` correspondant au bon niveau)
        else:
            st.info("ℹ️ Configurez vos paramètres dans la barre latérale, puis cliquez sur **▶ Lancer les prévisions**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ÉVALUATION DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
with tab_models:
    res = st.session_state.get("results", None)

    if res is None:
        st.info("ℹ️ Lancez d'abord les prévisions.")
    else:
        target_series = res["target_series"]
        target_col    = res["target_col"]


        # ── Metrics tableau ────────────────────────────────────────────────
        st.markdown('<div class="section-title">Métriques Comparatives</div>',
                    unsafe_allow_html=True)
        metrics_rows = []
        for name, m in [
            ("VECM", res["vecm_metrics"]),
            ("Random Forest", res["rf_metrics"]),
            ("XGBoost",  res["xgb_metrics"])
        ]:

            if m:
                metrics_rows.append({
                    "Modèle": name,
                    "MAE": round(m["MAE"], 5),
                    "RMSE": round(m["RMSE"], 5)
                })
        if metrics_rows:
            m_df = pd.DataFrame(metrics_rows).set_index("Modèle")
            st.dataframe(
                m_df.style.highlight_min(
                    ["MAE", "RMSE"],
                    color="#C8F5DA"
                ),
                use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – AIDE À LA DÉCISION
# ══════════════════════════════════════════════════════════════════════════════
with tab_diagnostic:

    res = st.session_state.get("results", None)

    if res is None:
        st.info("ℹ️ Lancez d'abord les prévisions.")
    else:

        target_series = res["target_series"]
        target_col = res["target_col"]
        horizon_val = res["horizon"]

        # ─────────────────────────────────────
        # Diagnostic
        # ─────────────────────────────────────
        forecasts = []

        if res["vecm_fc"] is not None:
            forecasts.extend(res["vecm_fc"])

        if res["rf_fc"] is not None:
            forecasts.extend(res["rf_fc"])

        if res["xgb_fc"] is not None:
            forecasts.extend(res["xgb_fc"])

        mean_forecast = np.mean(forecasts)
        last_value = target_series[-1]

        variation = (
            (mean_forecast - last_value)
            / (abs(last_value) + 1e-9)
        ) * 100


        if variation >= 5:
            signal = " Dynamique favorable"
            color = "success"

            diagnostic = (
                "Les prévisions indiquent une progression soutenue de "
                "l'investissement privé sur l'horizon considéré. "
                "Cette évolution traduit un environnement relativement favorable "
                "à l'accumulation du capital et devrait contribuer positivement "
                "à l'activité économique."
            )

            orientations = [
                "Maintenir les dispositifs de soutien à l'investissement.",
                "Poursuivre le suivi des déterminants de l'investissement privé.",
                "Consolider les politiques favorables au climat des affaires."
            ]

        elif variation <= -5:
            signal = " Ralentissement de l'investissement"
            color = "danger"

            diagnostic = (
                "Les modèles mettent en évidence une contraction de "
                "l'investissement privé sur l'horizon de prévision. "
                "Cette évolution pourrait refléter une détérioration des "
                "conditions économiques et nécessiter une surveillance accrue."
            )

            orientations = [
                "Renforcer le suivi des indicateurs conjoncturels.",
                "Identifier les facteurs susceptibles de freiner l'investissement privé.",
                "Évaluer l'opportunité de mesures d'accompagnement ou d'incitation."
            ]

        else:
            signal = " Évolution relativement stable"
            color = "gold"

            diagnostic = (
                "Les résultats des modèles suggèrent une évolution relativement "
                "stable de l'investissement privé sur l'horizon étudié. "
                "Aucun changement majeur de tendance n'est observé à court terme."
            )

            orientations = [
                "Poursuivre la veille conjoncturelle.",
                "Actualiser régulièrement les prévisions.",
                "Surveiller les indicateurs avancés de l'investissement."
            ]


        # ─────────────────────────────────────
        # Diagnostic
        # ─────────────────────────────────────
        st.markdown(
            '<div class="section-title">Diagnostic des Résultats</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-label">Signal</div>
            <div class="kpi-value {color}">
                {signal}
            </div>
            <div class="kpi-sub">
                Variation prévue : {variation:.2f} %
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="panel">
            <b>Analyse économique :</b><br><br>
            {diagnostic}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ─────────────────────────────────────
        # Orientations
        # ─────────────────────────────────────
        st.markdown(
            '<div class="section-title">Orientations à l\'attention des analystes et décideurs publics</div>',
            unsafe_allow_html=True
        )

        for item in orientations:
            st.markdown(f"- {item}")


        # ─────────────────────────────────────
        # Synthèse
        # ─────────────────────────────────────
        st.markdown(
            '<div class="section-title">Synthèse</div>',
            unsafe_allow_html=True
        )

        synthese = pd.DataFrame({
            "Indicateur": [
                "Variable étudiée",
                "Horizon de prévision",
                "Dernière observation",
                "Prévision moyenne",
                "Variation attendue"
            ],
            "Valeur": [
                target_col,
                f"{horizon_val} trimestres",
                round(last_value, 3),
                round(mean_forecast, 3),
                f"{variation:.2f} %"
            ]
        })

        st.dataframe(
            synthese,
            use_container_width=True,
            hide_index=True
        )

        # ── Export ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Exporter les résultats</div>',
                    unsafe_allow_html=True)
        export_rows = []
        dates_prevision_export = pd.date_range(
            start=date_debut,
            periods=horizon_val,
            freq="QE"
        )
        for t in range(horizon_val):
            row = {"Date": dates_prevision_export[t]}
            if res["vecm_fc"] is not None:
                row["VECM"] = round(res["vecm_fc"][t], 4)
            if res["rf_fc"] is not None:
                row["RF"]   = round(res["rf_fc"][t], 4)
            if res["xgb_fc"] is not None:
                row["XGB"]  = round(res["xgb_fc"][t], 4)
            export_rows.append(row)
        export_df = pd.DataFrame(export_rows)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Télécharger les prévisions (.csv)",
            csv_bytes,
            file_name=f"previsions_hcp_{target_col}_{horizon_val}T.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hcp-footer">
    Haut-Commissariat au Plan &nbsp;|&nbsp; Projet de Fin d'année 2025–2026 &nbsp;|&nbsp;
    VECM · Random Forest · XGBoost &nbsp;|&nbsp
</div>
""", unsafe_allow_html=True)
