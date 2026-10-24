"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   QUANTITATIVE FINANCE — EUROPEAN OPTION PRICING DASHBOARD                  ║
║   Black-Scholes · Monte Carlo · Euler-Maruyama · Machine Learning           ║
║   run :  streamlit run pricing.py                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    pip install streamlit numpy scipy scikit-learn plotly pandas
"""

import math, time, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── page config (PREMIER appel Streamlit) ────────────────────────────────────
st.set_page_config(
    page_title="Quant Finance · Option Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COULEURS  —  toutes en format rgba() pour Plotly, hex 6 chiffres pour CSS
# ─────────────────────────────────────────────────────────────────────────────
CB  = "#3b82f6"   # blue
CC  = "#22d3ee"   # cyan
CG  = "#10b981"   # green
CA  = "#f59e0b"   # amber
CR  = "#ef4444"   # red
CV  = "#8b5cf6"   # violet
CS  = "#334155"   # slate
CT  = "#2dd4bf"   # teal

# Couleurs semi-transparentes RGBA pour fillcolor Plotly
def rgba(hex6: str, alpha: float = 0.15) -> str:
    """Convertit #rrggbb en 'rgba(r,g,b,alpha)' compatible Plotly."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html,body,[class*="css"],.stApp{
  font-family:'IBM Plex Sans',sans-serif;
  background:#060b12!important;color:#d4e2f0;
}
section[data-testid="stSidebar"]{
  background:#07101a!important;border-right:1px solid #162333;
}
section[data-testid="stSidebar"]>div{padding-top:.4rem;}
.main .block-container{
  background:#060b12;padding:1.2rem 1.8rem 2.5rem;max-width:100%!important;
}
.sb-sep{
  font-family:'IBM Plex Mono',monospace;font-size:.57rem;letter-spacing:.18em;
  text-transform:uppercase;color:#1b6ea8;padding:.75rem 0 .22rem;
  border-bottom:1px solid #162333;margin-bottom:.45rem;
}
.page-hdr{
  background:linear-gradient(118deg,#071624 0%,#0b1e30 55%,#060e1c 100%);
  border:1px solid #162333;border-top:2px solid #1b6ea8;
  border-radius:8px;padding:1.3rem 1.9rem 1.1rem;margin-bottom:1.3rem;
}
.page-hdr h1{
  font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;
  color:#eaf3ff;margin:0 0 .15rem;
}
.page-hdr .sub{font-size:.76rem;color:#3d7aa5;font-family:'IBM Plex Mono',monospace;}
.pill{
  display:inline-block;background:rgba(27,110,168,.1);
  border:1px solid rgba(27,110,168,.32);color:#4fa0d0;
  font-family:'IBM Plex Mono',monospace;font-size:.53rem;
  padding:2px 8px;border-radius:3px;margin:.5rem 3px 0 0;
  letter-spacing:.09em;text-transform:uppercase;
}
.kpi-wrap{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:.9rem;}
.kpi{
  flex:1;min-width:120px;background:#08111c;border:1px solid #162333;
  border-radius:6px;padding:.85rem 1rem;position:relative;overflow:hidden;
}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.kpi-b::before{background:#2563eb;}.kpi-c::before{background:#0891b2;}
.kpi-g::before{background:#059669;}.kpi-a::before{background:#d97706;}
.kpi-r::before{background:#dc2626;}.kpi-v::before{background:#7c3aed;}
.kpi-s::before{background:#475569;}.kpi-t::before{background:#0d9488;}
.kpi-lbl{
  font-family:'IBM Plex Mono',monospace;font-size:.53rem;letter-spacing:.13em;
  text-transform:uppercase;color:#2a3d4f;margin-bottom:.3rem;
}
.kpi-val{font-family:'IBM Plex Mono',monospace;font-size:1.42rem;font-weight:700;line-height:1;}
.kpi-sub{font-size:.62rem;color:#2a3d4f;margin-top:.18rem;}
.kpi-b .kpi-val{color:#3b82f6;}.kpi-c .kpi-val{color:#22d3ee;}
.kpi-g .kpi-val{color:#10b981;}.kpi-a .kpi-val{color:#f59e0b;}
.kpi-r .kpi-val{color:#ef4444;}.kpi-v .kpi-val{color:#8b5cf6;}
.kpi-s .kpi-val{color:#94a3b8;}.kpi-t .kpi-val{color:#2dd4bf;}
.stitle{
  font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.14em;
  text-transform:uppercase;color:#2563eb;border-left:3px solid #2563eb;
  padding-left:.7rem;margin:1.3rem 0 .7rem;
}
.fbox{
  background:#07101a;border:1px solid #162333;border-left:3px solid #1b6ea8;
  border-radius:6px;padding:.9rem 1.1rem;font-family:'IBM Plex Mono',monospace;
  font-size:.76rem;color:#60a5fa;line-height:2;margin:.5rem 0 .8rem;
}
.fbox .ftitle{
  font-size:.56rem;letter-spacing:.12em;text-transform:uppercase;
  color:#2a3d4f;margin-bottom:.4rem;
}
.fhi{color:#38bdf8;font-weight:600;}.fdim{color:#1d4e6a;}
.ibox{
  background:rgba(37,99,235,.05);border:1px solid rgba(37,99,235,.18);
  border-radius:6px;padding:.65rem .95rem;font-size:.74rem;color:#6a9ec0;margin:.35rem 0;
}
.ctbl{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:.76rem;}
.ctbl th{
  background:#07101a;color:#2a3d4f;font-size:.53rem;letter-spacing:.1em;
  text-transform:uppercase;padding:.6rem .85rem;text-align:left;
  border-bottom:1px solid #162333;
}
.ctbl td{padding:.55rem .85rem;border-bottom:1px solid #0c1825;color:#8fb8d8;}
.ctbl tr:hover td{background:#07101a;}
.tg{color:#10b981;font-weight:700;}.ta{color:#f59e0b;font-weight:700;}
.tr{color:#ef4444;font-weight:700;}.tb{color:#3b82f6;font-weight:700;}
.tv{color:#8b5cf6;font-weight:700;}.tc{color:#22d3ee;font-weight:700;}
.gtbl{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:.75rem;}
.gtbl td{padding:.46rem .75rem;border-bottom:1px solid #0c1825;}
.gtbl tr:hover td{background:#07101a;}
.cc-grid{display:grid;grid-template-columns:1fr 1fr;gap:.85rem;margin-top:.5rem;}
.cc{background:#07101a;border:1px solid #162333;border-radius:8px;padding:.9rem 1.1rem;}
.cc-t{
  font-family:'IBM Plex Mono',monospace;font-size:.59rem;letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:.55rem;
}
.cc-b{font-size:.75rem;color:#4a7a95;line-height:1.72;}
.cc-b em{color:#2a3d4f;font-style:normal;display:block;margin-top:.35rem;}
div[data-baseweb="tab-list"]{
  background:#07101a!important;border-bottom:1px solid #162333!important;
}
button[data-baseweb="tab"]{
  font-family:'IBM Plex Mono',monospace!important;font-size:.62rem!important;
  letter-spacing:.06em!important;color:#2a3d4f!important;
  padding:.6rem 1.1rem!important;background:transparent!important;border-radius:0!important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color:#3b82f6!important;border-bottom:2px solid #2563eb!important;
}
hr{border-color:#162333;}
.stButton>button{
  font-family:'IBM Plex Mono',monospace;font-size:.62rem;letter-spacing:.06em;
  background:#0c1f31;border:1px solid #1b4d70;color:#4fa0d0;
  border-radius:4px;padding:.42rem .95rem;transition:all .2s;
}
.stButton>button:hover{background:#1b4d70;color:#eaf3ff;}
input[type="number"]{
  background:#07101a!important;border-color:#162333!important;
  color:#d4e2f0!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:.78rem!important;
}
div[data-baseweb="select"]>div{
  background:#07101a!important;border-color:#162333!important;
  font-family:'IBM Plex Mono',monospace!important;font-size:.76rem!important;
}
.stSlider>div>div>div>div{background:#2563eb!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PLOTLY DE BASE
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#07101a",
    plot_bgcolor="#060b12",
    font=dict(family="IBM Plex Mono,monospace", size=10, color="#4a6f88"),
    xaxis=dict(gridcolor="#0c1825", zerolinecolor="#0c1825", linecolor="#162333"),
    yaxis=dict(gridcolor="#0c1825", zerolinecolor="#0c1825", linecolor="#162333"),
    margin=dict(l=52, r=28, t=46, b=42),
    legend=dict(bgcolor="rgba(7,16,26,0.95)", bordercolor="#162333",
                borderwidth=1, font=dict(size=9)),
)

def fup(fig, title="", h=400):
    fig.update_layout(
        **PL, height=h,
        title=dict(text=title,
                   font=dict(family="IBM Plex Mono", size=12, color="#8fb8d8"),
                   x=0.01, xanchor="left", pad=dict(l=0, t=4)))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────
def kpi_row(items):
    html = '<div class="kpi-wrap">'
    for lbl, val, sub, c in items:
        html += (f'<div class="kpi kpi-{c}">'
                 f'<div class="kpi-lbl">{lbl}</div>'
                 f'<div class="kpi-val">{val}</div>'
                 f'<div class="kpi-sub">{sub}</div></div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def sec(t):
    st.markdown(f'<div class="stitle">{t}</div>', unsafe_allow_html=True)

def ibox(t):
    st.markdown(f'<div class="ibox">ℹ &nbsp;{t}</div>', unsafe_allow_html=True)

def fbox(title, lines):
    body = "<br>".join(lines)
    st.markdown(f'<div class="fbox"><div class="ftitle">{title}</div>{body}</div>',
                unsafe_allow_html=True)

def gtable(rows):
    """rows = list of (label, value, comment)"""
    tbl = "".join(
        f"<tr>"
        f"<td style='color:#3b82f6;font-weight:600;'>{r[0]}</td>"
        f"<td style='color:#e2e8f0;'>{r[1]}</td>"
        f"<td style='color:#2a3d4f;font-size:.63rem;'>{r[2]}</td>"
        f"</tr>"
        for r in rows)
    st.markdown(f"<table class='gtbl'>{tbl}</table>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════  MODÈLES FINANCIERS  ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def bs(S, K, T, r, sig, opt):
    S, K, T, r, sig = float(S), float(K), float(T), float(r), float(sig)
    if T < 1e-9:
        return max(0.0, S - K) if opt == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    if opt == "call":
        return float(S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2))
    return float(K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1))

def greeks(S, K, T, r, sig, opt):
    S, K, T, r, sig = float(S), float(K), float(T), float(r), float(sig)
    if T < 1e-6:
        return dict(delta=0., gamma=0., theta=0., vega=0., rho=0.)
    d1 = (math.log(S/K) + (r+0.5*sig**2)*T) / (sig*math.sqrt(T))
    d2 = d1 - sig*math.sqrt(T)
    phi = float(norm.pdf(d1))
    gam = phi / (S*sig*math.sqrt(T))
    veg = S*phi*math.sqrt(T) / 100.0
    if opt == "call":
        de  = float(norm.cdf(d1))
        th  = (-(S*phi*sig)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)) / 365.0
        rh  = K*T*math.exp(-r*T)*norm.cdf(d2) / 100.0
    else:
        de  = float(norm.cdf(d1)) - 1.0
        th  = (-(S*phi*sig)/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2)) / 365.0
        rh  = -K*T*math.exp(-r*T)*norm.cdf(-d2) / 100.0
    return dict(delta=de, gamma=gam, theta=th, vega=veg, rho=rh)

def implied_vol(S, K, T, r, mkt, opt):
    S, K, T, r, mkt = float(S), float(K), float(T), float(r), float(mkt)
    sig = 0.30
    for _ in range(200):
        p  = bs(S, K, T, r, sig, opt)
        d1 = (math.log(S/K) + (r+0.5*sig**2)*T) / (sig*math.sqrt(T))
        vg = S * float(norm.pdf(d1)) * math.sqrt(T)
        if abs(vg) < 1e-12: break
        sig -= (p - mkt) / vg
        sig  = max(1e-4, min(sig, 10.0))
        if abs(p - mkt) < 1e-6: break
    return float(sig)

def monte_carlo(S, K, T, r, sig, opt, N, seed=42):
    S, K, T, r, sig = float(S), float(K), float(T), float(r), float(sig)
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(int(N))
    ST  = S * np.exp((r - 0.5*sig**2)*T + sig*math.sqrt(T)*Z)
    pf  = np.maximum(ST-K, 0.) if opt=="call" else np.maximum(K-ST, 0.)
    d   = math.exp(-r*T)
    return float(d*np.mean(pf)), float(d*np.std(pf)/math.sqrt(N)), ST, pf

def mc_conv(S, K, T, r, sig, opt, maxN, npts=80, seed=42):
    S, K, T, r, sig = float(S), float(K), float(T), float(r), float(sig)
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(int(maxN))
    ST  = S * np.exp((r - 0.5*sig**2)*T + sig*math.sqrt(T)*Z)
    pf  = np.maximum(ST-K, 0.) if opt=="call" else np.maximum(K-ST, 0.)
    d   = math.exp(-r*T)
    ns  = np.unique(np.linspace(100, maxN, npts).astype(int))
    prices  = np.array([d*np.mean(pf[:n]) for n in ns])
    stderrs = np.array([d*np.std(pf[:n])/math.sqrt(n) for n in ns])
    return ns, prices, stderrs

def euler(S, K, T, r, sig, Ns, Np, opt, seed=42):
    S, K, T, r, sig = float(S), float(K), float(T), float(r), float(sig)
    rng  = np.random.default_rng(seed)
    dt   = T / int(Ns)
    paths = np.empty((int(Np), int(Ns)+1))
    paths[:, 0] = S
    for t in range(int(Ns)):
        Z = rng.standard_normal(int(Np))
        paths[:, t+1] = paths[:, t] * np.exp(
            (r - 0.5*sig**2)*dt + sig*math.sqrt(dt)*Z)
    pf = (np.maximum(paths[:,-1]-K, 0.) if opt=="call"
          else np.maximum(K-paths[:,-1], 0.))
    d  = math.exp(-r*T)
    return float(d*np.mean(pf)), float(d*np.std(pf)/math.sqrt(Np)), paths

def dt_study(S, K, T, r, sig, opt, Np=3000):
    ref   = bs(S, K, T, r, sig, opt)
    steps = [5, 10, 20, 50, 100, 252, 500, 1000]
    rows  = []
    for N in steps:
        p, se, _ = euler(S, K, T, r, sig, N, Np, opt)
        rows.append({"N_steps": N, "Δt": round(T/N, 5),
                     "Prix Euler": round(p, 5), "Std Error": round(se, 6),
                     "|Err vs BS|": round(abs(p-ref), 6),
                     "Err (%)": round(abs(p-ref)/ref*100, 4) if ref > 0 else 0.})
    return pd.DataFrame(rows), ref

# ─── ML ──────────────────────────────────────────────────────────────────────

def gen_dataset(n, seed=42):
    rng = np.random.default_rng(seed)
    S_  = rng.uniform(40.,  200., n)
    K_  = rng.uniform(40.,  200., n)
    T_  = rng.uniform(0.05,  3.0, n)
    r_  = rng.uniform(0.01, 0.12, n)
    sg_ = rng.uniform(0.08, 0.70, n)
    ot_ = rng.integers(0, 2, n)
    mo_ = S_ / K_
    y   = np.array([
        bs(float(S_[i]), float(K_[i]), float(T_[i]),
           float(r_[i]), float(sg_[i]),
           "call" if ot_[i] == 0 else "put")
        for i in range(n)], dtype=np.float64)
    X = np.column_stack([S_, K_, T_, r_, sg_, mo_,
                         ot_.astype(np.float64)]).astype(np.float64)
    return X, y

def train_models(n_samples):
    """
    Entraîne MLP, GBR, RF.
    JAMAIS décoré avec @st.cache_data → objets sklearn stockés dans session_state.
    """
    X, y = gen_dataset(n_samples)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)   # float64 (n,7)
    Xte_s = sc.transform(Xte)

    configs = {
        "MLP Neural Net": (
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                         max_iter=600, random_state=42, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=20), True),
        "Gradient Boosting": (
            GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                      learning_rate=0.05, random_state=42), False),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, max_depth=12,
                                  random_state=42, n_jobs=-1), False),
    }
    out = {}
    for name, (mdl, scaled) in configs.items():
        Xf = Xtr_s if scaled else Xtr
        Xe = Xte_s if scaled else Xte
        t0 = time.perf_counter(); mdl.fit(Xf, ytr); tr_t = time.perf_counter()-t0
        t0 = time.perf_counter(); pred = mdl.predict(Xe)
        inf_t = (time.perf_counter()-t0)/len(Xe)*1e6
        rmse  = float(math.sqrt(mean_squared_error(yte, pred)))
        out[name] = dict(
            model=mdl, scaler=sc if scaled else None, scaled=scaled,
            rmse=rmse, mae=float(mean_absolute_error(yte, pred)),
            r2=float(r2_score(yte, pred)),
            mape=float(np.mean(np.abs((yte-pred)/(yte+1e-8)))*100),
            tr_t=tr_t, inf_t=inf_t, y_test=yte, y_pred=pred)
    return out

def ml_pred1(models, S, K, T, r, sig, opt):
    """Prédiction scalaire — feat toujours float64 shape(1,7)."""
    feat = np.array([[float(S), float(K), float(T), float(r), float(sig),
                      float(S)/float(K),
                      0. if opt=="call" else 1.]], dtype=np.float64)
    return {nm: float(res["model"].predict(
                res["scaler"].transform(feat) if (res["scaled"] and res["scaler"] is not None)
                else feat)[0])
            for nm, res in models.items()}

def ml_pred_range(models, spots, K, T, r, sig, opt):
    """Prédiction vectorisée — feat float64 shape(n,7)."""
    n    = len(spots)
    feat = np.column_stack([
        spots.astype(np.float64),
        np.full(n, float(K)),  np.full(n, float(T)),
        np.full(n, float(r)),  np.full(n, float(sig)),
        spots.astype(np.float64)/float(K),
        np.full(n, 0. if opt=="call" else 1.)
    ]).astype(np.float64)
    return {nm: res["model"].predict(
                res["scaler"].transform(feat) if (res["scaled"] and res["scaler"] is not None)
                else feat).astype(float)
            for nm, res in models.items()}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — ML
# ─────────────────────────────────────────────────────────────────────────────
if "ml_models" not in st.session_state:
    st.session_state["ml_models"]    = None
    st.session_state["ml_n_samples"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  SIDEBAR  ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 .5rem;'>
      <div style='font-family:IBM Plex Mono,monospace;font-size:.9rem;
                  font-weight:700;color:#eaf3ff;'>📊 QuantPricing</div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:.55rem;
                  color:#1b6ea8;letter-spacing:.1em;margin-top:2px;'>
        EUROPEAN OPTIONS </div>
    </div>
    <hr style='margin:0 0 .5rem;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sep">Paramètres du Modèle</div>', unsafe_allow_html=True)
    S     = st.number_input("S₀  Prix spot",        min_value=1.,   max_value=5000., value=100.,  step=1.,   format="%.2f")
    K     = st.number_input("K   Strike",            min_value=1.,   max_value=5000., value=100.,  step=1.,   format="%.2f")
    T     = st.number_input("T   Maturité (ans)",    min_value=0.01, max_value=10.,   value=1.0,   step=0.05, format="%.3f")
    r     = st.number_input("r   Taux sans risque",  min_value=0.0,  max_value=0.30,  value=0.05,  step=0.005,format="%.4f")
    sigma = st.number_input("σ   Volatilité",        min_value=0.01, max_value=2.0,   value=0.20,  step=0.01, format="%.4f")
    opt   = st.selectbox("Type d'option", ["call", "put"])

    st.markdown('<div class="sb-sep">Simulation Numérique</div>', unsafe_allow_html=True)
    N_paths = st.select_slider("N simulations",
                                options=[1000,5000,10000,25000,50000,100000], value=10000)
    N_steps = st.select_slider("N pas de temps (Euler)",
                                options=[10,20,50,100,252,500,1000], value=252)

    st.markdown('<div class="sb-sep">Machine Learning</div>', unsafe_allow_html=True)
    N_samples = st.select_slider("Dataset ML",
                                  options=[1000,2000,5000,10000,20000], value=5000)
    train_btn = st.button("⚙  Entraîner / Ré-entraîner les modèles ML")

    # ── Infos rapides ──
    Sf   = float(S);  Kf  = float(K)
    Tf   = float(T);  rf  = float(r);  sigf = float(sigma)
    mono = Sf / Kf
    itm_l = ("ITM ✅" if ((opt=="call" and Sf>Kf) or (opt=="put" and Sf<Kf))
              else "ATM ⚖️" if abs(mono-1.)<0.02 else "OTM ❌")
    cp = bs(Sf, Kf, Tf, rf, sigf, "call")
    pp = bs(Sf, Kf, Tf, rf, sigf, "put")
    parity = cp - Sf + Kf*math.exp(-rf*Tf)

    st.markdown(f"""
    <div style='background:#07101a;border:1px solid #162333;border-radius:6px;
                padding:.7rem 1rem;font-family:IBM Plex Mono,monospace;
                font-size:.68rem;margin-top:.7rem;'>
      <div style='color:#2a3d4f;font-size:.53rem;letter-spacing:.12em;
                  text-transform:uppercase;margin-bottom:.45rem;'>Statut</div>
      <div style='color:#d4e2f0;margin-bottom:.2rem;'>{itm_l}</div>
      <div style='color:#3d7aa5;'>Moneyness : {mono:.4f}</div>
      <div style='color:#3d7aa5;'>S/K ratio  : {Sf/Kf:.4f}</div>
      <div style='color:#2a3d4f;margin-top:.4rem;font-size:.63rem;'>
        Put (parité) : ${parity:.4f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sep" style="margin-top:.8rem;">Volatilité Implicite</div>',
                unsafe_allow_html=True)
    mkt_def = round(bs(Sf, Kf, Tf, rf, sigf, opt), 4)
    iv_mkt  = st.number_input("Prix marché (IV)", min_value=0.0001, max_value=9999.,
                               value=mkt_def, step=0.01, format="%.4f")
    iv_val  = implied_vol(Sf, Kf, Tf, rf, float(iv_mkt), opt)
    st.markdown(f"""
    <div style='background:#07101a;border:1px solid #162333;border-radius:6px;
                padding:.6rem 1rem;font-family:IBM Plex Mono,monospace;
                font-size:.68rem;margin-top:.15rem;'>
      <div style='color:#2a3d4f;font-size:.53rem;letter-spacing:.12em;
                  text-transform:uppercase;margin-bottom:.25rem;'>Vol. Implicite (NR)</div>
      <div style='color:#22d3ee;font-size:1.05rem;font-weight:700;'>{iv_val*100:.3f} %</div>
      <div style='color:#2a3d4f;font-size:.62rem;'>Newton-Raphson &lt;200 itér.</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT ML
# ─────────────────────────────────────────────────────────────────────────────
if train_btn or st.session_state["ml_models"] is None:
    with st.spinner(f"Entraînement ML sur {N_samples:,} échantillons…"):
        st.session_state["ml_models"]    = train_models(int(N_samples))
        st.session_state["ml_n_samples"] = int(N_samples)

ML: dict = st.session_state["ml_models"]

# ─────────────────────────────────────────────────────────────────────────────
# PRÉ-CALCULS
# ─────────────────────────────────────────────────────────────────────────────
PBS = bs(Sf, Kf, Tf, rf, sigf, opt)
GR  = greeks(Sf, Kf, Tf, rf, sigf, opt)
d1v = (math.log(Sf/Kf) + (rf+0.5*sigf**2)*Tf) / (sigf*math.sqrt(Tf))
d2v = d1v - sigf*math.sqrt(Tf)
ML_CUR = ml_pred1(ML, Sf, Kf, Tf, rf, sigf, opt)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hdr">
  <h1>Quantitative Finance · Option Pricing Dashboard</h1>
  <p class="sub">Valorisation d'options européennes — Analytique · Numérique · Machine Learning</p>
  <div style='margin-top:.6rem;'>
    <span class="pill">Black-Scholes 1973</span>
    <span class="pill">Monte Carlo</span>
    <span class="pill">Euler-Maruyama EDS</span>
    <span class="pill">MLP · GBR · RF</span>
    <span class="pill">Greeks &amp; Sensibilités</span>
    <span class="pill">Vol. Implicite NR</span>
    <span class="pill">Surface 3D</span>
  </div>
</div>""", unsafe_allow_html=True)

kpi_row([
    ("Prix BS",  f"${PBS:.4f}",        f"{opt.upper()} · exact",   "b"),
    ("Delta Δ",  f"{GR['delta']:.5f}", "∂V/∂S",                    "c"),
    ("Gamma Γ",  f"{GR['gamma']:.6f}", "∂²V/∂S²",                  "g"),
    ("Vega ν",   f"{GR['vega']:.5f}",  "∂V/∂σ / 1%",               "a"),
    ("Theta Θ",  f"{GR['theta']:.5f}", "∂V/∂t / jour",             "r"),
    ("Rho ρ",    f"{GR['rho']:.5f}",   "∂V/∂r / 1%",               "v"),
    ("IV",       f"{iv_val*100:.2f}%", "Vol. implicite NR",         "t"),
])

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📐  Black-Scholes & Greeks  ",
    "  🎲  Monte Carlo  ",
    "  🔀  Euler-Maruyama  ",
    "  🤖  Machine Learning  ",
    "  ⚖️  Comparaison Globale  ",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — BLACK-SCHOLES
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    sec("Modèle Black-Scholes (1973) — Cadre Théorique")
    fbox("Équation Différentielle Stochastique & Formule Analytique", [
        '<span class="fhi">dS = μS dt + σS dW_t</span>'
        '  <span class="fdim">(GBM — Mouvement Brownien Géométrique)</span>',
        '<span class="fhi">C = S·N(d₁) − K·e^{−rT}·N(d₂)</span>'
        '  <span class="fdim">(Call Européen)</span>',
        '<span class="fhi">P = K·e^{−rT}·N(−d₂) − S·N(−d₁)</span>'
        '  <span class="fdim">(Put Européen)</span>',
        '<span class="fdim">d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)</span>',
        '<span class="fdim">d₂ = d₁ − σ√T</span>',
        f'<span class="fdim">→ d₁={d1v:.5f} · d₂={d2v:.5f}'
        f' · N(d₁)={norm.cdf(d1v):.5f} · N(d₂)={norm.cdf(d2v):.5f}</span>',
    ])

    cL, cR = st.columns([3, 1])
    with cR:
        sec("Greeks — Tableau Complet")
        gtable([
            ("Δ Delta", f"{GR['delta']:.6f}",  "∂V/∂S"),
            ("Γ Gamma", f"{GR['gamma']:.7f}",  "∂²V/∂S²"),
            ("Θ Theta", f"{GR['theta']:.6f}",  "∂V/∂t /jour"),
            ("ν Vega",  f"{GR['vega']:.6f}",   "∂V/∂σ /1%"),
            ("ρ Rho",   f"{GR['rho']:.6f}",    "∂V/∂r /1%"),
            ("d₁",      f"{d1v:.5f}",            "—"),
            ("d₂",      f"{d2v:.5f}",            "—"),
            ("N(d₁)",   f"{norm.cdf(d1v):.5f}", "Prob. log-normale"),
            ("N(d₂)",   f"{norm.cdf(d2v):.5f}", "Prob. risque-neutre"),
        ])
        sec("Parité Put-Call")
        st.markdown(f"""
        <div class='fbox' style='font-size:.71rem;'>
          <span class='fhi'>C − P = S − K·e^{{−rT}}</span><br>
          <span class='fdim'>Call  : ${cp:.5f}</span><br>
          <span class='fdim'>Put   : ${pp:.5f}</span><br>
          <span class='fdim'>C−P   : ${cp-pp:.5f}</span><br>
          <span class='fdim'>S−Ke⁻ʳᵀ: ${Sf-Kf*math.exp(-rf*Tf):.5f}</span>
        </div>""", unsafe_allow_html=True)

    with cL:
        sr   = np.linspace(max(1., Sf*0.4), Sf*1.6, 400)
        Tv   = sorted({Tf, Tf*0.5, max(0.05, Tf*0.25)})
        Tv   = [tv for tv in Tv if tv > 0.01]
        fig1 = go.Figure()
        for tv, col in zip(Tv, [CB, CC, CG]):
            yv = [bs(float(s), Kf, tv, rf, sigf, opt) for s in sr]
            fig1.add_trace(go.Scatter(x=sr.tolist(), y=yv,
                                       name=f"T={tv:.2f}a",
                                       line=dict(color=col, width=2)))
        pay = [max(0., s-Kf) if opt=="call" else max(0., Kf-s) for s in sr]
        fig1.add_trace(go.Scatter(x=sr.tolist(), y=pay, name="Payoff (T→0)",
                                   line=dict(color=CS, width=1.2, dash="dot")))
        fig1.add_vline(x=Sf, line=dict(color=CA, width=1.2, dash="dot"),
                       annotation_text=f"S={Sf}", annotation_font_color=CA)
        fig1.add_vline(x=Kf, line=dict(color=CR, width=1.2, dash="dot"),
                       annotation_text=f"K={Kf}", annotation_font_color=CR)
        fig1 = fup(fig1, "Prix de l'option vs. Spot — différentes maturités", 350)
        fig1.update_xaxes(title_text="Spot S")
        fig1.update_yaxes(title_text="Prix ($)")
        st.plotly_chart(fig1, use_container_width=True)

    sec("Greeks vs. Spot")
    gcols = st.columns(3)
    sr2   = sr[::2]
    greek_defs = [
        ("Delta Δ", lambda s: greeks(float(s),Kf,Tf,rf,sigf,opt)["delta"], CB, "Δ"),
        ("Gamma Γ", lambda s: greeks(float(s),Kf,Tf,rf,sigf,opt)["gamma"], CG, "Γ"),
        ("Theta Θ", lambda s: greeks(float(s),Kf,Tf,rf,sigf,opt)["theta"], CR, "Θ/j"),
        ("Vega ν",  lambda s: greeks(float(s),Kf,Tf,rf,sigf,opt)["vega"],  CA, "ν"),
        ("Rho ρ",   lambda s: greeks(float(s),Kf,Tf,rf,sigf,opt)["rho"],   CV, "ρ"),
        ("Payoff",  lambda s: max(0.,s-Kf) if opt=="call" else max(0.,Kf-s), CS, "$"),
    ]
    for (gname, gfn, gcol, gyl), col in zip(greek_defs, gcols*2):
        with col:
            yv = [gfn(s) for s in sr2]
            fg = go.Figure(go.Scatter(
                x=sr2.tolist(), y=yv,
                line=dict(color=gcol, width=2),
                fill="tozeroy",
                fillcolor=rgba(gcol, 0.08)))   # ← rgba() au lieu de hex+"18"
            fg.add_vline(x=Sf, line=dict(color=CA, width=1, dash="dot"))
            fg = fup(fg, gname, 215)
            fg.update_xaxes(title_text="Spot", tickfont=dict(size=8))
            fg.update_yaxes(title_text=gyl,    tickfont=dict(size=8))
            st.plotly_chart(fg, use_container_width=True)

    sec("Sensibilité à σ et r")
    cA, cB = st.columns(2)
    with cA:
        sigs = np.linspace(0.01, 0.90, 200)
        pvc  = [bs(Sf,Kf,Tf,rf,float(s),"call") for s in sigs]
        pvp  = [bs(Sf,Kf,Tf,rf,float(s),"put")  for s in sigs]
        fv = go.Figure()
        fv.add_trace(go.Scatter(x=sigs.tolist(), y=pvc, name="Call",
                                 line=dict(color=CB, width=2)))
        fv.add_trace(go.Scatter(x=sigs.tolist(), y=pvp, name="Put",
                                 line=dict(color=CR, width=2)))
        fv.add_vline(x=sigf, line=dict(color=CA, width=1.5, dash="dot"),
                     annotation_text=f"σ={sigf}", annotation_font_color=CA)
        fv = fup(fv, "Prix Call & Put vs. Volatilité σ", 310)
        fv.update_xaxes(title_text="σ"); fv.update_yaxes(title_text="Prix ($)")
        st.plotly_chart(fv, use_container_width=True)
    with cB:
        rates = np.linspace(0.001, 0.20, 200)
        prc   = [bs(Sf,Kf,Tf,float(ri),sigf,"call") for ri in rates]
        prp   = [bs(Sf,Kf,Tf,float(ri),sigf,"put")  for ri in rates]
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=rates.tolist(), y=prc, name="Call",
                                 line=dict(color=CB, width=2)))
        fr.add_trace(go.Scatter(x=rates.tolist(), y=prp, name="Put",
                                 line=dict(color=CR, width=2)))
        fr.add_vline(x=rf, line=dict(color=CA, width=1.5, dash="dot"),
                     annotation_text=f"r={rf}", annotation_font_color=CA)
        fr = fup(fr, "Prix Call & Put vs. Taux r", 310)
        fr.update_xaxes(title_text="r"); fr.update_yaxes(title_text="Prix ($)")
        st.plotly_chart(fr, use_container_width=True)

    sec("Surface de Prix 3D — (Spot, Volatilité)")
    S3  = np.linspace(max(1., Sf*0.5), Sf*1.5, 40)
    sg3 = np.linspace(0.05, 0.70, 40)
    Z3  = np.array([[bs(float(si),Kf,Tf,rf,float(sv),opt)
                     for sv in sg3] for si in S3])
    fig3d = go.Figure(go.Surface(
        x=S3.tolist(), y=sg3.tolist(), z=Z3.tolist(),
        colorscale=[[0,"#060b12"],[0.3,"#1b4d70"],[0.65,"#2563eb"],[1.0,"#22d3ee"]],
        showscale=True, opacity=0.92,
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor="#22d3ee", project_z=True))))
    fig3d.update_layout(
        **PL, height=500,
        title=dict(text=f"Surface de prix 3D — {opt.upper()} (K={Kf})",
                   font=dict(family="IBM Plex Mono", size=12, color="#8fb8d8")),
        scene=dict(
            xaxis=dict(title="Spot S",       gridcolor="#162333", backgroundcolor="#07101a"),
            yaxis=dict(title="Volatilité σ", gridcolor="#162333", backgroundcolor="#07101a"),
            zaxis=dict(title="Prix ($)",     gridcolor="#162333", backgroundcolor="#060b12"),
            bgcolor="#060b12"))
    st.plotly_chart(fig3d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    sec("Méthode Monte Carlo — Estimation par Simulation")
    fbox("Estimateur MC + Intervalle de Confiance 95%", [
        '<span class="fhi">Ĉ = e^{−rT} · (1/N) · Σᵢ payoff(Sᵢ_T)</span>',
        '<span class="fdim">S_T = S₀ · exp[(r − σ²/2)T + σ√T · Z],  Z ~ N(0,1)</span>',
        '<span class="fdim">Std Error = σ_payoff · e^{−rT} / √N  →  convergence O(1/√N)</span>',
        '<span class="fdim">IC 95% = Ĉ ± 1.96 · Std Error</span>',
    ])
    ibox("Variables antithétiques (Z et −Z) divisent la variance par 2 sans surcoût.")

    with st.spinner("Simulation Monte Carlo…"):
        mc_p, mc_se, ST_sim, pf_sim = monte_carlo(Sf,Kf,Tf,rf,sigf,opt,N_paths)
        ns_cv, prices_cv, se_cv     = mc_conv(Sf,Kf,Tf,rf,sigf,opt,N_paths)

    itm_r = float((ST_sim>Kf).mean() if opt=="call" else (ST_sim<Kf).mean())
    kpi_row([
        ("Prix MC",   f"${mc_p:.5f}",          f"N={N_paths:,}",                      "g"),
        ("Prix BS",   f"${PBS:.5f}",            "Exact analytique",                    "b"),
        ("Err. abs.", f"${abs(mc_p-PBS):.6f}", f"Rel: {abs(mc_p-PBS)/PBS*100:.4f}%",  "a"),
        ("Std Error", f"${mc_se:.6f}",          "= σ·e^{−rT}/√N",                      "v"),
        ("IC 95%",    f"±${1.96*mc_se:.5f}",   "Intervalle de confiance",             "c"),
        ("Prob ITM",  f"{itm_r*100:.2f}%",      "Fraction scénarios ITM",               "s"),
    ])

    cA, cB = st.columns(2)
    with cA:
        sec("Convergence du Prix vs. N")
        fc = go.Figure()
        fc.add_trace(go.Scatter(
            x=np.concatenate([ns_cv, ns_cv[::-1]]).tolist(),
            y=np.concatenate([prices_cv+1.96*se_cv,
                               (prices_cv-1.96*se_cv)[::-1]]).tolist(),
            fill="toself", fillcolor=rgba(CG, 0.07),
            line=dict(color="rgba(0,0,0,0)"), name="IC 95%"))
        fc.add_trace(go.Scatter(x=ns_cv.tolist(), y=prices_cv.tolist(),
                                 name="Prix MC", line=dict(color=CG, width=2)))
        fc.add_hline(y=PBS, line=dict(color=CB, width=1.8, dash="dash"),
                     annotation_text="BS exact", annotation_font_color=CB)
        fc = fup(fc, "Convergence Monte Carlo", 370)
        fc.update_xaxes(title_text="N simulations")
        fc.update_yaxes(title_text="Prix estimé ($)")
        st.plotly_chart(fc, use_container_width=True)

    with cB:
        sec("Erreur |MC − BS| — Échelle Log-Log")
        err_cv = np.abs(prices_cv - PBS)
        mask   = err_cv > 1e-10
        if mask.sum() > 2:
            ns_m  = ns_cv[mask].astype(float)
            err_m = err_cv[mask]
            ref_l = float(err_m[0])*math.sqrt(float(ns_m[0]))/np.sqrt(ns_m)
            fe = go.Figure()
            fe.add_trace(go.Scatter(x=ns_m.tolist(), y=err_m.tolist(),
                                     name="|MC−BS|", line=dict(color=CA, width=2)))
            fe.add_trace(go.Scatter(x=ns_m.tolist(), y=ref_l.tolist(),
                                     name="O(1/√N)",
                                     line=dict(color=CS, width=1.5, dash="dash")))
            fe = fup(fe, "Taux de convergence (log-log)", 370)
            fe.update_xaxes(title_text="N", type="log")
            fe.update_yaxes(title_text="|Erreur|", type="log")
            st.plotly_chart(fe, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        sec("Distribution de S_T — Simulée vs. Log-Normale Théorique")
        mu_ln  = math.log(Sf) + (rf-0.5*sigf**2)*Tf
        std_ln = sigf*math.sqrt(Tf)
        x_th   = np.linspace(float(ST_sim.min()), float(ST_sim.max()), 300)
        pdf_th = ((1./(x_th*std_ln*math.sqrt(2*math.pi)))
                  *np.exp(-0.5*((np.log(x_th)-mu_ln)/std_ln)**2))
        fd = go.Figure()
        fd.add_trace(go.Histogram(x=ST_sim.tolist(), nbinsx=80, name="MC simulé",
                                   marker=dict(color=CC, opacity=0.55),
                                   histnorm="probability density"))
        fd.add_trace(go.Scatter(x=x_th.tolist(), y=pdf_th.tolist(),
                                 name="Log-Normale (BS)", line=dict(color=CB, width=2)))
        fd.add_vline(x=Kf, line=dict(color=CR, width=1.5, dash="dot"),
                     annotation_text=f"K={Kf}", annotation_font_color=CR)
        EsT = Sf*math.exp(rf*Tf)
        fd.add_vline(x=EsT, line=dict(color=CA, width=1.2, dash="dot"),
                     annotation_text=f"E[S_T]={EsT:.1f}", annotation_font_color=CA)
        fd = fup(fd, "Distribution de S_T (densité)", 340)
        fd.update_xaxes(title_text="S_T"); fd.update_yaxes(title_text="Densité")
        st.plotly_chart(fd, use_container_width=True)

    with cD:
        sec("Distribution des Payoffs In-The-Money")
        itm_mask = pf_sim > 0
        pct      = float(itm_mask.mean()*100)
        fp = go.Figure()
        if itm_mask.any():
            fp.add_trace(go.Histogram(x=pf_sim[itm_mask].tolist(), nbinsx=70,
                                       name=f"Payoffs>0 ({pct:.1f}%)",
                                       marker=dict(color=CV, opacity=0.7)))
            fp.add_vline(x=float(np.mean(pf_sim[itm_mask])),
                         line=dict(color=CA, width=1.5, dash="dot"),
                         annotation_text="Moy. ITM", annotation_font_color=CA)
        fp = fup(fp, f"Payoffs ITM ({pct:.1f}% des scénarios)", 340)
        fp.update_xaxes(title_text="Payoff ($)")
        fp.update_yaxes(title_text="Fréquence")
        st.plotly_chart(fp, use_container_width=True)

    sec("Impact du Nombre de Simulations — Tableau")
    rows_mc = []
    for n in [500, 1000, 2000, 5000, 10000, 25000, 50000]:
        t0 = time.perf_counter()
        p_n, se_n, _, _ = monte_carlo(Sf,Kf,Tf,rf,sigf,opt,n)
        t_n = (time.perf_counter()-t0)*1000
        rows_mc.append({"N": n, "Prix MC": f"${p_n:.5f}", "Std Error": f"${se_n:.6f}",
                         "|Erreur vs BS|": f"${abs(p_n-PBS):.6f}",
                         "Erreur (%)": f"{abs(p_n-PBS)/PBS*100:.4f}%" if PBS>0 else "—",
                         "Temps (ms)": f"{t_n:.2f}"})
    st.dataframe(pd.DataFrame(rows_mc).set_index("N"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — EULER-MARUYAMA
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    sec("Schéma d'Euler-Maruyama — Discrétisation de l'EDS")
    fbox("Schéma Numérique", [
        '<span class="fhi">S_{t+Δt} = S_t · exp[(r − σ²/2)Δt + σ√Δt · Z_t]</span>',
        '<span class="fdim">Δt = T / N_steps  (schéma log-Euler, exact pour GBM)</span>',
        '<span class="fdim">Erreur de discrétisation ~ O(Δt) pour EDS générales</span>',
        f'<span class="fdim">→ N_steps={N_steps}, Δt={Tf/N_steps:.5f}, '
        f'total ops = {N_paths}×{N_steps} = {N_paths*N_steps:,}</span>',
    ])

    with st.spinner("Simulation Euler-Maruyama…"):
        eu_p, eu_se, paths = euler(Sf,Kf,Tf,rf,sigf,N_steps,N_paths,opt)
        df_ts, _           = dt_study(Sf,Kf,Tf,rf,sigf,opt)

    kpi_row([
        ("Prix Euler",  f"${eu_p:.5f}",         f"N_steps={N_steps}",                "v"),
        ("Prix BS",     f"${PBS:.5f}",           "Exact",                             "b"),
        ("Err. abs.",   f"${abs(eu_p-PBS):.6f}", f"{abs(eu_p-PBS)/PBS*100:.4f}%",     "a"),
        ("Std Error",   f"${eu_se:.6f}",         "Incertitude MC",                    "c"),
        ("Δt",          f"{Tf/N_steps:.5f}",     "T / N_steps",                       "g"),
        ("Ops totales", f"{N_paths*N_steps:,}",  "Calculs effectués",                 "s"),
    ])

    cA, cB = st.columns(2)
    with cA:
        sec("Trajectoires Simulées")
        n_show = min(100, paths.shape[0])
        tvec   = np.linspace(0, Tf, N_steps+1)
        fp2 = go.Figure()
        for i in range(n_show):
            fp2.add_trace(go.Scatter(x=tvec.tolist(), y=paths[i].tolist(),
                                      mode="lines",
                                      line=dict(color=CV, width=0.5),
                                      opacity=0.18, showlegend=False))
        mean_p = np.mean(paths[:n_show], axis=0)
        q5     = np.percentile(paths[:n_show],  5, axis=0)
        q95    = np.percentile(paths[:n_show], 95, axis=0)
        fp2.add_trace(go.Scatter(
            x=np.concatenate([tvec, tvec[::-1]]).tolist(),
            y=np.concatenate([q95, q5[::-1]]).tolist(),
            fill="toself", fillcolor=rgba(CV, 0.08),
            line=dict(color="rgba(0,0,0,0)"), name="Q5-Q95"))
        fp2.add_trace(go.Scatter(x=tvec.tolist(), y=mean_p.tolist(),
                                  name="Moyenne", line=dict(color=CC, width=2.5)))
        Est = Sf*np.exp(rf*tvec)
        fp2.add_trace(go.Scatter(x=tvec.tolist(), y=Est.tolist(),
                                  name="E[S_t]=S·e^{rt}",
                                  line=dict(color=CA, width=1.5, dash="dash")))
        fp2.add_hline(y=Kf, line=dict(color=CR, width=1.5, dash="dot"),
                      annotation_text=f"K={Kf}", annotation_font_color=CR)
        fp2 = fup(fp2, f"Trajectoires Euler-Maruyama (N_steps={N_steps})", 400)
        fp2.update_xaxes(title_text="t"); fp2.update_yaxes(title_text="S(t)")
        st.plotly_chart(fp2, use_container_width=True)

    with cB:
        sec("Convergence en Δt (Log-Log)")
        dt_v  = df_ts["Δt"].values.astype(float)
        err_v = df_ts["|Err vs BS|"].values.astype(float)
        fe2 = go.Figure()
        fe2.add_trace(go.Scatter(x=dt_v.tolist(), y=err_v.tolist(),
                                  mode="lines+markers", name="|Euler−BS|",
                                  line=dict(color=CA, width=2),
                                  marker=dict(color=CA, size=7)))
        fe2 = fup(fe2, "Erreur vs. Δt — plus Δt↓ → plus précis", 400)
        fe2.update_xaxes(title_text="Δt = T/N", type="log", autorange="reversed")
        fe2.update_yaxes(title_text="|Erreur| ($)", type="log")
        st.plotly_chart(fe2, use_container_width=True)

    sec("Impact du Pas de Temps — Tableau")
    st.dataframe(df_ts.set_index("N_steps"), use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        sec("S_T : Euler vs. Log-Normale Théorique")
        ST_eu = paths[:, -1].astype(float)
        mu2   = math.log(Sf) + (rf-0.5*sigf**2)*Tf
        std2  = sigf*math.sqrt(Tf)
        x2    = np.linspace(float(ST_eu.min()), float(ST_eu.max()), 300)
        pdf2  = ((1./(x2*std2*math.sqrt(2*math.pi)))
                 *np.exp(-0.5*((np.log(x2+1e-9)-mu2)/std2)**2))
        fd2 = go.Figure()
        fd2.add_trace(go.Histogram(x=ST_eu.tolist(), nbinsx=80, name="Euler S_T",
                                    marker=dict(color=CV, opacity=0.6),
                                    histnorm="probability density"))
        fd2.add_trace(go.Scatter(x=x2.tolist(), y=pdf2.tolist(),
                                  name="Log-Normale (BS)", line=dict(color=CB, width=2)))
        fd2.add_vline(x=Kf, line=dict(color=CR, width=1.5, dash="dot"),
                      annotation_text=f"K={Kf}", annotation_font_color=CR)
        fd2 = fup(fd2, "Distribution S_T : Euler vs. Théorique", 330)
        fd2.update_xaxes(title_text="S_T")
        st.plotly_chart(fd2, use_container_width=True)

    with cD:
        sec("Statistiques Descriptives de S_T")
        sts = {
            "Moyenne simulée":   float(np.mean(ST_eu)),
            "E[S_T] théorique":  Sf*math.exp(rf*Tf),
            "Médiane":           float(np.median(ST_eu)),
            "Écart-type":        float(np.std(ST_eu)),
            "Minimum":           float(ST_eu.min()),
            "Maximum":           float(ST_eu.max()),
            "Quantile 5%":       float(np.percentile(ST_eu, 5)),
            "Quantile 95%":      float(np.percentile(ST_eu, 95)),
            "Skewness":          float(((ST_eu-np.mean(ST_eu))**3).mean()/np.std(ST_eu)**3),
            "Kurtosis (excess)": float(((ST_eu-np.mean(ST_eu))**4).mean()/np.std(ST_eu)**4-3),
        }
        gtable([(k, f"{v:.5f}", "") for k, v in sts.items()])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    sec("Apprentissage Automatique — Approximation de la Fonction de Pricing")
    fbox("Architecture du Problème ML", [
        '<span class="fdim">Features  X = (S, K, T, r, σ, S/K, type)  →  7 dimensions</span>',
        '<span class="fdim">Target    y = Black-Scholes(S, K, T, r, σ, type)  →  scalaire</span>',
        f'<span class="fdim">Dataset   : {st.session_state["ml_n_samples"]:,} échantillons'
        f'  ·  Split 80/20 train/test</span>',
        '<span class="fdim">Modèles   : MLP(128→64→32·ReLU)  |  GBR(300 arbres)  |  RF(200 arbres)</span>',
    ])

    kpi_row([
        ("MLP RMSE",  f"${ML['MLP Neural Net']['rmse']:.5f}",
                      f"R²={ML['MLP Neural Net']['r2']:.5f}",     "v"),
        ("GBR RMSE",  f"${ML['Gradient Boosting']['rmse']:.5f}",
                      f"R²={ML['Gradient Boosting']['r2']:.5f}",  "g"),
        ("RF  RMSE",  f"${ML['Random Forest']['rmse']:.5f}",
                      f"R²={ML['Random Forest']['r2']:.5f}",      "c"),
        ("MLP train", f"{ML['MLP Neural Net']['tr_t']:.1f}s",    "Entraînement",   "s"),
        ("Inférence", f"{ML['MLP Neural Net']['inf_t']:.2f}μs",  "Par option MLP", "a"),
    ])

    sec("Prédiction — Paramètres Actuels")
    kpi_row([
        ("BS (réf.)", f"${PBS:.5f}", "Exact", "b"),
        ("MLP",  f"${ML_CUR['MLP Neural Net']:.5f}",
                 f"Err: ${abs(ML_CUR['MLP Neural Net']-PBS):.5f}", "v"),
        ("GBR",  f"${ML_CUR['Gradient Boosting']:.5f}",
                 f"Err: ${abs(ML_CUR['Gradient Boosting']-PBS):.5f}", "g"),
        ("RF",   f"${ML_CUR['Random Forest']:.5f}",
                 f"Err: ${abs(ML_CUR['Random Forest']-PBS):.5f}", "c"),
    ])

    cA, cB = st.columns(2)
    with cA:
        sec("Prédit vs. Réel — MLP Neural Net")
        r_mlp = ML["MLP Neural Net"]
        yt, yp = r_mlp["y_test"], r_mlp["y_pred"]
        mn_v, mx_v = float(yt.min()), float(yt.max())
        fs = go.Figure()
        fs.add_trace(go.Scatter(x=[mn_v,mx_v], y=[mn_v,mx_v], name="Idéal",
                                 line=dict(color=CS, width=1.5, dash="dash")))
        fs.add_trace(go.Scattergl(x=yt.tolist(), y=yp.tolist(),
                                   mode="markers", name="MLP",
                                   marker=dict(color=CV, size=2.5, opacity=0.4)))
        fs = fup(fs, f"MLP — RMSE={r_mlp['rmse']:.5f}  R²={r_mlp['r2']:.5f}", 370)
        fs.update_xaxes(title_text="Prix BS (réel)")
        fs.update_yaxes(title_text="Prix MLP prédit")
        st.plotly_chart(fs, use_container_width=True)

    with cB:
        sec("Prédit vs. Réel — Gradient Boosting")
        r_gbr = ML["Gradient Boosting"]
        yt2, yp2 = r_gbr["y_test"], r_gbr["y_pred"]
        mn2, mx2 = float(yt2.min()), float(yt2.max())
        fs2 = go.Figure()
        fs2.add_trace(go.Scatter(x=[mn2,mx2], y=[mn2,mx2], name="Idéal",
                                  line=dict(color=CS, width=1.5, dash="dash")))
        fs2.add_trace(go.Scattergl(x=yt2.tolist(), y=yp2.tolist(),
                                    mode="markers", name="GBR",
                                    marker=dict(color=CG, size=2.5, opacity=0.4)))
        fs2 = fup(fs2, f"GBR — RMSE={r_gbr['rmse']:.5f}  R²={r_gbr['r2']:.5f}", 370)
        fs2.update_xaxes(title_text="Prix BS (réel)")
        fs2.update_yaxes(title_text="Prix GBR prédit")
        st.plotly_chart(fs2, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        sec("Résidus — MLP Neural Net")
        res_mlp = (r_mlp["y_test"] - r_mlp["y_pred"]).astype(float)
        fr2 = go.Figure()
        fr2.add_trace(go.Histogram(x=res_mlp.tolist(), nbinsx=70,
                                    marker=dict(color=CV, opacity=0.75),
                                    name="Résidus MLP"))
        fr2.add_vline(x=0., line=dict(color=CR, width=1.5, dash="dot"))
        fr2.add_vline(x=float(res_mlp.mean()),
                      line=dict(color=CA, width=1.2, dash="dot"),
                      annotation_text=f"μ={res_mlp.mean():.4f}",
                      annotation_font_color=CA)
        fr2 = fup(fr2, "Distribution des résidus MLP", 310)
        fr2.update_xaxes(title_text="Résidu ($)")
        st.plotly_chart(fr2, use_container_width=True)

    with cD:
        sec("RMSE & MAE — Comparaison des Modèles ML")
        nms   = list(ML.keys())
        rmses = [ML[n]["rmse"] for n in nms]
        maes  = [ML[n]["mae"]  for n in nms]
        fb = go.Figure()
        fb.add_trace(go.Bar(name="RMSE", x=nms, y=rmses, marker_color=CA,
                             text=[f"${v:.5f}" for v in rmses],
                             textposition="outside", textfont=dict(size=9)))
        fb.add_trace(go.Bar(name="MAE",  x=nms, y=maes,  marker_color=CC,
                             text=[f"${v:.5f}" for v in maes],
                             textposition="outside", textfont=dict(size=9)))
        fb = fup(fb, "RMSE & MAE par modèle", 310)
        fb.update_layout(barmode="group")
        st.plotly_chart(fb, use_container_width=True)

    sec("Prix ML vs. Black-Scholes — Plage de Spots")
    spot_ml  = np.linspace(max(1., Sf*0.5), Sf*1.5, 60)
    bs_line  = np.array([bs(float(s),Kf,Tf,rf,sigf,opt) for s in spot_ml])
    ml_range = ml_pred_range(ML, spot_ml, Kf, Tf, rf, sigf, opt)
    fml = go.Figure()
    fml.add_trace(go.Scatter(x=spot_ml.tolist(), y=bs_line.tolist(),
                              name="Black-Scholes", line=dict(color=CB, width=2.5)))
    for nm, col in zip(ML.keys(), [CV, CG, CC]):
        fml.add_trace(go.Scatter(x=spot_ml.tolist(), y=ml_range[nm].tolist(),
                                  name=nm, line=dict(color=col, width=1.8, dash="dot")))
    fml.add_vline(x=Sf, line=dict(color=CA, width=1, dash="dot"),
                  annotation_text=f"S={Sf}", annotation_font_color=CA)
    fml = fup(fml, "Prix ML vs. Black-Scholes en fonction du Spot", 370)
    fml.update_xaxes(title_text="Spot S"); fml.update_yaxes(title_text="Prix ($)")
    st.plotly_chart(fml, use_container_width=True)

    sec("Métriques Complètes — Tableau")
    df_ml = pd.DataFrame([{
        "Modèle": nm, "RMSE ($)": f"{r['rmse']:.6f}", "MAE ($)": f"{r['mae']:.6f}",
        "MAPE (%)": f"{r['mape']:.4f}", "R²": f"{r['r2']:.6f}",
        "Train (s)": f"{r['tr_t']:.2f}", "Inférence (μs)": f"{r['inf_t']:.3f}",
    } for nm, r in ML.items()])
    st.dataframe(df_ml.set_index("Modèle"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARAISON GLOBALE
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    sec("Comparaison Globale — Analytique · Numérique · Machine Learning")

    # Benchmarking
    t0 = time.perf_counter()
    for _ in range(10000): bs(Sf,Kf,Tf,rf,sigf,opt)
    bs_us = (time.perf_counter()-t0)/10000*1e6

    t0 = time.perf_counter()
    mc_pc, mc_sec, _, _ = monte_carlo(Sf,Kf,Tf,rf,sigf,opt,N_paths)
    mc_ms = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    eu_pc, eu_sec, _ = euler(Sf,Kf,Tf,rf,sigf,N_steps,N_paths,opt)
    eu_ms = (time.perf_counter()-t0)*1000

    feat_b = np.array([[Sf, Kf, Tf, rf, sigf, Sf/Kf,
                         0. if opt=="call" else 1.]], dtype=np.float64)
    ml_inf_us = {}
    for nm, res in ML.items():
        sc = res["scaler"]
        Xb = sc.transform(feat_b) if (res["scaled"] and sc is not None) else feat_b
        t0 = time.perf_counter()
        for _ in range(500): res["model"].predict(Xb)
        ml_inf_us[nm] = (time.perf_counter()-t0)/500*1e6

    all_m = [
        ("Black-Scholes",  PBS,    0.,               bs_us, "μs","0",               "Analytique","Exacte"),
        ("Monte Carlo",    mc_pc,  abs(mc_pc-PBS),   mc_ms, "ms",f"±{mc_sec:.5f}", "Numérique", "O(1/√N)"),
        ("Euler-Maruyama", eu_pc,  abs(eu_pc-PBS),   eu_ms, "ms",f"±{eu_sec:.5f}", "Numérique", "O(Δt)"),
    ]
    for nm in ML:
        pml = ML_CUR[nm]
        all_m.append((nm, pml, abs(pml-PBS), ml_inf_us[nm], "μs",
                      f"RMSE≈{ML[nm]['rmse']:.5f}", "ML", "Approx."))

    rows_h = ""
    for meth, price, err, tv, tu, unc, cat, conv in all_m:
        ec  = "tg" if err<0.01 else ("ta" if err<0.10 else "tr")
        mc2 = {"Analytique":"tb","Numérique":"tc","ML":"tv"}.get(cat,"")
        rows_h += (f"<tr>"
                   f"<td><span class='{mc2}'>{meth}</span></td>"
                   f"<td style='color:#e2e8f0;font-weight:600;'>${price:.5f}</td>"
                   f"<td class='{ec}'>${err:.6f}</td>"
                   f"<td style='color:#5a8aaa;'>{tv:.2f} {tu}</td>"
                   f"<td style='color:#2a3d4f;font-size:.71rem;'>{unc}</td>"
                   f"<td style='color:#2a3d4f;font-size:.71rem;'>{cat}</td>"
                   f"<td style='color:#2a3d4f;font-size:.71rem;'>{conv}</td>"
                   f"</tr>")

    st.markdown(f"""
    <table class="ctbl">
      <thead><tr>
        <th>Méthode</th><th>Prix ($)</th><th>|Erreur vs BS|</th>
        <th>Temps</th><th>Incertitude</th><th>Catégorie</th><th>Convergence</th>
      </tr></thead>
      <tbody>{rows_h}</tbody>
    </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cA, cB = st.columns(2)

    with cA:
        sec("Erreurs Absolues — Toutes Méthodes")
        nms_nb  = [m[0] for m in all_m if m[0]!="Black-Scholes"]
        errs_nb = [m[2] for m in all_m if m[0]!="Black-Scholes"]
        c_nb    = [CG, CV, CA, CG, CC][:len(nms_nb)]
        fb2 = go.Figure(go.Bar(
            x=nms_nb, y=errs_nb, marker=dict(color=c_nb),
            text=[f"${e:.5f}" for e in errs_nb],
            textposition="outside", textfont=dict(size=9)))
        fb2.add_hline(y=0.01, line=dict(color=CS, width=1, dash="dot"),
                      annotation_text="Seuil $0.01", annotation_font_color=CS)
        fb2 = fup(fb2, "Erreur absolue vs. Black-Scholes", 360)
        fb2.update_yaxes(title_text="|Erreur| ($)")
        st.plotly_chart(fb2, use_container_width=True)

    with cB:
        sec("Radar — Profil Qualitatif (/10)")
        crit = ["Précision","Vitesse","Scalabilité","Stabilité","Interprétab.","Généralité"]
        radar_s = {
            "Black-Scholes":  [10,10, 6,10,10, 5],
            "Monte Carlo":    [ 7, 4,10, 8, 7,10],
            "Euler-Maruyama": [ 7, 4, 9, 7, 8, 9],
            "MLP Neural Net": [ 6,10, 9, 5, 4, 8],
        }
        fg_r = go.Figure()
        for (nm, sc2), col in zip(radar_s.items(), [CB, CG, CV, CA]):
            sc_cl = sc2 + [sc2[0]]
            cr_cl = crit + [crit[0]]
            fg_r.add_trace(go.Scatterpolar(
                r=sc_cl, theta=cr_cl, name=nm,
                line=dict(color=col, width=2),
                fill="toself",
                fillcolor=rgba(col, 0.08)))     # ← rgba() ici aussi
        fg_r.update_layout(
            **PL, height=360,
            polar=dict(
                bgcolor="#060b12",
                radialaxis=dict(visible=True, range=[0,10], gridcolor="#162333",
                                tickfont=dict(size=8, color=CS)),
                angularaxis=dict(gridcolor="#162333",
                                 tickfont=dict(size=9, color="#4a6f88"))),
            title=dict(text="Profil comparatif (score /10)",
                       font=dict(family="IBM Plex Mono", size=12, color="#8fb8d8")))
        st.plotly_chart(fg_r, use_container_width=True)

    sec("Convergence MC — Toutes Méthodes Superposées")
    ns_all, p_all, se_all = mc_conv(Sf,Kf,Tf,rf,sigf,opt,N_paths,npts=60)
    fca = go.Figure()
    fca.add_trace(go.Scatter(
        x=np.concatenate([ns_all, ns_all[::-1]]).tolist(),
        y=np.concatenate([p_all+1.96*se_all,
                           (p_all-1.96*se_all)[::-1]]).tolist(),
        fill="toself", fillcolor=rgba(CG, 0.07),
        line=dict(color="rgba(0,0,0,0)"), name="IC 95% MC"))
    fca.add_trace(go.Scatter(x=ns_all.tolist(), y=p_all.tolist(),
                              name="Monte Carlo", line=dict(color=CG, width=2)))
    fca.add_hline(y=PBS,   line=dict(color=CB, width=2,   dash="dash"),
                  annotation_text="Black-Scholes", annotation_font_color=CB)
    fca.add_hline(y=eu_pc, line=dict(color=CV, width=1.5, dash="dot"),
                  annotation_text="Euler-Maruyama", annotation_font_color=CV)
    for nm, col in zip(["MLP Neural Net","Gradient Boosting","Random Forest"],[CA,CG,CC]):
        fca.add_hline(y=ML_CUR[nm], line=dict(color=col, width=1.2, dash="dashdot"),
                      annotation_text=nm.split()[0], annotation_font_color=col)
    fca = fup(fca, "Convergence MC & positionnement inter-méthodes", 400)
    fca.update_xaxes(title_text="N simulations")
    fca.update_yaxes(title_text="Prix estimé ($)")
    st.plotly_chart(fca, use_container_width=True)

    sec("Synthèse Quantitative")
    df_syn = pd.DataFrame([{
        "Méthode": m[0], "Prix ($)": f"{m[1]:.5f}",
        "|Erreur| ($)": f"{m[2]:.6f}",
        "Erreur (%)": f"{m[2]/PBS*100:.5f}" if PBS>0 else "—",
        "Temps": f"{m[3]:.2f} {m[4]}", "Catégorie": m[6], "Convergence": m[7],
    } for m in all_m])
    st.dataframe(df_syn.set_index("Méthode"), use_container_width=True)

    sec("Conclusion & Recommandations d'Usage")
    st.markdown("""
    <div class="cc-grid">
      <div class="cc">
        <div class="cc-t" style="color:#3b82f6;">Black-Scholes — Analytique</div>
        <div class="cc-b">
          ✅ Prix exact · Greeks analytiques · Vitesse μs<br>
          ✅ Calibration de la volatilité implicite<br>
          ❌ σ constante · Pas de dividendes · Marché continu<br>
          ❌ Log-normalité des rendements supposée<br>
          <em>→ Standard de l'industrie pour options vanilles.</em>
        </div>
      </div>
      <div class="cc">
        <div class="cc-t" style="color:#10b981;">Monte Carlo — Simulation</div>
        <div class="cc-b">
          ✅ Flexible : Heston, SABR, HJM, sauts<br>
          ✅ Options exotiques · path-dependent<br>
          ✅ Portefeuilles multi-actifs<br>
          ❌ Convergence lente O(1/√N) · Coûteux<br>
          <em>→ Méthode universelle sans formule fermée.</em>
        </div>
      </div>
      <div class="cc">
        <div class="cc-t" style="color:#8b5cf6;">Euler-Maruyama — EDS</div>
        <div class="cc-b">
          ✅ Simule la trajectoire complète S(t)<br>
          ✅ Options asiatiques, barrière, lookback<br>
          ✅ Extensible : sauts, corrélations<br>
          ❌ Erreur O(Δt) si Δt trop grand<br>
          <em>→ Fondation de la simulation stochastique générale.</em>
        </div>
      </div>
      <div class="cc">
        <div class="cc-t" style="color:#f59e0b;">Machine Learning — Approx.</div>
        <div class="cc-b">
          ✅ Inférence μs · Pricing temps réel HFT<br>
          ✅ Non-linéarités · Scalable massivement<br>
          ✅ Surfaces de volatilité implicite<br>
          ❌ Boîte noire · Dataset requis · Extrapolation risquée<br>
          <em>→ Optimal pour pricing massivement parallèle.</em>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
        # Tableau synthèse

    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)    
    summary = pd.DataFrame({
        "Critère":     ["Précision", "Vitesse inférence", "Flexibilité", "Généralisation", "Scalabilité"],
        "Black-Scholes": ["🟢 Exacte", "🟢 Très rapide", "🔴 Limitée", "🔴 Faible", "🟡 Moyenne"],
        "Euler/MC":    ["🟡 Bonne", "🔴 Lente", "🟢 Très haute", "🟢 Haute", "🟡 Moyenne"],
        "ML":          ["🟡 Bonne", "🟢 Très rapide", "🟡 Moyenne", "🟡 Moyenne", "🟢 Haute"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;font-size:.56rem;
            color:#162333;padding:.35rem;letter-spacing:.08em;'>
  QUANTITATIVE FINANCE · OPTION PRICING DASHBOARD v3.1 &nbsp;·&nbsp;
  Black-Scholes · Monte Carlo · Euler-Maruyama · ML &nbsp;·&nbsp; European Vanilla Options
</div>""", unsafe_allow_html=True)