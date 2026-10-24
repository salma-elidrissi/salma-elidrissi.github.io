"""Pricing des Options : Black-Scholes, Binomial , Trinomial.
Installation : pip install streamlit numpy scipy pandas plotly yfinance
Lancement    : streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Pricing des Options", layout="wide")
BIN, TRI, BSC = "#4f46e5", "#d97706", "#0f172a"

# ───────────────────────── CSS Haute Lisibilité & Contraste ─────────────────────────
st.markdown("""<style>
/* Style global pour la lisibilité et fort contraste */
[data-testid=stAppViewContainer] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
p, span, div, label, h1, h2, h3, h4, h5, h6, li, .stMarkdown {
    color: #0f172a !important;
}

/* En-tête / Hero banner */
.hero {
    background: #ffffff !important;
    border: 2px solid #6366f1 !important;
    color: #0f172a !important;
    padding: 20px 26px !important;
    border-radius: 12px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}
.hero h1 {
    color: #1e1b4b !important;
    margin: 0 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
.hero p {
    color: #334155 !important;
    margin: 6px 0 0 !important;
    font-weight: 500 !important;
}

/* Cadres, Containers, Métriques et Formulaires */
[data-testid="stMetric"], div[data-testid="stForm"], div[data-testid="stVerticalBlock"] > div[style*="border"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}

/* Métriques */
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] div {
    color: #1e1b4b !important;
    font-weight: 700 !important;
}

/* Champs de saisie (Inputs, Selects, Sliders) */
div[data-baseweb=input] > div, div[data-baseweb=select] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}
input {
    color: #0f172a !important;
}

/* Boutons de navigation et de calcul */
.stButton > button {
    background-color: #ffffff !important;
    border: 2px solid #4f46e5 !important;
    border-radius: 999px !important;
    color: #1e1b4b !important;
    min-height: 2.9rem !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #4f46e5 !important;
    color: #ffffff !important;
    border-color: #4f46e5 !important;
}
.stButton > button[kind="primary"] *, .stButton > button[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
}

/* Radio buttons & Checkboxes */
div[role="radiogroup"] label p {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Expander & Accordions */
.streamlit-expanderHeader {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
}
</style>""", unsafe_allow_html=True)

# ───────────────────────── Moteurs de calcul ─────────────────────────
def payoff(S, K, call):
    return np.maximum(S - K, 0) if call else np.maximum(K - S, 0)

def bs(S, K, T, r, q, s, call=True):
    sq = s * np.sqrt(T); d1 = (np.log(S / K) + (r - q + .5 * s * s) * T) / sq; d2 = d1 - sq
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def greeks(S, K, T, r, q, s, call=True):
    sq = s * np.sqrt(T); d1 = (np.log(S / K) + (r - q + .5 * s * s) * T) / sq; d2 = d1 - sq
    w, eq, er, pdf = (1 if call else -1), np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)
    return {"Delta": w * eq * norm.cdf(w * d1), "Gamma": eq * pdf / (S * sq),
            "Vega (pour 1 %)": S * eq * pdf * np.sqrt(T) / 100,
            "Theta (par an)": -S * eq * pdf * s / (2 * np.sqrt(T)) - w * r * K * er * norm.cdf(w * d2) + w * q * S * eq * norm.cdf(w * d1),
            "Rho (pour 1 %)": w * K * T * er * norm.cdf(w * d2) / 100}

def tree(model, S, K, T, r, q, s, N, call=True, amer=False, keep=False):
    """Retourne (prix, couches[(S, V, exercice)]) ; prix = nan si probabilités hors [0,1]."""
    dt = T / N; disc = np.exp(-r * dt); g = np.exp((r - q) * dt)
    if model == "Binomial":
        u = np.exp(s * np.sqrt(dt)); p = (g - 1 / u) / (u - 1 / u); pr = (p, 1 - p)
        spot = lambda i: S * u ** (i - 2 * np.arange(i + 1))
    else:
        u = np.exp(s * np.sqrt(3 * dt)); a = (r - q - .5 * s * s) * np.sqrt(dt / (12 * s * s))
        pr = (1 / 6 + a, 2 / 3, 1 / 6 - a)
        spot = lambda i: S * u ** (i - np.arange(2 * i + 1))
    if min(pr) < 0 or max(pr) > 1:
        return np.nan, []
    V = payoff(spot(N), K, call); L = [(spot(N), V, np.zeros(len(V), bool))] if keep else []
    for i in range(N - 1, -1, -1):
        V = disc * (pr[0] * V[:-1] + pr[1] * V[1:]) if model == "Binomial" else \
            disc * (pr[0] * V[:-2] + pr[1] * V[1:-1] + pr[2] * V[2:])
        ex = np.zeros(len(V), bool)
        if amer:
            h = payoff(spot(i), K, call); ex = h > V; V = np.where(ex, h, V)
        if keep:
            L.append((spot(i), V, ex))
    return float(V[0]), L[::-1]

@st.cache_data(show_spinner=False)
def tp(model, P, call, amer, N):
    return tree(model, *P, N, call, amer)[0]

def reference(P, call, amer):
    if not amer or (call and P[4] == 0 and P[3] >= 0):
        return float(bs(*P, call)), "Black-Scholes (exact)"
    return tp("Binomial", P, call, True, 2500), "Binomial N=2500 (référence numérique)"

@st.cache_data(show_spinner=False)
def benchmark(P, call, amer):
    ref, _ = reference(P, call, amer); rows = []
    for N in (10, 25, 50, 100, 250, 500, 1000):
        for m in ("Binomial", "Trinomial"):
            t0 = time.perf_counter(); v = tree(m, *P, N, call, amer)[0]; ms = (time.perf_counter() - t0) * 1e3
            rows.append(dict(Modèle=m, N=N, Prix=v, Erreur=abs(v - ref), Temps_ms=ms))
    return pd.DataFrame(rows), ref

@st.cache_data(ttl=3600, show_spinner=False)
def market(tk):
    try:
        import yfinance as yf
        h = yf.Ticker(tk).history(period="1y")["Close"].dropna()
        return float(h.iloc[-1]), float(np.log(h / h.shift(1)).std() * np.sqrt(252))
    except Exception:
        return None

def draw_tree(L, color):
    ex, ey = [], []
    for i in range(len(L) - 1):
        d = len(L[i + 1][0]) - len(L[i][0])
        for k, s_ in enumerate(L[i][0]):
            for o in range(d + 1):
                ex += [i, i + 1, None]; ey += [s_, L[i + 1][0][k + o], None]
    X, Y, T_, C = [], [], [], []
    for i, (Si, Vi, Xi) in enumerate(L):
        for s_, v, x in zip(Si, Vi, Xi):
            X.append(i); Y.append(s_); T_.append(f"{v:.2f}"); C.append("#d1495b" if x else color)
    f = go.Figure()
    f.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#cbd5e1", width=1), hoverinfo="skip", showlegend=False)
    f.add_scatter(x=X, y=Y, mode="markers+text", text=T_, textposition="top center", marker=dict(size=9, color=C),
                  hovertemplate="pas %{x}<br>S=%{y:.2f}<br>V=%{text}<extra></extra>", showlegend=False)
    f.update_layout(height=520, template="plotly_white", xaxis_title="Pas i", yaxis_title="Sous-jacent S", margin=dict(t=20))
    return f

# ───────────────────────── ① Fondements ─────────────────────────
def fondements():
    b = st.radio(
        "Bloc",
        [
            "Concepts fondamentaux",
            "Black-Scholes",
            "Modèle binomial",
            "Modèle trinomial"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    if b.startswith("Concepts"):

        # 1. Définitions des concepts fondamentaux
        st.subheader("1. Définitions des concepts fondamentaux")

        st.markdown("""
        - **Call (option d'achat)** : contrat qui donne à son détenteur
          le droit d'acheter un actif sous-jacent au prix d'exercice
          \(K\), à la date d'échéance \(T\) ou selon les conditions
          prévues par le contrat.

        - **Put (option de vente)** : contrat qui donne à son détenteur
          le droit de vendre un actif sous-jacent au prix d'exercice
          \(K\), à la date d'échéance \(T\) ou selon les conditions
          prévues par le contrat.

        - **Valeur intrinsèque** : gain immédiat théorique lié à
          l'exercice de l'option. Pour une option européenne évaluée
          à l'échéance, elle correspond au payoff de l'option.

        - **Valeur temps** : partie du prix de l'option qui reflète
          notamment la possibilité que son évolution future devienne
          favorable avant l'échéance.
        """)

        # 2. Formules regroupées dans un bloc
        st.subheader("2. Formules des payoffs et du prix de l'option")

        with st.container(border=True):
            st.markdown("**Payoff du Call à l'échéance**")
            st.latex(r"C_T = (S_T-K)^+")

            st.markdown("**Payoff du Put à l'échéance**")
            st.latex(r"P_T = (K-S_T)^+")

            st.divider()

            st.markdown("**Valeur intrinsèque à une date donnée**")
            st.latex(r"VI_{Call} = max(S-K,0)")
            st.latex(r"VI_{Put} = max(K-S,0)")

            st.markdown("**Décomposition du prix de l'option**")
            st.latex(
                r"{Prix  de  l'option}"
                r"={Valeur   intrinsèque}"
                r"+{Valeur   temps}"
            )

        st.caption(
            "Notation : S est le prix actuel du sous-jacent, "
            "S_T son prix à l'échéance, K le prix d'exercice "
            "et T la date d'échéance."
        )

        # 3. Facteurs influençant le prix des options
        st.subheader("3. Facteurs influençant le prix des options")

        st.markdown("""
        Le prix d'une option dépend de plusieurs paramètres du marché
        et des caractéristiques du contrat.
        """)

        st.dataframe(
            pd.DataFrame({
                "Facteur": [
                    "Prix du sous-jacent (S)",
                    "Prix d'exercice (K)",
                    "Temps restant jusqu'à l'échéance",
                    "Volatilité (σ)",
                    "Taux d'intérêt (r)",
                    "Rendement du dividende (q)"
                ],
                "Effet sur le Call": [
                    "↑",
                    "↓",
                    "Généralement ↑",
                    "↑",
                    "Généralement ↑",
                    "↓"
                ],
                "Effet sur le Put": [
                    "↓",
                    "↑",
                    "Généralement ↑",
                    "↑",
                    "↓",
                    "↑"
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Les effets indiqués correspondent aux sensibilités "
            "théoriques usuelles des options européennes dans le "
            "cadre du modèle Black-Scholes. L'effet du temps dépend "
            "notamment des paramètres et de la proximité de l'échéance."
        )

        # 4. Guide de choix du modèle
        st.subheader("4. Guide de choix du modèle de valorisation")

        with st.container(border=True):
            st.markdown("### Situations adaptées au modèle Black-Scholes:")
            st.markdown("""
            - **Type d'option :** vous valorisez principalement des options vanilles européennes (Call ou Put).
            - **Rapidité :** vous souhaitez obtenir rapidement un prix théorique à l'aide d'une formule analytique.
            - **Analyse financière :** vous souhaitez calculer facilement les sensibilités du prix (Delta, Gamma, Vega, Theta et Rho).
            - **Cadre de modélisation :** les hypothèses du modèle, notamment celles relatives à la volatilité et à la dynamique du sous-jacent, sont adaptées à votre situation.
            """)

            st.divider()
            st.markdown("### Situations adaptées au modèle binomial :")
            st.markdown("""
            - **Type d'option :** vous souhaitez valoriser des options vanilles européennes ou américaines, ainsi que certains produits plus complexes.
            - **Exercice anticipé :** vous devez tenir compte de la possibilité d'exercer l'option avant son échéance.
            - **Compréhension du calcul :** vous souhaitez représenter l'évolution possible du sous-jacent étape par étape dans un arbre à deux branches.
            - **Précision :** vous souhaitez étudier la convergence du prix en augmentant le nombre de pas de l'arbre.
            """)

            st.divider()
            st.markdown("### Situations adaptées au modèle modèle trinomial :")
            st.markdown("""
            - **Type d'option :** vous valorisez des options européennes ou américaines, ou certains produits exotiques compatibles avec la méthode.
            - **Dynamique du sous-jacent :** vous souhaitez représenter trois mouvements possibles à chaque étape : hausse, baisse ou stabilité.
            - **Flexibilité numérique :** vous recherchez une méthode adaptable à différentes configurations de valorisation.
            - **Convergence :** vous souhaitez étudier la stabilité numérique du prix en faisant varier le nombre de pas et les paramètres de l'arbre.
            """)
        st.info(
            "À retenir : Black-Scholes est particulièrement pratique pour les "
            "options vanilles européennes dans ses hypothèses usuelles. Les "
            "arbres binomial et trinomial permettent notamment de modéliser "
            "l'exercice anticipé. Les options exotiques (asiatiques, à barrière, "
            "lookback, etc.) peuvent nécessiter des adaptations ou d'autres "
            "méthodes, comme Monte-Carlo ou les différences finies. Le choix "
            "dépend du contrat, des hypothèses et de la précision recherchée."
        )
        
        # 5. Remarque importante
        st.info(
            "Sans dividende et avec un taux d'intérêt non négatif, "
            "un Call américain vaut autant qu'un Call européen. "
            "Un Put américain vaut au moins autant que le Put européen "
            "correspondant, et peut valoir strictly plus."
        )
    elif b == "Black-Scholes":
        st.markdown("**Hypothèses** : prix log-normal à σ et r constants, marché sans friction ni arbitrage, option européenne.")
        st.latex(r"C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),\quad P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)")
        st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\quad d_2=d_1-\sigma\sqrt T")
        with st.expander("Démonstration"):
            st.markdown("Sous $dS=(\\mu-q)S\\,dt+\\sigma S\\,dW$, le portefeuille delta-neutre $\\Pi=V-\\Delta S$ n'a pas d'aléa (lemme d'Itô), donc rapporte $r$ :")
            st.latex(r"\partial_tV+\tfrac12\sigma^2S^2\partial_{SS}V+(r-q)S\partial_SV-rV=0")
            st.latex(r"V(S,t)=e^{-r(T-t)}\,\mathbb E^{\mathbb Q}\big[(S_T-K)^+\big]\ \text{(Feynman-Kac)}")
        st.markdown("#### Atouts du modèle Black-Scholes")
        st.markdown("""
        - **Référence professionnelle :** il sert de cadre de référence dans les marchés de dérivés pour analyser les prix et les risques.
        - **Volatilité implicite :** il permet de retrouver la volatilité implicite correspondant à un prix coté sur le marché.
        - **Comparaison des scénarios :** il facilite l'étude de l'effet d'une variation des paramètres sur la valorisation.
        - **Intégration informatique :** sa structure analytique facilite son intégration dans des applications financières et des outils de suivi des positions.
        """)

        st.markdown("#### Limites du modèle Black-Scholes")
        st.markdown("""
        - **Écart avec les prix de marché :** le prix théorique peut différer du prix coté, notamment lorsque les conditions de marché évoluent.
        - **Smile et skew de volatilité :** une volatilité unique ne permet pas de reproduire à elle seule toutes les volatilités implicites observées selon les strikes et les échéances.
        - **Couverture imparfaite :** les coûts de transaction, les sauts de prix et les ajustements discrets peuvent limiter l'efficacité d'une couverture fondée sur le modèle.
        - **Dépendance aux données :** des données de marché imprécises ou peu liquides peuvent rendre les résultats moins fiables.
        """)

    elif b == "Modèle binomial":
        st.latex(r"u=e^{\sigma\sqrt{\Delta t}},\ d=\tfrac1u,\ p=\frac{e^{(r-q)\Delta t}-d}{u-d},\ \Delta t=T/N")
        st.latex(r"V_i^j=\max\Big(\text{payoff},\ e^{-r\Delta t}\big[pV_{i+1}^{j}+(1-p)V_{i+1}^{j+1}\big]\Big)")
        with st.expander("Démonstration de p"):
            st.markdown("Neutralité au risque : $\\mathbb E^{\\mathbb Q}[S_{t+\\Delta t}]=Se^{(r-q)\\Delta t}$, soit $pu+(1-p)d=e^{(r-q)\\Delta t}$, d'où $p$.")
            st.markdown("Le $\\max$ ne s'applique que pour le style américain. Condition : $0\\le p\\le1$.")
        st.markdown("#### Atouts du modèle binomial")

        st.markdown("""
        - **Simplicité de mise en œuvre :** sa structure en arbre à deux branches permet de comprendre et de programmer progressivement le mécanisme de valorisation.
        - **Exercice anticipé :** il permet de comparer, à chaque nœud, la valeur de continuation et la valeur d'exercice immédiat, ce qui est particulièrement utile pour les options américaines.
        - **Flexibilité des payoffs :** il peut être adapté à différentes conditions d'exercice et à certains produits dont le payoff dépend du chemin suivi par le sous-jacent.
        - **Contrôle de la précision :** le nombre de pas peut être augmenté pour étudier la convergence du prix et la stabilité des résultats.
        """)

        st.markdown("#### Limites du modèle binomial")

        st.markdown("""
        - **Convergence numérique :** pour les options vanilles standards, l'erreur est souvent d'ordre O(1/N), où N représente le nombre de pas, mais des oscillations peuvent apparaître selon la discrétisation et les paramètres du contrat.
        - **Coût de calcul :** augmenter le nombre de pas améliore souvent la précision, mais augmente également le temps de calcul.
        - **Hypothèses de marché :** dans sa version classique, le modèle suppose généralement une volatilité constante et une dynamique simplifiée du sous-jacent.
        - **Produits complexes :** les options dépendantes du chemin ou comportant plusieurs facteurs de risque peuvent nécessiter des adaptations de l'arbre et accroître la complexité du calcul.
        """)

    else:
        st.latex(r"u=e^{\sigma\sqrt{3\Delta t}},\ d=\tfrac1u,\ p_m=\tfrac23,\ p_{u,d}=\tfrac16\pm(r-q-\tfrac{\sigma^2}2)\sqrt{\tfrac{\Delta t}{12\sigma^2}}")
        st.latex(r"V_i^k=\max\Big(\text{payoff},\ e^{-r\Delta t}\big[p_uV_{i+1}^{k}+p_mV_{i+1}^{k+1}+p_dV_{i+1}^{k+2}\big]\Big)")
        st.markdown("Trois issues par pas : le trinomial correspond à un **schéma de différences finies explicite** de l'EDP de Black-Scholes.")
        st.markdown("#### Atouts du modèle trinomial")

        st.markdown("""
        - **Convergence numérique :** il peut offrir une convergence plus régulière que certains arbres binomiaux, selon la construction de l'arbre et les paramètres du contrat.
        - **Flexibilité de modélisation :** les trois mouvements possibles (hausse, baisse ou stabilité) offrent davantage de possibilités pour représenter l'évolution du sous-jacent.
        - **Valorisation de différentes options :** il peut être utilisé pour des options européennes ou américaines et adapté à certains payoffs plus complexes.
        - **Étude de la précision :** il permet d'analyser la stabilité du prix calculé en augmentant le nombre de pas et en ajustant les paramètres numériques.
        """)

        st.markdown("#### Limites du modèle trinomial")

        st.markdown("""
        - **Coût de calcul :** chaque étape comporte trois branches au lieu de deux, ce qui augmente le nombre de calculs par étape.
        - **Mémoire nécessaire :** la taille de l'arbre et le stockage des valeurs peuvent devenir plus importants selon son implémentation.
        - **Paramétrage :** le choix des probabilités de transition et des mouvements doit respecter les conditions de cohérence du modèle.
        - **Précision non garantie :** une convergence plus régulière ne signifie pas que le modèle est toujours plus précis ; le résultat dépend de la construction de l'arbre et du nombre de pas retenu.
        """)

# ───────────────────────── ③ Comparaison ─────────────────────────
def comparaison(P, call, amer, N):
    S, K, T, r, q, sig = P
    df, ref = benchmark(P, call, amer); _, lab = reference(P, call, amer)
    st.caption(f"Référence utilisée : {lab} = {ref:.6f}")
    pb, pt = tp("Binomial", P, call, amer, N), tp("Trinomial", P, call, amer, N)
    c = st.columns(4)
    c[0].metric(f"Binomial (N={N})", f"{pb:.4f}", f"err {abs(pb - ref):.2e}", delta_color="off")
    c[1].metric(f"Trinomial (N={N})", f"{pt:.4f}", f"err {abs(pt - ref):.2e}", delta_color="off")
    c[2].metric("Référence", f"{ref:.4f}"); c[3].metric("Plus précis", "Binomial" if abs(pb - ref) < abs(pt - ref) else "Trinomial")
    f1, f2 = go.Figure(), go.Figure(); slopes = {}
    for m, col in (("Binomial", BIN), ("Trinomial", TRI)):
        d = df[df.Modèle == m]; e = d.Erreur.values + 1e-16
        slopes[m] = np.polyfit(np.log(d.N), np.log(e), 1)[0]
        f1.add_scatter(x=d.N, y=e, name=m, mode="lines+markers", line=dict(color=col))
        f2.add_scatter(x=d.Temps_ms, y=e, name=m, mode="lines+markers", line=dict(color=col), text=d.N, hovertemplate="N=%{text}<br>%{x:.2f} ms<br>err=%{y:.2e}")
    for f, tt, xx in ((f1, "Précision : erreur vs N", "N"), (f2, "Efficacité : erreur vs temps CPU", "Temps (ms)")):
        f.update_xaxes(type="log", title=xx); f.update_yaxes(type="log", title="Erreur absolue")
        f.update_layout(height=380, template="plotly_white", title=tt)
    a, b2 = st.columns(2); a.plotly_chart(f1, use_container_width=True); b2.plotly_chart(f2, use_container_width=True)
    st.caption(f"Pente log-log de l'erreur : Binomial {slopes['Binomial']:.2f} · Trinomial {slopes['Trinomial']:.2f} (≈ −1 attendu ; régression bruitée par les oscillations).")
    st.dataframe(df.pivot(index="N", columns="Modèle", values=["Prix", "Erreur", "Temps_ms"]).style.format("{:.3e}", subset=["Erreur"]).format("{:.6f}", subset=["Prix"]).format("{:.2f}", subset=["Temps_ms"]), use_container_width=True)

    bb, tt_ = df[df.Modèle == "Binomial"], df[df.Modèle == "Trinomial"]
    tcum = np.maximum.accumulate(tt_.Temps_ms.values); tgt = bb.Temps_ms.values[3]
    e_t = float(np.exp(np.interp(np.log(tgt), np.log(tcum), np.log(tt_.Erreur.values + 1e-16)))); e_b = bb.Erreur.values[3] + 1e-16
    w = lambda x, y: "Binomial" if x < y else "Trinomial"
    st.subheader("Verdict (calculé automatiquement)")
    st.dataframe(pd.DataFrame({"Critère": ["Précision à N égal (N=100)", "Temps de calcul (N=100)", "Précision à temps égal"],
        "Binomial": [f"{bb.Erreur.values[3]:.2e}", f"{bb.Temps_ms.values[3]:.2f} ms", f"{e_b:.2e}"],
        "Trinomial": [f"{tt_.Erreur.values[3]:.2e}", f"{tt_.Temps_ms.values[3]:.2f} ms", f"{e_t:.2e}"],
        "Gagnant": [w(bb.Erreur.values[3], tt_.Erreur.values[3]), w(bb.Temps_ms.values[3], tt_.Temps_ms.values[3]), w(e_b, e_t)]}), hide_index=True, use_container_width=True)

    st.subheader("Prime d'exercice anticipé  (américain − européen)")
    Ss = np.linspace(.6 * S, 1.4 * S, 25); f = go.Figure()
    for m, col in (("Binomial", BIN), ("Trinomial", TRI)):
        for cl, dash in ((False, "solid"), (True, "dot")):
            y = [tp(m, (x, K, T, r, q, sig), cl, True, 150) - tp(m, (x, K, T, r, q, sig), cl, False, 150) for x in Ss]
            f.add_scatter(x=Ss, y=y, name=f"{'Call' if cl else 'Put'} {m}", line=dict(color=col, dash=dash))
    f.update_layout(height=380, template="plotly_white", xaxis_title="S", yaxis_title="Prime"); st.plotly_chart(f, use_container_width=True)
    st.caption("Avec q = 0 la prime du call est nulle ; augmentez q dans le panneau latéral pour la voir apparaître. Le put garde une prime positive si r > 0.")

    st.subheader("Vérifications théoriques")
    S_, K_, T_, r_, q_, s_ = P; th = S_ * np.exp(-q_ * T_) - K_ * np.exp(-r_ * T_); P0 = (S_, K_, T_, max(r_, 0), 0.0, s_)
    rows = [("Call américain = call européen (q=0, r≥0)", abs(tp("Binomial", P0, True, True, 200) - tp("Binomial", P0, True, False, 200)), 1e-9),
            ("Put américain ≥ put européen", max(tp("Binomial", P, False, False, 200) - tp("Binomial", P, False, True, 200), 0), 1e-9),
            ("Parité call-put (Binomial)", abs(tp("Binomial", P, True, False, 200) - tp("Binomial", P, False, False, 200) - th), 1e-8),
            ("Parité call-put (Trinomial, approchée)", abs(tp("Trinomial", P, True, False, 200) - tp("Trinomial", P, False, False, 200) - th), 1e-2),
            ("Borne d'arbitrage C ≥ Se^{-qT} − Ke^{-rT}", max(th - float(bs(*P, True)), 0), 1e-9)]
    ck = pd.DataFrame(rows, columns=["Test", "Écart", "Tolérance"]); ck["Statut"] = np.where(ck.Écart <= ck.Tolérance, "✅", "❌")
    st.dataframe(ck, hide_index=True, use_container_width=True)

# ───────────────────────── Interface ─────────────────────────
HERO = ('<div class="hero"><h1> Pricing des Options </h1><p>Black-Scholes · Binomial · Trinomial</p></div>')
GREEK_INFO = {"Delta": "Variation du prix pour +1 sur S (ratio de couverture).", "Gamma": "Variation du Delta : convexité, risque de rebalancement.",
              "Vega (pour 1 %)": "Gain pour +1 point de volatilité.", "Theta (par an)": "Perte de valeur temps avec le passage du temps.",
              "Rho (pour 1 %)": "Gain pour +1 point de taux sans risque."}
THEORY = {
"BS": """**Idée.** Le prix d'une option est l'espérance actualisée de son payoff sous la probabilité risque-neutre, avec un sous-jacent log-normal.
1. **Modèle de prix** : dS = (r−q)S dt + σS dW (sous Q), donc ln S_T est gaussien.
2. **Calcul de d₁ et d₂** : ils mesurent la position du spot par rapport au strike, corrigée de la dérive et de la volatilité.
3. **Formule** : C = S e^{−qT}N(d₁) − K e^{−rT}N(d₂), où N(d₂) est la probabilité (risque-neutre) d'exercice et N(d₁) le delta.
4. **Grecs** : dérivées partielles de la formule, obtenues analytiquement.
5. **Parité** : C − P = S e^{−qT} − K e^{−rT} (européennes uniquement).
6. **Limites** : européen seulement, σ et r constants, pas de sauts, pas d'exercice anticipé.""",
"Binomial": """**Principe.** On découpe [0,T] en N pas Δt = T/N ; à chaque pas le sous-jacent est multiplié par u ou par d.
1. **Facteurs** : u = e^{σ√Δt}, d = 1/u. L'arbre est recombinant (u·d = 1) : N+1 nœuds au pas N.
2. **Probabilité risque-neutre** : p = (e^{(r−q)Δt} − d)/(u − d). Elle impose un rendement espéré de r − q.
3. **Payoff terminal** calculé sur les N+1 nœuds finaux.
4. **Remontée** : V = e^{−rΔt}[p V↑ + (1−p) V↓]. En américain : V = max(exercice immédiat, V).
5. **Prix** = valeur du nœud racine. Il converge vers Black-Scholes quand N → ∞ (erreur en O(1/N), avec oscillations).
6. **Condition** : 0 ≤ p ≤ 1, ce qui exige un N suffisant si r est grand ou σ petit.""",
"Trinomial": """**Principe.** Trois issues par pas : hausse (×u), stabilité (×1), baisse (×1/u).
1. **Facteurs** : u = e^{σ√(3Δt)}, d = 1/u. Le pas 3Δt rend la variance du log-rendement exacte.
2. **Probabilités** : p_m = 2/3 et p_{u,d} = 1/6 ± (r − q − σ²/2)√(Δt/(12σ²)).
3. **Nœuds** : 2i+1 au pas i, soit plus de nœuds qu'un binomial, mais une convergence plus lisse.
4. **Remontée** : V = e^{−rΔt}[p_u V↑ + p_m V→ + p_d V↓], avec le max d'exercice en américain.
5. **Lien théorique** : c'est un schéma de différences finies explicite de l'EDP de Black-Scholes.
6. **Compromis** : plus précis par pas, mais plus coûteux : l'efficacité se juge à temps de calcul égal (page Comparaison)."""}

@st.cache_data(ttl=3600, show_spinner="Récupération des données de marché…")
def market(tk):
    err = []
    try:
        import yfinance as yf
        h = yf.Ticker(tk).history(period="1y")["Close"].dropna()
        if len(h) > 30: return float(h.iloc[-1]), float(np.log(h / h.shift(1)).std() * np.sqrt(252)), "Yahoo Finance"
        err.append("Yahoo : historique vide")
    except Exception as e: err.append(f"Yahoo : {e}")
    try:
        h = pd.read_csv(f"https://stooq.com/q/d/l/?s={tk.lower() if '.' in tk else tk.lower() + '.us'}&i=d")["Close"].dropna().tail(252)
        if len(h) > 30: return float(h.iloc[-1]), float(np.log(h / h.shift(1)).std() * np.sqrt(252)), "Stooq"
        err.append("Stooq : historique vide")
    except Exception as e: err.append(f"Stooq : {e}")
    raise RuntimeError(" | ".join(err))  # une exception n'est jamais mise en cache

def params_form(key, trees):
    st.markdown("##### 1 · Données de marché")
    tk = st.text_input("Entrez le ticker de l'actif (ex : AAPL, MSFT, MC.PA)", "AAPL", key=key + "tk").strip().upper()
    S0, s0, src = 100.0, 0.20, "valeurs par défaut"
    try: S0, s0, src = market(tk)
    except Exception as e: st.warning(f"Données indisponibles ({e}). Saisie manuelle possible ci-dessous. Vérifiez le ticker et la connexion, puis `pip install -U yfinance`.")
    a, b = st.columns(2); a.info(f"Prix actuel récupéré : **{S0:.2f}** · source : {src}"); b.info(f"Volatilité annualisée (1 an) : **{s0:.4f}**")
    st.markdown("##### 2 · Paramètres de l'option")
    c1, c2 = st.columns(2); k = f"{key}{tk}"
    S_ = c1.number_input("Prix actuel de l'actif (S₀)", .01, value=round(S0, 2), key=k + "S")
    sg = c2.number_input("Volatilité (σ)", .01, 2.0, float(np.clip(round(s0, 4), .01, 2.0)), .01, format="%.4f", key=k + "s")
    K_ = c1.number_input("Prix d'exercice (K)", .01, value=round(S0, 2), key=k + "K")
    T_ = c2.number_input("Maturité (en années, T)", .05, 10.0, .5, .05, key=key + "T")
    r_ = c1.number_input("Taux sans risque (r)", -.02, .30, .05, .005, format="%.4f", key=key + "rf")
    q_ = c2.number_input("Rendement de dividende (q)", 0.0, .30, 0.0, .005, format="%.4f", key=key + "q")
    call = c1.selectbox("Type d'option", ["Call", "Put"], key=key + "c") == "Call"
    amer = c2.selectbox("Style d'exercice", ["Européenne", "Américaine"], key=key + "a") == "Américaine"
    N_ = st.slider("Nombre de pas N de l'arbre", 2, 1000, 100, key=key + "N") if trees else 100
    ok = st.button(f"Calculer le prix", key=key + "go", use_container_width=True)
    return (S_, K_, T_, r_, q_, sg), call, amer, N_, ok

def bs_page():
    st.header("📐 Modèle de Black-Scholes")
    with st.expander("📘 Théorie et méthode pas à pas", expanded=True): st.markdown(THEORY["BS"])
    P, call, amer, _, ok = params_form("bs", False)
    if ok: st.session_state["res_bs"] = (P, call, amer)
    if "res_bs" not in st.session_state: st.info("Renseignez les paramètres puis cliquez sur « Calculer le prix »."); return
    P, call, amer = st.session_state["res_bs"]; S, K, T, r, q, sig = P
    if amer: st.warning("Black-Scholes ne valorise que l'européen : le prix affiché est européen (exact aussi pour un call américain sans dividende, sinon borne inférieure).")
    d1 = (np.log(S / K) + (r - q + .5 * sig ** 2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    g = greeks(*P, call); c = st.columns(5); c[0].metric("Prix BS", f"{float(bs(*P, call)):.4f}")
    for col, (k_, x) in zip(c[1:], g.items()): col.metric(k_.split(" (")[0], f"{float(x):.4f}")
    st.caption(f"d₁ = {d1:.4f} · d₂ = {d2:.4f} · N(d₂) = probabilité risque-neutre d'exercice = {norm.cdf(d2):.2%} (call)")
    Ss = np.linspace(.5 * S, 1.5 * S, 120); G = greeks(Ss, K, T, r, q, sig, call)
    f = make_subplots(rows=2, cols=3, subplot_titles=list(G))
    for i, (k_, y) in enumerate(G.items()): f.add_scatter(x=Ss, y=y, line=dict(color=BSC), row=i // 3 + 1, col=i % 3 + 1)
    f.update_layout(height=520, template="plotly_white", showlegend=False); st.plotly_chart(f, use_container_width=True)
    st.markdown("**🔎 Lecture des Grecs**"); st.table(pd.DataFrame({"Grec": list(GREEK_INFO), "Interprétation": list(GREEK_INFO.values())}))
    f = go.Figure(); f.add_scatter(x=Ss, y=Ss * np.exp(-q * T) - K * np.exp(-r * T), name="Se^{-qT} − Ke^{-rT}", line=dict(color=BSC))
    f.add_scatter(x=Ss, y=bs(Ss, K, T, r, q, sig, True) - bs(Ss, K, T, r, q, sig, False), mode="markers", name="C − P (BS)", marker=dict(color=BIN, size=5))
    f.update_layout(height=340, template="plotly_white", title="Parité call-put (européennes)", xaxis_title="S"); st.plotly_chart(f, use_container_width=True)
    st.caption("Les points (C − P calculé) se superposent à la droite théorique : la parité est vérifiée. Pour l'américain elle devient un encadrement.")

def tree_params(m, P, N):
    S, K, T, r, q, s = P; dt = T / N
    if m == "Binomial":
        u = np.exp(s * np.sqrt(dt)); return {"Δt": dt, "u": u, "d": 1 / u, "p (hausse)": (np.exp((r - q) * dt) - 1 / u) / (u - 1 / u)}
    u = np.exp(s * np.sqrt(3 * dt)); a = (r - q - .5 * s * s) * np.sqrt(dt / (12 * s * s))
    return {"Δt": dt, "u": u, "d": 1 / u, "p_u": 1 / 6 + a, "p_m": 2 / 3, "p_d": 1 / 6 - a}

def model_page(m, color, key):
    st.header(f"{'🌳' if m == 'Binomial' else '🔱'} Modèle {m.lower()}")
    with st.expander("📘 Théorie et méthode pas à pas", expanded=True): st.markdown(THEORY[m])
    P, call, amer, N, ok = params_form(key, True)
    if ok: st.session_state["res_" + key] = (P, call, amer, N)
    if "res_" + key not in st.session_state: st.info("Renseignez les paramètres puis cliquez sur « Calculer le prix »."); return
    P, call, amer, N = st.session_state["res_" + key]; S, K, T, r, q, sig = P
    ref, lab = reference(P, call, amer); price = tp(m, P, call, amer, N)
    if np.isnan(price): st.error("Probabilités hors [0,1] : augmentez N."); return
    c = st.columns(4); c[0].metric(f"Prix {m} (N={N})", f"{price:.4f}"); c[1].metric("Référence", f"{ref:.4f}", help=lab)
    c[2].metric("Erreur absolue", f"{abs(price - ref):.2e}"); c[3].metric("Erreur relative", f"{abs(price - ref) / max(ref, 1e-12):.3%}")
    st.caption(f"Référence : {lab}")
    if amer: st.metric("Prime d'exercice anticipé (américain − européen)", f"{price - tp(m, P, call, False, N):.4f}")
    st.markdown("**🔎 Paramètres de l'arbre calculés**"); st.dataframe(pd.DataFrame([tree_params(m, P, N)]), hide_index=True)
    n = st.slider("Pas affichés dans l'arbre", 2, 8, 4, key=key + "n"); _, L = tree(m, *P, n, call, amer, True)
    st.plotly_chart(draw_tree(L, color), use_container_width=True)
    st.caption("Chaque nœud affiche la valeur de l'option (calculée en remontant de l'échéance vers le pas 0) ; 🔴 = exercice anticipé optimal.")
    Ns = list(range(2, 151)); pv = [tp(m, P, call, amer, k) for k in Ns]
    f = make_subplots(rows=1, cols=2, subplot_titles=("Prix en fonction de N", "Erreur |prix − référence| (log)"))
    f.add_scatter(x=Ns, y=pv, line=dict(color=color), row=1, col=1); f.add_hline(y=ref, line=dict(color=BSC, dash="dash"), row=1, col=1)
    f.add_scatter(x=Ns, y=np.abs(np.array(pv) - ref) + 1e-16, line=dict(color=color), row=1, col=2)
    f.update_yaxes(type="log", row=1, col=2); f.update_layout(height=380, template="plotly_white", showlegend=False); st.plotly_chart(f, use_container_width=True)
    st.info("**Lecture** : la courbe de gauche doit se rapprocher de la ligne pointillée (référence) ; celle de droite montre l'erreur qui décroît avec N. "
            + ("Les oscillations viennent de la position du strike par rapport aux nœuds finaux." if m == "Binomial" else "Le trinomial oscille généralement moins."))
    pc, pp = tp(m, P, True, False, N), tp(m, P, False, False, N); th = S * np.exp(-q * T) - K * np.exp(-r * T)
    st.markdown("**Parité call-put (européennes)**"); st.latex(r"C-P=Se^{-qT}-Ke^{-rT}")
    c = st.columns(3); c[0].metric("C − P (arbre)", f"{pc - pp:.6f}"); c[1].metric("Valeur théorique", f"{th:.6f}"); c[2].metric("Écart", f"{abs(pc - pp - th):.2e}")

def cmp_page():
    st.header("Comparaison et analyse")
    P, call, amer, N, ok = params_form("cmp", True)
    if ok: st.session_state["res_cmp"] = (P, call, amer, N)
    if "res_cmp" in st.session_state: comparaison(*st.session_state["res_cmp"])
    else: st.info("Renseignez les paramètres puis cliquez sur « Calculer le prix ».")

PAGES = [" Accueil", " Black-Scholes", " Binomial Model", " Trinomial Model", " Comparaison", " Rapport PDF"]
st.session_state.setdefault("page", PAGES[0])
for col, p in zip(st.columns(len(PAGES)), PAGES):
    if col.button(p, key="nav" + p, use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
        st.session_state.page = p; st.rerun()

st.markdown(HERO, unsafe_allow_html=True); pg = st.session_state.page
if pg == PAGES[0]: fondements()
elif pg == PAGES[1]: bs_page()
elif pg == PAGES[2]: model_page("Binomial", BIN, "bin")
elif pg == PAGES[3]: model_page("Trinomial", TRI, "tri")
elif pg == PAGES[4]: cmp_page()
else:
    import pathlib
    st.header("📖 Rapport PDF"); f_ = pathlib.Path(__file__).with_name("projet_pricing_bino_trino.pdf")
    if f_.exists(): st.download_button("Télécharger le rapport", f_.read_bytes(), "projet_pricing_bino_trino.pdf", "application/pdf")
    else: st.info("Placez le fichier `projet_pricing_bino_trino.pdf` à côté de `app.py` pour le rendre téléchargeable ici.")

st.markdown("""
<style>
.academic-footer {
    margin-top: 60px;
    padding: 20px;
    border-top: 1px solid #cbd5e1;
    text-align: center;
    color: #334155;
    font-size: 14px;
    line-height: 1.8;
}
.academic-footer .title {
    color: #4f46e5;
    font-size: 17px;
    font-weight: bold;
}
</style>

<div class="academic-footer">
    <div class="title">Projet académique : Pricing des Options</div>
    <div>Année universitaire : 2026–2027</div>
    <div>Réalisé par : SALMA EL IDRISSI</div>
    <div>Black-Scholes · Binomial · Trinomial</div>
</div>
""", unsafe_allow_html=True)
