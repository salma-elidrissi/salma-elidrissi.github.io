"""
Détecter les régimes cachés de marché : calme, euphorique ou crise
Projet académique - Salma El Idrissi - FID 1ère année - Université 2024-2025

Lancer :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Régimes de marché", page_icon="📉", layout="wide")

PALETTE = {"Calme": "#2E86AB", "Euphorique": "#F2A541", "Crise": "#D1495B"}
LABELS_FEATURES = {
    "rendement": "Rendement quotidien",
    "momentum": "Rendement cumulé (momentum)",
    "volatilite": "Volatilité annualisée",
    "volume_z": "Volume anormal (z-score)",
    "drawdown": "Drawdown",
    "correlation": "Corrélation moyenne entre actifs",
}


def show(fig):
    st.plotly_chart(fig, width="stretch")


def show_df(df):
    st.dataframe(df, width="stretch")


def couleur(nom, i=0):
    return PALETTE.get(nom, px.colors.qualitative.Set2[i % 8])


# ----------------------------------------------------------------------------
# 1. DONNÉES
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_yahoo(tickers, start, end):
    import yfinance as yf

    d = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    close, vol = d["Close"], d["Volume"]
    if isinstance(close, pd.Series):
        close, vol = close.to_frame(tickers[0]), vol.to_frame(tickers[0])
    close = close.dropna(how="all").ffill().dropna()
    vol = vol.reindex(close.index)
    return close, vol


@st.cache_data(show_spinner=False)
def simuler(n=2500, seed=42):
    """Marché simulé à 3 régimes cachés (chaîne de Markov) : calme, euphorique, crise."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.010, 0.005], [0.030, 0.960, 0.010], [0.040, 0.020, 0.940]])
    mu = np.array([0.0003, 0.0010, -0.0015])
    sig = np.array([0.006, 0.007, 0.025])
    idio = np.array([1.0, 1.0, 0.35])  # corrélations plus fortes en crise
    etat = np.zeros(n, dtype=int)
    for t in range(1, n):
        etat[t] = rng.choice(3, p=P[etat[t - 1]])
    m = rng.normal(mu[etat], sig[etat])
    betas = np.array([1.0, 1.1, -0.2, 0.3])
    rets = np.column_stack([b * m + rng.normal(0, sig[etat] * idio[etat]) * 0.8 for b in betas])
    dates = pd.bdate_range("2014-01-01", periods=n)
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates,
                         columns=["MARCHE", "ACTIF_B", "OBLIG", "OR"])
    vol = pd.DataFrame({"MARCHE": 1e6 * np.exp(rng.normal(0, 0.2, n)) * (1 + 1.5 * (etat == 2))},
                       index=dates)
    return close, vol


def lire_csv(fichier):
    df = pd.read_csv(fichier, index_col=0, parse_dates=True).sort_index()
    col_vol = [c for c in df.columns if c.lower() in ("volume", "vol")]
    vol = df[col_vol[0]].to_frame("MARCHE") if col_vol else None
    close = df.drop(columns=col_vol).dropna(how="all").ffill().dropna()
    if vol is not None:
        vol.columns = [close.columns[0]]
        vol = vol.reindex(close.index)
    return close, vol


# ----------------------------------------------------------------------------
# 2. VARIABLES (FEATURES) ET RISQUE FUTUR
# ----------------------------------------------------------------------------
def construire_features(close, volume, mkt, w_vol, w_mom, w_corr):
    ret = close.pct_change()
    r = ret[mkt]
    f = pd.DataFrame(index=close.index)
    f["rendement"] = r
    f["momentum"] = close[mkt].pct_change(w_mom)
    f["volatilite"] = r.rolling(w_vol).std() * np.sqrt(252)
    if volume is not None and mkt in volume.columns and volume[mkt].replace(0, np.nan).notna().mean() > 0.9:
        lv = np.log(volume[mkt].replace(0, np.nan))
        f["volume_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    f["drawdown"] = close[mkt] / close[mkt].cummax() - 1
    cols = list(ret.columns)
    if len(cols) > 1:
        paires = [ret[a].rolling(w_corr).corr(ret[b]) for i, a in enumerate(cols) for b in cols[i + 1:]]
        f["correlation"] = pd.concat(paires, axis=1).mean(axis=1)
    return f


def construire_risque(prix, h):
    """Risque réalisé sur les h jours SUIVANT chaque date (utilisé uniquement pour évaluer)."""
    r = prix.pct_change()
    vol_fut = r.rolling(h).std().shift(-h) * np.sqrt(252)
    dd_fut = pd.concat([prix.shift(-k) / prix - 1 for k in range(1, h + 1)], axis=1).min(axis=1)
    dd_fut[prix.shift(-h).isna()] = np.nan
    return pd.DataFrame({"vol_future": vol_fut, "dd_futur": dd_fut})


# ----------------------------------------------------------------------------
# 3. MODÈLES
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def courbes_k(X, kmax=7):
    res = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(2000, len(X)), random_state=42)
        res.append({"k": k, "inertie": km.inertia_, "silhouette": sil})
    return pd.DataFrame(res)


def nommer(centres, k):
    """Donne un nom économique à chaque cluster à partir de ses centres (unités d'origine)."""
    if k == 3:
        crise = centres["volatilite"].idxmax()
        reste = [c for c in centres.index if c != crise]
        if "momentum" in centres.columns:
            euph = centres.loc[reste, "momentum"].idxmax()
        elif "rendement" in centres.columns:
            euph = centres.loc[reste, "rendement"].idxmax()
        else:
            euph = centres.loc[reste, "drawdown"].idxmax()
        calme = [c for c in reste if c != euph][0]
        return {calme: "Calme", euph: "Euphorique", crise: "Crise"}
    ordre = list(centres.sort_values("volatilite").index)
    noms = {}
    for i, c in enumerate(ordre):
        suf = " (le plus calme)" if i == 0 else " (le plus risqué)" if i == k - 1 else ""
        noms[c] = f"Régime {i + 1}{suf}"
    return noms


def metriques(r):
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {}
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    return {"Rendement annualisé": ann, "Volatilité annualisée": vol,
            "Sharpe (rf=0)": ann / vol if vol > 0 else np.nan,
            "Max drawdown": mdd, "Calmar": ann / abs(mdd) if mdd < 0 else np.nan}


def durees(serie):
    blocs = (serie != serie.shift()).cumsum()
    return serie.groupby(blocs).agg(["first", "size"]).groupby("first")["size"].mean()


# ----------------------------------------------------------------------------
# INTERFACE : BARRE LATÉRALE
# ----------------------------------------------------------------------------
st.title("📉 Détecter les régimes cachés de marché")
st.caption("Marché calme, euphorique ou en crise — Projet académique : **Salma El Idrissi** · "
           "FID 1ère année · Université 2024-2025")

sb = st.sidebar
sb.header("1. Données")
source = sb.radio("Source", ["Yahoo Finance", "Fichier CSV", "Données simulées (démo)"])
close = volume = None
if source == "Yahoo Finance":
    txt = sb.text_input("Tickers (le 1er = marché de référence)", "SPY, QQQ, TLT, GLD, EEM")
    c1, c2 = sb.columns(2)
    d0 = c1.date_input("Début", pd.Timestamp("2008-01-01"))
    d1 = c2.date_input("Fin", pd.Timestamp.today())
    tickers = tuple(t.strip().upper() for t in txt.split(",") if t.strip())
    try:
        close, volume = charger_yahoo(tickers, str(d0), str(d1))
    except Exception as e:
        st.error(f"Téléchargement impossible ({e}). Essayez le CSV ou les données simulées.")
        st.stop()
elif source == "Fichier CSV":
    f = sb.file_uploader("CSV : Date + colonnes de prix (+ colonne Volume optionnelle)", type="csv")
    if f is None:
        st.info("Importez un fichier CSV (1ère colonne = date, puis un prix par colonne ; "
                "la 1ère colonne de prix est le marché de référence).")
        st.stop()
    close, volume = lire_csv(f)
else:
    close, volume = simuler()
    sb.caption("Série simulée avec 3 régimes connus : utile pour valider la méthode.")

if close is None or close.empty or len(close) < 400:
    st.error("Données insuffisantes (il faut au moins ~400 jours).")
    st.stop()

mkt = close.columns[0]
sb.header("2. Variables")
w_vol = sb.slider("Fenêtre volatilité (jours)", 5, 60, 20)
w_mom = sb.slider("Fenêtre momentum (jours)", 5, 60, 20)
w_corr = sb.slider("Fenêtre corrélation (jours)", 20, 120, 60)

feat_all = construire_features(close, volume, mkt, w_vol, w_mom, w_corr)
dispo = list(feat_all.columns)
cols = sb.multiselect("Variables utilisées", dispo, default=dispo,
                      format_func=lambda c: LABELS_FEATURES[c])
if "volatilite" not in cols:
    st.error("La variable « volatilité » est nécessaire pour nommer les régimes.")
    st.stop()

sb.header("3. Classification")
k = sb.slider("Nombre de régimes K (K-means)", 2, 6, 3)
k_cah = sb.slider("Nombre de familles CAH", 2, 6, 3)
methode = sb.selectbox("Critère d'agrégation CAH", ["ward", "complete", "average"])
n_cah = sb.slider("Jours échantillonnés pour la CAH", 100, 500, 250, step=50)
split = sb.slider("% de période d'apprentissage", 50, 100, 70,
                  help="100 % = tout en échantillon (illustratif). Moins = test hors-échantillon honnête.")

sb.header("4. Risque futur à anticiper")
h = sb.slider("Horizon h (jours)", 3, 20, 5)
type_ev = sb.radio("Définition de l'événement de risque",
                   ["Volatilité future élevée", "Baisse future importante"])
if type_ev == "Volatilité future élevée":
    q = sb.slider("Seuil = quantile (%) de la vol future (apprentissage)", 60, 95, 80) / 100
else:
    seuil_dd = sb.slider("Baisse minimale sur h jours (%)", 1.0, 10.0, 3.0, step=0.5) / 100
eval_test = sb.checkbox("Évaluer uniquement hors-échantillon", value=split < 100)

# ----------------------------------------------------------------------------
# CALCULS
# ----------------------------------------------------------------------------
risque = construire_risque(close[mkt], h)
df = feat_all[cols].join(risque).join(close[mkt].pct_change().rename("ret_mkt"))
df = df.dropna(subset=cols)
n_train = int(len(df) * split / 100)
train = df.iloc[:n_train]

scaler = StandardScaler().fit(train[cols])
X = scaler.transform(df[cols])
Xtr = X[:n_train]

km = KMeans(k, n_init=20, random_state=42).fit(Xtr)
lab = km.predict(X)
centres = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=cols)
noms = nommer(centres, k)
df["regime"] = pd.Series(lab, index=df.index).map(noms)
ordre = [noms[c] for c in centres.sort_values("volatilite").index]
id_risque = centres["volatilite"].idxmax()
df["score_crise"] = -km.transform(X)[:, id_risque]

if type_ev == "Volatilité future élevée":
    seuil = train["vol_future"].quantile(q)
    df["event"] = (df["vol_future"] > seuil).astype(float).where(df["vol_future"].notna())
    desc_ev = f"volatilité des {h} prochains jours > {seuil:.1%} (quantile {q:.0%} de l'apprentissage)"
else:
    df["event"] = (df["dd_futur"] < -seuil_dd).astype(float).where(df["dd_futur"].notna())
    desc_ev = f"baisse > {seuil_dd:.1%} au cours des {h} prochains jours"

ev = df.iloc[n_train:] if (eval_test and n_train < len(df) - 50) else df
ev = ev.dropna(subset=["event"])

# ----------------------------------------------------------------------------
# ONGLETS
# ----------------------------------------------------------------------------
tabs = st.tabs([" Données", " CAH", " K-means", " Régimes",
                " Anticipation du risque", " Application", " Conclusion"])

# ---- Données ---------------------------------------------------------------
with tabs[0]:
    st.subheader("Données et variables construites")
    st.write(f"Marché de référence : **{mkt}** · {len(df)} jours exploitables "
             f"({df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y}) · "
             f"apprentissage : {n_train} jours")
    base = (close / close.iloc[0] * 100)
    show(px.line(base, title="Prix normalisés (base 100)"))
    st.markdown("**Statistiques descriptives des variables**")
    show_df(df[cols].describe().T.round(4))
    c1, c2 = st.columns(2)
    corr = df[cols].corr().round(2)
    c1.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Corrélations entre variables"),
                    width="stretch")
    c2.plotly_chart(px.line(df["volatilite"], title="Volatilité annualisée"),
                    width="stretch")
    with st.expander("Définition des variables"):
        st.markdown(
            "- **Rendement** : variation quotidienne du marché de référence.\n"
            "- **Momentum** : rendement cumulé sur la fenêtre choisie.\n"
            "- **Volatilité** : écart-type glissant des rendements × √252.\n"
            "- **Volume anormal** : écart du log-volume à sa moyenne sur 60 jours (z-score).\n"
            "- **Drawdown** : perte relative par rapport au plus haut historique.\n"
            "- **Corrélation** : moyenne des corrélations glissantes entre tous les actifs "
            "(elle monte typiquement en période de stress).")

# ---- CAH -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Classification ascendante hiérarchique : familles de comportements")
    idx = np.linspace(0, n_train - 1, min(n_cah, n_train)).astype(int)
    Xs = Xtr[idx]
    Z = linkage(Xs, method=methode)
    fig = ff.create_dendrogram(Xs, linkagefun=lambda x: linkage(x, methode))
    thr = (Z[-(k_cah - 1), 2] + Z[-k_cah, 2]) / 2
    fig.add_hline(y=thr, line_dash="dash", line_color="red",
                  annotation_text=f"Coupure → {k_cah} familles")
    fig.update_layout(title=f"Dendrogramme ({methode}, {len(Xs)} jours échantillonnés)",
                      height=450, xaxis=dict(showticklabels=False), yaxis_title="Distance")
    show(fig)

    c1, c2 = st.columns(2)
    hauteurs = Z[-12:, 2][::-1]
    c1.plotly_chart(px.bar(x=list(range(2, len(hauteurs) + 2)), y=hauteurs,
                           labels={"x": "Nombre de familles", "y": "Hauteur de fusion"},
                           title="Sauts de fusion : un grand saut suggère le bon nombre de familles"),
                    width="stretch")
    fam = fcluster(Z, k_cah, "maxclust")
    prof = pd.DataFrame(Xs, columns=cols).groupby(fam).mean()
    prof.index = [f"Famille {i}" for i in prof.index]
    c2.plotly_chart(px.imshow(prof.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              aspect="auto", title="Profil des familles (z-scores)"),
                    width="stretch")
    effectifs = pd.Series(fam).value_counts().sort_index()
    st.write("Effectifs : " + " · ".join(f"Famille {i} : {n}" for i, n in effectifs.items()))

    km_s = KMeans(k_cah, n_init=10, random_state=42).fit(Xs)
    ari = adjusted_rand_score(fam, km_s.labels_)
    st.metric("Concordance CAH / K-means (Rand ajusté, même échantillon)", f"{ari:.2f}",
              help="1 = partitions identiques ; 0 = accord dû au hasard.")
    st.caption("La CAH sert à **découvrir** la structure (nombre de familles plausible, "
               "profils types) ; K-means sert ensuite à **classer chaque jour**.")

# ---- K-means ---------------------------------------------------------------
with tabs[2]:
    st.subheader("K-means : affectation de chaque journée à un régime")
    ck = courbes_k(Xtr)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(ck, x="k", y="inertie", markers=True, title="Méthode du coude (inertie)"),
                    width="stretch")
    c2.plotly_chart(px.line(ck, x="k", y="silhouette", markers=True, title="Score de silhouette"),
                    width="stretch")
    st.info(f"Meilleur silhouette : K = {int(ck.loc[ck.silhouette.idxmax(), 'k'])} · "
            f"K choisi : {k}")

    c3, c4 = st.columns(2)
    cz = pd.DataFrame(km.cluster_centers_, columns=cols)
    cz.index = [noms[i] for i in cz.index]
    c3.plotly_chart(px.imshow(cz.loc[ordre].round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              aspect="auto", title="Centres des régimes (z-scores)"),
                    width="stretch")
    pca = PCA(2).fit(Xtr)
    P = pca.transform(X)
    dfp = pd.DataFrame(P, columns=["PC1", "PC2"], index=df.index).assign(regime=df["regime"])
    figp = px.scatter(dfp, x="PC1", y="PC2", color="regime", opacity=0.6,
                      color_discrete_map={r: couleur(r, i) for i, r in enumerate(ordre)},
                      title=f"Projection ACP ({pca.explained_variance_ratio_.sum():.0%} de variance)")
    c4.plotly_chart(figp, width="stretch")
    st.markdown("**Centres des régimes (unités d'origine)**")
    cr = centres.copy()
    cr.index = [noms[i] for i in cr.index]
    show_df(cr.loc[ordre].round(4))
    st.caption("Nommage automatique (K=3) : *Crise* = volatilité la plus élevée ; "
               "*Euphorique* = momentum le plus fort parmi les deux autres ; *Calme* = le reste.")

# ---- Régimes ---------------------------------------------------------------
with tabs[3]:
    st.subheader("Chronologie et caractéristiques des régimes")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.04,
                        subplot_titles=("Prix (log) coloré par régime", "Volatilité", "Drawdown"))
    px_mkt = close[mkt].reindex(df.index)
    for i, r in enumerate(ordre):
        m = df["regime"] == r
        fig.add_trace(go.Scatter(x=df.index[m], y=px_mkt[m], mode="markers", name=r,
                                 marker=dict(size=3, color=couleur(r, i))), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["volatilite"], line=dict(color="#555", width=1),
                             showlegend=False), row=2, col=1)
    if "drawdown" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["drawdown"], fill="tozeroy",
                                 line=dict(color="#D1495B", width=1), showlegend=False), row=3, col=1)
    if n_train < len(df):
        fig.add_vline(x=df.index[n_train], line_dash="dash", line_color="black")
        fig.add_annotation(x=df.index[n_train], y=1, yref="paper", text="Début hors-échantillon",
                           showarrow=False, xanchor="left")
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(height=700, legend=dict(orientation="h"))
    show(fig)

    c1, c2 = st.columns(2)
    freq = df["regime"].value_counts(normalize=True).reindex(ordre)
    dur = durees(df["regime"]).reindex(ordre)
    tab = pd.DataFrame({"Fréquence": freq, "Durée moyenne (jours)": dur})
    tab = tab.join(df.groupby("regime")[cols].mean().reindex(ordre))
    c1.markdown("**Résumé par régime**")
    c1.dataframe(tab.round(4), width="stretch")
    T = pd.crosstab(df["regime"].iloc[:-1].values, df["regime"].iloc[1:].values, normalize="index")
    T = T.reindex(index=ordre, columns=ordre).fillna(0)
    c2.plotly_chart(px.imshow(T.round(3), text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Régime demain", y="Régime aujourd'hui"),
                              title="Matrice de transition (persistance des régimes)"),
                    width="stretch")

# ---- Anticipation du risque ------------------------------------------------
with tabs[4]:
    st.subheader("Les régimes détectés anticipent-ils une hausse du risque ?")
    st.write(f"**Événement de risque :** {desc_ev}. "
             f"Évaluation sur **{'hors-échantillon' if ev is not df else 'toute la période'}** "
             f"({len(ev)} jours).")
    p0 = ev["event"].mean()
    rows = []
    for r in ordre:
        s = ev[ev.regime == r]
        if len(s) == 0:
            continue
        rows.append({"Régime": r, "Jours": len(s),
                     "Vol actuelle": s["volatilite"].mean(),
                     "Vol future (h jours)": s["vol_future"].mean(),
                     "Ratio futur/actuel": s["vol_future"].mean() / s["volatilite"].mean(),
                     "P(événement)": s["event"].mean(),
                     "Lift vs base": s["event"].mean() / p0 if p0 > 0 else np.nan})
    tr = pd.DataFrame(rows).set_index("Régime")
    st.metric("Probabilité de base de l'événement (tous jours confondus)", f"{p0:.1%}")
    show_df(tr.round(3))

    c1, c2 = st.columns(2)
    figb = px.box(ev.dropna(subset=["vol_future"]), x="regime", y="vol_future",
                  category_orders={"regime": ordre}, color="regime",
                  color_discrete_map={r: couleur(r, i) for i, r in enumerate(ordre)},
                  title=f"Volatilité réalisée sur les {h} jours suivants")
    c1.plotly_chart(figb, width="stretch")
    figp = px.bar(tr.reset_index(), x="Régime", y="P(événement)", color="Régime",
                  color_discrete_map={r: couleur(r, i) for i, r in enumerate(ordre)},
                  title="Probabilité de l'événement de risque selon le régime")
    figp.add_hline(y=p0, line_dash="dash", annotation_text="Base")
    c2.plotly_chart(figp, width="stretch")

    groupes = [g["vol_future"].dropna().values for _, g in ev.groupby("regime") if g["vol_future"].notna().sum() > 5]
    if len(groupes) >= 2:
        stat, pval = kruskal(*groupes)
        st.write(f"**Test de Kruskal-Wallis** (vol future identique entre régimes ?) : "
                 f"H = {stat:.1f}, p-value = {pval:.2e}")

    st.markdown("#### Le régime apporte-t-il plus que la simple volatilité du jour ?")
    ytrue = ev["event"].values
    if 0 < ytrue.sum() < len(ytrue):
        auc_reg = roc_auc_score(ytrue, ev["score_crise"])
        auc_vol = roc_auc_score(ytrue, ev["volatilite"])
        c3, c4 = st.columns(2)
        c3.metric("AUC — score de régime (proximité du centre « crise »)", f"{auc_reg:.3f}")
        c4.metric("AUC — volatilité actuelle seule (référence)", f"{auc_vol:.3f}",
                  delta=f"{auc_reg - auc_vol:+.3f} pour le régime")
        st.caption("AUC = 0,5 : aucun pouvoir d'anticipation ; 1 : parfait. La volatilité étant très "
                   "persistante, elle constitue un point de comparaison exigeant.")
    else:
        st.warning("Un seul type de journée dans l'échantillon d'évaluation : AUC non calculable.")

    st.markdown("#### Système d'alerte")
    alerte_reg = st.multiselect("Déclencher l'alerte quand le régime est :", ordre, default=[ordre[-1]])
    al = ev["regime"].isin(alerte_reg)
    if al.any() and ev["event"].sum() > 0:
        prec = ev.loc[al, "event"].mean()
        rec = al[ev["event"] == 1].mean()
        fa = al[ev["event"] == 0].mean()
        a, b, c = st.columns(3)
        a.metric("Précision (alertes justifiées)", f"{prec:.1%}")
        b.metric("Rappel (événements précédés d'une alerte)", f"{rec:.1%}")
        c.metric("Taux de fausses alertes", f"{fa:.1%}")

    st.markdown("#### Entrée en régime risqué : signal précoce ?")
    risque_nom = ordre[-1]
    entree = (df["regime"] == risque_nom) & (df["regime"].shift(1) != risque_nom)
    ee = ev[entree.reindex(ev.index).fillna(False)]
    if len(ee) > 0:
        st.write(f"**{len(ee)} entrées** dans le régime « {risque_nom} » : vol future moyenne "
                 f"**{ee['vol_future'].mean():.1%}** contre {ev['vol_future'].mean():.1%} en moyenne générale ; "
                 f"vol du jour d'entrée : {ee['volatilite'].mean():.1%}.")
    st.warning("**Limites :** les fenêtres de risque futur se chevauchent (autocorrélation) → les p-values "
               "sont optimistes ; la persistance de la volatilité explique une bonne part du résultat ; "
               "il s'agit d'**anticipation probabiliste d'une hausse du risque**, pas de prédiction "
               "systématique des krachs.")

# ---- Application -----------------------------------------------------------
with tabs[5]:
    st.subheader("Adapter l'exposition au risque selon le régime détecté")
    st.write("Règle : l'exposition du jour *t* dépend du régime observé en *t-1* "
             "(aucun biais d'anticipation). Le reste du capital est en liquidités (rendement 0).")
    defaut = np.linspace(100, 20, len(ordre)).astype(int)
    if k == 3:
        defaut = [100, 80, 20]
    cols_sl = st.columns(len(ordre))
    expo_map = {r: cols_sl[i].slider(f"Exposition {r} (%)", 0, 150, int(defaut[i]), key=f"e{i}") / 100
                for i, r in enumerate(ordre)}
    cout = st.slider("Coût de transaction (points de base par unité d'exposition échangée)", 0, 20, 5)

    bt = df.iloc[n_train:] if (eval_test and n_train < len(df) - 50) else df
    expo = bt["regime"].map(expo_map).shift(1).fillna(1.0)
    turn = expo.diff().abs().fillna(0)
    strat = expo * bt["ret_mkt"] - turn * cout / 1e4
    bh = bt["ret_mkt"]
    comp = pd.DataFrame({"Buy & Hold": metriques(bh), "Stratégie par régimes": metriques(strat)}).T
    cum = pd.DataFrame({"Buy & Hold": (1 + bh).cumprod(), "Stratégie par régimes": (1 + strat).cumprod()})

    logy = st.checkbox("Échelle logarithmique", value=True)
    figc = px.line(cum, title="Valeur cumulée (base 1)", log_y=logy)
    show(figc)
    fmt = comp.copy()
    for c in ["Rendement annualisé", "Volatilité annualisée", "Max drawdown"]:
        fmt[c] = (fmt[c] * 100).round(2).astype(str) + " %"
    fmt["Sharpe (rf=0)"] = fmt["Sharpe (rf=0)"].round(2)
    fmt["Calmar"] = fmt["Calmar"].round(2)
    show_df(fmt)
    dd = pd.DataFrame({"Buy & Hold": cum["Buy & Hold"] / cum["Buy & Hold"].cummax() - 1,
                       "Stratégie par régimes": cum["Stratégie par régimes"] / cum["Stratégie par régimes"].cummax() - 1})
    show(px.line(dd, title="Drawdowns comparés"))
    figx = px.area(expo, title="Exposition appliquée dans le temps")
    figx.update_layout(showlegend=False)
    show(figx)
    st.caption("Résultats illustratifs, non optimisés, hors fiscalité et hors glissement. "
               "Ne constitue pas un conseil en investissement.")

# ---- Conclusion ------------------------------------------------------------
with tabs[6]:
    st.subheader("Synthèse")
    msg = []
    if len(tr) > 0:
        pire = tr["P(événement)"].idxmax()
        msg.append(f"- Le régime « **{pire}** » est associé à la plus forte probabilité d'événement de risque "
                   f"(**{tr.loc[pire, 'P(événement)']:.1%}** contre {p0:.1%} en moyenne, soit un lift de "
                   f"**{tr.loc[pire, 'Lift vs base']:.2f}×**).")
    if 0 < ytrue.sum() < len(ytrue):
        msg.append(f"- AUC du score de régime : **{auc_reg:.3f}** vs **{auc_vol:.3f}** pour la volatilité seule.")
    if not strat.empty:
        msg.append(f"- Application : Sharpe **{comp.loc['Stratégie par régimes', 'Sharpe (rf=0)']:.2f}** "
                   f"vs {comp.loc['Buy & Hold', 'Sharpe (rf=0)']:.2f} (Buy & Hold) ; "
                   f"max drawdown **{comp.loc['Stratégie par régimes', 'Max drawdown']:.1%}** "
                   f"vs {comp.loc['Buy & Hold', 'Max drawdown']:.1%}.")
    st.markdown("\n".join(msg))
    st.markdown("""
**Démarche :** variables de marché → **CAH** (découverte des familles) → **K-means** (classement quotidien)
→ nommage économique des régimes → **test d'anticipation** du risque → **gestion de l'exposition**.

**Apport :** on teste si les régimes détectés *signalent une hausse probable du risque dans les jours
suivants*, sans prétendre prédire systématiquement les krachs.

**Limites & pistes :** K-means suppose des régimes sphériques et sans mémoire (alternative : modèles de
Markov cachés / mélanges gaussiens) ; dépendance au choix des fenêtres et de K ; chevauchement des
fenêtres de risque ; résultats sensibles à la période ; coûts de transaction simplifiés.
""")
    st.caption("Projet académique — Salma El Idrissi · FID 1ère année · 2024-2025")
