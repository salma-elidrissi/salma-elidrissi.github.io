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
    st.header("📖 Rapport PDF"); f_ = pathlib.Path(__file__).with_name("projet_pricing_bino_trino.pdf")
    if f_.exists(): st.download_button("Télécharger le rapport", f_.read_bytes(), "projet_pricing_bino_trino.pdf", "application/pdf")
    else: st.info("Placez le fichier `rapport.pdf` à côté de `app.py` pour le rendre téléchargeable ici.")



st.markdown("""
<style>
.academic-footer {
    margin-top: 60px;
    padding: 20px;
    border-top: 1px solid #c7d2fe;
    text-align: center;
    color: #374151;
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
    <div>Réalisé par : EL IDRISSI Salma</div>
    <div>Black-Scholes · Binomial · Trinomial</div>
</div>
""", unsafe_allow_html=True)    
