# pages/2_Personnalisation_client.py
# ---------------------------------------------------------------------
# Personnalisation client — SQUELETTE (UI + hooks, sans calculs)
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# --- Charte (mêmes codes que la page 1) --------------------------------
PRIMARY = "#4E26DF"
SECONDARY = "#7CEF17"

# --- Helpers / Hooks à implémenter plus tard ---------------------------
def run_backtest(profile: dict) -> dict:
    """
    TODO: implémenter le backtest avec vos portefeuilles/règles.
    Retour attendu (exemple) :
      {
        "returns": pd.Series,            # rendements périodiques du portefeuille perso
        "nav": pd.Series,                # NAV base 100
        "metrics": dict,                 # Sharpe, MaxDD, etc.
      }
    """
    return {"returns": pd.Series(dtype=float),
            "nav": pd.Series(dtype=float),
            "metrics": {}}

def run_monte_carlo(profile: dict) -> dict:
    """
    TODO: implémenter la simulation MC.
    Retour attendu (exemple) :
      {
        "paths": pd.DataFrame,           # simulations de NAV (colonnes = chemins)
        "summary": pd.DataFrame,         # stats (P5, P50, P95, VAR/CVAR $)
        "metrics": dict,                 # agrégats utiles
      }
    """
    return {"paths": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "metrics": {}}

# --- UI ----------------------------------------------------------------
st.title("🎯 Personnalisation client")

st.markdown(
    "Renseignez le **profil** ci-dessous. "
    "La page prépare les entrées pour un **backtest** et/ou une **simulation Monte Carlo**."
)

with st.form("client_inputs"):
    st.subheader("1) Profil & Contraintes")

    c1, c2, c3 = st.columns(3)
    with c1:
        patrimoine = st.number_input("Patrimoine total (USD)", min_value=0.0, value=500_000.0, step=1_000.0, format="%.2f")
        investissement = st.number_input("Montant investi (USD)", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f")
    with c2:
        horizon_annees = st.number_input("Horizon d’investissement (années)", min_value=1, value=5, step=1)
        apports_annuels = st.number_input("Apports annuels (USD)", min_value=0.0, value=0.0, step=1_000.0, format="%.2f")
    with c3:
        appetence = st.slider("Appétence au risque", 1, 10, 5, help="1 = très prudent ; 10 = très dynamique")
        dd_tol = st.slider("Tolérance drawdown max (%)", 5, 80, 30)

    st.divider()
    st.subheader("2) Paramètres d’estimation")

    c4, c5, c6 = st.columns(3)
    with c4:
        mode_calc = st.multiselect(
            "Méthodes à lancer",
            ["Backtest", "Monte Carlo"],
            default=["Backtest", "Monte Carlo"]
        )
    with c5:
        var_conf = st.slider("Confiance VaR/CVaR", 0.80, 0.995, 0.95, 0.005)
    with c6:
        freq = st.selectbox("Fréquence de calcul", ["Daily", "Weekly"], index=0)

    st.caption("Les champs ci-dessus ne déclenchent aucun calcul tant que vous n’avez pas cliqué sur **Lancer**.")
    submitted = st.form_submit_button("🚀 Lancer l’analyse personnalisée", use_container_width=True)

# --- Construction du profil (persisté en session) ----------------------
profile = {
    "patrimoine_total": patrimoine,
    "montant_investi": investissement,
    "horizon_annees": horizon_annees,
    "apports_annuels": apports_annuels,
    "appetence": appetence,
    "dd_tolerance_pct": dd_tol,
    "var_conf": float(var_conf),
    "freq": freq,
    "created": str(date.today()),
    # Hooks à enrichir: mapping portefeuille cible, contraintes ESG, etc.
}

# Sauvegarde pour usage inter-pages
st.session_state["client_profile"] = profile

# --- Lancement (sans implémentation des calculs) -----------------------
if submitted:
    results = {}

    if "Backtest" in mode_calc:
        with st.spinner("Backtest en préparation…"):
            results["backtest"] = run_backtest(profile)

    if "Monte Carlo" in mode_calc:
        with st.spinner("Monte Carlo en préparation…"):
            results["mc"] = run_monte_carlo(profile)

    st.session_state["client_results"] = results
    st.success("Entrées enregistrées. Hooks prêts pour l’implémentation des calculs.")

# --- Affichage minimal des emplacements (placeholder) ------------------
if "client_results" in st.session_state:
    res = st.session_state["client_results"]

    tabs = []
    names = []
    if "backtest" in res: names.append("Backtest")
    if "mc" in res: names.append("Monte Carlo")
    if names:
        tabs = st.tabs(names)

    if "backtest" in res:
        with tabs[names.index("Backtest")]:
            st.subheader("Résultats Backtest (placeholder)")
            bk = res["backtest"]
            st.write("• returns:", bk.get("returns").shape)
            st.write("• nav:", bk.get("nav").shape)
            st.write("• metrics:", bk.get("metrics"))

    if "mc" in res:
        with tabs[names.index("Monte Carlo")]:
            st.subheader("Résultats Monte Carlo (placeholder)")
            mc = res["mc"]
            st.write("• paths:", mc.get("paths").shape)
            st.write("• summary:", mc.get("summary").shape)
            st.write("• metrics:", mc.get("metrics"))

# --- Zone export (sera branchée sur vos générateurs PDF/DOCX) ----------
st.divider()
st.caption("Zone export (à relier plus tard aux générateurs de rapports).")
