import streamlit as st

st.set_page_config(page_title="Personnalisation client", layout="wide")
st.title("🧩 Personnalisation client — Backtest + Monte-Carlo")
st.info("Cette page accueillera : inputs client, backtest personnalisé et simulations Monte-Carlo.")

# Hooks / état (rien d’actif pour l’instant)
if "client_inputs" not in st.session_state:
    st.session_state["client_inputs"] = {}

with st.form("inputs_client_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        profil = st.selectbox("Profil de risque", ["Prudent", "Équilibré", "Dynamique"])
    with c2:
        horizon = st.number_input("Horizon (années)", min_value=1, max_value=50, value=5)
    with c3:
        capital = st.number_input("Capital investi ($)", min_value=1000, value=100000, step=1000)

    submitted = st.form_submit_button("Enregistrer")
    if submitted:
        st.session_state["client_inputs"] = {
            "profil": profil,
            "horizon": horizon,
            "capital": capital,
        }
        st.success("Paramètres enregistrés. (Calculs à venir)")

st.divider()
st.subheader("Résultats personnalisés (à venir)")
st.caption("Ici s’afficheront : rendement attendu en $, max drawdown estimé, VaR/CVaR en $, etc.")

