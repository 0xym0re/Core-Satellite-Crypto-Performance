import streamlit as st

st.set_page_config(page_title="Alphacap Digital Assets", layout="wide")

st.title("🏠 Accueil — Alphacap Digital Assets")
st.markdown(
    "Choisissez une section ci-dessous ou via le menu latéral."
)

# Liens vers les pages (fonctionne avec Streamlit récent). Fallback propre si indisponible.
try:
    st.page_link("pages/1_Analyses_générales.py", label="📊 Analyses générales")
    st.page_link("pages/2_Personnalisation_client.py", label="🎯 Personnalisation client")
except Exception:
    st.info("Utilisez le menu latéral « Pages » pour naviguer.")

# Option: bouton pour basculer directement (si st.switch_page dispo)
col1, col2 = st.columns(2)
with col1:
    if st.button("➡️ Ouvrir Analyses générales"):
        try:
            st.switch_page("pages/1_Analyses_générales.py")
        except Exception:
            pass
with col2:
    if st.button("➡️ Ouvrir Personnalisation client"):
        try:
            st.switch_page("pages/2_Personnalisation_client.py")
        except Exception:
            pass
