# Core-Satellite-Crypto-Performance (robustifié + Sharpe avec RF)
# ------------------------------------------------
# Correctifs majeurs :
# - Defaults multiselect sûrs (pas de clés inexistantes)
# - Mapping Yahoo nettoyé (tickers valides)
# - Téléchargement yfinance robuste + cache + Adj Close
# - ffill/bfill limité aux actifs traditionnels
# - Renormalisation des poids si colonnes manquantes
# - Annualisation adaptée : 252 (traditionnels) / 365 (cryptos)
# - Portefeuilles : facteur d'annualisation choisi selon la présence de crypto
# - st.dataframe sans .style (compat Streamlit)
# - Avertissements explicites sur tickers indisponibles
# - NOUVEAU : Taux sans risque (annualisé, %) paramétrable pour le Sharpe

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from datetime import timedelta
from matplotlib.backends.backend_pdf import PdfPages

# ------------------------------------------------
# Charte graphique
primary_color = "#4E26DF"
secondary_color = "#7CEF17"
heatmap_colors = ["#B8A8F2", "#C1E5F5", "#C3F793"]
performance_colors = ["#4E26DF", "#7CEF17", "#35434B", "#B8A8F2", "#C1E5F5", "#C3F793", "#F2CFEE","#F2F2F2","#FCD9C4", "#A7C7E7", "#D4C2FC", "#F9F6B2", "#C4FCD2"]

# ------------------------------------------------
# Mapping des actifs (Yahoo tickers)
asset_mapping = {
    "MSCI World": "URTH",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",          # attention : taux, pas un "prix"
    "Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "iShares Bonds Agregate": "AGGG.L"
}

crypto_mapping = {
    "Bitcoin (BTC$)": "BTC-USD",
    "Bitcoin (BTC€)": "BTC-EUR",
    "Ethereum (ETH$)": "ETH-USD",
    "Ethereum (ETH€)": "ETH-EUR",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Stellar (XLM)": "XLM-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "Filecoin (FIL)": "FIL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Tron (TRX)": "TRX-USD",
    "Bitcoin Cash (BCH)": "BCH-USD",
    "Monero (XMR)": "XMR-USD",
    "Uniswap (UNI)": "UNI-USD",
    "Near Protocol (NEAR)": "NEAR-USD",
    "Aave (AAVE)": "AAVE-USD",
    "Cronos (CRO)": "CRO-USD",
    "Vechain (VET)": "VET-USD",
    "Celestia (TIA)": "TIA-USD",
    "Arbitrum (ARB)": "ARB-USD",
    "Render (RNDR)": "RNDR-USD",
    "Optimism (OP)": "OP-USD",
    "Fetch.AI (FET)": "FET-USD",
    "Maker (MKR)": "MKR-USD",
    "Jupiter (JUP)": "JUP-USD",
    "Synthetix (SNX)": "SNX-USD",
    "Flux (FLUX)": "FLUX-USD",
}

us_equity_mapping = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA"
}

full_asset_mapping = {**asset_mapping, **crypto_mapping, **us_equity_mapping}
asset_names_map = {v: k for k, v in full_asset_mapping.items()}
crypto_tickers_set = set(crypto_mapping.values())
traditional_tickers_set = set(asset_mapping.values()) | set(us_equity_mapping.values())

# ------------------------------------------------
# Config Streamlit
st.set_page_config(page_title="Alphacap", layout="wide")
st.title("Comparaison de performances d'actifs")

# ------------------------------------------------
# Portefeuilles de base
portfolio_allocations = {
    "Portfolio 1": {  # 60/40
        "^GSPC": 0.60,
        "AGGG.L": 0.40
    },
    "Portfolio 2": {  # 60/40 + 5% or
        "^GSPC": 0.57,  # 95% * 60%
        "AGGG.L": 0.38, # 95% * 40%
        "GC=F": 0.05
    }
}

# ------------------------------------------------
# Helpers robustes

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    """
    Retourne un DataFrame calendar daily (index jour) d'Adj Close (ou Close fallback)
    avec toutes les colonnes demandées ; ajoute les colonnes manquantes en NA.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers, start=start, end=end + pd.Timedelta(days=1),
        interval="1d", auto_adjust=False, group_by="column", threads=True, progress=False
    )

    # MultiIndex -> privilégier Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            df = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            df = data["Close"].copy()
        else:
            first_level = data.columns.levels[0][0]
            df = data[first_level].copy()
    else:
        # Single ticker -> en DataFrame
        if "Adj Close" in data.columns:
            df = data["Adj Close"].to_frame(name=tickers[0])
        elif "Close" in data.columns:
            df = data["Close"].to_frame(name=tickers[0])
        else:
            df = data.to_frame(name=tickers[0])

    # Index calendrier
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx)

    # S'assurer que toutes les colonnes existent
    for t in tickers:
        if t not in df.columns:
            df[t] = pd.NA

    return df.sort_index()

def is_crypto_ticker(t):
    return t in crypto_tickers_set

def annualization_factor_for_portfolio(allocations):
    # 365 si au moins une crypto présente avec poids > 0, sinon 252
    for t, w in allocations.items():
        if w > 0 and is_crypto_ticker(t):
            return 365
    return 252

def renormalize_weights_if_needed(prices_df, allocations):
    # Garder seulement les tickers présents dans prices_df
    tickers = [t for t in allocations if t in prices_df.columns]
    if not tickers:
        return {}, []
    w = np.array([allocations[t] for t in tickers], dtype=float)
    if w.sum() <= 0:
        return {}, []
    w = w / w.sum()
    return dict(zip(tickers, w)), tickers

def build_portfolio3(portfolio1_alloc, crypto_global_pct, crypto_allocation_pairs):
    """
    portfolio1_alloc : dict ticker -> weight (ex: 60/40)
    crypto_global_pct : % du portefeuille total pour la poche crypto (0..100)
    crypto_allocation_pairs : [(crypto_name, pct_in_crypto_pocket), ...] (somme ~100)
    """
    portfolio3 = {}
    classic_weight = 1 - crypto_global_pct / 100.0
    # Part classique
    for t, w in portfolio1_alloc.items():
        portfolio3[t] = portfolio3.get(t, 0.0) + w * classic_weight
    # Part crypto (pondération interne)
    for name, pct in crypto_allocation_pairs:
        ticker = crypto_mapping[name]
        weight = (pct / 100.0) * (crypto_global_pct / 100.0)
        portfolio3[ticker] = portfolio3.get(ticker, 0.0) + weight
    return portfolio3

# ------------------------------------------------
# UI : paramètres généraux + poche crypto

# NOUVEAU : Taux sans risque (annualisé, en %) – utilisé dans le Sharpe
with st.sidebar:
    st.header("Paramètres")
    risk_free_rate_percent = st.number_input(
        "Taux sans risque annuel (%)",
        min_value=-5.0, max_value=20.0, value=0.0, step=0.1,
        help="Ce taux est annualisé. Il sera soustrait au rendement annualisé dans le calcul du Sharpe."
    )
    st.caption("Le Sharpe utilise : (Rendement annualisé − RF) / Volatilité annualisée.")

st.markdown("## 💼 Composition du portefeuille crypto")

crypto_options = list(crypto_mapping.keys())
crypto_allocation = []
crypto_global_pct = st.number_input(
    "% du portefeuille total alloué à l'allocation crypto",
    min_value=0.0, max_value=100.0, value=5.0, step=0.5
)
num_crypto = st.number_input(
    "Nombre d'actifs cryptoactifs dans la poche", min_value=1, max_value=15, step=1, value=1
)

total_pct = 0.0
for i in range(num_crypto):
    cols = st.columns([3, 1])
    with cols[0]:
        selected_crypto = st.selectbox(f"Crypto {i+1}", crypto_options, key=f"crypto_{i}")
    with cols[1]:
        pct = st.number_input(
            f"% de la crypto {i+1} dans la poche", min_value=0.0, max_value=100.0, step=0.1, key=f"pct_{i}"
        )
    crypto_allocation.append((selected_crypto, pct))
    total_pct += pct

# Validation
if not np.isclose(total_pct, 100.0, atol=0.01):
    st.warning(f"⚠️ La somme des pourcentages de la poche crypto est {total_pct:.2f}%. Elle doit être ≈ 100%.")
elif crypto_global_pct <= 0:
    st.warning("⚠️ Le pourcentage global alloué à la poche crypto doit être > 0.")
else:
    st.success("✅ Répartition valide du portefeuille.")

    # Crée le Portefeuille 3 dès que valide
    portfolio3 = build_portfolio3(portfolio_allocations["Portfolio 1"], crypto_global_pct, crypto_allocation)
    portfolio_allocations["Portfolio 3"] = portfolio3

# ------------------------------------------------
# Sélection d'actif
available_assets = list(full_asset_mapping.keys())
selected_asset = st.selectbox("📌 Sélectionnez un actif :", available_assets)
asset_ticker = full_asset_mapping[selected_asset]

# ------------------------------------------------
# Périodes
timeframes = {
    "1 semaine": "7d", "1 mois": "30d", "3 mois": "90d", "6 mois": "180d",
    "1 an": "365d", "2 ans": "730d", "3 ans": "1095d", "5 ans": "1825d"
}
period_label = st.selectbox("⏳ Période :", list(timeframes.keys()))

# Période personnalisée
use_custom_period = st.checkbox("Période personnalisée")
custom_col1, custom_col2 = st.columns(2)
with custom_col1:
    custom_start = st.date_input(
        "Date de début", value=pd.Timestamp.today() - pd.Timedelta(days=30), disabled=not use_custom_period
    )
with custom_col2:
    custom_end = st.date_input(
        "Date de fin", value=pd.Timestamp.today() - pd.Timedelta(days=1), disabled=not use_custom_period
    )

# ------------------------------------------------
# Actifs de comparaison
st.markdown("**Liste des actifs à comparer**")
compare_assets = [a for a in available_assets if a != selected_asset]

preselect = [
    "Bitcoin (BTC$)", "Ethereum (ETH$)",
    "MSCI World", "Nasdaq", "S&P 500",
    "US 10Y Yield", "Dollar Index", "Gold"
]
safe_default = [a for a in preselect if a in compare_assets]

selected_comparisons = st.multiselect(
    "📊 Actifs à comparer :", compare_assets, default=safe_default
)
compare_tickers = [full_asset_mapping[a] for a in selected_comparisons]

# ------------------------------------------------
# Calculs
if st.button("🔎 Analyser"):
    try:
        # Tickers nécessaires
        tickers_graphiques = list(set(compare_tickers + [asset_ticker]))
        tickers_portefeuilles = set()
        for alloc in portfolio_allocations.values():
            tickers_portefeuilles.update(alloc.keys())
        tickers_dl = list(set(tickers_graphiques + list(tickers_portefeuilles)))

        # Détermination période
        if use_custom_period:
            start_date = pd.to_datetime(custom_start)
            end_date = pd.to_datetime(custom_end)
        else:
            nb_days = int(timeframes[period_label].replace('d', ''))
            end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
            start_date = end_date - pd.Timedelta(days=nb_days - 1)

        # Téléchargement robuste
        df = download_prices(tickers_dl, start_date, end_date)

        # Détecter colonnes totalement vides (tickers invalides)
        invalid_tickers = [t for t in tickers_dl if df[t].isna().all()]
        if invalid_tickers:
            pretty = [asset_names_map.get(t, t) for t in invalid_tickers]
            st.warning("⚠️ Tickers indisponibles sur Yahoo (colonnes vides) : " + ", ".join(pretty))

        # Séparation pour graph/analyses
        df_graph = df[tickers_graphiques].copy()

        # Remplissage limité aux actifs traditionnels (pour alignement/calendrier)
        traditional_tickers = [t for t in traditional_tickers_set if t in df.columns]
        if traditional_tickers:
            df[traditional_tickers] = df[traditional_tickers].ffill().bfill()

        # Label période
        label_period = (
            f"sur {period_label.lower()}" if not use_custom_period
            else f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}"
        )

        # -------------------------------
        # MATRICE DE CORRÉLATION
        df_graph_corr = df_graph.ffill().bfill()
        returns_corr = df_graph_corr.pct_change().dropna(how="all")
        correlation_matrix = returns_corr.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")

        asset_names = {v: k for k, v in full_asset_mapping.items()}
        correlation_matrix.rename(index=asset_names, columns=asset_names, inplace=True)

        fig_width = max(4, len(correlation_matrix.columns) * 0.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom", ["#4E26DF", "#a993fa", "#CAE5F5", "#F2F2F2", "#C3F793", "#7CEF17"], N=256
        )

        sns.heatmap(
            correlation_matrix, annot=True, fmt=".2f", cmap=custom_cmap, cbar=True,
            cbar_kws={'shrink': 0.4, 'format': '%.2f'}, ax=ax,
            annot_kws={"fontsize": 5, "color": "#35434B"},
            linewidths=1, linecolor="white"
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=5, color="#35434B")
        ax.set_yticklabels(ax.get_xticklabels(), rotation=30, va='top', fontsize=5, color="#35434B")
        ax.set_title(f"Matrice de corrélation {label_period}", fontsize=7, color="#35434B", pad=10)
        fig.tight_layout(pad=2.0)
        st.pyplot(fig)

        # -------------------------------
        # PERFORMANCES CUMULÉES (chart colonnes)
        df_graph_disp = df_graph.ffill().bfill()
        df_graph_disp = df_graph_disp[df_graph_disp.columns[df_graph_disp.notna().any()]]
        performance = df_graph_disp.iloc[-1] / df_graph_disp.iloc[0] - 1
        performance = performance.sort_values(ascending=False)
        perf_pct = (df_graph_disp / df_graph_disp.iloc[0] - 1) * 100

        perf_df = performance.reset_index()
        perf_df.columns = ['Actif', 'Performance (%)']
        perf_df['Performance (%)'] = (perf_df['Performance (%)'] * 100).round(2)
        perf_df["Actif"] = perf_df["Actif"].map(asset_names_map).fillna(perf_df["Actif"])

        custom_colors = ["#7CEF17", "#4E26DF", "#35434B", "#949494", "#EDEDED", "#C3F793", "#B8A8F2", "#C1E5F5", "#F2F2F2", "#F2CFEE"]
        palette = custom_colors[:len(perf_df)] if len(perf_df) <= len(custom_colors) else custom_colors + sns.color_palette("husl", n_colors=len(perf_df) - len(custom_colors))

        fig_perf, ax_perf = plt.subplots(figsize=(5, 4))
        bars = ax_perf.bar(perf_df["Actif"], perf_df["Performance (%)"], color=palette)
        for bar in bars:
            h = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width() / 2, h + (0.2 if h >= 0 else -0.4), f'{h:.2f}%', ha='center',
                         va='bottom' if h >= 0 else 'top', fontsize=7, color='#35434B')
        ax_perf.axhline(0, color='#949494', linewidth=1, linestyle='--')
        y_range = max(abs(perf_df['Performance (%)'].max()), abs(perf_df['Performance (%)'].min())) * 1.2
        ax_perf.set_ylim(-y_range, y_range)
        ax_perf.set_xticklabels(ax_perf.get_xticklabels(), rotation=30, fontsize=6, color='#35434B')
        ax_perf.tick_params(axis='y', labelsize=6, labelcolor='#35434B')
        ax_perf.set_title(f"Performances cumulées {label_period}", fontsize=9, color="#35434B", pad=10)
        fig_perf.tight_layout(pad=2.0)
        st.pyplot(fig_perf)

        # -------------------------------
        # SÉRIES NORMALISÉES (base 100)
        df_norm = df_graph_disp / df_graph_disp.iloc[0] * 100.0
        fig_price, ax_price = plt.subplots(figsize=(6, 4))
        for idx, col in enumerate(df_norm.columns):
            color = performance_colors[idx % len(performance_colors)]
            ax_price.plot(df_norm.index, df_norm[col], label=asset_names_map.get(col, col), color=color, linewidth=1.5)
        ax_price.set_title(f"Évolution de la performance des actifs {label_period}", fontsize=11, color="#35434B", pad=10)
        ax_price.legend(fontsize=6)
        ax_price.tick_params(axis='x', labelsize=6, labelcolor="#35434B")
        ax_price.tick_params(axis='y', labelsize=6, labelcolor="#35434B")
        ax_price.grid(True, linestyle='--', alpha=0.4)
        fig_price.tight_layout(pad=2.0)
        st.pyplot(fig_price)

        # -------------------------------
        # VOLATILITÉ ANNUALISÉE par actif (252 vs 365)
        def asset_vol(series, ticker):
            s = series.dropna()
            if is_crypto_ticker(ticker):
                # crypto : returns journaliers calendaire (365)
                r = s.pct_change().dropna()
                return (r.std() * np.sqrt(365) * 100.0)
            else:
                # traditionnel : returns business days (252)
                sb = s.asfreq("B").ffill()
                rb = sb.pct_change().dropna()
                return (rb.std() * np.sqrt(252) * 100.0)

        vol_list = []
        for t in df_graph.columns:
            if df_graph[t].notna().any():
                vol_list.append((t, round(float(asset_vol(df_graph[t], t)), 2)))

        vol_df = pd.DataFrame(vol_list, columns=["Actif", "Volatilité (%)"])
        vol_df["Nom"] = vol_df["Actif"].map(asset_names_map).fillna(vol_df["Actif"])
        vol_df = vol_df.sort_values("Volatilité (%)", ascending=False)

        fig_vol, ax_vol = plt.subplots(figsize=(5, 4))
        bars_vol = ax_vol.bar(vol_df["Nom"], vol_df["Volatilité (%)"], color=palette[:len(vol_df)])
        for bar in bars_vol:
            h = bar.get_height()
            ax_vol.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f'{h:.2f}%', ha='center', va='bottom', fontsize=7, color='#35434B')
        ax_vol.set_xticklabels(ax_vol.get_xticklabels(), rotation=30, fontsize=6, color='#35434B')
        ax_vol.tick_params(axis='y', labelsize=6, labelcolor='#35434B')
        ax_vol.set_title(f"Volatilité annualisée {label_period}", fontsize=9, color="#35434B", pad=10)
        fig_vol.tight_layout(pad=2.0)
        st.pyplot(fig_vol)

        # -------------------------------
        # MÉTRIQUES PORTEFEUILLES (renormalisation + annualisation adaptée + RF)
        rf_annual = risk_free_rate_percent / 100.0  # <- NOUVEAU : RF annualisé (décimal)

        def compute_portfolio_metrics(prices, allocations, reference_returns=None):
            # Renormaliser si colonnes manquantes
            alloc_norm, tickers = renormalize_weights_if_needed(prices, allocations)
            if not tickers:
                return {
                    "Annualized Return": 0, "Cumulative Return": 0, "Volatility": 0,
                    "Sharpe Ratio": None, "Max Drawdown": 0, "Correlation with Portfolio 1": "-"
                }

            weights = np.array([alloc_norm[t] for t in tickers], dtype=float)

            # Returns calendaires
            data = prices[tickers].copy()
            returns = data.pct_change().dropna(how="all")

            if returns.empty:
                return {
                    "Annualized Return": 0, "Cumulative Return": 0, "Volatility": 0,
                    "Sharpe Ratio": None, "Max Drawdown": 0, "Correlation with Portfolio 1": "-"
                }

            weighted_returns = (returns * weights).sum(axis=1)

            # Facteur d'annualisation : 365 si crypto présente, sinon 252
            dpy = annualization_factor_for_portfolio(allocations)

            cumulative_return = (1 + weighted_returns).prod() - 1
            n = len(weighted_returns)
            annualized_return = (1 + cumulative_return) ** (dpy / n) - 1 if n > 0 else 0.0
            volatility = weighted_returns.std() * np.sqrt(dpy)

            # NOUVEAU : Sharpe avec taux sans risque annualisé (même base annuelle)
            excess_return = annualized_return - rf_annual
            sharpe = excess_return / volatility if volatility != 0 else np.nan

            cumulative = (1 + weighted_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()

            correlation = weighted_returns.corr(reference_returns) if reference_returns is not None else np.nan

            return {
                "Annualized Return": round(annualized_return * 100, 2),
                "Cumulative Return": round(cumulative_return * 100, 2),
                "Volatility": round(volatility * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Max Drawdown": round(max_drawdown * 100, 2),
                "Correlation with Portfolio 1": round(float(correlation), 2) if not np.isnan(correlation) else "-"
            }

        def analyze_all_portfolios(prices, portfolio_allocations):
            port_metrics = {}
            ref_returns = None

            crypto_label = f"Portefeuille 3 (60/40 + {crypto_global_pct:.0f}% Crypto)"
            name_mapping = {
                "Portfolio 1": "Portefeuille 1 (60/40)",
                "Portfolio 2": "Portefeuille 2 (60/40 + 5% Gold)",
                "Portfolio 3": crypto_label
            }

            for i, (name, alloc) in enumerate(portfolio_allocations.items()):
                metrics = compute_portfolio_metrics(prices, alloc, reference_returns=ref_returns)
                port_metrics[name_mapping.get(name, name)] = metrics

                # Définir la série de référence pour la corrélation
                if i == 0:
                    alloc_norm, tks = renormalize_weights_if_needed(prices, alloc)
                    if tks:
                        w = np.array([alloc_norm[t] for t in tks], dtype=float)
                        rets = prices[tks].pct_change().dropna(how="all")
                        if not rets.empty:
                            ref_returns = (rets * w).sum(axis=1)

            df_metrics = pd.DataFrame(port_metrics).T
            df_metrics_display = df_metrics.T
            return df_metrics_display

        # S'assurer que "Portfolio 3" existe si répartition crypto valide
        if "Portfolio 3" not in portfolio_allocations and np.isclose(total_pct, 100.0, atol=0.01) and crypto_global_pct > 0:
            portfolio_allocations["Portfolio 3"] = build_portfolio3(portfolio_allocations["Portfolio 1"], crypto_global_pct, crypto_allocation)

        df_portfolios = analyze_all_portfolios(df, portfolio_allocations)

        # Renommer colonnes pour exports (tickers -> noms)
        df_export = df.copy()
        df_export.rename(columns=asset_names_map, inplace=True)

        # -------------------------------
        # TABLEAU COMPARAISON PORTEFEUILLES
        st.markdown("### Comparaison de portefeuilles")
        st.dataframe(df_portfolios, use_container_width=True, height=320)
        st.caption(f"Taux sans risque utilisé pour le Sharpe : {risk_free_rate_percent:.2f}% (annualisé).")

        # -------------------------------
        # EXPORTS
        st.subheader("📥 Exporter les résultats")

        # Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, sheet_name="Prix")
            perf_pct.to_excel(writer, sheet_name="Performance (%)")
            vol_df[["Nom", "Volatilité (%)"]].to_excel(writer, sheet_name="Volatilité (%)", index=False)
            df_portfolios.to_excel(writer, sheet_name="Résumé Portefeuilles")
        excel_buffer.seek(0)
        st.download_button("📄 Télécharger les données complètes (.xlsx)", data=excel_buffer, file_name="donnees_completes.xlsx")

        # PDF multi-graph
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdfp:
            for fig_to_save in [fig, fig_perf, fig_price, fig_vol]:
                fig_to_save.tight_layout()
                pdfp.savefig(fig_to_save)
        pdf_buffer.seek(0)
        st.download_button(
            "🖼️ Télécharger les graphiques en PDF",
            data=pdf_buffer,
            file_name="graphique_actifs.pdf",
            mime="application/pdf"
        )

        # Infos complémentaires
        if "US 10Y Yield" in selected_comparisons or selected_asset == "US 10Y Yield":
            st.info("ℹ️ Note : 'US 10Y Yield' (^TNX) est un rendement, pas un prix. Les comparaisons de performance/volatilité avec des actifs 'prix' doivent être interprétées avec prudence.")

    except Exception as e:
        st.error("❌ Erreur lors du chargement ou de l’analyse des données.")
        st.code(str(e))
        st.info("💡 Réessayez avec une période ou un actif différent.")

