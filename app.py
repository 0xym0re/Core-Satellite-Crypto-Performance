# Core-Satellite-Crypto-Performance — v2 (rebalancing + risk metrics + gaps + Plotly + PDF)
# ----------------------------------------------------------------------------------------
# Nouveautés :
# - Rebalancing : Buy & Hold (drift), Monthly, Quarterly
# - Risk metrics : Sharpe (avec RF), Sortino, Calmar, VaR/CVaR (historiques, daily)
# - Data gaps : détection détaillée + alertes
# - Graphiques : Plotly interactifs (cumul, barres, heatmap)
# - PDF pro : logo + couleurs via ReportLab, charts Plotly exportés avec kaleido

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
from datetime import timedelta
from matplotlib.colors import LinearSegmentedColormap  # conservé si tu veux garder le cmap
from matplotlib.backends.backend_pdf import PdfPages   # utilisé avant; on bascule sur ReportLab
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Image
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------------------------
# Charte graphique (UI + PDF)
primary_color = "#4E26DF"
secondary_color = "#7CEF17"
performance_colors = ["#4E26DF", "#7CEF17", "#35434B", "#B8A8F2", "#C1E5F5", "#C3F793",
                      "#F2CFEE","#F2F2F2","#FCD9C4", "#A7C7E7", "#D4C2FC", "#F9F6B2", "#C4FCD2"]

# --------------------------------------------------------------------
# Mappings Yahoo (nettoyés)
asset_mapping = {
    "MSCI World": "URTH",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",
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

# --------------------------------------------------------------------
# Streamlit config
st.set_page_config(page_title="Alphacap", layout="wide")
st.title("Comparaison de performances d'actifs")

# --------------------------------------------------------------------
# Portefeuilles de base
portfolio_allocations = {
    "Portfolio 1": {"^GSPC": 0.60, "AGGG.L": 0.40},
    "Portfolio 2": {"^GSPC": 0.57, "AGGG.L": 0.38, "GC=F": 0.05}
}

# ------------------ Utils & Core ------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(
        tickers, start=start, end=end + pd.Timedelta(days=1),
        interval="1d", auto_adjust=False, group_by="column",
        threads=True, progress=False
    )
    # MultiIndex -> Adj Close prioritaire
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            df = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            df = data["Close"].copy()
        else:
            level0 = data.columns.levels[0][0]
            df = data[level0].copy()
    else:
        if "Adj Close" in data.columns:
            df = data["Adj Close"].to_frame(name=tickers[0])
        elif "Close" in data.columns:
            df = data["Close"].to_frame(name=tickers[0])
        else:
            df = data.to_frame(name=tickers[0])
    # Index calendrier complet
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx).sort_index()
    # Colonnes manquantes -> NA
    for t in tickers:
        if t not in df.columns:
            df[t] = pd.NA
    return df

def is_crypto_ticker(t): return t in crypto_tickers_set

def annualization_factor_for_portfolio(allocations):
    for t, w in allocations.items():
        if w > 0 and is_crypto_ticker(t):
            return 365
    return 252

def renormalize_weights_if_needed(prices_df, allocations):
    tickers = [t for t in allocations if t in prices_df.columns]
    if not tickers: return {}, []
    w = np.array([allocations[t] for t in tickers], dtype=float)
    if w.sum() <= 0: return {}, []
    w = w / w.sum()
    return dict(zip(tickers, w)), tickers

def build_portfolio3(portfolio1_alloc, crypto_global_pct, crypto_allocation_pairs):
    portfolio3 = {}
    classic_weight = 1 - crypto_global_pct / 100.0
    for t, w in portfolio1_alloc.items():
        portfolio3[t] = portfolio3.get(t, 0.0) + w * classic_weight
    for name, pct in crypto_allocation_pairs:
        ticker = crypto_mapping[name]
        weight = (pct / 100.0) * (crypto_global_pct / 100.0)
        portfolio3[ticker] = portfolio3.get(ticker, 0.0) + weight
    return portfolio3

# ------------------ Data quality / gaps ------------------------------
def detect_data_gaps(df, expected_freq="D"):
    """Retourne un DataFrame avec : %NA, nb_days, nb_na, plus long gap consécutif."""
    stats = []
    idx = df.index
    for col in df.columns:
        s = df[col]
        total = len(s)
        na = int(s.isna().sum())
        # Longest consecutive NaNs
        max_gap = 0
        current = 0
        for v in s.isna().values:
            if v:
                current += 1
                max_gap = max(max_gap, current)
            else:
                current = 0
        stats.append({
            "Ticker": col,
            "Nom": asset_names_map.get(col, col),
            "Days": total,
            "NA": na,
            "NA_%": round(100*na/total, 2) if total > 0 else np.nan,
            "Longest_NA_Streak": max_gap
        })
    return pd.DataFrame(stats).sort_values(["NA_%","Longest_NA_Streak"], ascending=False)

# ------------------ Rebalancing engine -------------------------------
def portfolio_returns_buy_and_hold(prices, allocations):
    """Buy & Hold (weights driftent) — calcule les retours quotidiens du portefeuille."""
    alloc_norm, tickers = renormalize_weights_if_needed(prices, allocations)
    if not tickers: return pd.Series(dtype=float)
    P = prices[tickers].copy()
    P = P.ffill()  # nécessaire pour ratio
    base = P.iloc[0].replace(0, np.nan)
    norm = P.divide(base)
    w = np.array([alloc_norm[t] for t in tickers], dtype=float)
    nav = (norm * w).sum(axis=1)
    rets = nav.pct_change().dropna()
    return rets

def _block_weighted_returns(block_returns, target_weights):
    """Retour pondéré avec weights constants sur un bloc (rebalance au début du bloc)."""
    cols = [c for c in block_returns.columns if c in target_weights]
    if len(cols) == 0: return pd.Series(dtype=float)
    w = np.array([target_weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (block_returns[cols] * w).sum(axis=1)

def portfolio_returns_with_rebalancing(prices, allocations, freq="M"):
    """Rebalance au début de chaque période (M/Q). Returns calendaire quotidiens concatenés."""
    alloc_norm, tickers = renormalize_weights_if_needed(prices, allocations)
    if not tickers: return pd.Series(dtype=float)
    P = prices[tickers].copy().ffill()
    R = P.pct_change().dropna(how="all")
    # dates de rebalance : début de mois/trimestre
    if freq == "M":
        keys = R.index.to_period("M")
    elif freq == "Q":
        keys = R.index.to_period("Q")
    else:
        raise ValueError("freq must be 'M' or 'Q'")
    pieces = []
    for _, g in R.groupby(keys):
        # sur ce bloc, weights cibles constants (renormalisés aux cols présentes)
        cols = [c for c in g.columns if c in alloc_norm]
        if not cols: continue
        w = np.array([alloc_norm[c] for c in cols], dtype=float)
        if w.sum() <= 0: continue
        w = w / w.sum()
        rp = (g[cols] * w).sum(axis=1)
        pieces.append(rp)
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()

# ------------------ Risk metrics -------------------------------------
def drawdown_stats(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1.0
    max_dd = dd.min() if len(dd) else np.nan
    return dd, max_dd

def compute_metrics_from_returns(r, dpy=252, rf_annual=0.0,
                                 want_sortino=True, want_calmar=True,
                                 want_var=False, want_cvar=False, var_alpha=0.95):
    """r: daily returns series (calendar)."""
    if r is None or len(r) == 0:
        return {}
    # Annualized stats (arithmétique simple)
    cum_ret = (1 + r).prod() - 1
    ann_ret = (1 + cum_ret)**(dpy/len(r)) - 1 if len(r) > 0 else np.nan
    vol = r.std() * np.sqrt(dpy)

    # Sharpe (RF annual en décimal)
    excess = ann_ret - rf_annual
    sharpe = excess/vol if vol and vol != 0 else np.nan

    # MaxDD & Calmar
    dd, max_dd = drawdown_stats(r)
    calmar = (ann_ret/abs(max_dd)) if want_calmar and max_dd and max_dd != 0 else np.nan

    # Sortino
    sortino = np.nan
    if want_sortino:
        downside = r.copy()
        downside[downside > 0] = 0
        down_stdev = downside.std() * np.sqrt(dpy)
        sortino = (excess/down_stdev) if down_stdev and down_stdev != 0 else np.nan

    # VaR / CVaR (historique, daily)
    var_val = cvar_val = np.nan
    if want_var or want_cvar:
        # on travaille sur la distribution quotidienne
        losses = -r.dropna()
        if len(losses) > 0:
            q = np.quantile(losses, var_alpha)  # ex: 95% → perte dépassée 5% du temps
            if want_var: var_val = q
            if want_cvar:
                tail = losses[losses >= q]
                cvar_val = tail.mean() if len(tail) > 0 else q

    return {
        "Annualized Return %": round(ann_ret*100, 2),
        "Cumulative Return %": round(cum_ret*100, 2),
        "Volatility %": round(vol*100, 2),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown %": round(max_dd*100, 2) if pd.notna(max_dd) else np.nan,
        "Calmar": round(calmar, 2) if pd.notna(calmar) else np.nan,
        "Sortino": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "VaR (daily)": round(var_val*100, 2) if pd.notna(var_val) else np.nan,
        "CVaR (daily)": round(cvar_val*100, 2) if pd.notna(cvar_val) else np.nan,
    }

# ------------------ Plotly charts ------------------------------------
def plot_cumulative_lines(df_prices, names_map, title):
    df_norm = df_prices.ffill().bfill()
    df_norm = df_norm / df_norm.iloc[0] * 100
    fig = go.Figure()
    for col in df_norm.columns:
        fig.add_trace(go.Scatter(
            x=df_norm.index, y=df_norm[col],
            mode='lines', name=names_map.get(col, col)
        ))
    fig.update_layout(
        title=title, xaxis_title="", yaxis_title="Base 100",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=20, r=20, t=60, b=60), template="plotly_white"
    )
    return fig

def plot_perf_bars(df_prices, names_map, title):
    df = df_prices.ffill().bfill()
    perf = (df.iloc[-1]/df.iloc[0] - 1).sort_values(ascending=False)
    fig = px.bar(
        perf.rename(index=names_map).rename("Performance"),
        text=perf.apply(lambda x: f"{x*100:.2f}%")
    )
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(
        title=title, yaxis_title="Performance", xaxis_title="",
        uniformtext_minsize=8, uniformtext_mode='hide',
        margin=dict(l=20, r=20, t=60, b=60), template="plotly_white"
    )
    return fig

def plot_heatmap_corr(df_prices, names_map, title):
    R = df_prices.ffill().bfill().pct_change().dropna(how="all")
    C = R.corr()
    C.index = [names_map.get(c, c) for c in C.index]
    C.columns = [names_map.get(c, c) for c in C.columns]
    fig = px.imshow(
        C, text_auto=".2f", aspect="auto", color_continuous_scale=["#4E26DF","#a993fa","#CAE5F5","#F2F2F2","#C3F793","#7CEF17"]
    )
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

# ------------------ PDF report (ReportLab + Plotly->PNG via kaleido) --
def fig_to_png_bytes(fig, scale=2):
    return fig.to_image(format="png", scale=scale)

def generate_pdf_report(company_name, logo_file, primary_hex, secondary_hex,
                        charts_dict, metrics_df):
    """
    charts_dict: {"Heatmap": fig, "Perf bars": fig, "Cum lines": fig, ...}
    metrics_df: pandas DataFrame (petit tableau)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    elements = []
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal = styles["Normal"]

    # Header: logo + title
    if logo_file is not None:
        try:
            logo = Image(logo_file, width=3*cm, height=3*cm)
            elements.append(logo)
        except Exception:
            pass
    elements.append(Paragraph(f"{company_name} — Portfolio Report", title_style))
    elements.append(Paragraph(" ", normal))

    # Charts
    for name, fig in charts_dict.items():
        try:
            png = fig_to_png_bytes(fig)
            img = Image(io.BytesIO(png), width=17*cm, height=9*cm)
            elements.append(Paragraph(name, styles["Heading2"]))
            elements.append(img)
            elements.append(Paragraph(" ", normal))
        except Exception:
            continue

    # Metrics table
    if metrics_df is not None and not metrics_df.empty:
        elements.append(Paragraph("Portfolio Metrics", styles["Heading2"]))
        data = [ [str(x) for x in ["Metric"] + metrics_df.columns.tolist()] ]
        for idx, row in metrics_df.iterrows():
            data.append([str(idx)] + [str(v) for v in row.values])
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), rl_colors.HexColor(primary_hex)),
            ('TEXTCOLOR',(0,0),(-1,0), rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('GRID',(0,0),(-1,-1),0.3, rl_colors.grey),
            ('ROWBACKGROUNDS',(0,1),(-1,-1), [rl_colors.whitesmoke, rl_colors.HexColor("#F7F7F7")]),
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------ UI : paramètres ----------------------------------
with st.sidebar:
    st.header("Paramètres")
    risk_free_rate_percent = st.number_input(
        "Taux sans risque annuel (%)", min_value=-5.0, max_value=20.0, value=0.0, step=0.1
    )
    rebal_mode = st.selectbox(
        "Rebalancing", ["Buy & Hold (no rebalance)", "Monthly", "Quarterly"],
        help="Buy & Hold = drift ; Monthly/Quarterly = rebalance au début de période."
    )
    risk_measures = st.multiselect(
        "Mesures de risque à afficher",
        ["Sharpe","Sortino","Calmar","VaR (daily)","CVaR (daily)"],
        default=["Sharpe","Sortino","Calmar"]
    )
    var_conf = st.slider("Confiance VaR/CVaR (daily)", 0.80, 0.995, 0.95, 0.005)
    st.divider()
    st.subheader("Rapport PDF")
    company_name = st.text_input("Nom société", "Alphacap")
    logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
    include_pdf = st.checkbox("Générer un rapport PDF à l'export", value=True)

# ------------------ UI poche crypto ---------------------------------
st.markdown("## 💼 Composition du portefeuille crypto")
crypto_options = list(crypto_mapping.keys())
crypto_allocation = []
crypto_global_pct = st.number_input("% du portefeuille total alloué à l'allocation crypto", 0.0, 100.0, 5.0, 0.5)
num_crypto = st.number_input("Nombre d'actifs cryptoactifs dans la poche", 1, 15, 1, 1)

total_pct = 0.0
for i in range(num_crypto):
    cols = st.columns([3, 1])
    with cols[0]:
        selected_crypto = st.selectbox(f"Crypto {i+1}", crypto_options, key=f"crypto_{i}")
    with cols[1]:
        pct = st.number_input(f"% de la crypto {i+1} dans la poche", 0.0, 100.0, 100.0 if i==0 and num_crypto==1 else 0.0, 0.1, key=f"pct_{i}")
    crypto_allocation.append((selected_crypto, pct))
    total_pct += pct

if not np.isclose(total_pct, 100.0, atol=0.01):
    st.warning(f"⚠️ La somme des pourcentages de la poche crypto est {total_pct:.2f}%. Elle doit être ≈ 100%.")
elif crypto_global_pct <= 0:
    st.warning("⚠️ Le pourcentage global alloué à la poche crypto doit être > 0.")
else:
    st.success("✅ Répartition valide du portefeuille.")
    portfolio3 = build_portfolio3(portfolio_allocations["Portfolio 1"], crypto_global_pct, crypto_allocation)
    portfolio_allocations["Portfolio 3"] = portfolio3

# ------------------ Sélections d'actifs, période ---------------------
available_assets = list(full_asset_mapping.keys())
selected_asset = st.selectbox("📌 Sélectionnez un actif :", available_assets)
asset_ticker = full_asset_mapping[selected_asset]

timeframes = {"1 semaine":"7d","1 mois":"30d","3 mois":"90d","6 mois":"180d","1 an":"365d","2 ans":"730d","3 ans":"1095d","5 ans":"1825d"}
period_label = st.selectbox("⏳ Période :", list(timeframes.keys()))

use_custom_period = st.checkbox("Période personnalisée")
c1, c2 = st.columns(2)
with c1:
    custom_start = st.date_input("Date de début", value=pd.Timestamp.today() - pd.Timedelta(days=30), disabled=not use_custom_period)
with c2:
    custom_end = st.date_input("Date de fin", value=pd.Timestamp.today() - pd.Timedelta(days=1), disabled=not use_custom_period)

st.markdown("**Liste des actifs à comparer**")
compare_assets = [a for a in available_assets if a != selected_asset]
preselect = ["Bitcoin (BTC$)","Ethereum (ETH$)","MSCI World","Nasdaq","S&P 500","US 10Y Yield","Dollar Index","Gold"]
safe_default = [a for a in preselect if a in compare_assets]
selected_comparisons = st.multiselect("📊 Actifs à comparer :", compare_assets, default=safe_default)
compare_tickers = [full_asset_mapping[a] for a in selected_comparisons]

# ------------------ ANALYSE ------------------------------------------
if st.button("🔎 Analyser"):
    try:
        tickers_graphiques = list(set(compare_tickers + [asset_ticker]))
        tickers_portefeuilles = set()
        for alloc in portfolio_allocations.values():
            tickers_portefeuilles.update(alloc.keys())
        tickers_dl = list(set(tickers_graphiques + list(tickers_portefeuilles)))

        # Période
        if use_custom_period:
            start_date = pd.to_datetime(custom_start); end_date = pd.to_datetime(custom_end)
        else:
            nb_days = int(timeframes[period_label].replace('d',''))
            end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
            start_date = end_date - pd.Timedelta(days=nb_days - 1)

        # Data
        df = download_prices(tickers_dl, start_date, end_date)
        # Remplissage limité aux actifs traditionnels (aligne les trous calendrier)
        traditional_tickers = [t for t in traditional_tickers_set if t in df.columns]
        if traditional_tickers:
            df[traditional_tickers] = df[traditional_tickers].ffill().bfill()

        # Gaps
        gaps = detect_data_gaps(df[tickers_graphiques])
        critical = gaps[(gaps["NA_%"] > 5) | (gaps["Longest_NA_Streak"] >= 5)]
        with st.expander("🔎 Qualité des données (gaps, NA%)", expanded=len(critical)>0):
            st.dataframe(gaps.assign(**{"NA_%":gaps["NA_%"].map(lambda x: f"{x:.2f}%")}),
                         use_container_width=True, height=220)
            if not critical.empty:
                st.warning("Certaines séries présentent des trous significatifs (NA%>5% ou streak≥5j). Vérifiez les tickers concernés.")

        # Label période
        label_period = (f"sur {period_label.lower()}" if not use_custom_period
                        else f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")

        # Graphiques Plotly
        df_graph = df[tickers_graphiques]
        fig_heat = plot_heatmap_corr(df_graph, asset_names_map, f"Matrice de corrélation {label_period}")
        fig_perf = plot_perf_bars(df_graph, asset_names_map, f"Performances cumulées {label_period}")
        fig_lines = plot_cumulative_lines(df_graph, asset_names_map, f"Évolution des actifs {label_period}")

        st.plotly_chart(fig_heat, use_container_width=True)
        st.plotly_chart(fig_perf, use_container_width=True)
        st.plotly_chart(fig_lines, use_container_width=True)

        # ---------------- Portefeuilles & métriques -------------------
        rf_annual = risk_free_rate_percent / 100.0

        def portfolio_daily_returns(prices, allocations):
            if rebal_mode.startswith("Buy"):
                return portfolio_returns_buy_and_hold(prices, allocations)
            elif rebal_mode.startswith("Monthly"):
                return portfolio_returns_with_rebalancing(prices, allocations, freq="M")
            else:
                return portfolio_returns_with_rebalancing(prices, allocations, freq="Q")

        metrics_dict = {}
        port_returns_dict = {}
        for name, alloc in portfolio_allocations.items():
            # annualization factor
            dpy = annualization_factor_for_portfolio(alloc)
            r = portfolio_daily_returns(df, alloc)
            port_returns_dict[name] = r
            want_var = "VaR (daily)" in risk_measures
            want_cvar = "CVaR (daily)" in risk_measures
            m = compute_metrics_from_returns(
                r, dpy=dpy, rf_annual=rf_annual,
                want_sortino=("Sortino" in risk_measures),
                want_calmar=("Calmar" in risk_measures),
                want_var=want_var, want_cvar=want_cvar, var_alpha=var_conf
            )
            metrics_dict[name] = m

        metrics_df = pd.DataFrame(metrics_dict)
        # réduire aux colonnes sélectionnées + basiques
        cols_order = ["Annualized Return %","Cumulative Return %","Volatility %","Sharpe"]
        if "Sortino" in risk_measures: cols_order.append("Sortino")
        if "Calmar" in risk_measures: cols_order.append("Calmar")
        if "VaR (daily)" in risk_measures: cols_order.append("VaR (daily)")
        if "CVaR (daily)" in risk_measures: cols_order.append("CVaR (daily)")
        metrics_df = metrics_df.reindex(index=cols_order)

        st.markdown("### Comparaison de portefeuilles")
        st.dataframe(metrics_df, use_container_width=True, height=320)
        st.caption(f"Rebalancing : **{rebal_mode}** | RF utilisé pour Sharpe/Sortino/Calmar : **{risk_free_rate_percent:.2f}%** (annualisé).")

        # ---------------- Export Excel & PDF --------------------------
        st.subheader("📥 Exporter les résultats")
        # Excel
        perf_pct = (df_graph.ffill().bfill()/df_graph.ffill().bfill().iloc[0]-1)*100
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.rename(columns=asset_names_map).to_excel(writer, sheet_name="Prix")
            perf_pct.rename(columns=asset_names_map).to_excel(writer, sheet_name="Performance (%)")
            metrics_df.to_excel(writer, sheet_name="Résumé Portefeuilles")
            gaps.to_excel(writer, sheet_name="Data Gaps", index=False)
        excel_buffer.seek(0)
        st.download_button("📄 Télécharger les données & métriques (.xlsx)", data=excel_buffer, file_name="donnees_completes.xlsx")

        # PDF (ReportLab)
        if include_pdf:
            charts_for_pdf = {
                "Matrice de corrélation": fig_heat,
                "Performances cumulées": fig_perf,
                "Évolution (base 100)": fig_lines
            }
            pdf_buf = generate_pdf_report(company_name, logo_file, primary_color, secondary_color,
                                          charts_for_pdf, metrics_df)
            st.download_button("🖼️ Télécharger le rapport PDF", data=pdf_buf, file_name="rapport_portefeuille.pdf", mime="application/pdf")

        # Note sur ^TNX
        if "US 10Y Yield" in selected_comparisons or selected_asset == "US 10Y Yield":
            st.info("ℹ️ 'US 10Y Yield' (^TNX) est un rendement (pas un prix). Interpréter les comparaisons avec prudence.")

    except Exception as e:
        st.error("❌ Erreur lors du chargement ou de l’analyse des données.")
        st.code(str(e))
        st.info("💡 Réessayez avec une période, un actif, ou un sous-ensemble plus restreint.")
