# Core-Satellite-Crypto-Performance — v3
# ----------------------------------------------------------------------------------------
# Nouveautés vs v2 :
# - Benchmark dédié (sélecteur) + graphique de performance relative
# - Crypto list dynamique (CoinGecko, mcap>200M) + validation Yahoo + fallback statique
# - Composition des portefeuilles affichée (ex: 60/40 ; 60/40 + X% crypto)
# - Max Drawdown remis sous la volatilité dans le tableau comparatif
# - Rapport PDF amélioré : logo non déformé (ratio), tableau plus propre, + graph "3 portefeuilles"
# - Graphs Plotly conservés (heatmap, barres, base 100) + relative vs benchmark

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io, math, requests
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys, subprocess, os, shutil
import plotly.io as pio

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import PageBreak
from PIL import Image as PILImage

def ensure_kaleido() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaleido"])
            import kaleido  # noqa: F401
            return True
        except Exception:
            return False

def _possible_chrome_paths():
    # PATH binaries
    names = ["google-chrome", "chrome", "chromium", "chromium-browser"]
    for n in names:
        p = shutil.which(n)
        if p:
            yield p
    # Emplacements classiques
    if sys.platform.startswith("win"):
        for p in [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        ]:
            if os.path.exists(p): yield p
    elif sys.platform == "darwin":
        p = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(p): yield p
    else:
        # linux containers fréquents
        for p in ["/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]:
            if os.path.exists(p): yield p

#def ensure_plotly_chrome(verbose=False) -> bool:
#    """
#    1) Cherche Chrome localement
#    2) Si absent, essaie l'installeur Plotly (CLI): plotly_get_chrome / plotly-get-chrome / python -m plotly get-chrome
#    3) Configure pio.kaleido.scope.chromium_executable + variables d'env
#    """
#    def _configure(path: str) -> bool:
#        if not path or not os.path.exists(path):
#            return False
#        os.environ["PLOTLY_CHROME"] = path   # reconnu par Plotly
#        os.environ["KAL_CHROME_PATH"] = path # pour certaines builds Kaleido
#        try:
#            # Certaines versions nécessitent de réinitialiser la scope
#            # (on la crée si elle n'existe pas encore)
#            _ = pio.kaleido.scope
#            pio.kaleido.scope.chromium_executable = path
#        except Exception:
#            pass
#        return True
#
#    # 1) déjà disponible ?
#    try:
#        exe = getattr(pio.kaleido.scope, "chromium_executable", None)
#        if exe and os.path.exists(exe):
#            return True
#    except Exception:
#        pass
#    for p in _possible_chrome_paths():
#        if _configure(p): 
#            if verbose: print(f"[plotly] Using Chrome at: {p}")
#           return True
#
#    # 2) tenter l'installation via la CLI plotly_get_chrome
#    cmds = [
#        ["plotly_get_chrome"],
#        ["plotly-get-chrome"],
#        [sys.executable, "-m", "plotly", "get-chrome"],
#    ]
#    for cmd in cmds:
#       try:
#            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
#            if verbose: print(out)
#            # après installation, re-scanne
#            for p in _possible_chrome_paths():
#                if _configure(p):
#                    return True
#       except Exception as e:
#            if verbose: print(f"[plotly] get-chrome attempt failed: {cmd} -> {e}")
#            continue
#    return False


def _to_bytes(uploaded_file):
        if not uploaded_file:
            return None
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return uploaded_file.read()


# --------------------------------------------------------------------
# Charte graphique
PRIMARY = "#4E26DF"
SECONDARY = "#7CEF17"
PERF_COLORS = ["#4E26DF","#7CEF17","#35434B","#B8A8F2","#C1E5F5","#C3F793",
               "#F2CFEE","#F2F2F2","#FCD9C4","#A7C7E7","#D4C2FC","#F9F6B2","#C4FCD2"]

# --------------------------------------------------------------------
# Mappings de base (fallback)
asset_mapping = {
    "MSCI World": "URTH",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "US 10Y Yield": "^TNX",     # rendement, pas un prix
    "Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "iShares Bonds Agregate": "AGGG.L",
}
# Liste crypto statique (fallback si API indispo)
crypto_static = {
    "Bitcoin (BTC$)": "BTC-USD",
    "Ethereum (ETH$)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "XRP (XRP)": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "TRON (TRX)": "TRX-USD",
    "Toncoin (TON)": "TON11419-USD",  # peut être indispo sur Yahoo selon régions
    "Polkadot (DOT)": "DOT-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Bitcoin Cash (BCH)": "BCH-USD",
    "Chainlink (LINK)": "LINK-USD",
    "Stellar (XLM)": "XLM-USD",
    "Monero (XMR)": "XMR-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "Aptos (APT)": "APT-USD",
    "NEAR Protocol (NEAR)": "NEAR-USD",
    "Arbitrum (ARB)": "ARB-USD",
    "Render (RNDR)": "RNDR-USD",
    "Optimism (OP)": "OP-USD",
    "Uniswap (UNI)": "UNI-USD",
    "Cosmos (ATOM)": "ATOM-USD",
    "Filecoin (FIL)": "FIL-USD",
    "Aave (AAVE)": "AAVE-USD",
    "Sui (SUI)": "SUI-USD",
    "Maker (MKR)": "MKR-USD",
    "IOTA (IOTA)": "IOTA-USD",
    "Algorand (ALGO)": "ALGO-USD",
    "VeChain (VET)": "VET-USD",
    "Injective (INJ)": "INJ-USD",
    "Celestia (TIA)": "TIA-USD",
    "Jupiter (JUP)": "JUP-USD",
    "Synthetix (SNX)": "SNX-USD",
    "The Graph (GRT)": "GRT-USD",
    "Fetch.AI (FET)": "FET-USD",
    "Hyperliquid (HYPE)": "HYPE32196-USD", 
    "Bittensor (TAO)": "TAO22974-USD", 
    "Shiba Inu (SHIB)": "SHIB-USD", 
    "Mantle (MNT)": "MNT-USD", 
    "PEPE (PEPE)": "PEPE-USD", 
    "Ondo (ONDO)": "ONDO-USD", 
    "Stacks (STX)": "STX-USD", 
    "Chiliz (CHZ)": "CHZ-USD", 
    "Raydium (RAY)": "RAY-USD", 
    "dogwifhat (WIF)": "WIF-USD", 
    "Theta (THETA)": "THETA-USD", 
    "Tezos (XTZ)": "XTZ-USD", 
    "Morpho (MORPHO)": "MORPHO-USD",
}
us_equity_mapping = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
}

# --------------------------------------------------------------------
# App config
st.set_page_config(page_title="Alphacap Digital Assets", layout="wide")
st.title("Comparaison de performances d'actifs")

# --------------------------------------------------------------------
# Portefeuilles de base
portfolio_allocations = {
    "Portfolio 1": {"^GSPC": 0.60, "AGGG.L": 0.40},
    "Portfolio 2": {"^GSPC": 0.57, "AGGG.L": 0.38, "GC=F": 0.05},
}
def portfolio1_label(): return "Portefeuille 1 (60/40)"
def portfolio2_label(): return "Portefeuille 2 (60/40 + 5% Or)"

# --------------------------------------------------------------------
# Helpers data
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    if isinstance(tickers, str): tickers = [tickers]
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
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx).sort_index()
    for t in tickers:
        if t not in df.columns:
            df[t] = pd.NA
    return df

def is_crypto_ticker(t, crypto_set):
    return t in crypto_set

def annualization_factor_for_portfolio(allocations, crypto_set):
    for t, w in allocations.items():
        if w > 0 and is_crypto_ticker(t, crypto_set):
            return 365
    return 252

def renormalize_weights_if_needed(prices_df, allocations):
    tickers = [t for t in allocations if t in prices_df.columns]
    if not tickers: return {}, []
    w = np.array([allocations[t] for t in tickers], dtype=float)
    if w.sum() <= 0: return {}, []
    w = w / w.sum()
    return dict(zip(tickers, w)), tickers

def detect_data_gaps(df):
    stats = []
    for col in df.columns:
        s = df[col]
        total = len(s)
        na = int(s.isna().sum())
        max_gap = 0; current = 0
        for v in s.isna().values:
            if v: current += 1; max_gap = max(max_gap, current)
            else: current = 0
        stats.append({
            "Ticker": col,
            "Nom": col,  # remplacé plus tard
            "Days": total,
            "NA": na,
            "NA_%": round(100*na/total, 2) if total > 0 else np.nan,
            "Longest_NA_Streak": max_gap
        })
    return pd.DataFrame(stats).sort_values(["NA_%","Longest_NA_Streak"], ascending=False)

# ------------------ Rebalancing engine -------------------------------
def portfolio_returns_buy_and_hold(prices, allocations):
    alloc_norm, tickers = renormalize_weights_if_needed(prices, allocations)
    if not tickers: return pd.Series(dtype=float)
    P = prices[tickers].copy().ffill()
    base = P.iloc[0].replace(0, np.nan)
    nav = (P.divide(base) * np.array([alloc_norm[t] for t in tickers])).sum(axis=1)
    return nav.pct_change().dropna()

def portfolio_returns_with_rebalancing(prices, allocations, freq="M"):
    alloc_norm, tickers = renormalize_weights_if_needed(prices, allocations)
    if not tickers: return pd.Series(dtype=float)
    P = prices[tickers].copy().ffill()
    R = P.pct_change().dropna(how="all")
    keys = R.index.to_period("M" if freq=="M" else "Q")
    parts = []
    w_full = np.array([alloc_norm[c] for c in tickers], dtype=float)
    for _, g in R.groupby(keys):
        cols = [c for c in g.columns if c in alloc_norm]
        if not cols: continue
        w = np.array([alloc_norm[c] for c in cols], dtype=float)
        if w.sum() <= 0: continue
        w = w / w.sum()
        parts.append((g[cols] * w).sum(axis=1))
    if not parts: return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()

def portfolio_daily_returns(prices, allocations, rebal_mode):
    if rebal_mode.startswith("Buy"):
        return portfolio_returns_buy_and_hold(prices, allocations)
    elif rebal_mode.startswith("Monthly"):
        return portfolio_returns_with_rebalancing(prices, allocations, "M")
    else:
        return portfolio_returns_with_rebalancing(prices, allocations, "Q")

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
    if r is None or len(r) == 0: return {}
    cum_ret = (1 + r).prod() - 1
    ann_ret = (1 + cum_ret)**(dpy/len(r)) - 1 if len(r) > 0 else np.nan
    vol = r.std() * np.sqrt(dpy)
    excess = ann_ret - rf_annual
    sharpe = excess/vol if vol and vol != 0 else np.nan
    dd, max_dd = drawdown_stats(r)
    calmar = (ann_ret/abs(max_dd)) if want_calmar and max_dd and max_dd != 0 else np.nan
    sortino = np.nan
    if want_sortino:
        downside = r.copy(); downside[downside > 0] = 0
        down_stdev = downside.std() * np.sqrt(dpy)
        sortino = (excess/down_stdev) if down_stdev and down_stdev != 0 else np.nan
    var_val = cvar_val = np.nan
    if want_var or want_cvar:
        losses = -r.dropna()
        if len(losses) > 0:
            q = np.quantile(losses, var_alpha)
            if want_var: var_val = q
            if want_cvar:
                tail = losses[losses >= q]
                cvar_val = tail.mean() if len(tail) > 0 else q
    return {
        "Annualized Return %": round(ann_ret*100, 2),
        "Cumulative Return %": round(cum_ret*100, 2),
        "Volatility %": round(vol*100, 2),
        "Max Drawdown %": round(max_dd*100, 2) if pd.notna(max_dd) else np.nan,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Calmar": round(calmar, 2) if pd.notna(calmar) else np.nan,
        "VaR (daily)": round(var_val*100, 2) if pd.notna(var_val) else np.nan,
        "CVaR (daily)": round(cvar_val*100, 2) if pd.notna(cvar_val) else np.nan,
    }

# ------------------ Plotly charts ------------------------------------
def plot_cumulative_lines(df_prices, names_map, title):
    df_norm = df_prices.ffill().bfill()
    df_norm = df_norm / df_norm.iloc[0] * 100
    fig = go.Figure()
    for col in df_norm.columns:
        fig.add_trace(go.Scatter(x=df_norm.index, y=df_norm[col], mode='lines',
                                 name=names_map.get(col, col)))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Base 100",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

def plot_perf_bars(df_prices, names_map, title):
    df = df_prices.ffill().bfill()
    perf = (df.iloc[-1]/df.iloc[0] - 1).sort_values(ascending=False)
    s = perf.rename(index=names_map).rename("Performance")
    fig = px.bar(s, text=s.apply(lambda x: f"{x*100:.2f}%"))
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(title=title, yaxis_title="Performance", xaxis_title="",
                      uniformtext_minsize=8, uniformtext_mode='hide',
                      margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

def plot_heatmap_corr(df_prices, names_map, title):
    R = df_prices.ffill().bfill().pct_change().dropna(how="all")
    C = R.corr()
    C.index = [names_map.get(c, c) for c in C.index]
    C.columns = [names_map.get(c, c) for c in C.columns]
    fig = px.imshow(C, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#4E26DF","#a993fa","#CAE5F5","#F2F2F2","#C3F793","#7CEF17"])
    fig.update_layout(title=title, margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

def plot_relative_vs_benchmark(df_all, benchmark_ticker, names_map, title):
    df = df_all.ffill().bfill()
    if benchmark_ticker not in df.columns:
        return go.Figure()
    bench = (df[benchmark_ticker]/df[benchmark_ticker].iloc[0])
    rel = (df.divide(bench, axis=0) - 1.0) * 100  # en %
    fig = go.Figure()
    for col in [c for c in df.columns if c != benchmark_ticker]:
        fig.add_trace(go.Scatter(x=rel.index, y=rel[col], mode='lines',
                                 name=names_map.get(col, col)))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Sur/ss perf vs benchmark (%)",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

def plot_portfolios_cum(nav_dict, title):
    fig = go.Figure()
    for name, r in nav_dict.items():
        if r is None or r.empty: continue
        cum = (1+r).cumprod()*100
        fig.add_trace(go.Scatter(x=cum.index, y=cum, mode='lines', name=name))
    fig.update_layout(title=title, xaxis_title="", yaxis_title="Base 100",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=20, r=20, t=60, b=60), template="plotly_white")
    return fig

def fig_to_png_bytes(fig, scale=2):
    """
    Exporte une figure Plotly en PNG via Kaleido UNIQUEMENT (pas besoin de Chrome).
    Tente d'installer kaleido si absent, puis exporte. Remonte l'erreur si échec.
    """
    try:
        import kaleido  # noqa: F401
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaleido"])
            import kaleido  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"Installation de kaleido impossible: {e}")

    try:
        return fig.to_image(format="png", scale=scale, engine="kaleido")
    except Exception as e:
        raise RuntimeError(f"Export Plotly→PNG (kaleido) a échoué: {e}")


def build_crypto_sleeve_nav(df_prices, crypto_allocation_pairs, crypto_mapping):
    """
    Construit la NAV (base 1.0) de la poche crypto pondérée selon la répartition choisie.
    On combine les rendements journaliers des tickers crypto sélectionnés.
    """
    if not crypto_allocation_pairs:
        return pd.Series(dtype=float)

    # Normalisation des poids à 100% (tolérant si la somme != 100)
    total_pct = sum(p for _, p in crypto_allocation_pairs) or 0.0
    if total_pct <= 0:
        return pd.Series(dtype=float)

    tw = []
    for name, pct in crypto_allocation_pairs:
        t = crypto_mapping.get(name)
        if t in df_prices.columns and pct > 0:
            tw.append((t, pct / total_pct))
    if not tw:
        return pd.Series(dtype=float)

    P = df_prices[[t for t, _ in tw]].copy().ffill().bfill()
    R = P.pct_change().dropna()
    w = np.array([w for _, w in tw], dtype=float)
    sleeve_r = (R * w).sum(axis=1)
    nav = (1.0 + sleeve_r).cumprod()
    nav.index = R.index
    return nav


def plot_crypto_sleeve_vs_benchmark(df_all, benchmark_ticker, sleeve_nav, names_map, title):
    """
    Trace la sur/sous-perf (%) de la poche crypto (seule) vs le benchmark.
    """
    df = df_all.ffill().bfill()
    if sleeve_nav is None or sleeve_nav.empty or benchmark_ticker not in df.columns:
        return go.Figure()

    # Aligner les dates sur la NAV de la poche
    bench = df[benchmark_ticker].reindex(sleeve_nav.index).dropna()
    sleeve_nav = sleeve_nav.reindex(bench.index).dropna()
    if sleeve_nav.empty or bench.empty:
        return go.Figure()

    bench_norm = bench / bench.iloc[0]
    sleeve_norm = sleeve_nav / sleeve_nav.iloc[0]
    rel = (sleeve_norm / bench_norm - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rel.index, y=rel, mode='lines',
                             name="Poche Crypto (pondérée)"))
    fig.update_layout(title=title,
                      xaxis_title="", yaxis_title="Sur/ss perf vs benchmark (%)",
                      legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=20, r=20, t=60, b=60),
                      template="plotly_white")
    return fig

# ------------------ PDF report ---------------------------------------
def keep_aspect_image(file_like, target_width_cm):
    try:
        im = PILImage.open(file_like)
        w, h = im.size
        ar = w / h
        target_w = target_width_cm * cm
        target_h = target_w / ar
        file_like.seek(0)
        return RLImage(file_like, width=target_w, height=target_h)
    except Exception:
        file_like.seek(0)
        return RLImage(file_like, width=3*cm, height=3*cm)

def generate_pdf_report(company_name, logo_file, charts_dict, metrics_df, composition_lines):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.2*cm, bottomMargin=1.2*cm)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    normal = styles["BodyText"]

    # Styles de cellules avec word-wrap
    cell_left = ParagraphStyle("cell_left", parent=normal, fontSize=9, leading=11,
                               spaceAfter=0, wordWrap="CJK", alignment=0)
    cell_right = ParagraphStyle("cell_right", parent=normal, fontSize=9, leading=11,
                                spaceAfter=0, wordWrap="CJK", alignment=2)

    # --- Header : logo + titre
    if logo_file is not None:
        try:
            logo_file.seek(0)
            logo_buf = io.BytesIO(logo_file.read())
            logo_img = keep_aspect_image(logo_buf, target_width_cm=4.5)
            elements.append(logo_img)
            elements.append(Spacer(1, 0.2*cm))
        except Exception:
            pass
    elements.append(Paragraph(f"{company_name} — Portfolio Report", title_style))
    elements.append(Spacer(1, 0.3*cm))

    # --- Compositions
    if composition_lines:
        elements.append(Paragraph("Compositions des portefeuilles", h2))
        for line in composition_lines:
            elements.append(Paragraph(line, normal))
        elements.append(Spacer(1, 0.2*cm))

    # --- Graphiques
    # Tente un pré-check Kaleido pour éviter une boucle d'échecs silencieux
    kaleido_ok = ensure_kaleido()
    for name, fig in charts_dict.items():
        try:
            png = fig_to_png_bytes(fig, scale=2)  # lèvera si impossible
            elements.append(Paragraph(name, h2))
            elements.append(RLImage(io.BytesIO(png), width=17*cm, height=9*cm))
            elements.append(Spacer(1, 0.3*cm))
        except Exception as e:
            # on laisse une trace visible dans le PDF
            elements.append(Paragraph(f"⚠️ Impossible d’exporter le graphique « {name} » : {str(e)}", normal))
            elements.append(Spacer(1, 0.2*cm))

    # --- Tableau des métriques (header vert + wrap)
    if metrics_df is not None and not metrics_df.empty:
        elements.append(Paragraph("Portfolio Metrics", h2))

        df = metrics_df.copy()
        header_cells = [Paragraph("Metric", cell_left)] + \
                       [Paragraph(str(c), cell_left) for c in df.columns.tolist()]
        data = [header_cells]
        for idx, row in df.iterrows():
            row_cells = [Paragraph(str(idx), cell_left)]
            for v in row.values:
                txt = "" if pd.isna(v) else str(v)
                # aligner à droite si nombre
                try:
                    _ = float(str(v).replace(",", "."))
                    row_cells.append(Paragraph(txt, cell_right))
                except Exception:
                    row_cells.append(Paragraph(txt, cell_left))
            data.append(row_cells)

        avail_w = A4[0] - doc.leftMargin - doc.rightMargin
        first_w = 0.35 * avail_w
        other_w = (avail_w - first_w) / max(1, len(df.columns))
        col_widths = [first_w] + [other_w]*len(df.columns)

        table = Table(data, repeatRows=1, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), rl_colors.HexColor(SECONDARY)),  # <<< vert de la charte
            ('TEXTCOLOR',(0,0),(-1,0), rl_colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,0),10),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ALIGN',(1,1),(-1,-1),'RIGHT'),
            ('ALIGN',(0,0),(0,-1),'LEFT'),
            ('GRID',(0,0),(-1,-1),0.3, rl_colors.HexColor("#DDDDDD")),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
                [rl_colors.whitesmoke, rl_colors.HexColor("#F7F7F7")]),
            ('BOTTOMPADDING',(0,0),(-1,0),6),
            ('TOPPADDING',(0,0),(-1,0),6),
            ('LEFTPADDING',(0,0),(-1,-1),4),
            ('RIGHTPADDING',(0,0),(-1,-1),4),
        ]))
        elements.append(table)

    # --- Glossaire
    elements.append(PageBreak())
    elements.append(Paragraph("Glossaire des indicateurs de risque (*)", h2))
    for line in [
        "<b>Volatilité*</b> : écart-type des rendements journaliers, annualisé (base 252).",
        "<b>Max Drawdown*</b> : pire baisse (pic-creux) cumulée sur la période.",
        "<b>Sharpe*</b> : (Rendement annualisé − Taux sans risque) / Volatilité.",
        "<b>Sortino*</b> : variante du Sharpe ne pénalisant que la volatilité baissière.",
        "<b>Calmar*</b> : Rendement annualisé / |Max Drawdown|.",
        "<b>VaR (historique, daily)*</b> : perte seuil, dépassée dans (1−α) des cas.",
        "<b>CVaR (Expected Shortfall)*</b> : perte moyenne conditionnelle au-delà de la VaR.",
    ]:
        elements.append(Paragraph(line, normal))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ------------------ UI : sidebar -------------------------------------
with st.sidebar:
    st.header("Paramètres")
    risk_free_rate_percent = st.number_input("Taux sans risque annuel (%)", -5.0, 20.0, 0.0, 0.1)
    rebal_mode = st.selectbox("Rebalancing", ["Buy & Hold (no rebalance)", "Monthly", "Quarterly"])
    risk_measures = st.multiselect("Mesures de risque à afficher",
                                   ["Sharpe","Sortino","Calmar","VaR (daily)","CVaR (daily)"],
                                   default=["Sharpe","Sortino","Calmar"])
    var_conf = st.slider("Confiance VaR/CVaR (daily)", 0.80, 0.995, 0.95, 0.005)

    st.divider()
    st.subheader("Rapport PDF")
    company_name = st.text_input("Nom société", "Alphacap Digital Assets")
    logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
    include_pdf = st.checkbox("Générer un rapport PDF à l'export", value=True)

# ------------------ Crypto list dynamique -----------------------------
@st.cache_data(ttl=24*3600, show_spinner=False)
def build_crypto_mapping_dynamic(min_mcap_usd=2e8, pages=4):
    """
    Récupère via CoinGecko la liste des cryptos (mcap USD), garde > min_mcap,
    génère tickers Yahoo 'SYMBOL-USD' et valide sommairement via yfinance (history).
    """
    out = {}
    session = requests.Session()
    for page in range(1, pages+1):
        url = ("https://api.coingecko.com/api/v3/coins/markets"
               f"?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}")
        r = session.get(url, timeout=15)
        if r.status_code != 200: break
        arr = r.json()
        if not arr: break
        for it in arr:
            mcap = it.get("market_cap")
            if mcap is None or mcap < min_mcap_usd: continue
            name = it.get("name","").strip()
            sym = (it.get("symbol","") or "").upper()
            if not sym: continue
            # candidat Yahoo
            y_ticker = f"{sym}-USD"
            # Quick validate (limité)
            try:
                hist = yf.Ticker(y_ticker).history(period="5d")
                if hist is None or hist.empty: continue
                out[f"{name} ({sym})"] = y_ticker
            except Exception:
                continue
    # exceptions / corrections possibles
    # (si besoin : corriger TON, TAO, ENA ... selon disponibilités Yahoo)
    return out

st.markdown("## 💼 Composition du portefeuille crypto")
use_dynamic_crypto = st.checkbox("Utiliser la liste crypto dynamique (mcap>200M)", value=True,
                                 help="Récupère la liste via CoinGecko et vérifie la dispo sur Yahoo. Fallback : liste statique.")

try:
    crypto_mapping = build_crypto_mapping_dynamic() if use_dynamic_crypto else crypto_static
except Exception:
    st.warning("CoinGecko indisponible — fallback sur la liste crypto statique.")
    crypto_mapping = crypto_static

# Ensemble complet + map noms
full_asset_mapping = {**asset_mapping, **crypto_mapping, **us_equity_mapping}
asset_names_map = {v: k for k, v in full_asset_mapping.items()}
crypto_tickers_set = set(crypto_mapping.values())
traditional_tickers_set = set(asset_mapping.values()) | set(us_equity_mapping.values())

# ------------------ Poche crypto -------------------------------------
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
        default_pct = 100.0 if i==0 and num_crypto==1 else 0.0
        pct = st.number_input(f"% de la crypto {i+1} dans la poche", 0.0, 100.0, default_pct, 0.1, key=f"pct_{i}")
    crypto_allocation.append((selected_crypto, pct))
    total_pct += pct

if not np.isclose(total_pct, 100.0, atol=0.01):
    st.warning(f"⚠️ La somme des pourcentages de la poche crypto est {total_pct:.2f}%. Elle doit être ≈ 100%.")
elif crypto_global_pct <= 0:
    st.warning("⚠️ Le pourcentage global alloué à la poche crypto doit être > 0.")
else:
    st.success("✅ Répartition valide du portefeuille.")
    # Portefeuille 3 (60/40 + crypto)
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
    portfolio_allocations["Portfolio 3"] = build_portfolio3(portfolio_allocations["Portfolio 1"], crypto_global_pct, crypto_allocation)

# ------------------ Sélections d'actifs & période --------------------
available_assets = list(full_asset_mapping.keys())

# >>> Benchmark (au lieu de "sélectionner un actif")
benchmark_label = st.selectbox("📌 Sélectionnez le benchmark :", available_assets, index=available_assets.index("S&P 500") if "S&P 500" in available_assets else 0)
benchmark_ticker = full_asset_mapping[benchmark_label]

timeframes = {"1 semaine":"7d","1 mois":"30d","3 mois":"90d","6 mois":"180d","1 an":"365d","2 ans":"730d","3 ans":"1095d","5 ans":"1825d"}
period_label = st.selectbox("⏳ Période :", list(timeframes.keys()))

use_custom_period = st.checkbox("Période personnalisée")
c1, c2 = st.columns(2)
with c1:
    custom_start = st.date_input("Date de début", value=pd.Timestamp.today() - pd.Timedelta(days=30), disabled=not use_custom_period)
with c2:
    custom_end = st.date_input("Date de fin", value=pd.Timestamp.today() - pd.Timedelta(days=1), disabled=not use_custom_period)

# ------------------ Comparaison d'actifs -----------------------------
st.markdown("**Liste des actifs à comparer**")
compare_assets = [a for a in available_assets]  # on compare tout ce que l'utilisateur choisit
preselect = ["Bitcoin (BTC$)","Ethereum (ETH$)","MSCI World","Nasdaq","S&P 500","US 10Y Yield","Dollar Index","Gold"]
safe_default = [a for a in preselect if a in compare_assets]
selected_comparisons = st.multiselect("📊 Actifs à comparer :", compare_assets, default=safe_default)
compare_tickers = [full_asset_mapping[a] for a in selected_comparisons if a in full_asset_mapping]

# ------------------ ANALYSE ------------------------------------------
if st.button("🔎 Analyser"):
    try:
        tickers_graphiques = sorted(set(compare_tickers + [benchmark_ticker]))
        tickers_portefeuilles = set()
        for alloc in portfolio_allocations.values():
            tickers_portefeuilles.update(alloc.keys())
        tickers_dl = sorted(set(tickers_graphiques + list(tickers_portefeuilles)))

        # Période
        if use_custom_period:
            start_date = pd.to_datetime(custom_start); end_date = pd.to_datetime(custom_end)
        else:
            nb_days = int(timeframes[period_label].replace('d',''))
            end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
            start_date = end_date - pd.Timedelta(days=nb_days - 1)

        # Data
        df = download_prices(tickers_dl, start_date, end_date)
        # Remplissage limité aux actifs "traditionnels"
        traditional_tickers = [t for t in traditional_tickers_set if t in df.columns]
        if traditional_tickers:
            df[traditional_tickers] = df[traditional_tickers].ffill().bfill()

        # Gaps
        gaps = detect_data_gaps(df[tickers_graphiques])
        gaps["Nom"] = gaps["Ticker"].map(asset_names_map).fillna(gaps["Ticker"])
        critical = gaps[(gaps["NA_%"] > 5) | (gaps["Longest_NA_Streak"] >= 5)]
        with st.expander("🔎 Qualité des données (gaps, NA%)", expanded=len(critical)>0):
            gdisp = gaps.copy(); gdisp["NA_%"] = gdisp["NA_%"].map(lambda x: f"{x:.2f}%")
            st.dataframe(gdisp[["Nom","Ticker","Days","NA","NA_%","Longest_NA_Streak"]],
                         use_container_width=True, height=220)
            if not critical.empty:
                st.warning("Certaines séries présentent des trous significatifs (NA%>5% ou streak≥5j).")

        label_period = (f"sur {period_label.lower()}" if not use_custom_period
                        else f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")

        # Graphs
        df_graph = df[tickers_graphiques]
        fig_heat = plot_heatmap_corr(df_graph, asset_names_map, f"Matrice de corrélation {label_period}")
        fig_perf = plot_perf_bars(df_graph, asset_names_map, f"Performances cumulées {label_period}")
        fig_lines = plot_cumulative_lines(df_graph, asset_names_map, f"Évolution des actifs {label_period}")
        sleeve_nav = build_crypto_sleeve_nav(df, crypto_allocation, crypto_mapping)
        fig_rel = plot_crypto_sleeve_vs_benchmark(
            df, benchmark_ticker, sleeve_nav, asset_names_map,
            f"Poche crypto vs benchmark ({asset_names_map.get(benchmark_ticker, benchmark_ticker)}) {label_period}"
)
        

        st.plotly_chart(fig_heat, use_container_width=True)
        st.plotly_chart(fig_perf, use_container_width=True)
        st.plotly_chart(fig_lines, use_container_width=True)
        st.plotly_chart(fig_rel, use_container_width=True)

        # ---------------- Portefeuilles & métriques -------------------
        rf_annual = risk_free_rate_percent / 100.0
        port_returns = {}
        port_names_display = {}

        # Noms jolis
        port_names_display["Portfolio 1"] = portfolio1_label()
        port_names_display["Portfolio 2"] = portfolio2_label()
        # Portfolio 3 (avec % crypto)
        if "Portfolio 3" in portfolio_allocations:
            port_names_display["Portfolio 3"] = f"Portefeuille 3 (60/40 + {crypto_global_pct:.0f}% Crypto)"

        # Calcul des retours quotidiens selon rebalancing
        for key, alloc in portfolio_allocations.items():
            r = portfolio_daily_returns(df, alloc, rebal_mode)
            port_returns[port_names_display.get(key, key)] = r

        # Metrics
        def want(x): return x in risk_measures
        metrics_dict = {}
        for key, alloc in portfolio_allocations.items():
            disp = port_names_display.get(key, key)
            dpy = annualization_factor_for_portfolio(alloc, crypto_tickers_set)
            r = port_returns.get(disp, pd.Series(dtype=float))
            m = compute_metrics_from_returns(
                r, dpy=dpy, rf_annual=rf_annual,
                want_sortino=want("Sortino"), want_calmar=want("Calmar"),
                want_var=want("VaR (daily)"), want_cvar=want("CVaR (daily)"), var_alpha=var_conf
            )
            metrics_dict[disp] = m
        metrics_df = pd.DataFrame(metrics_dict)

        # Ordre : Vol, MaxDD sous Vol, puis Sharpe, Sortino, Calmar, Var/CVar
        cols_order = ["Annualized Return %","Cumulative Return %","Volatility %","Max Drawdown %","Sharpe"]
        if "Sortino" in risk_measures: cols_order.append("Sortino")
        if "Calmar" in risk_measures: cols_order.append("Calmar")
        if "VaR (daily)" in risk_measures: cols_order.append("VaR (daily)")
        if "CVaR (daily)" in risk_measures: cols_order.append("CVaR (daily)")
        metrics_df = metrics_df.reindex(index=cols_order)

        st.markdown("### Comparaison de portefeuilles")
        st.dataframe(metrics_df, use_container_width=True, height=320)
        st.caption(f"Rebalancing : **{rebal_mode}** | RF utilisé : **{risk_free_rate_percent:.2f}%** (annualisé).")

        # Composition textuelle
        def alloc_to_text(alloc):
            items = []
            for t, w in sorted(alloc.items(), key=lambda x: -x[1]):
                if w <= 0: continue
                items.append(f"{asset_names_map.get(t, t)} {w*100:.1f}%")
            return ", ".join(items)

        comp_lines = []
        for key, alloc in portfolio_allocations.items():
            disp = port_names_display.get(key, key)
            comp_lines.append(f"<b>{disp}</b> : {alloc_to_text(alloc)}")
        st.markdown("**Compositions :**<br>" + "<br>".join(comp_lines), unsafe_allow_html=True)

        # Graph Perf des 3 portefeuilles
        fig_ports = plot_portfolios_cum(port_returns, f"Performance cumulée des portefeuilles ({rebal_mode})")
        st.plotly_chart(fig_ports, use_container_width=True)

            # ---------------- Export Excel & PDF --------------------------
        st.subheader("📥 Exporter les résultats")

        # On prépare tout ce qu'il faut pour exporter, et on le persiste.
        perf_pct = (df_graph.ffill().bfill()/df_graph.ffill().bfill().iloc[0]-1)*100

        charts_for_pdf = {
            "Matrice de corrélation": fig_heat,
            "Performances cumulées": fig_perf,
            "Évolution (base 100)": fig_lines,
            "Perf relative vs benchmark": fig_rel,
            "Portefeuilles (base 100)": fig_ports,
        }
        # Composition en texte simple pour PDF (sans HTML)
        comp_plain = []
        for line in comp_lines:
            p = Paragraph(line.replace("<b>","").replace("</b>",""), getSampleStyleSheet()["BodyText"])
            comp_plain.append(p.getPlainText())
 

        st.session_state["export_payload"] = {
            "df_graph": df_graph,                       # prix des tickers choisis (index date)
            "perf_pct": perf_pct,                       # perfs cumulées en %
            "metrics_df": metrics_df,                   # tableau des métriques
            "gaps": gaps,                               # contrôle qualité data
            "charts_for_pdf": charts_for_pdf,           # figures Plotly
            "comp_lines_plain": comp_plain,             # compositions en texte simple
            "company_name": company_name,
            "logo_bytes": _to_bytes(logo_file),         # bytes du logo (optionnel)
        }
        st.success("Résultats prêts pour export (Excel / PDF) dans la section en bas de page.")


        if "US 10Y Yield" in selected_comparisons or benchmark_label == "US 10Y Yield":
            st.info("ℹ️ 'US 10Y Yield' (^TNX) est un rendement (pas un prix). Interpréter les comparaisons avec prudence.")


        # ---------------- Notes / Glossaire risques -------------------
        st.markdown("---")
        st.markdown("### ℹ️ Notes sur les indicateurs de risque (*)")
        st.caption(
            "- **Volatilité*** : écart-type des rendements journaliers, annualisé (base 252). Plus élevé = plus instable.\n"
            "- **Max Drawdown*** : pire baisse en % entre un pic et le creux suivant (mesure la profondeur des pertes).\n"
            "- **Sharpe*** : (Rendement annualisé − Taux sans risque) / Volatilité. "
            "Ex.: un Sharpe de 1,0 signifie ~1 point de rendement excédentaire par point de volatilité.\n"
            "- **Sortino*** : variante du Sharpe qui ne pénalise que la volatilité **baissière** (downside deviation).\n"
            "- **Calmar*** : Rendement annualisé / |Max Drawdown|. Plus il est élevé, meilleur est le couple rendement/perte maximale.\n"
            "- **VaR*** (historique, daily) : perte **seuil** telle qu’elle n’est dépassée que dans (1−α) des cas (ex. α=95% ⇒ 5% des pires jours au-delà de la VaR).\n"
            "- **CVaR*** (ou Expected Shortfall) : **perte moyenne** conditionnelle au-delà de la VaR (mesure la gravité des pires jours)."
        )

            
    except Exception as e:
        st.error("❌ Erreur lors du chargement ou de l’analyse des données.")
        st.code(str(e))
        st.info("💡 Réessayez avec une période, un actif, ou un sous-ensemble plus restreint.")


# ================== ZONE EXPORT PERSISTANTE ==================
        if "export_payload" in st.session_state:
            payload = st.session_state["export_payload"]

            st.markdown("---")
            st.subheader("📥 Exporter les résultats")

            # --- Excel ---
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                # On renomme les colonnes à l'export (tickers -> noms)
                payload["df_graph"].rename(columns=asset_names_map).to_excel(writer, sheet_name="Prix")
                payload["perf_pct"].rename(columns=asset_names_map).to_excel(writer, sheet_name="Performance (%)")
                payload["metrics_df"].to_excel(writer, sheet_name="Résumé Portefeuilles")
                payload["gaps"].to_excel(writer, sheet_name="Data Gaps", index=False)
            excel_buffer.seek(0)
            st.download_button("📄 Télécharger les données & métriques (.xlsx)",
                               data=excel_buffer, file_name="donnees_completes.xlsx")

            # --- PDF ---
            if include_pdf:
                # 1 clic pour générer le PDF, puis un bouton de téléchargement qui persiste
                gen_pdf = st.button("🖼️ Générer le rapport PDF", key="gen_pdf")
                if gen_pdf:
                    logo_io = io.BytesIO(payload["logo_bytes"]) if payload["logo_bytes"] else None
                    pdf_buf = generate_pdf_report(
                        payload["company_name"],
                        logo_io,
                        payload["charts_for_pdf"],
                        payload["metrics_df"],
                        composition_lines=payload["comp_lines_plain"]
                    )
                    st.session_state["pdf_bytes"] = pdf_buf.getvalue()

                if "pdf_bytes" in st.session_state:
                    st.download_button("⬇️ Télécharger le rapport PDF",
                                       data=st.session_state["pdf_bytes"],
                                       file_name="rapport_portefeuille.pdf",
                                       mime="application/pdf")
