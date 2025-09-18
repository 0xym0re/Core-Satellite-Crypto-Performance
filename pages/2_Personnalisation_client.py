# pages/2_Personnalisation_client.py
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt

from io import StringIO
from scipy.stats import norm
from datetime import timedelta, date
from math import sqrt
from pages.shared_assets import asset_mapping, crypto_static, us_equity_mapping
from pages.shared_quant import (
    download_prices, portfolio_daily_returns, compute_metrics_from_returns
)

# --- Charte (mêmes codes que la page 1) --------------------------------
PRIMARY = "#4E26DF"
SECONDARY = "#7CEF17"

# --- Helpers (dérivés de la page 1, versions compactes) ----------------

def download_prices_simple(tickers, start, end):
    if isinstance(tickers, str): tickers = [tickers]
    data = yf.download(
        tickers, start=start, end=end + pd.Timedelta(days=1),
        interval="1d", auto_adjust=False, group_by="column",
        threads=True, progress=False
    )
    # Gère MultiIndex vs simple
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            df = data["Adj Close"].copy()
        elif "Close" in data.columns.get_level_values(0):
            df = data["Close"].copy()
        else:
            lvl0 = data.columns.levels[0][0]
            df = data[lvl0].copy()
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close" if "Close" in data.columns else None
        if col:
            df = data[col].to_frame(name=tickers[0])
        else:
            df = data.to_frame(name=tickers[0])

    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx).sort_index()
    for t in tickers:
        if t not in df.columns:
            df[t] = pd.NA
    return df

def align_to_business_days(df): return df.resample("B").last().ffill()
def align_to_weekly(df, rule="W-FRI"): return df.resample(rule).last().ffill()

def normalize_clock(df, mode):
    if mode == "Daily":  return align_to_business_days(df), 252
    else:                return align_to_weekly(df), 52

def renormalize_weights_if_needed(prices_df, allocations):
    tickers = [t for t in allocations if t in prices_df.columns]
    if not tickers: return {}, []
    w = np.array([allocations[t] for t in tickers], dtype=float)
    s = w.sum()
    if s <= 0: return {}, []
    w = w / s
    return dict(zip(tickers, w)), tickers

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
    for _, g in R.groupby(keys):
        cols = [c for c in g.columns if c in alloc_norm]
        if not cols: continue
        w = np.array([alloc_norm[c] for c in cols], dtype=float)
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

def drawdown_stats(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1.0
    max_dd = dd.min() if len(dd) else np.nan
    return dd, max_dd

def run_backtest(profile: dict) -> dict:
    try:
        alloc = dict(profile.get("custom_alloc", {}))
        tickers = sorted(alloc.keys())
        if not tickers:
            return {"returns": pd.Series(dtype=float), "nav": pd.Series(dtype=float), "metrics": {}}

        # Fenêtre de données = max(5 ans, horizon backtest choisi)
        dpy0 = 252 if profile["freq_backtest"]=="Daily" else 52
        window_years = max(5, float(profile["horizon_backtest_annees"]))
        window_days = int(window_years * (252 if profile["freq_backtest"]=="Daily" else 52) * (252/dpy0))
        end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=window_days)

        raw = download_prices_simple(tickers, start_date, end_date)
        df, dpy = normalize_clock(raw, profile["freq_backtest"])
        r = portfolio_daily_returns(df, alloc, profile["rebal_mode"])
        nav = (1 + r).cumprod() * 100.0

        # métriques simples (Sharpe, Vol, MDD, CAGR)
        metrics = compute_metrics_from_returns(r, dpy=dpy, rf_annual=0.0)

        # VaR/CVaR en $ à l'horizon d'investissement (utilise l'horizon MC)
        T = int(profile["horizon_mc_annees"] * (252 if profile["freq"]=="Daily" else 52))
        mu_d, sig_d = r.mean(), r.std()
        if not np.isnan(mu_d) and not np.isnan(sig_d) and T>0:
            alpha = profile["var_conf"]
            z = norm.ppf(1 - alpha)
            ret_h = mu_d*T + z*sig_d*np.sqrt(T)
            var_pct = -ret_h
            cvar_pct = -(mu_d*T - sig_d*np.sqrt(T) * norm.pdf(z)/(1-alpha))
            metrics.update({
                f"VaR {int(alpha*100)}% ($)": round(profile["montant_investi"]*max(0.0, var_pct), 2),
                f"CVaR {int(alpha*100)}% ($)": round(profile["montant_investi"]*max(0.0, cvar_pct), 2)
            })

        return {"returns": r, "nav": nav, "metrics": metrics}
    except Exception as e:
        st.warning(f"Backtest: {e}")
        return {"returns": pd.Series(dtype=float), "nav": pd.Series(dtype=float), "metrics": {}}

def run_monte_carlo(profile: dict) -> dict:
    try:
        # 1) Calibrage mu/sigma sur le backtest (portefeuille custom)
        bk = run_backtest(profile)
        r = bk["returns"]
        if r.empty:
            return {"paths": pd.DataFrame(), "summary": pd.DataFrame(), "metrics": {}}

        mu_d, sig_d = r.mean(), r.std()
        capital0 = float(profile["montant_investi"])
        T = int(profile["horizon_mc_annees"] * (252 if profile["freq"]=="Daily" else 52))
        N = int(profile["mc_paths"])
        np.random.seed(profile["seed"])

        # 2) Simulations
        if profile["mc_model"] == "GBM":
            eps = np.random.normal(loc=mu_d, scale=sig_d, size=(T, N))
            paths_ret = eps
        else:
            series = r.values
            if len(series) < 20:
                eps = np.random.normal(loc=mu_d, scale=sig_d, size=(T, N))
                paths_ret = eps
            else:
                B = int(profile["mc_block"])
                paths_ret = np.zeros((T, N))
                for j in range(N):
                    out = []
                    while len(out) < T:
                        start = np.random.randint(0, max(1, len(series)-B))
                        blk = series[start:start+B]
                        out.extend(blk)
                    paths_ret[:, j] = np.array(out[:T])

        # 3) NAV chemins (en $)
        nav_paths = np.empty((T+1, N), dtype=float)
        nav_paths[0, :] = capital0
        nav_paths[1:, :] = capital0 * np.cumprod(1 + paths_ret, axis=0)

        # 4) Fan chart percentiles
        prc = [5, 25, 50, 75, 95]
        fan = pd.DataFrame({f"P{p}": np.percentile(nav_paths, p, axis=1) for p in prc})

        # 5) Distribution finale + VaR/CVaR numéraire
        terminal = nav_paths[-1, :]
        alpha = profile["var_conf"]
        q = np.quantile(terminal, 1 - alpha)
        losses = np.clip(capital0 - terminal, a_min=0, a_max=None)
        var_usd = float(capital0 - q)
        cvar_usd = float(losses[terminal <= q].mean() if np.any(terminal <= q) else var_usd)

        # 6) Drawdowns par trajectoire (MDD)
        cummax_paths = np.maximum.accumulate(nav_paths, axis=0)
        drawdowns = nav_paths / cummax_paths - 1.0
        mdd_per_path = drawdowns.min(axis=0)            # valeurs négatives
        mdd_perc_per_path = -mdd_per_path * 100.0       # en %
        mdd_mean_pct = float(mdd_perc_per_path.mean())
        mdd_p95_pct  = float(np.percentile(mdd_perc_per_path, 95))

        # 7) CAGR par trajectoire
        years = float(profile["horizon_mc_annees"])
        cagr_per_path = (terminal / capital0) ** (1.0 / max(1e-9, years)) - 1.0
        cagr_median_pct = float(np.median(cagr_per_path) * 100.0)

        summary = pd.DataFrame({
            "Metric": [
                "Capital initial", "Espérance finale",
                f"VaR {int(alpha*100)}%", f"CVaR {int(alpha*100)}%",
                "P5", "P50", "P95",
                "Max drawdown moyen (%)", "Max drawdown P95 (%)",
                "CAGR médian (%)"
            ],
            "Value ($)": [
                round(capital0,2),
                round(float(terminal.mean()),2),
                round(var_usd,2),
                round(cvar_usd,2),
                round(float(np.percentile(terminal, 5)),2),
                round(float(np.percentile(terminal, 50)),2),
                round(float(np.percentile(terminal, 95)),2),
                None, None, None
            ]
        })

        # Ajout des valeurs en % dans une seconde colonne pour lisibilité
        summary["Value (%)]"] = [
            None, None, None, None, None, None, None,
            round(mdd_mean_pct, 2), round(mdd_p95_pct, 2),
            round(cagr_median_pct, 2)
        ]

        metrics = {
            "mu_daily": float(mu_d),
            "sigma_daily": float(sig_d),
            "horizon_steps": T,
            "n_paths": N,
            "model": profile["mc_model"],
            "mdd_mean_pct": mdd_mean_pct,
            "mdd_p95_pct": mdd_p95_pct,
            "cagr_median_pct": cagr_median_pct
        }

        paths_df = pd.DataFrame(nav_paths)
        fan["t"] = np.arange(fan.shape[0])

        return {"paths": paths_df, "summary": summary, "metrics": metrics, "fan": fan}
    except Exception as e:
        st.warning(f"Monte Carlo: {e}")
        return {"paths": pd.DataFrame(), "summary": pd.DataFrame(), "metrics": {}}


# --- UI ----------------------------------------------------------------
st.title("🎯 Personnalisation client")

st.markdown(
    "Renseignez le **profil** ci-dessous. "
    "La page prépare les entrées pour un **backtest** et/ou une **simulation Monte Carlo**."
)

st.subheader("Profil & Contraintes")

c1, c2, c3 = st.columns(3)
with c1:
    patrimoine = st.number_input(
        "Patrimoine total (USD)", min_value=0.0, value=500_000.0, step=1_000.0, format="%.2f",
        help="Valeur totale du patrimoine du client (utile pour le contexte, ne change pas les calculs)."
    )
    investissement = st.number_input(
        "Montant investi (USD)", min_value=0.0, value=100_000.0, step=1_000.0, format="%.2f",
        help="Capital de départ utilisé pour le backtest et la simulation Monte Carlo (pour VaR/CVaR en $)."
    )
with c2:
    horizon_backtest_annees = st.number_input(
        "Horizon **historique** backtest (années)", min_value=1, value=5, step=1,
        help="Longueur de l'historique utilisé pour calibrer le portefeuille (min 5 ans appliqué)."
    )
    horizon_mc_annees = st.number_input(
        "Horizon **prévision** Monte Carlo (années)", min_value=1, value=5, step=1,
        help="Durée de projection des simulations."
    )
    apports_annuels = st.number_input(
        "Versements complémentaires (USD)", min_value=0.0, value=0.0, step=1_000.0, format="%.2f",
        help="(Optionnel) Versements annuels supplémentaires. (Non utilisés pour l’instant dans le calcul.)"
    )
with c3:
    objectif = st.radio(
        "Objectif principal",
        ["Tolérance drawdown max (%)", "Rendement annuel attendu (%)"],
        index=0,
        help="Critère prioritaire évalué **sur les simulations Monte Carlo**."
    )
    dd_tol = st.slider(
        "Tolérance drawdown max (%)", 5, 80, 30,
        help="Perte max tolérée ; comparée au MDD P95 des simulations.",
        disabled = (objectif != "Tolérance drawdown max (%)")
    )
    expected_return_pct = st.number_input(
        "Rendement annuel attendu (%)", min_value=-50.0, value=6.0, step=0.5, format="%.2f",
        help="Comparé à la CAGR médiane simulée.",
        disabled = (objectif != "Rendement annuel attendu (%)")
    )

st.divider()
st.subheader("Patrimoine du client")

# Univers d'actifs
full_asset_mapping = {**asset_mapping, **crypto_static, **us_equity_mapping}
asset_names_map = {v: k for k, v in full_asset_mapping.items()}

# Nombre d'actifs & éditeur de poids (hors form -> feedback live)
n_assets = st.number_input(
    "Nombre d'actifs dans le portefeuille", 1, 20, 3, 1,
    help="Ajoute des lignes pour sélectionner les actifs et leurs poids."
)
custom_alloc_pairs = []
used = set()

for i in range(int(n_assets)):
    cA, cB = st.columns([3, 1])
    with cA:
        choice = st.selectbox(
            f"Actif {i+1}",
            list(full_asset_mapping.keys()),
            key=f"cust_asset_{i}",
            help="Choisis l’actif par son nom. Les données viennent de Yahoo Finance."
        )
    with cB:
        w = st.number_input(
            f"% poids {i+1}", 0.0, 100.0, 0.0, 0.1, key=f"cust_w_{i}",
            help="Poids de l’actif dans le portefeuille (en %). La somme doit être 100%."
        )
    if choice in full_asset_mapping and full_asset_mapping[choice] not in used:
        custom_alloc_pairs.append((full_asset_mapping[choice], w/100.0))
        used.add(full_asset_mapping[choice])

# Feedback live sur la somme des poids
sum_w = round(sum(w for _, w in custom_alloc_pairs)*100, 2)
if np.isclose(sum_w, 100.0, atol=0.01):
    st.success("✅ La somme des poids est bien de 100%.")
else:
    st.warning(f"⚠️ La somme des poids est de {sum_w:.2f}%, elle doit être 100%.")

custom_alloc = {t: w for t, w in custom_alloc_pairs if w > 0}

# --- Poche crypto (financée par "Montant investi (USD)") -------------------
st.markdown("### Allocation crypto appliquée au Montant investi")
crypto_tickers_set = set(crypto_static.values())  # pour filtrer la base non-crypto

nb_crypto = st.number_input("Nombre d'actifs crypto", 0, 15, 0, 1,
                            help="Sélectionne les cryptoactifs à acheter avec le Montant investi.")
crypto_pairs = []
for i in range(int(nb_crypto)):
    c1_, c2_ = st.columns([3, 1])
    with c1_:
        cchoice = st.selectbox(
            f"Crypto {i+1}",
            list(crypto_static.keys()),
            key=f"crypto_sel_{i}",
            help="Uniquement des cryptoactifs (liste statique)."
        )
    with c2_:
        cw = st.number_input(
            f"% crypto {i+1}", 0.0, 100.0, 0.0, 0.1,
            key=f"crypto_w_{i}",
            help="Répartition de la poche crypto (somme ≈ 100%)."
        )
    if cchoice in crypto_static:
        crypto_pairs.append((crypto_static[cchoice], cw/100.0))

sum_cw = round(sum(w for _, w in crypto_pairs)*100, 2)
if nb_crypto > 0:
    if np.isclose(sum_cw, 100.0, atol=0.01):
        st.success("✅ La somme des poids crypto est bien de 100%.")
    else:
        st.warning(f"⚠️ La somme des poids crypto est de {sum_cw:.2f}%, elle doit être ≈ 100%.")
crypto_alloc = {t: w for t, w in crypto_pairs if w > 0}

st.divider()
st.subheader("Paramètres d’estimation")

c4, c5 = st.columns(2)
with c4:
    var_conf = st.slider(
        "Confiance VaR/CVaR", 0.80, 0.995, 0.95, 0.005,
        help="Niveau de confiance pour la VaR/CVaR. 0.95 = perte dépassée dans 5% des cas."
    )
with c5:
    freq = st.selectbox(
        "Fréquence de calcul", ["Daily", "Weekly"], index=0,
        help="Fréquence des rendements/échantillonnage : 'Daily' ≈ 252 j/an ; 'Weekly' ≈ 52 sem/an."
    )

st.divider()
st.subheader("Backtest & Monte Carlo — réglages")

# Backtest
rebal_mode = st.selectbox(
    "Rebalancing", ["Buy & Hold (no rebalance)", "Monthly", "Quarterly"], index=1,
    help="Buy&Hold = pas de rééquilibrage ; Monthly/Quarterly = rééquilibrage périodique aux poids cibles."
)

# Monte Carlo
mc_model = st.selectbox(
    "Modèle Monte Carlo",
    ["GBM (normal i.i.d.)", "Block bootstrap"],
    index=0,
    help="GBM: tirages normaux i.i.d. calibrés sur mu/sigma historiques. "
         "Block bootstrap: ré-échantillonnage par blocs des rendements historiques (préserve des dépendances locales)."
)
mc_paths = st.number_input(
    "N (nombre de chemins)", 100, 20000, 2000, 100,
    help="Nombre de trajectoires simulées. 2 000–10 000 = compromis précision/temps."
)
seed = st.number_input(
    "Seed (graine aléatoire)", 0, 10**6, 42, 1,
    help="Pour rendre les simulations reproductibles."
)
if mc_model == "Block bootstrap":
    mc_block = st.number_input(
        "Taille de bloc (jours/semaines)", 5, 60, 20, 1,
        help="Longueur des blocs consécutifs rééchantillonnés. Plus grand = plus d’autocorrélations conservées."
    )
else:
    mc_block = 20  # non utilisé en GBM

# Bouton RUN (hors form)
run_clicked = st.button("🚀 Lancer l’analyse personnalisée", use_container_width=True)

# --- Construction du profil (persisté en session) ----------------------
profile = {
    "patrimoine_total": patrimoine,
    "montant_investi": investissement,
    "horizon_backtest_annees": float(horizon_backtest_annees),
    "horizon_mc_annees": float(horizon_mc_annees),
    "apports_annuels": apports_annuels,
    "dd_tolerance_pct": dd_tol,
    "var_conf": float(var_conf),
    "freq": freq,              # fréquence pour MC & métriques
    "freq_backtest": freq,     # aligne le backtest sur cette fréquence
    "rebal_mode": rebal_mode,
    "created": str(date.today()),
    "custom_alloc": custom_alloc,
    "objective": ("MDD" if objectif.startswith("Tolérance") else "TARGET_RETURN"),
    "expected_return_annual": float(expected_return_pct)/100.0,

    # Monte Carlo
    "mc_model": ("GBM" if mc_model.startswith("GBM") else "BOOT"),
    "mc_paths": int(mc_paths),
    "mc_block": int(mc_block),
    "seed": int(seed),
}

# Sauvegarde pour usage inter-pages
st.session_state["client_profile"] = profile

# --- Lancement ---------------------------------------------------------
if run_clicked:
    # bornes temporelles basiques pour le backtest
    end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
    start_date = end_date - pd.Timedelta(days=int(horizon_backtest_annees*365))

    # Portefeuilles benchmark
    bench_60_40 = {"^GSPC": 0.60, "AGGG.L": 0.40}
    bench_60_40_gold = {"^GSPC": 0.57, "AGGG.L": 0.38, "GC=F": 0.05}

    # Tickers à télécharger (custom + crypto + benchmarks)
    all_tickers = set(custom_alloc.keys()) | set(crypto_alloc.keys()) | set(bench_60_40.keys()) | set(bench_60_40_gold.keys())
    df = download_prices(sorted(all_tickers), start_date, end_date)

    # Fréquence
    if profile["freq"] == "Daily":
        dpy = 252
        dfA = df.resample("B").last().ffill()
    else:
        dpy = 52
        dfA = df.resample("W-FRI").last().ffill()

    # ------------------ 4 PORTEFEUILLES ------------------
    # 1) Patrimoine (hors crypto)
    crypto_tickers_set = set(crypto_static.values())
    base_alloc_non_crypto = {t: w for t, w in custom_alloc.items() if t not in crypto_tickers_set}
    s_base = sum(base_alloc_non_crypto.values())
    if s_base > 0:
        base_alloc_non_crypto = {t: w/s_base for t, w in base_alloc_non_crypto.items()}

    r_base = portfolio_daily_returns(dfA, base_alloc_non_crypto, rebal_mode) if base_alloc_non_crypto else pd.Series(dtype=float)

    # 2) Poche crypto
    r_crypto = portfolio_daily_returns(dfA, crypto_alloc, rebal_mode) if crypto_alloc else pd.Series(dtype=float)

    # NAV en $ pour combiner patrimoine et crypto
    def nav_usd_from_returns(r, capital0):
        if r is None or r.empty or capital0 <= 0:
            return pd.Series(dtype=float)
        nav = (1 + r).cumprod() * float(capital0)
        nav.name = "NAV"
        return nav

    nav_base_usd = nav_usd_from_returns(r_base, patrimoine)
    nav_crypto_usd = nav_usd_from_returns(r_crypto, investissement)

    # Portefeuille 2 = Patrimoine + Crypto (overlay en $)
    if not nav_base_usd.empty and not nav_crypto_usd.empty:
        idx = nav_base_usd.index.intersection(nav_crypto_usd.index)
        nav_combined = nav_base_usd.reindex(idx).ffill() + nav_crypto_usd.reindex(idx).ffill()
        r_combined = nav_combined.pct_change().dropna()
    elif not nav_base_usd.empty:
        r_combined = r_base.copy()
    elif not nav_crypto_usd.empty:
        r_combined = r_crypto.copy()
    else:
        r_combined = pd.Series(dtype=float)

    # 3) 60/40  |  4) 60/40 + 5% Or
    r_60_40 = portfolio_daily_returns(dfA, bench_60_40, rebal_mode)
    r_60_40_gold = portfolio_daily_returns(dfA, bench_60_40_gold, rebal_mode)

    # Dictionnaire des retours (pour tableaux/graphs)
    port_returns = {
        "Patrimoine (hors crypto)": r_base,
        "Patrimoine + Crypto (overlay)": r_combined,
        "60/40": r_60_40,
        "60/40 + 5% Or": r_60_40_gold
    }

    # Metrics
    metrics = {}
    for name, r in port_returns.items():
        metrics[name] = compute_metrics_from_returns(
            r, dpy=dpy, rf_annual=0.0,
            want_sortino=True, want_calmar=True, want_var=True, want_cvar=True, var_alpha=profile["var_conf"]
        )
    metrics_df = pd.DataFrame(metrics)

    # Composition capital-pondérée du portefeuille "Patrimoine + Crypto"
    asset_names_map = {v: k for k, v in full_asset_mapping.items()}
    tot_cap = float(patrimoine) + float(investissement)
    w_base_cap = (float(patrimoine)/tot_cap) if tot_cap > 0 else 0.0
    w_crypto_cap = (float(investissement)/tot_cap) if tot_cap > 0 else 0.0

    combined_alloc = {}
    for t, w in base_alloc_non_crypto.items():
        combined_alloc[t] = combined_alloc.get(t, 0.0) + w_base_cap * w
    for t, w in crypto_alloc.items():
        combined_alloc[t] = combined_alloc.get(t, 0.0) + w_crypto_cap * w

    ports = {
        "Patrimoine (hors crypto)": base_alloc_non_crypto,
        "Patrimoine + Crypto (overlay)": combined_alloc,
        "60/40": bench_60_40,
        "60/40 + 5% Or": bench_60_40_gold
    }
    
    # Monte Carlo (calibré sur l'historique du portefeuille personnalisé)
    mc = run_monte_carlo(profile)

    st.session_state["client_results"] = {
        "backtest": {
            "returns": port_returns,
            "metrics_df": metrics_df,
            "ports": ports,
        },
        "mc": mc
    }

    # --- Contrôle de cohérence basé sur Monte Carlo ----------------------
    mc_metrics = mc.get("metrics", {})
    if profile["objective"] == "MDD":
        mdd_p95 = mc_metrics.get("mdd_p95_pct", np.nan)
        ok = (not np.isnan(mdd_p95)) and (mdd_p95 <= float(profile["dd_tolerance_pct"]))
        msg = f"MDD P95 simulé : {mdd_p95:.1f}%  |  Tolérance : {profile['dd_tolerance_pct']:.0f}%."
        if ok:
            st.success("✅ Cohérent (Monte Carlo) — " + msg)
        else:
            st.error("❌ Non cohérent (Monte Carlo) — " + msg)
    else:
        cagr_median = mc_metrics.get("cagr_median_pct", np.nan)
        target = float(profile["expected_return_annual"]) * 100.0
        ok = (not np.isnan(cagr_median)) and (cagr_median >= target)
        msg = f"CAGR médiane simulée : {cagr_median:.1f}%  |  Objectif : {target:.1f}%."
        if ok:
            st.success("✅ Cohérent (Monte Carlo) — " + msg)
        else:
            st.error("❌ Non cohérent (Monte Carlo) — " + msg)

    st.success("Backtest + Monte Carlo terminés.")

# --- Affichage minimal des emplacements (placeholder) ------------------
if "client_results" in st.session_state:
    res = st.session_state["client_results"]

    tab_backtest, tab_mc = st.tabs(["Backtest", "Monte Carlo"])

    with tab_backtest:
        st.subheader("Métriques (4 portefeuilles)")
        st.dataframe(res["backtest"]["metrics_df"], use_container_width=True)

        ports_saved = res["backtest"].get("ports", {})
        if ports_saved:
            def alloc_to_text(alloc):
                items = []
                for t, w in sorted(alloc.items(), key=lambda x: -x[1]):
                    if w <= 0:
                        continue
                    items.append(f"{asset_names_map.get(t, t)} {w*100:.1f}%")
                return ", ".join(items)

            comp_lines = []
            for name, alloc in ports_saved.items():
                comp_lines.append(f"<b>{name}</b> : {alloc_to_text(alloc)}")
            st.markdown("**Compositions :**<br>" + "<br>".join(comp_lines), unsafe_allow_html=True)

        st.subheader("Performance du patrimoine contre les deux portefeuilles Benchmark")
        port_returns = res["backtest"]["returns"]
        nav_df = pd.DataFrame({k: (1+v).cumprod()*100 for k, v in port_returns.items() if v is not None})
        st.line_chart(nav_df)

    with tab_mc:
        st.subheader("Résumé Monte Carlo")
        st.dataframe(res["mc"]["summary"], use_container_width=True)

        # Trajectoires simulées (sans légende par trajectoire)
        if "paths" in res["mc"] and not res["mc"]["paths"].empty:
            paths = res["mc"]["paths"]
            n_show = min(50, paths.shape[1])  # 50 chemins pour lisibilité
            df_plot = paths.iloc[:, :n_show].copy()
            df_plot["t"] = np.arange(df_plot.shape[0])
            df_long = df_plot.melt(id_vars="t", var_name="Path", value_name="Capital ($)")
            title = f"Trajectoires simulées — N={res['mc']['metrics']['n_paths']}, horizon={int(profile['horizon_mc_annees'])} an(s)"
            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("t:Q", title="Pas de temps"),
                    y=alt.Y("Capital ($):Q", title="Capital ($)"),
                    detail="Path:N"  # pas de couleur -> pas de légende
                )
                .properties(height=300, title=title)
            )
            st.altair_chart(chart, use_container_width=True)

        # Fan chart percentiles (5 courbes seulement, légende lisible)
        if "fan" in res["mc"] and not res["mc"]["fan"].empty:
            fan = res["mc"]["fan"].copy()
            if "t" not in fan.columns:
                fan["t"] = np.arange(fan.shape[0])
            fan_long = fan.melt(id_vars="t", var_name="Percentile", value_name="Capital ($)")
            fan_chart = (
                alt.Chart(fan_long)
                .mark_line()
                .encode(
                    x=alt.X("t:Q", title="Pas de temps"),
                    y=alt.Y("Capital ($):Q", title="Capital ($)"),
                    color=alt.Color("Percentile:N", title="Percentile")
                )
                .properties(height=300, title="Plage de scénarios (P5–P95)")
            )
            st.altair_chart(fan_chart, use_container_width=True)

# --- Zone export (à relier plus tard aux générateurs PDF/DOCX) ----------
st.divider()
st.caption("Zone export (à relier plus tard aux générateurs de rapports).")

st.divider()
st.markdown("### ℹ️ Glossaire des paramètres")
st.markdown(
    "- **Fréquence** : cadence d’échantillonnage (Daily ≈ 252, Weekly ≈ 52). "
    "Impacte l’annualisation et la longueur de l’horizon.\n"
    "- **Rebalancing** : rééquilibrage du portefeuille aux poids cibles (Buy&Hold = aucun, Monthly/Quarterly = périodique).\n"
    "- **Confiance VaR/CVaR** : probabilité de ne pas dépasser la perte (VaR) et perte moyenne conditionnelle au-delà (CVaR).\n"
    "- **Modèle Monte Carlo** :\n"
    "   - **GBM (normal i.i.d.)** : tirages indépendants avec moyenne/vol empirique.\n"
    "   - **Block bootstrap** : ré-échantillonnage par blocs des rendements historiques pour préserver certaines dépendances.\n"
    "- **N (nombre de chemins)** : plus grand = intervalles plus stables mais plus de calcul.\n"
    "- **Taille de bloc** : pour le bootstrap ; plus grand = plus d’autocorrélations conservées, moins de diversité.\n"
    "- **MDD P95** : drawdown que l’on ne dépasse que dans 5% des trajectoires (mesure robuste).\n"
    "- **CAGR médiane** : croissance annualisée de la médiane des capitalisations simulées."
)
