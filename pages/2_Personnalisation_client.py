# pages/2_Personnalisation_client.py
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from io import StringIO
from scipy.stats import norm
from datetime import timedelta, date
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

def compute_metrics_from_returns(r, dpy=252, rf_annual=0.0):
    if r is None or len(r)==0: 
        return {"Annualized Return %": np.nan, "Volatility %": np.nan,
                "Max Drawdown %": np.nan, "Sharpe": np.nan}
    cum_ret = (1 + r).prod() - 1
    cagr = (1 + cum_ret)**(dpy/len(r)) - 1
    mu_ann = r.mean() * dpy
    vol_ann = r.std() * np.sqrt(dpy)
    sharpe = (mu_ann - rf_annual) / vol_ann if vol_ann else np.nan
    _, mdd = drawdown_stats(r)
    return {
        "Annualized Return %": round(cagr*100, 2),
        "Volatility %": round(vol_ann*100, 2),
        "Max Drawdown %": round(mdd*100, 2) if pd.notna(mdd) else np.nan,
        "Sharpe": round(sharpe, 2)
    }

def _build_allocations_from_profile(profile):
    if profile["portfolio_choice"].startswith("60/40 (S&P"):
        return {"^GSPC": 0.60, "AGGG.L": 0.40}
    elif profile["portfolio_choice"].startswith("60/40 + 5% Or"):
        return {"^GSPC": 0.57, "AGGG.L": 0.38, "GC=F": 0.05}
    else:
        try:
            d = pd.read_json(pd.io.common.StringIO(profile["custom_weights_json"]), typ="series").to_dict()
        except Exception:
            d = {}
        # sécurité: normalise si somme != 1
        s = sum(max(0.0, float(v)) for v in d.values()) or 1.0
        return {k: max(0.0, float(v))/s for k, v in d.items()}

def run_backtest(profile: dict) -> dict:
    try:
        alloc = _build_allocations_from_profile(profile)
        tickers = sorted(alloc.keys())
        if not tickers:
            return {"returns": pd.Series(dtype=float), "nav": pd.Series(dtype=float), "metrics": {}}

        # Fenêtre de données = max(5 ans, horizon user)
        dpy0 = 252 if profile["freq_backtest"]=="Daily" else 52
        window_days = int(max(5, profile["horizon_annees"]) * (252 if profile["freq_backtest"]=="Daily" else 52) * (252/dpy0))
        end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=window_days)

        raw = download_prices_simple(tickers, start_date, end_date)
        df, dpy = normalize_clock(raw, profile["freq_backtest"])
        r = portfolio_daily_returns(df, alloc, profile["rebal_mode"])
        nav = (1 + r).cumprod() * 100.0

        # métriques simples (Sharpe, Vol, MDD, CAGR)
        metrics = compute_metrics_from_returns(r, dpy=dpy, rf_annual=0.0)

        # VaR/CVaR en $ à l'horizon (approx normale sur r daily)
        T = int(profile["horizon_annees"] * (252 if profile["freq"]=="Daily" else 52))
        mu_d, sig_d = r.mean(), r.std()
        if not np.isnan(mu_d) and not np.isnan(sig_d) and T>0:
            from scipy.stats import norm
            alpha = profile["var_conf"]
            z = norm.ppf(1 - alpha)
            # rendement horizon approx: mu*T + z*sig*sqrt(T)
            ret_h = mu_d*T + z*sig_d*np.sqrt(T)
            var_pct = -ret_h  # perte
            # CVaR approx classique (normale)
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
        # 1) Re-utilise le backtest pour estimer mu/sigma du portefeuille
        bk = run_backtest(profile)
        r = bk["returns"]
        if r.empty:
            return {"paths": pd.DataFrame(), "summary": pd.DataFrame(), "metrics": {}}

        mu_d, sig_d = r.mean(), r.std()
        capital0 = float(profile["montant_investi"])
        T = int(profile["horizon_annees"] * (252 if profile["freq"]=="Daily" else 52))
        N = int(profile["mc_paths"])
        np.random.seed(profile["seed"])

        # 2) Simulations
        if profile["mc_model"] == "GBM":
            # Simule des rendements quotidiens i.i.d. ~ N(mu_d, sig_d)
            eps = np.random.normal(loc=mu_d, scale=sig_d, size=(T, N))
            paths_ret = eps
        else:
            # Block bootstrap sur les rendements historiques
            series = r.values
            if len(series) < 20:  # trop court
                eps = np.random.normal(loc=mu_d, scale=sig_d, size=(T, N))
                paths_ret = eps
            else:
                B = int(profile["mc_block"])
                # assemble T rendements par blocs tirés avec remise
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
        fan = pd.DataFrame(
            {f"P{p}": np.percentile(nav_paths, p, axis=1) for p in prc}
        )
        # 5) Distribution finale + VaR/CVaR numéraire
        terminal = nav_paths[-1, :]
        alpha = profile["var_conf"]
        q = np.quantile(terminal, 1 - alpha)
        losses = np.clip(capital0 - terminal, a_min=0, a_max=None)
        var_usd = float(capital0 - q)
        cvar_usd = float(losses[terminal <= q].mean() if np.any(terminal <= q) else var_usd)

        summary = pd.DataFrame({
            "Metric": ["Capital initial", "Espérance finale", f"VaR {int(alpha*100)}%", f"CVaR {int(alpha*100)}%", "P5", "P50", "P95"],
            "Value ($)": [
                round(capital0,2),
                round(float(terminal.mean()),2),
                round(var_usd,2),
                round(cvar_usd,2),
                round(float(np.percentile(terminal, 5)),2),
                round(float(np.percentile(terminal, 50)),2),
                round(float(np.percentile(terminal, 95)),2),
            ]
        })

        metrics = {
            "mu_daily": float(mu_d),
            "sigma_daily": float(sig_d),
            "horizon_steps": T,
            "n_paths": N,
            "model": profile["mc_model"],
        }

        # retourne DataFrame pour compat UI (index = t)
        paths_df = pd.DataFrame(nav_paths)

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

    c4, c5 = st.columns(2)
    with c4:
        var_conf = st.slider("Confiance VaR/CVaR", 0.80, 0.995, 0.95, 0.005)
    with c5:
        freq = st.selectbox("Fréquence de calcul", ["Daily", "Weekly"], index=0)

    st.divider()
    st.subheader("3) Portefeuille & Monte Carlo")

    # Univers d'actifs (identique page 1)
    full_asset_mapping = {**asset_mapping, **crypto_static, **us_equity_mapping}
    asset_names_map = {v: k for k, v in full_asset_mapping.items()}

    n_assets = st.number_input("Nombre d'actifs dans le portefeuille", 1, 20, 3, 1)
    custom_alloc_pairs = []
    used = set()

    cols = st.columns([3, 1])
    for i in range(int(n_assets)):
        with cols[0]:
            choice = st.selectbox(
                f"Actif {i+1}",
                list(full_asset_mapping.keys()),
                key=f"cust_asset_{i}"
            )
        with cols[1]:
            w = st.number_input(f"% poids {i+1}", 0.0, 100.0, 0.0, 0.1, key=f"cust_w_{i}")
        if choice in full_asset_mapping and full_asset_mapping[choice] not in used:
            custom_alloc_pairs.append((full_asset_mapping[choice], w/100.0))
            used.add(full_asset_mapping[choice])

    sum_w = sum(w for _, w in custom_alloc_pairs)*100
    if not np.isclose(sum_w, 100.0, atol=0.01):
        st.warning(f"⚠️ La somme des poids est {sum_w:.2f}%, elle doit être 100%.")
    custom_alloc = {t: w for t, w in custom_alloc_pairs if w > 0}

    # Option : bouton de pré-remplissage par appétence
    def suggested_weights_from_risk(score):
        # 3 ETFs simples: S&P500 (^GSPC), AGGG.L (agg bond), Gold (GC=F)
        # Interp linéaire equity/bond, or ~5%
        gold = 0.05
        eq = np.interp(score, [1, 10], [0.20, 0.85])
        bond = 1.0 - gold - eq
        return {"^GSPC": eq, "AGGG.L": max(0.0, bond), "GC=F": gold}

    if st.button("🎚️ Pré-remplir les poids selon l’appétence"):
        sw = suggested_weights_from_risk(appetence)
        # Remplit les 3 premières lignes si possible
        for i, (ticker, w) in enumerate(sw.items()):
            st.session_state[f"cust_asset_{i}"] = asset_names_map.get(ticker, ticker)
            st.session_state[f"cust_w_{i}"] = round(w*100, 1)
        st.experimental_rerun()

    st.divider()
    rebal_mode = st.selectbox("Rebalancing", ["Buy & Hold (no rebalance)", "Monthly", "Quarterly"], index=1)
    
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
    "portfolio_choice": port_choice,
    "custom_weights_json": custom_json,
    "rebal_mode": rebal_mode,
    "freq_backtest": freq_backtest,
    "mc_model": mc_model,
    "mc_paths": int(mc_paths),
    "mc_block": int(mc_block),
    "seed": int(seed),
    # Hooks à enrichir: mapping portefeuille cible, contraintes ESG, etc.
}

    
# Sauvegarde pour usage inter-pages
st.session_state["client_profile"] = profile

# --- Lancement (sans implémentation des calculs) -----------------------
if submitted:
    results = {}
    # bornes temporelles basiques (1 an glissant par défaut)
    end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).normalize()
    start_date = end_date - pd.Timedelta(days=int(horizon_annees*365))

    # Portefeuilles benchmark
    bench_60_40 = {"^GSPC": 0.60, "AGGG.L": 0.40}
    bench_60_40_gold = {"^GSPC": 0.57, "AGGG.L": 0.38, "GC=F": 0.05}

    # Tickers à télécharger (custom + benchmarks)
    all_tickers = set(custom_alloc.keys()) | set(bench_60_40.keys()) | set(bench_60_40_gold.keys())
    df = download_prices(sorted(all_tickers), start_date, end_date)

    # Fréquence
    if profile["freq"] == "Daily":
        dpy = 252
        dfA = df.resample("B").last().ffill()
    else:
        dpy = 52
        dfA = df.resample("W-FRI").last().ffill()

    # Backtest (3 portefeuilles)
    from pages.shared_quant import portfolio_daily_returns
    ports = {
        "Personnalisé": custom_alloc,
        "60/40": bench_60_40,
        "60/40 + 5% Or": bench_60_40_gold
    }
    port_returns = {}
    for name, alloc in ports.items():
        port_returns[name] = portfolio_daily_returns(dfA, alloc, rebal_mode)

    # Metrics
    metrics = {}
    for name, r in port_returns.items():
        metrics[name] = compute_metrics_from_returns(
            r, dpy=dpy, rf_annual=0.0,  # tu peux mettre le RF plus tard
            want_sortino=True, want_calmar=True,
            want_var=True, want_cvar=True, var_alpha=profile["var_conf"]
        )
    metrics_df = pd.DataFrame(metrics)

    # Monte Carlo sur le portefeuille Personnalisé (piloté par l’appétence)
    def risk_controls(score):
        vol_scale = float(np.interp(score, [1, 10], [0.6, 1.6]))
        df_tail   = int(round(np.interp(score, [1, 10], [12, 4])))
        mu_shift  = float(np.interp(score, [1, 10], [-0.01, 0.01]))
        return vol_scale, df_tail, mu_shift

    r_hist = port_returns["Personnalisé"].dropna()
    if len(r_hist) < 10:
        mc = {"paths": pd.DataFrame(), "summary": pd.DataFrame(), "metrics": {"note": "pas assez d'historique"}}
    else:
        mu = r_hist.mean() * dpy
        vol = r_hist.std() * np.sqrt(dpy)
        vol_scale, df_tail, mu_shift = risk_controls(appetence)

        horizon_days = int(horizon_annees * (252 if profile["freq"]=="Daily" else 52))
        n_paths = 2000

        # génère des rendements annualisés puis reconvertit à la fréquence choisie
        step_per_year = 252 if profile["freq"]=="Daily" else 52
        step_mu  = (1+mu+mu_shift)**(1/step_per_year) - 1
        step_vol = (vol * vol_scale) / np.sqrt(step_per_year)

        np.random.seed(42)
        shocks = np.random.standard_t(df_tail, size=(horizon_days, n_paths)) * (step_vol/np.sqrt(df_tail/(df_tail-2)))  # t->vol target
        step_r = step_mu + shocks
        nav = np.cumprod(1 + step_r, axis=0)
        nav = pd.DataFrame(nav, index=pd.date_range(start=end_date, periods=horizon_days, freq=("B" if profile["freq"]=="Daily" else "W-FRI")))
        nav.iloc[0,:] = 1.0  # base 1

        # VaR/CVaR en $ à horizon
        final_r = nav.iloc[-1,:] - 1.0
        alpha = profile["var_conf"]
        capital0 = investissement
        q = np.quantile(final_r, 1-alpha)
        var_cash = float(capital0 * -q)
        cvar_cash = float(capital0 * -final_r[final_r <= q].mean())

        mc = {
            "paths": nav,
            "summary": pd.DataFrame({
                "P5": [np.quantile(final_r, 0.05)],
                "P50": [np.quantile(final_r, 0.50)],
                "P95": [np.quantile(final_r, 0.95)],
                f"VaR {int(alpha*100)}% ($)": [var_cash],
                f"CVaR {int(alpha*100)}% ($)": [cvar_cash],
            }),
            "metrics": {"mu_ann_hist": float(mu), "vol_ann_hist": float(vol),
                        "vol_scale": vol_scale, "df": df_tail, "mu_shift": mu_shift}
        }

    results["backtest"] = {"returns": port_returns, "metrics_df": metrics_df}
    results["mc"] = mc
    st.session_state["client_results"] = results
    st.success("Backtest + Monte Carlo terminés.")

# --- Affichage minimal des emplacements (placeholder) ------------------
if "client_results" in st.session_state:
    res = st.session_state["client_results"]

    tabs = st.tabs(["Backtest", "Monte Carlo"])

    with tabs[0]:
        st.subheader("Métriques (3 portefeuilles)")
        st.dataframe(res["backtest"]["metrics_df"], use_container_width=True)

        st.subheader("NAV (base 100)")
        port_returns = res["backtest"]["returns"]
        nav_df = pd.DataFrame({k: (1+v).cumprod()*100 for k,v in port_returns.items() if v is not None})
        st.line_chart(nav_df)

    with tabs[1]:
        st.subheader("Résumé Monte Carlo")
        st.dataframe(res["mc"]["summary"], use_container_width=True)
        if not res["mc"]["paths"].empty:
            st.subheader("Trajectoires simulées (échantillon)")
            st.line_chart(res["mc"]["paths"].iloc[:, :50])  # 50 chemins pour lisibilité

# --- Zone export (sera branchée sur vos générateurs PDF/DOCX) ----------
st.divider()
st.caption("Zone export (à relier plus tard aux générateurs de rapports).")
