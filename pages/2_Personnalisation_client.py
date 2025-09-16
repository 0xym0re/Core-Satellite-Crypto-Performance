# pages/2_Personnalisation_client.py
# ---------------------------------------------------------------------
# Personnalisation client — SQUELETTE (UI + hooks, sans calculs)
# ---------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from io import StringIO
from scipy.stats import norm
from datetime import timedelta, datetime

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
        var_$ = float(capital0 - q)
        cvar_$ = float(losses[terminal <= q].mean() if np.any(terminal <= q) else var_$)

        summary = pd.DataFrame({
            "Metric": ["Capital initial", "Espérance finale", f"VaR {int(alpha*100)}%", f"CVaR {int(alpha*100)}%", "P5", "P50", "P95"],
            "Value ($)": [
                round(capital0,2),
                round(float(terminal.mean()),2),
                round(var_$,2),
                round(cvar_$,2),
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

    st.divider()
    st.subheader("3) Portefeuille & Monte Carlo")

    c7, c8, c9 = st.columns(3)
    with c7:
        port_choice = st.selectbox(
            "Portefeuille",
            ["60/40 (S&P 500 / AGGG.L)", "60/40 + 5% Or", "Personnalisé (JSON)"],
            index=0
        )
    with c8:
        rebal_mode = st.selectbox("Rebalancing", ["Buy & Hold (no rebalance)", "Monthly", "Quarterly"], index=1)
    with c9:
        freq_backtest = st.selectbox("Fréquence données (backtest)", ["Daily", "Weekly"], index=0)

    custom_json = st.text_area(
        "Poids personnalisés (JSON {\"^GSPC\":0.6, \"AGGG.L\":0.4})",
        value='{"^GSPC": 0.6, "AGGG.L": 0.4}',
        help="Utilisé uniquement si 'Personnalisé (JSON)' est sélectionné."
    )

    c10, c11, c12 = st.columns(3)
    with c10:
        mc_model = st.selectbox("Modèle Monte Carlo", ["GBM", "Block bootstrap"], index=0)
    with c11:
        mc_paths = st.number_input("Trajectoires MC (N)", 100, 20000, 5000, 100)
    with c12:
        mc_block = st.number_input("Taille de bloc (bootstrap)", 5, 60, 10, 1)
    seed = st.number_input("Graine aléatoire", 0, 10**9, 42, 1)
    
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
            bk = res["backtest"]
            st.line_chart(bk["nav"], height=220)
            st.json(bk["metrics"])
    if "mc" in res:
        with tabs[names.index("Monte Carlo")]:
            mc = res["mc"]
            if "fan" in mc:
                st.area_chart(mc["fan"], height=220)
            st.dataframe(mc["summary"], use_container_width=True)

# --- Zone export (sera branchée sur vos générateurs PDF/DOCX) ----------
st.divider()
st.caption("Zone export (à relier plus tard aux générateurs de rapports).")
