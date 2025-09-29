import pandas as pd
import numpy as np
import yfinance as yf

PPY = 52  # periods per year (weekly)

def download_prices(tickers, start, end):
    if isinstance(tickers, str): tickers = [tickers]
    data = yf.download(
        tickers, start=start, end=end + pd.Timedelta(days=1),
        interval="1d", auto_adjust=False, group_by="column",
        threads=True, progress=False
    )
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
    return df.sort_index()

def renormalize_weights_if_needed(prices_df, allocations):
    tickers = [t for t in allocations if t in prices_df.columns]
    if not tickers: return {}, []
    w = np.array([allocations[t] for t in tickers], dtype=float)
    if w.sum() <= 0: return {}, []
    w = w / w.sum()
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
        if w.sum() <= 0: continue
        w = w / w.sum()
        parts.append((g[cols] * w).sum(axis=1))
    if not parts: return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()

def portfolio_periodic_returns(prices, allocations, rebal_mode):
    if rebal_mode.startswith("Buy"):
        return portfolio_returns_buy_and_hold(prices, allocations)
    elif rebal_mode.startswith("Monthly"):
        return portfolio_returns_with_rebalancing(prices, allocations, "M")
    else:
        return portfolio_returns_with_rebalancing(prices, allocations, "Q")

def compute_metrics_from_returns(r, ppy=PPY, rf_annual=0.0,
                                 want_sortino=True, want_calmar=True,
                                 want_var=False, want_cvar=False, var_alpha=0.95):
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    # Cumulative & CAGR (robuste aux périodes partielles)
    cum_ret = (1 + r).prod() - 1
    if len(r.index) >= 2 and hasattr(r.index, "to_series"):
        years = (r.index[-1] - r.index[0]).days / 365.25
    else:
        years = len(r) / ppy
    cagr = (1 + cum_ret)**(1/years) - 1 if years and years > 0 else np.nan

    # Annualisation
    mu_ann  = r.mean() * ppy
    vol_ann = r.std(ddof=1) * np.sqrt(ppy)

    # Sharpe
    excess_mu = mu_ann - rf_annual
    sharpe = excess_mu/vol_ann if vol_ann and vol_ann != 0 else np.nan

    # Sortino: semidéviation (RMS des rendements négatifs)
    sortino = np.nan
    if want_sortino:
        downside = np.minimum(r, 0.0)
        semidev_ann = np.sqrt((downside**2).mean()) * np.sqrt(ppy)
        sortino = (excess_mu/semidev_ann) if semidev_ann and semidev_ann != 0 else np.nan

    # Drawdown / Calmar
    cum = (1+r).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1.0
    max_dd = dd.min() if not dd.empty else np.nan
    calmar = (cagr/abs(max_dd)) if (want_calmar and pd.notna(max_dd) and max_dd != 0) else np.nan

    # VaR / CVaR (à la fréquence de r → ici hebdo)
    var_val = cvar_val = np.nan
    if want_var or want_cvar:
        base = -r
        base = base.dropna()
        if not base.empty:
            q = np.quantile(base, var_alpha)
            if want_var: var_val = q
            if want_cvar:
                tail = base[base >= q]
                cvar_val = tail.mean() if len(tail) > 0 else q

    return {
        "Annualized Return %": round(cagr*100, 2),
        "Cumulative Return %": round(cum_ret*100, 2),
        "Volatility %": round(vol_ann*100, 2),
        "Max Drawdown %": round(max_dd*100, 2) if pd.notna(max_dd) else np.nan,
        "Sharpe": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "Sortino": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Calmar": round(calmar, 2) if pd.notna(calmar) else np.nan,
        "VaR (weekly)": round(var_val*100, 2) if pd.notna(var_val) else np.nan,
        "CVaR (weekly)": round(cvar_val*100, 2) if pd.notna(cvar_val) else np.nan,
    }
