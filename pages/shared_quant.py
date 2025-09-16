import pandas as pd
import numpy as np
import yfinance as yf

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
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx).sort_index()
    for t in tickers:
        if t not in df.columns:
            df[t] = pd.NA
    return df

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

def portfolio_daily_returns(prices, allocations, rebal_mode):
    if rebal_mode.startswith("Buy"):
        return portfolio_returns_buy_and_hold(prices, allocations)
    elif rebal_mode.startswith("Monthly"):
        return portfolio_returns_with_rebalancing(prices, allocations, "M")
    else:
        return portfolio_returns_with_rebalancing(prices, allocations, "Q")

def compute_metrics_from_returns(r, dpy=252, rf_annual=0.0,
                                 want_sortino=True, want_calmar=True,
                                 want_var=False, want_cvar=False, var_alpha=0.95):
    if r is None or len(r) == 0: return {}
    cum_ret = (1 + r).prod() - 1
    cagr = (1 + cum_ret)**(dpy/len(r)) - 1
    mu_ann = r.mean() * dpy
    vol_ann = r.std() * np.sqrt(dpy)
    excess_mu = mu_ann - rf_annual
    sharpe = excess_mu/vol_ann if vol_ann else np.nan
    sortino = np.nan
    if want_sortino:
        downside = r.clip(upper=0)
        down_stdev_ann = downside.std() * np.sqrt(dpy)
        sortino = (excess_mu/down_stdev_ann) if down_stdev_ann else np.nan
    cum = (1+r).cumprod()
    peak = cum.cummax()
    dd = cum/peak - 1.0
    max_dd = dd.min() if len(dd) else np.nan
    calmar = (cagr/abs(max_dd)) if want_calmar and max_dd else np.nan
    var_val = cvar_val = np.nan
    if want_var or want_cvar:
        q = np.quantile(-r.dropna(), var_alpha) if len(r.dropna())>0 else np.nan
        if want_var:  var_val = q
        if want_cvar:
            tail = -r.dropna()
            tail = tail[tail >= q]
            cvar_val = tail.mean() if len(tail) > 0 else q
    return {
        "Annualized Return %": round(cagr*100, 2),
        "Cumulative Return %": round(cum_ret*100, 2),
        "Volatility %": round(vol_ann*100, 2),
        "Max Drawdown %": round(max_dd*100, 2) if pd.notna(max_dd) else np.nan,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Calmar": round(calmar, 2) if pd.notna(calmar) else np.nan,
        "VaR (daily)": round(var_val*100, 2) if pd.notna(var_val) else np.nan,
        "CVaR (daily)": round(cvar_val*100, 2) if pd.notna(cvar_val) else np.nan,
    }
