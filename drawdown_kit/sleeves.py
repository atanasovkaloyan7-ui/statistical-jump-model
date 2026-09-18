"""Static multi-asset sleeves -- the second thing that worked.

WHY STATIC
----------
The obvious upgrade is to pick sleeves dynamically from a per-asset regime
model. We tested exactly that (the BMDA/BMGA construction from Luo & Mulvey
2026) on a 10-sleeve universe over 14.5 years out-of-sample. It scored Sharpe
0.249 against 0.826 for a plain 60/40, with 0 of 8 sleeves improved by their own
regime signal and 2.0%/yr of cost drag. Static weights beat it comfortably.
FINDINGS.md has the numbers.

WHY IT WORKS
------------
Not because the sleeves earn much on their own. Over 2008-2025 the broad
commodity sleeve returned -2.3%/yr. It works because the sleeves take turns:

    sleeve      2008-2021                 2022-2025
    treasury    +6.3%/yr, corr -0.41      -9.3%/yr, corr +0.10
    commodity   -3.5%/yr, corr +0.47      +5.5%/yr, corr +0.24
    gold        +5.0%/yr, corr +0.05     +23.5%/yr, corr +0.09

The best single-sleeve bet in the first regime was the worst in the second, and
nothing available at the start told you which was coming. Holding all of them in
small size is the response to that, not a return forecast.

PICK WEIGHTS ON REASONING, NOT ON A BACKTEST
--------------------------------------------
Each sleeve should be justified by the regime it survives. Round numbers.
Declare them with freeze_config() before you run anything. If you find yourself
searching weights, you are no longer doing this.
"""
import numpy as np
import pandas as pd

from .overlay import apply_overlay, max_drawdown

# One-way cost per sleeve, basis points. Flat assumptions flatter whichever
# sleeve is most expensive to trade -- usually credit, commodities and small
# caps, usually in exactly the stressed markets where an overlay trades most.
DEFAULT_COST_BP = {
    "SPY": 2, "IVV": 2, "VTI": 2, "IJH": 3, "IWM": 3, "IJR": 3, "EFA": 5,
    "EEM": 8, "VNQ": 6,
    "SHY": 3, "IEF": 4, "TLT": 4, "GOVT": 3, "TIP": 5, "BIL": 2,
    "GLD": 5, "IAU": 5, "SLV": 10,
    "DBC": 20, "PDBC": 15, "GSG": 20, "DJP": 20, "DBA": 20, "USO": 20,
    "HYG": 25, "LQD": 15,
    "BOOK": 20,          # a hand-built stock book; measure yours
}

# A starting point, not a recommendation. Justify every line before you use it.
DEFAULT_WEIGHTS = {"EQUITY": 0.60, "TLT": 0.15, "GLD": 0.15, "DBC": 0.10}


def stats(r, periods=252):
    """CAGR, vol, drawdown, Sharpe, Calmar for a daily return series."""
    r = pd.Series(r).dropna()
    yrs = len(r) / periods
    mdd, _ = max_drawdown(r)
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    sd = r.std(ddof=1)
    return dict(years=round(yrs, 2), cagr=float(cagr),
                vol=float(sd * np.sqrt(periods)), max_drawdown=float(mdd),
                sharpe=float(np.sqrt(periods) * r.mean() / sd) if sd > 0 else np.nan,
                calmar=float(cagr / abs(mdd)) if mdd else np.nan)


def load_sleeves(path, tickers=None):
    """Load an adjusted-close frame and return daily simple returns.

    The bundled data/sleeve_closeadj.parquet holds 27 ETFs, 1997-2026, on
    total-return adjusted closes. Swap in your own frame with the same shape.
    """
    px = pd.read_parquet(path)
    if tickers:
        px = px[[t for t in tickers if t in px.columns]]
    return px.sort_index().pct_change()


def overlay_sleeve(returns, regime, riskfree=None, cost_bp=20,
                   bear_exposure=0.50, execution_lag=1):
    """Scale ONE sleeve by a regime signal; the de-risked part earns cash.

    Apply this to equity and equity-like sleeves only. We measured the same
    overlay destroying Treasury Sharpe (0.230 -> 0.089) and investment-grade
    credit (0.347 -> -0.033). Low-volatility mean-reverting series are the wrong
    home for a trend filter. Hold duration straight.
    """
    ov = apply_overlay(returns, regime, riskfree=riskfree, cost_bp=cost_bp,
                       execution_lag=execution_lag, bear_exposure=bear_exposure)
    return ov["net"], float(ov["exposure"].mean())


def rebalanced(returns, weights, cost_bp=None, freq="Q"):
    """Fixed-weight portfolio: drift between rebalances, turnover charged.

    returns : DataFrame of daily sleeve returns
    weights : {column: weight}; normalised for you
    cost_bp : {column: one-way bp}; DEFAULT_COST_BP used for anything missing
    freq    : 'Q', 'M' or 'A' -- rebalance at the first session of each period

    Returns (portfolio_returns, annual_gross_turnover).

    Turnover is charged per sleeve, on |target - drifted| at each rebalance.
    Nothing is charged between rebalances, which is the point of holding static
    weights: the portfolio only trades when you tell it to.
    """
    cost_bp = cost_bp or DEFAULT_COST_BP
    cols = [c for c in weights if c in returns.columns]
    missing = [c for c in weights if c not in returns.columns]
    if missing:
        raise KeyError("weights reference columns not in returns: %s" % missing)
    w = pd.Series({c: float(weights[c]) for c in cols})
    if (w < 0).any():
        raise ValueError("negative weights: this layer is long-only")
    w = w / w.sum()

    sub = returns[cols].dropna()
    if sub.empty:
        raise ValueError("no overlapping dates across the requested sleeves")
    cost = np.array([cost_bp.get(c, 10) for c in cols], dtype=float) / 1e4
    marks = sub.index.to_period(freq)

    cur, out, turns, prev = w.to_numpy().copy(), [], [], marks[0]
    for i, row in enumerate(sub.to_numpy()):
        if marks[i] != prev:
            trade = np.abs(w.to_numpy() - cur)
            out.append(float((cur * row).sum()) - float((trade * cost).sum()))
            turns.append(float(trade.sum()))
            cur, prev = w.to_numpy().copy(), marks[i]
        else:
            out.append(float((cur * row).sum()))
            turns.append(0.0)
        cur = cur * (1 + row)
        s = cur.sum()
        cur = cur / s if s > 0 else w.to_numpy().copy()

    port = pd.Series(out, index=sub.index, name="portfolio")
    ann_turnover = float(np.sum(turns) / (len(sub) / 252.0))
    return port, ann_turnover


def build_portfolio(sleeve_returns, equity_col, regime=None, weights=None,
                    cost_bp=None, bear_exposure=0.50, freq="Q",
                    riskfree_col="BIL"):
    """The whole stack in one call: overlay the equity sleeve, then blend.

    Set regime=None to get the un-overlaid version -- which you need anyway, as
    one of the controls.

    Returns dict with the portfolio series, turnover and mean equity exposure.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    R = sleeve_returns.copy()
    rf = R[riskfree_col] if riskfree_col in R.columns else None

    if "EQUITY" in weights:
        weights[equity_col + "_EQ"] = weights.pop("EQUITY")
    eq_key = equity_col + "_EQ" if equity_col + "_EQ" in weights else equity_col

    mean_exp = 1.0
    if regime is not None:
        cb = (cost_bp or DEFAULT_COST_BP).get(equity_col, 20)
        R[eq_key], mean_exp = overlay_sleeve(
            R[equity_col], regime.reindex(R.index).ffill(), riskfree=rf,
            cost_bp=cb, bear_exposure=bear_exposure)
    elif eq_key != equity_col:
        R[eq_key] = R[equity_col]

    cb = dict(cost_bp or DEFAULT_COST_BP)
    cb.setdefault(eq_key, cb.get(equity_col, 20))
    port, turn = rebalanced(R, weights, cb, freq)
    return dict(portfolio=port, turnover=turn, mean_equity_exposure=mean_exp,
                weights=weights, stats=stats(port))
