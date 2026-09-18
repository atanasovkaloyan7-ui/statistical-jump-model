"""Baseline signals to measure your own against.

The checklist in the docs says "does it beat a 200-day moving average?" -- so the
package should ship one rather than making you write it. These are deliberately
trivial. If your signal cannot beat them, it is not earning its complexity.

Every function here returns the same thing the detector returns: a 0/1 Series,
1 = bull, aligned to the input index. Anything with that shape can be fed to
`apply_overlay` and `compare_signals`, including a signal of your own from
somewhere else entirely.
"""
import numpy as np
import pandas as pd


def _as_returns(returns):
    return pd.Series(returns).astype(float).dropna()


def always_bull(returns):
    """The control. Never trades, so its Sharpe must be unchanged by the harness.

    If this row moves when you run it through your pipeline, your pipeline has a
    timing bug. It is the cheapest and most informative test you can run.
    """
    r = _as_returns(returns)
    return pd.Series(1, index=r.index, name="always_bull")


def moving_average_regime(returns, window=200, prices=None):
    """Bull when price is above its own moving average. The benchmark to beat.

    This is the honest competitor for any trend-based regime model, because that
    is what a trend-based regime model is. Two lines of code, no fitting, no
    hyperparameters worth arguing about.
    """
    r = _as_returns(returns)
    px = (pd.Series(prices).reindex(r.index).ffill() if prices is not None
          else (1 + r).cumprod())
    ma = px.rolling(window, min_periods=window).mean()
    sig = (px > ma).astype(int)
    sig[ma.isna()] = 1                      # before the window fills, stay invested
    sig.name = "ma_%d" % window
    return sig


def dual_moving_average_regime(returns, fast=50, slow=200, prices=None):
    """Bull when the fast average is above the slow one. Classic crossover."""
    r = _as_returns(returns)
    px = (pd.Series(prices).reindex(r.index).ffill() if prices is not None
          else (1 + r).cumprod())
    f = px.rolling(fast, min_periods=fast).mean()
    s = px.rolling(slow, min_periods=slow).mean()
    sig = (f > s).astype(int)
    sig[s.isna()] = 1
    sig.name = "ma_%d_%d" % (fast, slow)
    return sig


def momentum_regime(returns, lookback=252, skip=21):
    """Bull when trailing return over the lookback (skipping the last month) is
    positive. The time-series momentum rule from the academic literature."""
    r = _as_returns(returns)
    cum = (1 + r).cumprod()
    past = cum.shift(skip) / cum.shift(skip + lookback) - 1
    sig = (past > 0).astype(int)
    sig[past.isna()] = 1
    sig.name = "tsmom_%d_%d" % (lookback, skip)
    return sig


def volatility_regime(returns, window=63, quantile=0.80):
    """Bear when trailing volatility is in its own top quantile.

    The quantile is expanding, not full-sample -- a full-sample quantile would
    be a look-ahead leak, and a subtle one.
    """
    r = _as_returns(returns)
    vol = r.rolling(window, min_periods=window).std()
    thresh = vol.expanding(min_periods=window * 2).quantile(quantile)
    sig = (vol <= thresh).astype(int)
    sig[thresh.isna()] = 1
    sig.name = "vol_q%d" % int(100 * quantile)
    return sig


def random_regime(returns, switches_per_year=7, seed=0):
    """Placebo. Random regimes with a realistic switching rate.

    Include this in every comparison. It is the row that tells you how much of
    your result is the signal and how much is simply being out of the market
    some fraction of the time -- which, in a market with fat left tails, is
    worth something all by itself.
    """
    r = _as_returns(returns)
    rng = np.random.default_rng(seed)
    p = switches_per_year / 252.0
    state, out = 1, []
    for _ in range(len(r)):
        if rng.random() < p:
            state = 1 - state
        out.append(state)
    s = pd.Series(out, index=r.index, name="random_seed%d" % seed)
    return s


def random_ensemble(returns, switches_per_year=7, n=25):
    """Many placebo draws, so you can compare against a distribution rather than
    one lucky or unlucky coin flip. Returns a dict of Series."""
    return {"random_%02d" % i: random_regime(returns, switches_per_year, seed=i)
            for i in range(n)}


ALL_BENCHMARKS = {
    "always bull (control)": always_bull,
    "200d moving average": moving_average_regime,
    "50/200 crossover": dual_moving_average_regime,
    "12-1 momentum": momentum_regime,
    "vol quantile": volatility_regime,
    "random (placebo)": random_regime,
}


def build_benchmarks(returns, prices=None, include=None):
    """All baselines at once, ready to hand to compare_signals."""
    out = {}
    for name, fn in ALL_BENCHMARKS.items():
        if include and name not in include:
            continue
        try:
            out[name] = (fn(returns, prices=prices)
                         if fn in (moving_average_regime, dual_moving_average_regime)
                         else fn(returns))
        except TypeError:
            out[name] = fn(returns)
    return out
