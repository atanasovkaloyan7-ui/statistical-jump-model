"""Apply a regime signal to a portfolio, and measure honestly what it did.

WHAT THIS IS FOR
----------------
This is a DE-RISKING OVERLAY. In our testing over 1997-2026 it cut maximum
drawdown by roughly 40-45% and left the Sharpe ratio approximately unchanged.
That is the honest claim, and it is a useful one: most investors will trade a
flat Sharpe for half the drawdown.

It is NOT an alpha source. Do not size it on expected return. If you find
yourself reporting a Sharpe improvement above about 0.15 from this overlay,
check your execution lag before you believe it (see `assert_execution_lag`).

The measured evidence, 8 sleeves, 1997-2026, out-of-sample, T+1 fills:

    sleeve       B&H Sharpe   overlay Sharpe   B&H MDD   overlay MDD
    LargeCap        0.443         0.474         -55.3%      -30.6%
    MidCap          0.395         0.424         -58.2%      -42.7%
    SmallCap        0.407         0.448         -59.8%      -36.8%
    EAFE            0.311         0.402         -59.3%      -29.4%
    REIT            0.340         0.353         -76.3%      -35.4%
    HighYield       0.681         0.637         -30.2%      -12.3%
    Treasury        0.230         0.089         -46.4%      -21.6%
    Corporate       0.347        -0.033         -36.3%      -28.2%

Two things to read from that table. Drawdown improves everywhere, often by
half. And the overlay HURTS on the two bond sleeves -- low-volatility,
mean-reverting series are the wrong home for a trend-based regime filter. Apply
it to equities, credit and real assets; leave duration alone.
"""
import numpy as np
import pandas as pd


def assert_execution_lag(lag):
    """The single most important line in this package.

    Filling on the same bar that generated the signal is worth roughly +1.0 to
    +1.5 Sharpe on every strategy we tested, and 0.00 on buy-and-hold (which
    never trades -- that is the control that proves the effect is a fill
    artefact, not a signal). It is not achievable in practice.

    Signal computed from data through the close of day t may be filled at the
    close of day t at the earliest. That is lag=1 in this convention.
    """
    if lag < 1:
        raise ValueError(
            "execution_lag=%d fills on the bar that generated the signal. "
            "In our tests that single change moved a Sharpe of 0.50 to 1.58. "
            "It is look-ahead, not alpha. Use lag >= 1." % lag)
    return True


def apply_overlay(returns, regime, riskfree=None, cost_bp=10, execution_lag=1,
                  bear_exposure=0.0, bull_exposure=1.0):
    """Scale exposure by regime.

    returns        : daily returns of the asset or portfolio being overlaid
    regime         : 1 = bull, 0 = bear (from RegimeDetector)
    riskfree       : daily risk-free return earned when out of the market
    bear_exposure  : 0.0 = fully out. Use 0.5 for a partial de-risk, which
                     costs less in whipsaws and is easier to hold through.
    """
    assert_execution_lag(execution_lag)
    r = pd.Series(returns).astype(float)
    g = pd.Series(regime).reindex(r.index).ffill()
    idx = g.dropna().index
    r, g = r.reindex(idx), g.reindex(idx)
    rf = (pd.Series(riskfree).reindex(idx).ffill().fillna(0.0)
          if riskfree is not None else pd.Series(0.0, index=idx))

    target = g.map({1: bull_exposure, 0: bear_exposure}).astype(float)
    w = target.shift(execution_lag)

    gross = w * r + (1 - w) * rf
    turn = w.diff().abs().fillna(0.0)
    tc = turn * (cost_bp / 1e4) * 2.0          # both legs of the switch
    net = (gross - tc).dropna()
    return pd.DataFrame(dict(net=net, gross=gross.reindex(net.index),
                             exposure=w.reindex(net.index),
                             turnover=turn.reindex(net.index),
                             cost=tc.reindex(net.index),
                             rf=rf.reindex(net.index)))


def max_drawdown(r):
    wealth = (1 + pd.Series(r).fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min()), dd


def evaluate(overlay_df, baseline_returns, periods=252):
    """Compare overlaid vs unoverlaid on the metrics that matter here."""
    net = overlay_df["net"]
    base = pd.Series(baseline_returns).reindex(net.index)
    rf = overlay_df["rf"]

    def stats(x):
        ex = (x - rf).dropna()
        vol = ex.std(ddof=1) * np.sqrt(periods)
        mdd, dd = max_drawdown(x)
        yrs = len(x) / periods
        cagr = (1 + x.fillna(0)).prod() ** (1 / max(yrs, 1e-9)) - 1
        return dict(cagr=float(cagr), excess_ret=float(ex.mean() * periods),
                    vol=float(vol),
                    sharpe=float(ex.mean() * periods / vol) if vol > 0 else np.nan,
                    max_dd=mdd, avg_dd=float(dd.mean()),
                    calmar=float(cagr / abs(mdd)) if mdd else np.nan)

    a, b = stats(net), stats(base)
    return dict(
        overlay=a, baseline=b,
        d_sharpe=a["sharpe"] - b["sharpe"],
        d_maxdd_pp=100 * (a["max_dd"] - b["max_dd"]),
        maxdd_reduction=(1 - a["max_dd"] / b["max_dd"]) if b["max_dd"] else np.nan,
        turnover=float(overlay_df["turnover"].mean()),
        cost_drag=float(overlay_df["cost"].mean() * periods),
        time_in_market=float((overlay_df["exposure"] > 0).mean()),
    )


def print_evaluation(ev, name="overlay"):
    a, b = ev["overlay"], ev["baseline"]
    print("\n   %-22s %12s %12s %10s" % (name, "baseline", "overlay", "change"))
    print("   " + "-" * 60)
    for k, lab, pct in [("cagr", "CAGR", True), ("vol", "volatility", True),
                        ("sharpe", "Sharpe", False), ("max_dd", "max drawdown", True),
                        ("calmar", "Calmar", False)]:
        if pct:
            print("   %-22s %11.2f%% %11.2f%% %9.2fpp"
                  % (lab, 100 * b[k], 100 * a[k], 100 * (a[k] - b[k])))
        else:
            print("   %-22s %12.3f %12.3f %+10.3f" % (lab, b[k], a[k], a[k] - b[k]))
    print("   " + "-" * 60)
    print("   drawdown reduced by %.0f%%   time in market %.0f%%   cost drag %.2f%%/yr"
          % (100 * ev["maxdd_reduction"], 100 * ev["time_in_market"],
             100 * ev["cost_drag"]))
    if ev["d_sharpe"] > 0.15:
        print("   NOTE: a Sharpe gain above 0.15 is larger than we measured.")
        print("         Verify execution_lag >= 1 before trusting it.")


def crisis_report(regime, crises=None):
    """How often was the signal defensive during known stress episodes?

    This is the diagnostic that matters for a de-risking overlay. A good
    detector is bearish through crises and quiet in calm markets. Ours ran
    72-87% bear through every major episode and 2% through 2013-14.

    IMPORTANT: the default episode list is US-equity-centric and assumes REAL
    market dates. On synthetic data, on a series with its own idiosyncratic
    history, or on an asset whose stress periods differ (commodities, single
    names, crypto), these windows mean nothing and the numbers will be noise.
    Pass your own `crises` list in that case, or ignore the column.
    """
    g = pd.Series(regime)
    crises = crises or [
        ("1998-07", "1998-10", "LTCM"),
        ("2000-09", "2002-10", "dot-com"),
        ("2007-10", "2009-03", "GFC"),
        ("2011-07", "2011-10", "Euro / debt ceiling"),
        ("2015-08", "2016-02", "China / oil"),
        ("2018-10", "2018-12", "Q4 2018"),
        ("2020-02", "2020-04", "COVID"),
        ("2022-01", "2022-10", "rate hikes"),
    ]
    rows = []
    for lo, hi, nm in crises:
        w = (g == 0).loc[lo:hi]
        if len(w):
            rows.append(dict(episode=nm, window="%s..%s" % (lo, hi),
                             days=len(w), bear_pct=round(100 * float(w.mean()), 1)))
    return pd.DataFrame(rows)
