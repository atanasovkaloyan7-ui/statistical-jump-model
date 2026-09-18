"""Head-to-head comparison of regime signals on one return series.

This is the module to use when you already HAVE a signal -- from another model,
another project, a vendor, or your own intuition encoded as a rule -- and want to
know whether it is any good.

`compare_signals` takes a dict of {name: 0/1 Series} and puts them all through
the identical overlay, costs and fill convention. Anything with that shape works;
nothing here knows or cares where the signal came from.

Include the control and the placebo. They are the two rows that tell you whether
a result is real:

  always bull      must be unchanged by the harness. If it moves, you have a
                   timing bug, not a signal.
  random           shows what you get from simply being out of the market some
                   of the time. In a market with fat left tails that is worth
                   something on its own, and your signal has to beat it.
"""
import numpy as np
import pandas as pd

from .overlay import apply_overlay, evaluate, crisis_report
from .benchmarks import random_ensemble


def _switches_per_year(regime):
    g = pd.Series(regime).dropna()
    if len(g) < 2:
        return np.nan
    return float((g.diff().abs() == 1).sum() / (len(g) / 252.0))


def score_signal(returns, regime, riskfree=None, cost_bp=5, execution_lag=1,
                 bear_exposure=0.0, crises=None, calm=("2013", "2014")):
    """Every metric that matters for one signal."""
    ov = apply_overlay(returns, regime, riskfree=riskfree, cost_bp=cost_bp,
                       execution_lag=execution_lag, bear_exposure=bear_exposure)
    ev = evaluate(ov, pd.Series(returns).reindex(ov.index))
    g = pd.Series(regime).reindex(ov.index).ffill()

    cr = crisis_report(g, crises)
    hit = float(cr["bear_pct"].mean()) if len(cr) else np.nan
    try:
        fp = 100 * float((g == 0).loc[calm[0]:calm[1]].mean())
    except Exception:
        fp = np.nan

    return dict(
        sharpe=ev["overlay"]["sharpe"],
        d_sharpe=ev["d_sharpe"],
        max_dd=100 * ev["overlay"]["max_dd"],
        dd_reduction=100 * ev["maxdd_reduction"],
        calmar=ev["overlay"]["calmar"],
        switches_yr=_switches_per_year(g),
        bear_share=100 * float((g == 0).mean()),
        cost_drag=100 * ev["cost_drag"],
        crisis_hit=hit,
        calm_false_pos=fp,
    )


def compare_signals(returns, signals, riskfree=None, cost_bp=5, execution_lag=1,
                    bear_exposure=0.0, crises=None, add_placebo=True,
                    n_placebo=25):
    """Score every signal through the same harness.

    signals : {name: 0/1 Series}. Bring your own -- anything indexed like
              `returns` with 1 = risk-on works.

    Returns a DataFrame sorted by drawdown reduction, because that is the
    product. A `placebo_pct` column reports where each signal's Sharpe falls in
    the distribution of random signals with the same switching rate.
    """
    r = pd.Series(returns).astype(float).dropna()

    # Align every signal to a COMMON window before scoring. Without this a
    # walk-forward model that needs six years of warm-up gets compared against a
    # moving average over a different (usually longer) period, and the table is
    # not a comparison at all. The common window is the intersection.
    common = r.index
    for g in signals.values():
        gi = pd.Series(g).dropna().index
        if len(gi):
            common = common.intersection(gi)
    if len(common) < 500:
        raise ValueError(
            "signals share only %d common days -- too little overlap to compare. "
            "Check that each signal covers a similar period." % len(common))
    r = r.reindex(common)
    signals = {k: pd.Series(v).reindex(common) for k, v in signals.items()}

    rows = {}
    for name, g in signals.items():
        try:
            rows[name] = score_signal(r, g, riskfree, cost_bp, execution_lag,
                                      bear_exposure, crises)
        except Exception as e:
            rows[name] = dict(sharpe=np.nan, note=str(e)[:40])
    df = pd.DataFrame(rows).T
    df.attrs["window"] = (str(common.min().date()), str(common.max().date()),
                          len(common))

    if add_placebo and len(df):
        med_sw = float(np.nanmedian(df["switches_yr"])) or 7.0
        null = []
        for _, g in random_ensemble(r, switches_per_year=med_sw, n=n_placebo).items():
            try:
                null.append(score_signal(r, g, riskfree, cost_bp,
                                         execution_lag, bear_exposure, crises)["sharpe"])
            except Exception:
                pass
        null = np.array([x for x in null if np.isfinite(x)])
        if len(null):
            df["placebo_pct"] = [
                100 * float((null < s).mean()) if np.isfinite(s) else np.nan
                for s in df["sharpe"]]
            df.attrs["placebo_median"] = float(np.median(null))
            df.attrs["placebo_p95"] = float(np.percentile(null, 95))
            df.attrs["placebo_n"] = int(len(null))
    return df.sort_values("dd_reduction", ascending=False)


def print_comparison(df, title="signal comparison"):
    print("\n" + "=" * 96)
    print(title.upper())
    print("=" * 96)
    print("   %-26s %7s %8s %8s %8s %7s %7s %7s %7s"
          % ("signal", "Sharpe", "dSharpe", "maxDD", "ddRedn", "sw/yr",
             "crisis", "calmFP", "vs rand"))
    print("   " + "-" * 91)
    for name, r in df.iterrows():
        if not np.isfinite(r.get("sharpe", np.nan)):
            print("   %-26s  (failed: %s)" % (name, r.get("note", "?")))
            continue
        pp = r.get("placebo_pct", np.nan)
        print("   %-26s %7.3f %+8.3f %7.1f%% %7.0f%% %7.1f %6.0f%% %6.1f%% %6s"
              % (name[:26], r["sharpe"], r["d_sharpe"], r["max_dd"],
                 r["dd_reduction"], r["switches_yr"], r["crisis_hit"],
                 r["calm_false_pos"],
                 ("%.0f%%" % pp) if np.isfinite(pp) else "-"))
    print("   " + "-" * 91)
    if "placebo_median" in df.attrs:
        print("   placebo: %d random signals, median Sharpe %.3f, 95th pct %.3f"
              % (df.attrs["placebo_n"], df.attrs["placebo_median"],
                 df.attrs["placebo_p95"]))
    print("\n   crisis = mean % of days flagged bear across stress episodes (higher better)")
    print("   calmFP = % of days flagged bear in a calm bull market (lower better)")
    print("            both assume REAL US-equity dates -- meaningless on synthetic")
    print("            data or assets with their own stress history. Pass `crises=`.")
    print("   vs rand = percentile of this Sharpe within the random-signal distribution")

    ctrl = [i for i in df.index if "control" in str(i).lower() or "always" in str(i).lower()]
    if ctrl:
        d = df.loc[ctrl[0], "d_sharpe"]
        if abs(d) > 0.02:
            print("\n   WARNING: the control moved by %+.3f Sharpe. It never trades, so it" % d)
            print("            should be ~0.000. Your harness has a timing bug.")
        else:
            print("\n   control unchanged (%+.3f) -- harness timing is clean." % d)


def verdict(df, signal_name, ma_name="200d moving average", tol=0.15):
    """Blunt answer to: is this signal worth its complexity?"""
    if signal_name not in df.index:
        return "signal %r not in comparison" % signal_name
    s = df.loc[signal_name]
    lines = []
    pp = s.get("placebo_pct", np.nan)
    if np.isfinite(pp) and pp < 80:
        lines.append("FAILS PLACEBO: Sharpe is at the %.0fth percentile of random "
                     "signals with the same switching rate." % pp)
    if ma_name in df.index:
        d = s["sharpe"] - df.loc[ma_name, "sharpe"]
        dd = s["dd_reduction"] - df.loc[ma_name, "dd_reduction"]
        if d < tol and dd < 5:
            lines.append("DOES NOT BEAT THE MOVING AVERAGE: %+.3f Sharpe and %+.0f pp "
                         "drawdown versus a 200-day rule. Ship the moving average."
                         % (d, dd))
        else:
            lines.append("Beats the 200-day rule by %+.3f Sharpe and %+.0f pp drawdown."
                         % (d, dd))
    if s["switches_yr"] > 8:
        lines.append("TURNOVER HIGH: %.1f switches/yr, cost drag %.2f%%/yr."
                     % (s["switches_yr"], s["cost_drag"]))
    if np.isfinite(s.get("calm_false_pos", np.nan)) and s["calm_false_pos"] > 15:
        lines.append("NOISY IN CALM MARKETS: bear on %.0f%% of days in a quiet bull run."
                     % s["calm_false_pos"])
    if not lines:
        lines.append("Clears placebo, beats the moving average, turnover controlled.")
    return "\n".join("   - " + l for l in lines)
