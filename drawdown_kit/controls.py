"""The controls. Run these before believing any drawdown result.

Every one of these exists because something passed a naive test and failed a
real one. They are cheap. Run all of them.

    1. exposure_matched_control -- the one that kills most de-risking ideas.
       Cutting exposure ALWAYS cuts drawdown. The question is whether your
       timing beat simply holding less, on average, the whole time.
    2. vol_matched -- the "is this just a de-risked index fund?" test.
    3. weight_perturbation -- fitted results collapse when nudged.
    4. regime_split / episode_table -- 18 years of daily data is not 18 years of
       evidence. It is a handful of independent episodes.
    5. full_battery -- all of the above plus the placebo and moving-average
       comparisons from compare.py.
"""
import numpy as np
import pandas as pd

from .overlay import apply_overlay, evaluate, max_drawdown
from .sleeves import rebalanced, stats

CRISES = [("1998-07", "1998-10", "LTCM"),
          ("2000-09", "2002-10", "dot-com"),
          ("2007-10", "2009-03", "GFC"),
          ("2011-07", "2011-10", "Euro / debt ceiling"),
          ("2015-08", "2016-02", "China / oil"),
          ("2018-10", "2018-12", "Q4 2018"),
          ("2020-02", "2020-04", "COVID"),
          ("2022-01", "2022-10", "rate hikes"),
          ("2025-02", "2025-05", "tariffs")]


def exposure_matched_control(returns, regime, riskfree=None, cost_bp=20,
                             bear_exposure=0.0, execution_lag=1):
    """THE decisive control. Same average exposure, held flat, no timing.

    If your overlay averages 81% invested, the honest comparison is a book held
    at a constant 81% -- not the fully invested book. A flat cut also reduces
    drawdown, and it costs nothing in turnover.

    Returns a dict; read d_maxdd_pp (negative = the overlay's drawdown is
    deeper) and d_calmar. On US equity 2004-2023 we measured the overlay
    beating its matched constant by 17.5 percentage points of drawdown, which
    is the strongest evidence in the whole programme that the timing does
    something beyond owning less.
    """
    r = pd.Series(returns).dropna()
    g = pd.Series(regime).reindex(r.index).ffill()
    ov = apply_overlay(r, g, riskfree=riskfree, cost_bp=cost_bp,
                       execution_lag=execution_lag, bear_exposure=bear_exposure)
    o = evaluate(ov, r)["overlay"]
    mean_exp = float(ov["exposure"].mean())

    flat_sig = pd.Series(1, index=ov.index)
    fm = apply_overlay(r.reindex(ov.index), flat_sig, riskfree=riskfree,
                       cost_bp=cost_bp, execution_lag=execution_lag,
                       bull_exposure=mean_exp, bear_exposure=mean_exp)
    f = evaluate(fm, r.reindex(ov.index))["overlay"]

    return dict(mean_exposure=mean_exp, overlay=o, flat_control=f,
                d_cagr_pp=100 * (o["cagr"] - f["cagr"]),
                d_maxdd_pp=100 * (o["max_dd"] - f["max_dd"]),
                d_sharpe=o["sharpe"] - f["sharpe"],
                d_calmar=o["calmar"] - f["calmar"],
                verdict=("timing adds drawdown protection beyond owning less"
                         if o["max_dd"] > f["max_dd"] and o["calmar"] > f["calmar"]
                         else "NOT clearly better than simply holding less"))


def vol_matched(strategy, benchmark, riskfree=None, periods=252):
    """Is your strategy just a de-risked version of the benchmark?

    Scales the benchmark with cash to your strategy's volatility, then compares.
    A lower-risk portfolio will always look better on Sharpe; this asks whether
    it beats the trivial way of getting the same risk.
    """
    s = pd.Series(strategy).dropna()
    b = pd.Series(benchmark).reindex(s.index)
    rf = (pd.Series(riskfree).reindex(s.index).ffill().fillna(0.0)
          if riskfree is not None else pd.Series(0.0, index=s.index))
    w = s.std() / b.std()
    scaled = w * b + (1 - w) * rf
    return dict(benchmark_weight=float(w), strategy=stats(s),
                benchmark_vol_matched=stats(scaled),
                d_cagr_pp=100 * (stats(s)["cagr"] - stats(scaled)["cagr"]),
                d_sharpe=stats(s)["sharpe"] - stats(scaled)["sharpe"])


def weight_perturbation(sleeve_returns, base_weights, cost_bp=None, freq="Q",
                        nudges=((0, 0), (-.10, .10), (.10, -.10),
                                (-.05, .05), (.05, -.05))):
    """Fitted weights collapse when nudged; robust ones do not.

    Shifts weight between the largest sleeve and the rest and reports the spread
    of outcomes. A Sharpe range of a few hundredths across five mixes is a good
    sign. A range of several tenths means you fitted the weights.
    """
    rows = []
    big = max(base_weights, key=lambda k: base_weights[k])
    others = [k for k in base_weights if k != big]
    for d_big, d_rest in nudges:
        w = dict(base_weights)
        w[big] = max(0.0, w[big] + d_big)
        for k in others:
            w[k] = max(0.0, w[k] + d_rest / len(others))
        r, t = rebalanced(sleeve_returns, w, cost_bp, freq)
        s = stats(r)
        s["weights"] = {k: round(v, 3) for k, v in w.items()}
        s["turnover"] = t
        rows.append(s)
    sh = [x["sharpe"] for x in rows]
    return dict(rows=rows, sharpe_min=min(sh), sharpe_max=max(sh),
                sharpe_range=max(sh) - min(sh),
                verdict=("robust to weight choice" if max(sh) - min(sh) < 0.12
                         else "FRAGILE -- the weights are doing the work"))


def regime_split(series_map, buckets=(("2008", "2021"), ("2022", "2026"))):
    """Stats per period for several strategies at once.

    series_map : {name: daily return Series}

    Correlations and sleeve behaviour are not stationary. Treasuries diversified
    equities for four decades and then stopped in 2022. Always report the
    buckets separately; never pool them and quote one number.
    """
    out = {}
    for name, r in series_map.items():
        out[name] = {}
        for lo, hi in buckets:
            seg = pd.Series(r).loc[lo:hi].dropna()
            out[name]["%s-%s" % (lo, hi)] = stats(seg) if len(seg) > 60 else None
    return out


def episode_table(series_map, crises=None):
    """Worst drawdown per named stress episode, per strategy.

    The package's checklist asks how many INDEPENDENT episodes support a claim.
    Twenty-five years holds perhaps eight macro events, not 6,300 independent
    days. If the benefit lives in one of them, say so.
    """
    crises = crises or CRISES
    rows = []
    for lo, hi, nm in crises:
        row = {"episode": nm, "window": "%s..%s" % (lo, hi)}
        for name, r in series_map.items():
            seg = pd.Series(r).loc[lo:hi].dropna()
            row[name] = 100 * max_drawdown(seg)[0] if len(seg) > 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def full_battery(returns, regime, riskfree=None, cost_bp=20,
                 bear_exposure=0.0, benchmark=None, n_placebo=40):
    """Everything at once. Returns a printable report string.

    Pass `benchmark` (e.g. an index return series) to include the vol-matched
    test. Anything that fails here does not go into production.
    """
    from .compare import compare_signals
    from .benchmarks import build_benchmarks

    r = pd.Series(returns).dropna()
    g = pd.Series(regime).reindex(r.index).ffill().dropna()
    r = r.reindex(g.index)

    L = []
    L.append("=" * 78)
    L.append("CONTROL BATTERY   %s .. %s   (%d sessions, %.1f years)"
             % (r.index.min().date(), r.index.max().date(), len(r), len(r) / 252))
    L.append("=" * 78)

    # 1. exposure-matched
    em = exposure_matched_control(r, g, riskfree, cost_bp, bear_exposure)
    L.append("\n1. EXPOSURE-MATCHED CONTROL  (mean exposure %.1f%%)"
             % (100 * em["mean_exposure"]))
    L.append("   overlay      max DD %7.2f%%   Calmar %.3f   Sharpe %.3f"
             % (100 * em["overlay"]["max_dd"], em["overlay"]["calmar"],
                em["overlay"]["sharpe"]))
    L.append("   flat control max DD %7.2f%%   Calmar %.3f   Sharpe %.3f"
             % (100 * em["flat_control"]["max_dd"], em["flat_control"]["calmar"],
                em["flat_control"]["sharpe"]))
    L.append("   difference   %+.2f pp drawdown, %+.3f Calmar   -> %s"
             % (em["d_maxdd_pp"], em["d_calmar"], em["verdict"]))

    # 2. placebo, control and moving average
    sigs = build_benchmarks(r)
    sigs["your signal"] = g
    try:
        df = compare_signals(r, sigs, riskfree=riskfree, cost_bp=cost_bp,
                             bear_exposure=bear_exposure, n_placebo=n_placebo)
        row = df.loc["your signal"]
        ctrl = [i for i in df.index if "always" in str(i).lower()]
        L.append("\n2. PLACEBO AND BENCHMARK COMPARISON")
        L.append("   your signal   Sharpe %.3f   DD cut %3.0f%%   %.1f switches/yr"
                 % (row["sharpe"], row["dd_reduction"], row["switches_yr"]))
        if "placebo_pct" in df.columns and np.isfinite(row.get("placebo_pct", np.nan)):
            L.append("   vs %d random signals at the same switching rate: %.0fth percentile%s"
                     % (df.attrs.get("placebo_n", n_placebo), row["placebo_pct"],
                        "" if row["placebo_pct"] >= 80 else "   <-- FAILS PLACEBO"))
        if "200d moving average" in df.index:
            d = row["sharpe"] - df.loc["200d moving average", "sharpe"]
            L.append("   vs a 200-day moving average: %+.3f Sharpe%s"
                     % (d, "" if d >= 0.15 else "   <-- ship the moving average instead"))
        if ctrl:
            dc = df.loc[ctrl[0], "d_sharpe"]
            L.append("   harness check: buy-and-hold moved %+.3f Sharpe%s"
                     % (dc, "  (clean)" if abs(dc) <= 0.02 else "   <-- TIMING BUG"))
    except Exception as e:
        L.append("\n2. comparison unavailable: %s" % str(e)[:70])

    # 3. vol-matched
    if benchmark is not None:
        ov = apply_overlay(r, g, riskfree=riskfree, cost_bp=cost_bp,
                           bear_exposure=bear_exposure)["net"]
        vm = vol_matched(ov, pd.Series(benchmark).reindex(ov.index), riskfree)
        L.append("\n3. VOL-MATCHED BENCHMARK  (benchmark scaled to %.0f%% weight)"
                 % (100 * vm["benchmark_weight"]))
        L.append("   strategy            CAGR %6.2f%%   Sharpe %.3f"
                 % (100 * vm["strategy"]["cagr"], vm["strategy"]["sharpe"]))
        L.append("   benchmark, matched  CAGR %6.2f%%   Sharpe %.3f"
                 % (100 * vm["benchmark_vol_matched"]["cagr"],
                    vm["benchmark_vol_matched"]["sharpe"]))
        L.append("   difference          %+.2f pp CAGR, %+.3f Sharpe%s"
                 % (vm["d_cagr_pp"], vm["d_sharpe"],
                    "" if vm["d_cagr_pp"] > 0 else "   <-- just a de-risked benchmark"))

    # 4. episodes
    ov = apply_overlay(r, g, riskfree=riskfree, cost_bp=cost_bp,
                       bear_exposure=bear_exposure)["net"]
    ep = episode_table({"baseline": r, "overlaid": ov})
    ep = ep.dropna(subset=["baseline"])
    if len(ep):
        won = int((ep["overlaid"] > ep["baseline"]).sum())
        L.append("\n4. EPISODES  (%d stress windows in this sample)" % len(ep))
        for _, e in ep.iterrows():
            L.append("   %-22s baseline %7.2f%%   overlaid %7.2f%%"
                     % (e["episode"], e["baseline"], e["overlaid"]))
        L.append("   overlay shallower in %d of %d episodes" % (won, len(ep)))
        L.append("   NOTE: this is the count that matters, not the number of days.")

    # 5. noise
    from .validate import sharpe_standard_error
    se = sharpe_standard_error(0.6, len(r) / 252)
    L.append("\n5. HOW MUCH OF THIS IS NOISE")
    L.append("   Sharpe standard error over %.1f years: %.3f" % (len(r) / 252, se))
    L.append("   Differences smaller than that are not differences. Count every")
    L.append("   configuration you tried and compare against deflation_hurdle().")
    L.append("=" * 78)
    return "\n".join(L)
