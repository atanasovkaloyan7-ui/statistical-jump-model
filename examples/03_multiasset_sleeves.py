"""The full stack: overlay the equity sleeve, then blend static sleeves."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawdown_kit import (load_sleeves, build_portfolio, rebalanced, stats,
                          freeze_config, regime_split, weight_perturbation)

HERE = Path(__file__).resolve().parents[1]
R = load_sleeves(HERE / "data/sleeve_closeadj.parquet",
                 ["SPY", "TLT", "GLD", "DBC", "BIL", "IEF"]).loc["2008-04-01":]
regime = pd.read_csv(HERE / "data/regime_spy_long.csv",
                     parse_dates=[0], index_col=0)["regime"]
R = R.reindex(R.index.intersection(regime.dropna().index)).dropna()

# Declare the weights BEFORE running anything, and hash them.
WEIGHTS = {"EQUITY": 0.60, "TLT": 0.15, "GLD": 0.15, "DBC": 0.10}
cfg_hash = freeze_config(dict(weights=WEIGHTS, bear=0.50, freq="Q"))
print("frozen config:", cfg_hash, "\n")

print("%-40s %8s %9s %8s %8s" % ("", "CAGR", "max DD", "Sharpe", "Calmar"))


def show(name, s):
    print("%-40s %7.2f%% %8.2f%% %8.3f %8.3f"
          % (name, 100 * s["cagr"], 100 * s["max_drawdown"], s["sharpe"], s["calmar"]))


show("equity sleeve alone", stats(R["SPY"]))
plain = build_portfolio(R, "SPY", regime=None, weights=WEIGHTS)
show("sleeves, no overlay", plain["stats"])
full = build_portfolio(R, "SPY", regime=regime, weights=WEIGHTS, bear_exposure=0.50)
show("sleeves + overlay", full["stats"])
print("   mean equity exposure %.1f%%, %.2fx annual turnover"
      % (100 * full["mean_equity_exposure"], full["turnover"]))

print("\n--- is it fragile? ---")
wp = weight_perturbation(R, {"SPY": .60, "TLT": .15, "GLD": .15, "DBC": .10})
print("Sharpe across five weight mixes: %.3f to %.3f  ->  %s"
      % (wp["sharpe_min"], wp["sharpe_max"], wp["verdict"]))

print("\n--- does it hold in both correlation regimes? ---")
rs = regime_split({"sleeves + overlay": full["portfolio"], "equity alone": R["SPY"]})
for nm, buckets in rs.items():
    for per, s in buckets.items():
        if s:
            print("   %-22s %-12s CAGR %6.2f%%  Sharpe %.3f"
                  % (nm, per, 100 * s["cagr"], s["sharpe"]))
