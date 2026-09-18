"""Already have a 0/1 signal? Skip the detector. This is the whole interface."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawdown_kit import (apply_overlay, evaluate, print_evaluation,
                          exposure_matched_control)

HERE = Path(__file__).resolve().parents[1]
px = pd.read_parquet(HERE / "data/sleeve_closeadj.parquet")
returns = px["SPY"].pct_change().dropna()          # <- your book's returns
rf = px["BIL"].pct_change()                        # <- your cash sleeve

# Any 0/1 Series indexed like returns works. Nothing here knows or cares where
# it came from: another model, a vendor feed, a hand-written rule.
regime = pd.read_csv(HERE / "data/regime_spy_long.csv",
                     parse_dates=[0], index_col=0)["regime"]

ov = apply_overlay(returns, regime, riskfree=rf, cost_bp=5, bear_exposure=0.50)
print_evaluation(evaluate(ov, returns.reindex(ov.index)))

print("\n--- the control that matters ---")
em = exposure_matched_control(returns, regime, riskfree=rf, cost_bp=5,
                              bear_exposure=0.50)
print("mean exposure %.1f%%" % (100 * em["mean_exposure"]))
print("overlay vs a flat book at the SAME average exposure:")
print("   %+.2f pp drawdown, %+.3f Calmar" % (em["d_maxdd_pp"], em["d_calmar"]))
print("   ->", em["verdict"])
