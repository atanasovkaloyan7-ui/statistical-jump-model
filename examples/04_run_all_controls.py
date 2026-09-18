"""The battery. Nothing goes to production until this is clean."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawdown_kit import full_battery, assert_causal, build_features

HERE = Path(__file__).resolve().parents[1]
px = pd.read_parquet(HERE / "data/sleeve_closeadj.parquet")
returns = px["SPY"].pct_change().dropna()
rf = px["BIL"].pct_change()
regime = pd.read_csv(HERE / "data/regime_spy_long.csv",
                     parse_dates=[0], index_col=0)["regime"]

# Prove the feature code is causal before anything else. This scrambles every
# observation after a cutoff and requires the past to come back bit-identical.
assert_causal(lambda d: build_features(d["ret"]),
              returns.to_frame("ret"), name="features")
print("assert_causal: PASSED\n")

print(full_battery(returns, regime, riskfree=rf, cost_bp=5,
                   bear_exposure=0.0, benchmark=returns))
