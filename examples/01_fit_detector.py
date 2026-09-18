"""Fit the regime detector on any daily price series.

Takes 2-3 minutes for 30 years. The output is a 0/1 Series you save once and
reuse -- never refit it inside a backtest loop.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawdown_kit import RegimeDetector, crisis_report

HERE = Path(__file__).resolve().parents[1]

px = pd.read_parquet(HERE / "data/sleeve_closeadj.parquet")["SPY"].dropna()
returns = px.pct_change().dropna()
print("input: %d sessions, %s .. %s"
      % (len(returns), returns.index.min().date(), returns.index.max().date()))

det = RegimeDetector()          # defaults: 5+1yr windows, annual refit,
                                # penalties 41/61/81, threshold 0.70
res = det.fit_predict(returns, cost_bp=5)
res.print_summary()
print(crisis_report(res.regime).to_string(index=False))

out = HERE / "data/my_regime.csv"
res.regime.to_frame("regime").to_csv(out)
print("\nsaved ->", out)
print("You need train_years + val_years of history before the FIRST signal.")
print("Fit on a long proxy series, not on a short book, or the warm-up will eat")
print("the only crisis you wanted to test against.")
