"""Deployment: refit occasionally on closed history, signal daily.

The jump model runs ONCE here, at refit. The daily loop never touches it.
"""
import sys, shutil
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drawdown_kit import fit_and_freeze, LiveSignal, check_label_stability

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "artifacts_demo"
if ART.exists():
    shutil.rmtree(ART)

px = pd.read_parquet(HERE / "data/sleeve_closeadj.parquet")["SPY"].dropna()
returns = px.pct_change().dropna()

# ---- STEP 1: refit, on a CLOSED window ---------------------------------
TRAIN_END = "2024-12-31"
print("refitting on data through %s ..." % TRAIN_END)
meta = fit_and_freeze(returns.loc[:TRAIN_END], ART)
print("   trained %s .. %s  (%d sessions), bull share %.1f%%\n"
      % (meta["train_start"], meta["train_end"], meta["n_train"],
         100 * meta["bull_share"]))

# ---- STEP 2: the daily loop -------------------------------------------
# Everything after TRAIN_END is unseen. Walk it one day at a time, exactly as
# production would: yesterday's state in, today's target exposure out.
sig = LiveSignal(ART, bear_exposure=0.50)
live = returns.loc["2025-01-01":]
print("walking %d unseen sessions, one at a time\n" % len(live))

switches = []
for day in live.index:
    hist = returns.loc[:day]          # only data that exists on that date
    out = sig.update(hist, asof=day)
    if out.get("switched"):
        switches.append(out)

print("%-12s %8s %8s  %s" % ("date", "P(bull)", "target", "note"))
for s in switches:
    print("%-12s %8.3f %7.0f%%  %s"
          % (s["asof"], s["p_bull"], 100 * s["target_exposure"], s["note"]))
print("\n%d switches in %d sessions" % (len(switches), len(live)))

final = sig.update(returns, asof=returns.index[-1])
print("\ntoday: %s  P(bull) %.3f  ->  %s, hold %.0f%%  (fill at the %s)"
      % (final["asof"], final["p_bull"], final["regime_label"],
         100 * final["target_exposure"], final["fill"]))
print("model trained through %s, %d days ago -> refit needed: %s"
      % (final["model_trained_through"], sig.days_since_refit(),
         sig.needs_refit()))

# ---- STEP 3: what to check at the NEXT refit --------------------------
print("\nat the next refit, before shipping it:")
new_meta = fit_and_freeze(returns, ART / "_candidate")
for p in check_label_stability(meta, new_meta) or ["   no label-stability problems"]:
    print("   " + p if not p.startswith("   ") else p)
shutil.rmtree(ART / "_candidate", ignore_errors=True)
