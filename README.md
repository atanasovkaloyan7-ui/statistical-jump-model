# drawdown_kit

Drawdown-preservation machinery, extracted so it can be bolted onto **any** signal.

There is **no stock selection in here.** No factors, no universe construction, no
alpha model. This layer only ever answers *how much to hold*. You bring the
*what*.

---

## Why this exists

Over ~18 years of testing on a hand-built equity book, a long list of
risk-reduction ideas were tried. Almost all of them cut drawdown by cutting
return by more. Two survived their controls:

| | What it does | Measured |
|---|---|---|
| **The regime overlay** | Scales equity exposure with a walk-forward bull/bear signal | Max DD −43.8% → −33.8% on the book; beats an exposure-matched flat book by 17.5 pp |
| **Static multi-asset sleeves** | Fixed weights across weakly correlated assets, rebalanced | Sharpe 0.533 → 0.819; Calmar 0.216 → 0.411 |

Together, on an index equity sleeve, 2008–2025: **CAGR 9.13%, max drawdown
−22.5%, Sharpe 0.922** against SPY's 11.55% / −51.5% / 0.649.

Full numbers and every failure in [`FINDINGS.md`](FINDINGS.md).

---

## Install

```bash
pip install -r requirements.txt
```

Then copy `drawdown_kit/` into your project, or add this folder to your path.
`jumpmodels` and `xgboost` are needed **only** to fit a new detector — the
overlay, sleeves, controls, benchmarks and leakage guards all work without them.

**Price data is not bundled** (licensed vendor terms). Rebuild it with your own
key, or use the free fallback — see [`DATA.md`](DATA.md):

```bash
python data/fetch_sleeves.py --source sharadar   # your own SHARADAR_API_KEY
python data/fetch_sleeves.py --source yfinance   # free, slightly different
```

The pre-fitted regime signal (`data/regime_spy_long.csv`) **does** ship, so you
can start overlaying your own book immediately without fitting anything.

---

## Three ways to use it

### 1. You already have a signal

The whole interface is one function. Any 0/1 Series indexed like your returns
works — another model, a vendor feed, a hand-written rule.

```python
from drawdown_kit import apply_overlay, evaluate, print_evaluation

ov = apply_overlay(returns, my_regime, riskfree=rf, cost_bp=5, bear_exposure=0.5)
print_evaluation(evaluate(ov, returns))
```

### 2. You want a signal

```python
from drawdown_kit import RegimeDetector

det = RegimeDetector()                       # defaults are the tuned ones
res = det.fit_predict(returns, riskfree=rf)  # ~2-3 min for 30 years
res.regime.to_csv("my_regime.csv")           # save it; never refit in a loop
```

A pre-fitted US-equity regime is bundled: `data/regime_spy_long.csv`, walk-forward
on SPY, 2004-01 → 2025-12, 17.2% bear days, 4.4 switches/yr. Use it directly if
your book is US-equity-shaped.

### 3. The full stack

```python
from drawdown_kit import load_sleeves, build_portfolio, freeze_config

WEIGHTS = {"EQUITY": 0.60, "TLT": 0.15, "GLD": 0.15, "DBC": 0.10}
freeze_config(dict(weights=WEIGHTS, bear=0.50, freq="Q"))   # hash it FIRST

R = load_sleeves("data/sleeve_closeadj.parquet", ["SPY","TLT","GLD","DBC","BIL"])
out = build_portfolio(R, "SPY", regime=my_regime, weights=WEIGHTS)
print(out["stats"])
```

Run the examples in order — they are the documentation:

```bash
python examples/01_fit_detector.py         # fit a detector on any price series
python examples/02_overlay_your_signal.py  # bring your own 0/1 signal
python examples/03_multiasset_sleeves.py   # the full stack
python examples/04_run_all_controls.py     # the battery (run this before believing anything)
python examples/05_deploy_live.py          # refit + daily signal loop
```

### Deploying it

The jump model is a **training-time object** — it never runs in production.
Today's signal is one forward pass through a frozen classifier:

```python
from drawdown_kit import fit_and_freeze, LiveSignal

fit_and_freeze(returns_through_last_year, "artifacts/")   # occasionally
sig = LiveSignal("artifacts/", bear_exposure=0.50)
out = sig.update(returns_through_today)                   # every day
out["target_exposure"]   # -> trade this at the NEXT close
```

`LiveSignal` persists the hysteresis state to disk between runs. That matters:
the gate is path-dependent, so a process restart that reinitialised to bull
would silently put you fully invested in the middle of a bear regime.

---

## The controls are the actual product

Anyone can produce a backtest where de-risking reduces drawdown — cutting
exposure *always* reduces drawdown. These are the tests that separate a signal
from a switching rate. Run them all:

```python
from drawdown_kit import full_battery
print(full_battery(returns, my_regime, riskfree=rf, cost_bp=5, benchmark=spy))
```

| Control | The question it answers |
|---|---|
| `exposure_matched_control` | **The decisive one.** Did the timing beat simply holding less, on average, the whole time? |
| `vol_matched` | Is this just a de-risked index fund? |
| `weight_perturbation` | Do the results collapse when you nudge the weights? |
| `episode_table` | How many *independent* episodes support the claim — not how many days? |
| `regime_split` | Does it hold in both correlation regimes, or only the friendly one? |
| `assert_causal` | Does the past come back bit-identical when you scramble the future? |
| `deflation_hurdle` | What would the best of N pure-noise configurations have scored? |
| placebo (in `compare_signals`) | Does it beat random signals with the same switching rate? |
| always-bull control | Does buy-and-hold come out unchanged? If not, it's a timing bug. |

---

## Non-negotiables

1. **`execution_lag >= 1`.** Filling on the bar that generated the signal is
   worth +1.0 to +1.5 Sharpe of pure illusion and 0.00 on buy-and-hold —
   which is the control that proves it's a fill artefact. `apply_overlay`
   raises below 1. Do not remove that check.
2. **Jump-model labels never trade.** The SJM optimises the whole state path by
   dynamic programming, so the label on a Tuesday depends on the following
   April. They are training targets for the classifier and nothing else.
3. **Never overlay duration or IG credit.** We measured this overlay taking
   Treasury Sharpe 0.230 → 0.089 and Corporate 0.347 → −0.033. Trend filters
   belong on trending, fat-left-tailed series. Hold bonds straight.
4. **Fit the detector on a long proxy, not on a short book.** You need
   `train_years + val_years` before the first signal. Fit on a 6-year book and
   the warm-up eats the only crisis you wanted to test.
5. **Sell it on drawdown, never on Sharpe.** The drawdown reduction is robust
   and large. The Sharpe change is usually inside the standard error.
6. **Declare weights before you run.** `freeze_config()` is one line. A
   specification changed after seeing a result is a new trial — count it.

---

## Layout

```
drawdown_kit/
  detector.py     SJM -> XGBoost -> hysteresis gate. The signal factory.
  features.py     12 causal features; TrainScaler (fit on train only)
  overlay.py      apply_overlay, evaluate, crisis_report, the lag guard
  sleeves.py      static multi-asset layer: rebalanced(), build_portfolio()
  live.py         deployment: fit_and_freeze(), LiveSignal, label-flip alarm
  controls.py     the battery
  benchmarks.py   MA200, 50/200, momentum, vol-quantile, random, always-bull
  compare.py      score many signals through one identical harness
  validate.py     assert_causal, freeze_config, deflation_hurdle, contracts
data/
  regime_spy_long.csv       pre-fitted SPY regime, 2004-2025 (ready to use)
  prob_spy_long.csv         the probabilities behind it, before the gate
  fetch_sleeves.py          rebuild the price frame with your own credentials
  sleeve_closeadj.parquet   (gitignored -- see DATA.md)
docs/
  Regime_Overlay_How_It_Works.pdf   17-page technical doc on the detector
examples/                   01 -> 04, run them in order
FINDINGS.md                 every result, including the failures
```

---

## Provenance

The detector derives from a replication study of Luo, Y. and J. M. Mulvey
(2026), *Regime-Aware Asset Allocation with Dual-Regime Signals and
Regime-Dependent Asset Selection*, SSRN 6933278. The method — statistical jump
models for regime identification followed by supervised forecasting — is theirs
and is well founded. **The paper's 1.43 Sharpe headline did not replicate**; under
an honest fill convention it measured 0.25–0.52, and the paper's own dynamic
asset-selection layer scored 0.249 against 0.826 for a plain 60/40. See
`FINDINGS.md` § "What we did not keep".

Every performance figure in this folder is our own out-of-sample measurement.

Not investment advice.
