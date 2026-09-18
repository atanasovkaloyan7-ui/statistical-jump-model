# Findings

Everything measured while building this, including what failed. Kept so the same
ground is not re-walked.

All figures are our own out-of-sample measurements, after costs, with T+1 fills.
Unless stated otherwise the window is **2 Apr 2008 – 31 Dec 2025** (17.7 years)
and the equity sleeve pays **20 bp per traded side**.

---

## 1. What worked

### 1.1 The regime overlay

Walk-forward bull/bear detector (jump model → XGBoost → 0.70 hysteresis gate),
fit on **SPY alone**, applied to the equity exposure.

Frozen signal in `data/regime_spy_long.csv`: 2004-01 → 2025-12, 5,535 sessions,
bear on 17.2% of days, 96 switches (4.4/yr).

**Crisis detection** — the most robust finding in the whole programme:

| Episode | Days flagged bear |
|---|---|
| Financial crisis 2007-10 → 2009-03 | 86.5% |
| Q4 2018 | 88.9% |
| Rate hikes 2022 | 67.9% |
| COVID 2020 | 59.7% |
| Euro / debt ceiling 2011 | 56.5% |
| China / oil 2015-16 | 34.5% |
| **Calm market 2013–14** | **0.0%** |

Defensive through crises, silent in quiet markets. Weakest on slow grinding
declines — it finds sharp breaks, not long bleeds.

**Against the exposure-matched control** (same average exposure, held flat, no
timing) — the test that killed every other de-risking idea tried:

| Book, 2004–2023, 20 bp | Mean exposure | Overlay max DD | Flat control max DD | Extra DD saved | Calmar gain |
|---|---|---|---|---|---|
| bear 0% | 81.3% | −33.79% | −51.26% | **17.5 pp** | +0.033 |
| bear 25% | 85.9% | −36.64% | −53.44% | 16.8 pp | +0.045 |
| bear 50% | 90.6% | −44.74% | −55.54% | 10.8 pp | +0.027 |

It clears it. That is the strongest single result here.

### 1.2 Static multi-asset sleeves

| Sleeve | 2008–2021 | 2022–2025 |
|---|---|---|
| **TLT** treasury | +6.3%/yr, corr −0.41 | **−9.3%/yr, corr +0.10** |
| **DBC** commodity | −3.5%/yr, corr +0.47 | **+5.5%/yr**, corr +0.24 |
| **GLD** gold | +5.0%/yr, corr +0.05 | +23.5%/yr, corr +0.09 |

The best single-sleeve bet in the first regime was the worst in the second.
Correspondingly, so was the best weight scheme:

| Weight set (overlay on equity) | Sharpe 2008–2021 | Sharpe 2022–2025 |
|---|---|---|
| no commodity 60/20/20 | **0.883** | 0.430 |
| primary 60/15/15/10 | 0.755 | 0.436 |
| gold only 75/25 | 0.698 | 0.639 |
| no treasury 60/25/15 | 0.594 | **0.736** |

**Nothing available at the start told you which was coming.** The ensemble is
mid-table in both — that is what it is for, and why it should be declared rather
than fitted.

### 1.3 The two together

| SMC stock book as the equity sleeve | CAGR | Max DD | Sharpe | Calmar |
|---|---|---|---|---|
| book alone | 9.44% | −43.78% | 0.533 | 0.216 |
| + overlay | 8.11% | −33.76% | 0.544 | 0.240 |
| + sleeves, no overlay | 8.28% | −30.93% | 0.664 | 0.268 |
| **+ sleeves + overlay** | **7.30%** | **−19.99%** | **0.680** | **0.365** |

| Index equity sleeve (SPY) | CAGR | Vol | Max DD | Sharpe | Calmar |
|---|---|---|---|---|---|
| **60 SPY / 15 TLT / 15 GLD / 10 DBC, overlaid** | **9.13%** | 10.0% | **−22.51%** | **0.922** | **0.406** |
| SPY buy and hold | 11.55% | 19.9% | −51.49% | 0.649 | 0.224 |
| SPY + cash at matched 10% vol | 6.82% | 10.0% | −28.45% | 0.710 | 0.240 |

Against **vol-matched** SPY it adds **+2.31 pp of CAGR at identical risk** — so
it is not merely a de-risked index fund. Roughly two-thirds of that comes from
the overlay, one-third from the sleeves.

Robust to weight choice: Sharpe 0.875–0.944 across five different mixes.

---

## 2. What failed

### 2.1 Dynamic asset selection (the source paper's own mechanism)

BMDA/BMGA: a second regime model per sleeve, rebuilding the investable universe
daily. Fully implemented and run twice on the 10-sleeve universe.

| 14.5 years OOS, 2012–2026 | Sharpe | Max DD | Cost drag |
|---|---|---|---|
| Plain 60/40 | **0.826** | −26.4% | 0% |
| SPY buy and hold | 0.809 | −33.8% | 0% |
| 0/1 RiskFree–SP500 (global layer only) | 0.680 | −20.3% | 0.13% |
| **0/1 BMDA–BMGA** (paper claims 1.43) | **0.249** | −33.8% | 2.01% |
| same, global layer removed | 0.164 | −32.1% | 1.98% |

Three diagnosed causes:

1. **The per-sleeve signal has no skill.** Against the paper's claim that the
   0/1 rule roughly doubles Sharpe for every asset: **0 of 8 sleeves improved by
   their own regime signal**, mean uplift −0.045. It actively *hurts* Treasury
   (0.230 → 0.089) and Corporate (0.347 → −0.033).
2. **The basket doesn't behave as described.** The paper says equities are
   rarely selected in bear markets; measured bear-regime selection was MidCap
   38.6%, EAFE 39.8%, HighYield 45.3%, REIT 32.5%. Treasury only 46.3%.
3. **Turnover eats it.** 9.2% daily turnover, 2.0%/yr drag. Scores 0.470 at zero
   cost, 0.215 at 10 bp.

Ablation: removing the global layer entirely costs only 0.069–0.085 Sharpe,
below the 0.15 bar for keeping a complication. Deflated Sharpe probability 0.002.

The paper's cost sensitivity table is also arithmetically inconsistent — implied
turnover backs out to ~1.35%/day regardless of the 8.6% reported.

### 2.2 Fitting the detector on the book's own NAV

| Detector fitted on | First signal | Bear days | Crisis in test window? | Placebo | DD cut |
|---|---|---|---|---|---|
| the book's own NAV | Apr 2014 | 40.0% | no, warm-up ate it | 64th, **fails** | 25% |
| SPY, 2008 onward | Apr 2014 | 15.7% | no, warm-up ate it | 76th, **fails** | 27% |
| **SPY, 1998 onward** | Jan 2004 | 17.2% | yes, fully OOS | 85th–90th, clears | **43%** |

Bear on 40% of days is the tell — fitted on its own NAV the detector was
defensive through 100% of Q4 2018 and 100% of 2022, which is a model with too
little history latching onto a level.

### 2.3 Everything else tried on the equity book

Each cut drawdown but cut return by more, and each lost to its exposure-matched
control:

- Volatility targeting (20% target, 50–100% budget): CAGR 9.37% → 7.14%, Sharpe 0.521 → 0.449
- HMM / credit / ISM macro filter: 5.83% vs 8.08% for a flat 70% — lost to simply holding less
- Fixed exposure budgets (60/65/70%)
- SMA50 / SMA100 / SMA200 entry screens

### 2.4 Levers that don't work for CAGR

Priced in CAGR gained per point of extra drawdown:

| Lever | Δ CAGR | Δ Drawdown | Rate |
|---|---|---|---|
| Cut trading cost 20 → 5 bp | +0.43pp | 0 | **free** |
| Split equity 30% book / 30% index | +0.93pp | 0.0pp | **free** |
| Drop a trend entry screen | +0.61pp | +1.1pp | 0.55 |
| Turn the overlay off | +0.95pp | +10.9pp | 0.09 |
| **Raise equity 60% → 100%** | +0.82pp | **+13.8pp** | **0.06** |

**Turning up risk is the worst available lever.** It just undoes the work.

---

## 3. Structural limits worth knowing before you start

### 3.1 You cannot measure the improvement you are hoping for

| Sample | Sharpe SE | 95% interval around a measured 0.55 |
|---|---|---|
| 3 years | 0.619 | [−0.66, 1.76] |
| 15 years | 0.278 | **[0.01, 1.09]** |
| 30 years | 0.196 | [0.17, 0.93] |

A Sharpe of 1.0 sits inside the confidence interval of a measured 0.55.
Distinguishing them at 95% needs ~44 years.

### 3.2 The search itself manufactures the number

Expected best Sharpe from **pure noise** at 15 years:

| Trials | 5 | 20 | 100 | 500 | 1000 |
|---|---|---|---|---|---|
| Best by luck | 0.33 | 0.53 | **0.70** | 0.85 | 0.91 |

Count every configuration, including abandoned ones. Use `deflation_hurdle()`.

### 3.3 Correlated signals cannot escape their ceiling

Combining N streams of Sharpe 0.50:

| Correlation | N=1 | N=3 | N=5 | N=10 | N=30 |
|---|---|---|---|---|---|
| 0.0 | 0.50 | 0.87 | **1.12** | 1.58 | 2.74 |
| 0.5 | 0.50 | 0.61 | 0.65 | 0.67 | 0.70 |
| 0.8 | 0.50 | 0.54 | 0.55 | 0.55 | **0.56** |

Stacking more equity-correlated signals is the bottom row. This is why
diversification moved the numbers roughly ten times as much as any signal work,
and it was predicted by the arithmetic before it was measured.

---

## 4. Honest caveats on everything above

- **The overlay's Sharpe benefit is close to one episode.** On the stock book it
  added +0.183 Sharpe in 2008–09 and was *negative* in 2015–19 (−0.264) and
  2020–22 (−0.060). The drawdown benefit is broader — shallower in 6 of 7 stress
  episodes — but the Sharpe case rests on the financial crisis.
- **Costs are assumed, not measured.** At 40 bp per side the overlay loses 5.0 pp
  of CAGR and Calmar falls to 0.107. This is the single input that decides
  whether any of it is worth running, and it has never been measured on real
  fills. Measure yours first.
- **The treasury sleeve may be permanently impaired.** Its diversification
  through 2021 was a 40-year duration bull market. Correlation flipped to +0.10
  in 2022–2025 with −9.3%/yr returns.
- **2008–2025 flatters US large cap.** SPY's standalone Sharpe of 0.649 over the
  window is well above its long-run ~0.4–0.5. Anything loaded toward large cap is
  partly a bet that continues.
- **None of this is out-of-sample.** The window was examined repeatedly while
  the work was done. A genuinely untouched test needs forward paper testing or a
  market never run on.
- **Sharpe above 1 is not reachable** in a long-only, unlevered, single-asset-class
  mandate. Reaching 1.0 from a market Sharpe of 0.55 needs a sustained
  information ratio of 0.84. The unlevered constraint is also what stops you
  converting a Sharpe 0.92 portfolio into more money than SPY.

---

## 5. Sources

- Detector and technical documentation: `docs/Regime_Overlay_How_It_Works.pdf`
- Luo, Y. and J. M. Mulvey (2026), SSRN 6933278 — the method's origin; headline
  did not replicate
- Bemporad et al. (2018), *Fitting jump models*, Automatica 96 — the SJM
- Lo (2002) — Sharpe standard errors
- Bailey et al. (2015) — deflated Sharpe / backtest overfitting
- Sleeve prices: Sharadar SFP `closeadj` (total-return adjusted), 27 ETFs
