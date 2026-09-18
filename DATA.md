# Data

## What ships with this repo

| File | What it is | Redistributable |
|---|---|---|
| `data/regime_spy_long.csv` | Pre-fitted bull/bear regime, 2004-01 → 2025-12. A 0/1 series. | Yes — derived signal, not prices |
| `data/prob_spy_long.csv` | The classifier probabilities behind it, before the hysteresis gate | Yes |

The regime file is the useful one. It is a completed walk-forward fit on SPY —
17.2% bear days, 4.4 switches/yr — so you can apply the overlay to your own book
today without fitting anything or installing `jumpmodels`/`xgboost`.

## What does not ship

`data/sleeve_closeadj.parquet` — the 27-ETF total-return price frame. This is
licensed vendor data (Sharadar SFP) and the subscription terms do not permit
redistribution. It is gitignored.

Rebuild it with your own credentials:

```bash
export SHARADAR_API_KEY=...            # your own key
python data/fetch_sleeves.py --source sharadar
```

No Sharadar access? There is a free fallback:

```bash
pip install yfinance
python data/fetch_sleeves.py --source yfinance
```

Coverage and adjustment conventions differ slightly, so numbers will not tie out
to `FINDINGS.md` to the basis point. None of the conclusions depend on that.

## Bring your own

Nothing downstream knows where prices came from. Any frame in this shape works:

```
index    DatetimeIndex, daily
columns  one per ticker
values   TOTAL-RETURN adjusted close (distributions reinvested)
```

One trap worth naming: use an **adjusted** close, not a raw one. Raw prices
silently understate every bond and commodity sleeve by their entire yield, which
is most of the return for those sleeves.
