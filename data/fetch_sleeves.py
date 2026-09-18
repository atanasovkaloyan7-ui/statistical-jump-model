"""Fetch the sleeve price frame.

The price data itself is NOT redistributed with this repo -- it is licensed
vendor data and the terms do not allow it. This script rebuilds the exact frame
the examples expect, using your own credentials.

    python data/fetch_sleeves.py --source sharadar
    python data/fetch_sleeves.py --source yfinance     # free fallback

Output: data/sleeve_closeadj.parquet
    index   DatetimeIndex, daily
    columns one per ticker, TOTAL-RETURN adjusted close
             (distributions reinvested -- NOT raw price)

Any frame in that shape works. If you have your own data source, write your own
loader and skip this entirely; nothing downstream knows where the prices came
from.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "sleeve_closeadj.parquet"

SLEEVES = {
    # equity
    "SPY": "us large cap", "IJH": "us mid cap", "IWM": "us small cap",
    "IJR": "us small cap 600", "EFA": "developed ex-US", "EEM": "emerging",
    "VNQ": "us reit",
    # duration and cash
    "SHY": "treasury 1-3y", "IEF": "treasury 7-10y", "TLT": "treasury 20y+",
    "GOVT": "treasury broad", "TIP": "TIPS", "BIL": "t-bills",
    # real assets
    "GLD": "gold", "IAU": "gold", "SLV": "silver",
    "DBC": "commodity broad", "PDBC": "commodity no-K-1", "GSG": "commodity GSCI",
    "DJP": "commodity BCOM", "DBA": "agriculture", "USO": "oil",
    # currency -- included for completeness; see FINDINGS.md, they do not work
    # long-only unlevered
    "UUP": "USD bullish", "UDN": "USD bearish", "FXE": "euro",
    "FXY": "yen", "FXB": "sterling",
}


def from_sharadar(tickers):
    """Sharadar SFP (fund prices). Needs SHARADAR_API_KEY in the environment.

    `closeadj` is Sharadar's total-return adjusted close -- distributions
    reinvested. Do not substitute `close`, which is raw price and will silently
    understate every bond and commodity sleeve.
    """
    import io
    import urllib.request

    key = os.getenv("SHARADAR_API_KEY")
    if not key:
        raise SystemExit("SHARADAR_API_KEY not set. Put it in your environment "
                         "or use --source yfinance.")
    frames = {}
    for i, tk in enumerate(tickers, 1):
        url = ("https://api.sharadar.com/v1.0/data/SFP?api_key=%s&ticker=%s"
               % (key, tk))
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                df = pd.read_csv(io.BytesIO(r.read()))
        except Exception as e:
            print("  %-6s FAILED  %s" % (tk, str(e)[:60]))
            continue
        if df.empty or "closeadj" not in df.columns:
            print("  %-6s no closeadj" % tk)
            continue
        df["date"] = pd.to_datetime(df["date"])
        s = df.sort_values("date").set_index("date")["closeadj"].astype(float)
        frames[tk] = s[~s.index.duplicated(keep="last")]
        print("  %-6s %-20s %5d rows  %s .. %s"
              % (tk, SLEEVES.get(tk, ""), len(s), s.index.min().date(),
                 s.index.max().date()))
        time.sleep(0.2)
    return frames


def from_yfinance(tickers):
    """Free fallback. `auto_adjust=True` gives a total-return series.

    Coverage and adjustment conventions differ slightly from Sharadar, so
    numbers will not tie out to FINDINGS.md to the basis point. The conclusions
    do not depend on that.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("pip install yfinance")
    frames = {}
    data = yf.download(list(tickers), auto_adjust=True, progress=False,
                       group_by="ticker")
    for tk in tickers:
        try:
            s = (data[tk]["Close"] if len(tickers) > 1 else data["Close"]).dropna()
        except Exception:
            print("  %-6s FAILED" % tk)
            continue
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None)
        frames[tk] = s.astype(float)
        print("  %-6s %-20s %5d rows  %s .. %s"
              % (tk, SLEEVES.get(tk, ""), len(s), s.index.min().date(),
                 s.index.max().date()))
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("sharadar", "yfinance"),
                    default="sharadar")
    ap.add_argument("--tickers", nargs="*", default=sorted(SLEEVES),
                    help="default: the full 27-sleeve set")
    args = ap.parse_args()

    print("fetching %d tickers from %s" % (len(args.tickers), args.source))
    frames = (from_sharadar(args.tickers) if args.source == "sharadar"
              else from_yfinance(args.tickers))
    if not frames:
        raise SystemExit("nothing fetched")

    px = pd.DataFrame(frames).sort_index()
    px.to_parquet(OUT)
    cov = pd.DataFrame([dict(ticker=t, description=SLEEVES.get(t, ""),
                             rows=int(px[t].notna().sum()),
                             start=str(px[t].dropna().index.min().date()),
                             end=str(px[t].dropna().index.max().date()))
                        for t in px.columns])
    cov.to_csv(HERE / "sleeve_coverage.csv", index=False)
    print("\nwrote %s  (%d tickers, %d dates, %s .. %s)"
          % (OUT, px.shape[1], px.shape[0], px.index.min().date(),
             px.index.max().date()))
    print("wrote %s" % (HERE / "sleeve_coverage.csv"))


if __name__ == "__main__":
    main()
