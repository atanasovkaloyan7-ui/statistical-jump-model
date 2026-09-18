"""Feature construction.

Twelve features from a single return series: exponentially smoothed mean return,
downside deviation, and their ratio (Sortino), each at four half-lives.

Be clear about what these are. A smoothed mean return at a 1-21 day half-life is
a short-horizon momentum measure; downside deviation is one-sided volatility.
So the detector is a trend-and-downside-risk classifier with a switching
penalty. That is not a criticism - it is what makes it work, and knowing it
tells you what to benchmark against (a moving average, not a factor model).

Every function here is causal: the value at time t uses only data up to t.
`validate.assert_causal` proves it rather than asserting it.
"""
import numpy as np
import pandas as pd

HALF_LIVES = (1, 5, 10, 21)


def ewm_mean(r, hl):
    return r.ewm(halflife=hl, min_periods=max(2, hl)).mean()


def ewm_downside_dev(r, hl):
    """EW root-mean-square of the negative part only."""
    neg = r.clip(upper=0.0)
    return np.sqrt((neg ** 2).ewm(halflife=hl, min_periods=max(2, hl)).mean())


def ewm_sortino(r, hl, eps=1e-8):
    return ewm_mean(r, hl) / (ewm_downside_dev(r, hl) + eps)


def build_features(returns, half_lives=HALF_LIVES):
    """Return series -> 12-column feature frame (3 statistics x 4 half-lives)."""
    r = pd.Series(returns).astype(float)
    out = {}
    for hl in half_lives:
        out["ret_hl%d" % hl] = ewm_mean(r, hl)
        out["dd_hl%d" % hl] = ewm_downside_dev(r, hl)
        out["sortino_hl%d" % hl] = ewm_sortino(r, hl)
    return pd.DataFrame(out, index=r.index)


def add_macro(features, macro, lag_days=1):
    """Optionally join exogenous columns (VIX, curve slope, whatever you have).

    `lag_days` shifts them so a value dated t could genuinely have been known at
    t. Published macro series are revised and released late; market series are
    not. Lag anything you did not observe in real time.

    In our testing the macro block carried roughly a tenth of the model's
    explanatory weight, so this is optional. Start without it.
    """
    if macro is None or len(macro.columns) == 0:
        return features
    m = pd.DataFrame(macro).reindex(features.index).ffill()
    if lag_days:
        m = m.shift(lag_days)
    return features.join(m, how="left")


class TrainScaler:
    """Standardise using TRAINING-window statistics only, then clip.

    Fitting a scaler on the full sample is the most common silent leak in this
    kind of pipeline: it is invisible in the code and it inflates every result.
    Fit on train, transform everything.
    """

    def __init__(self, clip=3.0):
        self.clip, self.mu, self.sd = clip, None, None

    def fit(self, X):
        self.mu = X.mean()
        self.sd = X.std().replace(0, np.nan)
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        if self.clip:
            Z = Z.clip(-self.clip, self.clip)
        return Z.fillna(0.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
