"""Deployment. The part where there is no future to see.

THE POINT
---------
The jump model is a TRAINING-TIME object. It never runs in production. To
produce today's signal you need three things, all frozen at the last refit:

    a fitted classifier, the scaler it was trained with, and yesterday's state

That is it. Today's signal is one forward pass:

    features(today) -> scaler -> classifier -> P(bull) -> hysteresis -> position

No dynamic programming, no whole-path optimisation, nothing that needs tomorrow.
The look-ahead lives entirely inside the refit, and a refit only ever runs on a
window that has already closed.

    ┌── ANNUALLY, on closed history ─────────────────────────────┐
    │  jump model labels the training window  (non-causal, fine: │
    │  that window is in the past)                               │
    │          ↓                                                 │
    │  classifier learns features(t) -> label(t+1)               │
    │          ↓                                                 │
    │  freeze classifier + scaler to disk                        │
    └────────────────────────────────────────────────────────────┘
    ┌── EVERY DAY, after the close ──────────────────────────────┐
    │  features(today) -> frozen scaler -> frozen classifier     │
    │          ↓                                                 │
    │  P(bull) -> hysteresis(persisted state) -> target exposure │
    │          ↓                                                 │
    │  trade at the NEXT close                                   │
    └────────────────────────────────────────────────────────────┘

THE OPERATIONAL TRAP
--------------------
The hysteresis gate is path-dependent. Today's state depends on yesterday's.
If you restart the process and reinitialise to bull, you silently re-enter the
market in the middle of a bear regime. `LiveSignal` persists state to disk for
exactly this reason. Do not remove it.
"""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, TrainScaler
from .detector import fit_jump_model, fit_forecaster


# ------------------------------------------------------------------- refit
def fit_and_freeze(returns, path, jump_penalty=41, seed=0, model="xgboost",
                   half_lives=(1, 5, 10, 21)):
    """Run at refit time only. Trains on the window you pass and freezes it.

    Pass a CLOSED window -- data up to the last completed period, nothing from
    the period you are about to trade. The jump model looks ahead inside this
    window, which is exactly why the window must already be history.

    Writes: <path>/model.pkl, <path>/meta.json
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    r = pd.Series(returns).dropna().astype(float)
    X = build_features(r, half_lives).dropna()
    r = r.reindex(X.index)

    labels, _, _ = fit_jump_model(X, r, jump_penalty, seed=seed)
    clf, scaler = fit_forecaster(X, labels, model=model, seed=seed)
    if clf is None:
        raise RuntimeError("forecaster did not train -- too little history, or "
                           "the training window contains only one regime")

    # The bull cluster is whichever had the higher mean return. Record it: if
    # this flips on a later refit without the data justifying it, every position
    # reverses on a labelling artefact. See check_label_stability below.
    bull_mean = float(r[labels == 1].mean())
    bear_mean = float(r[labels == 0].mean())

    with open(path / "model.pkl", "wb") as f:
        pickle.dump(dict(clf=clf, scaler=scaler), f)
    meta = dict(
        fitted_at=datetime.now(timezone.utc).isoformat(),
        train_start=str(r.index.min().date()), train_end=str(r.index.max().date()),
        n_train=len(r), jump_penalty=jump_penalty, seed=seed, model=model,
        half_lives=list(half_lives), feature_columns=list(X.columns),
        bull_mean_daily=bull_mean, bear_mean_daily=bear_mean,
        bull_share=float((labels == 1).mean()),
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def check_label_stability(old_meta, new_meta, tol=0.0):
    """Call on every refit. Raises if the regimes stopped meaning what they meant.

    Cluster identity is not stable across refits. The mapping-by-mean-return in
    fit_jump_model fixes it in principle, but when two clusters have similar
    means the assignment can still flip -- and a flip reverses every position
    you hold, caused by nothing. Log it, alarm on it, require sign-off.
    """
    problems = []
    if new_meta["bull_mean_daily"] <= new_meta["bear_mean_daily"] + tol:
        problems.append("bull cluster does not have the higher mean return "
                        "(%.5f vs %.5f)" % (new_meta["bull_mean_daily"],
                                            new_meta["bear_mean_daily"]))
    drift = abs(new_meta["bull_share"] - old_meta["bull_share"])
    if drift > 0.25:
        problems.append("bull share moved %.1f pp between refits (%.1f%% -> %.1f%%)"
                        % (100 * drift, 100 * old_meta["bull_share"],
                           100 * new_meta["bull_share"]))
    return problems


# -------------------------------------------------------------- daily signal
class LiveSignal:
    """Today's position from a frozen model. No jump model, no future.

        sig = LiveSignal("artifacts/", bear_exposure=0.50)
        out = sig.update(returns_through_today)
        print(out["target_exposure"], "-> trade at the NEXT close")

    State persists to <path>/state.json so a restart does not silently reset
    you to fully invested in the middle of a bear market.
    """

    def __init__(self, path, threshold=0.70, bear_exposure=0.50,
                 bull_exposure=1.00, max_staleness_days=5):
        self.path = Path(path)
        with open(self.path / "model.pkl", "rb") as f:
            art = pickle.load(f)
        self.clf, self.scaler = art["clf"], art["scaler"]
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.threshold = threshold
        self.bear_exposure = float(bear_exposure)
        self.bull_exposure = float(bull_exposure)
        self.max_staleness_days = max_staleness_days
        self.state_file = self.path / "state.json"
        self.state = self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        # First run only. Starting bull is a choice -- make it deliberately.
        return dict(regime=1, last_date=None, last_prob=None, history=[])

    def _save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def update(self, returns, asof=None, persist=True):
        """Compute today's target exposure. Trade it at the NEXT close.

        returns : the full daily return series up to and including today.
                  Needs at least ~60 sessions so the 21-day half-life features
                  are warmed up; pass more.
        """
        r = pd.Series(returns).dropna().astype(float)
        if len(r) < 60:
            raise ValueError("need at least ~60 sessions of history to warm up "
                             "the features; got %d" % len(r))

        X = build_features(r, tuple(self.meta["half_lives"]))
        missing = [c for c in self.meta["feature_columns"] if c not in X.columns]
        if missing:
            raise ValueError("feature mismatch against the frozen model: %s" % missing)
        X = X[self.meta["feature_columns"]]

        row = X.iloc[[-1]].replace([np.inf, -np.inf], np.nan)
        asof = pd.Timestamp(asof or X.index[-1])
        if row.isna().any(axis=1).iloc[0]:
            # A bad feature row must not silently become an all-clear. Hold.
            out = self._emit(asof, None, "features unavailable; state held")
            if persist:
                self._save_state()
            return out

        # Staleness guard: a stalled data feed should not keep trading on a
        # week-old signal.
        if self.state["last_date"]:
            gap = (asof - pd.Timestamp(self.state["last_date"])).days
            if gap > self.max_staleness_days:
                out = self._emit(asof, None,
                                 "data gap of %d days; state held, INVESTIGATE" % gap)
                if persist:
                    self._save_state()
                return out

        p = float(self.clf.predict_proba(self.scaler.transform(row))[:, 1][0])

        # The hysteresis gate, applied to the PERSISTED state.
        prev = int(self.state["regime"])
        new = prev
        if prev == 1 and (1.0 - p) > self.threshold:
            new = 0
        elif prev == 0 and p > self.threshold:
            new = 1

        self.state["regime"] = new
        out = self._emit(asof, p, "switched to %s" % ("BULL" if new else "BEAR")
                         if new != prev else "no change")
        out["switched"] = new != prev
        if persist:
            self._save_state()
        return out

    def _emit(self, asof, p, note):
        g = int(self.state["regime"])
        self.state["last_date"] = str(asof.date())
        self.state["last_prob"] = p
        self.state["history"] = (self.state.get("history", []) +
                                 [dict(date=str(asof.date()), p=p, regime=g)])[-500:]
        return dict(asof=str(asof.date()), p_bull=p, regime=g,
                    regime_label="BULL" if g else "BEAR",
                    target_exposure=self.bull_exposure if g else self.bear_exposure,
                    fill="next close", note=note,
                    model_trained_through=self.meta["train_end"])

    def days_since_refit(self, asof=None):
        asof = pd.Timestamp(asof or datetime.now(timezone.utc).date())
        return int((asof - pd.Timestamp(self.meta["train_end"])).days)

    def needs_refit(self, asof=None, every_days=365):
        return self.days_since_refit(asof) > every_days
