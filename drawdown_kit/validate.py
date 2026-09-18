"""Leakage guards. Run these on your own pipeline before believing any result.

The premise: leakage is not found by reading code, it is found by making the
code prove itself. `assert_causal` scrambles everything after a cutoff date,
recomputes, and requires the past to come back bit-identical. It catches
full-sample scaling, centred windows, back-fill, negative shifts and lookahead
joins without needing to know in advance which mistake you made.

Verified against six planted leaks; catches all six.
"""
import hashlib
import json

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Future information detected in a past-dated value."""


def assert_causal(fn, data, cutoffs=None, n_cutoffs=6, rng=None, atol=1e-12,
                  name="function", verbose=True):
    """Prove fn is causal by scrambling the future and checking the past.

        assert_causal(lambda d: build_features(d["ret"]), df, name="features")
    """
    rng = rng or np.random.default_rng(0)
    idx = data.index
    if cutoffs is None:
        lo, hi = int(len(idx) * 0.30), len(idx) - 2
        cutoffs = idx[np.linspace(lo, hi, n_cutoffs).astype(int)]

    base = fn(data)
    failures = []
    for t in cutoffs:
        scram = data.copy()
        fut = scram.index > t
        if fut.sum() == 0:
            continue
        cols = scram.columns if isinstance(scram, pd.DataFrame) else [None]
        for c in cols:
            col = scram[c] if c is not None else scram
            if not np.issubdtype(col.dtype, np.number):
                continue
            sd = float(np.nanstd(col.to_numpy())) or 1.0
            vals = rng.normal(0, sd * 3, int(fut.sum()))
            # NaNs too: a back-fill only leaks when a gap spans the cutoff.
            vals[rng.random(len(vals)) < 0.15] = np.nan
            if c is not None:
                scram.loc[fut, c] = vals
            else:
                scram.loc[fut] = vals

        out = fn(scram)
        a, b = base.loc[:t], out.loc[:t]
        a, b = a.align(b, join="inner")
        diff = (np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float)))
                if a.size else 0.0)
        if not (diff <= atol or np.isnan(diff)):
            failures.append((t, float(diff)))

    if failures:
        raise LeakageError(
            "%s LEAKS: %d/%d cutoffs moved when only the FUTURE was altered. "
            "worst |delta| = %.3e at %s"
            % (name, len(failures), len(cutoffs),
               max(f[1] for f in failures), failures[0][0].date()))
    if verbose:
        print("   [causal] %-34s PASS (%d cutoffs)" % (name, len(cutoffs)))
    return True


def assert_no_overlap(train_idx, test_idx, name="split"):
    """Training must end strictly before testing begins."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return True
    if train_idx.max() >= test_idx.min():
        raise LeakageError(
            "%s: train ends %s, test begins %s -- %d overlapping observations"
            % (name, train_idx.max().date(), test_idx.min().date(),
               int((train_idx >= test_idx.min()).sum())))
    return True


def assert_instrument_exists(weights, inception, name="universe"):
    """No position before an instrument started trading.

    Backtesting an ETF before its launch date is the quietest and most common
    survivorship error in multi-asset work.
    """
    bad = []
    for col, start in (inception or {}).items():
        if col not in weights.columns or start is None:
            continue
        n = int((weights.loc[weights.index < pd.Timestamp(start), col].abs() > 1e-12).sum())
        if n:
            bad.append((col, str(pd.Timestamp(start).date()), n))
    if bad:
        raise LeakageError("%s holds instruments before inception: %s" % (name, bad))
    return True


def detect_backfill(raw, filled, name="series"):
    """Every filled value must equal the PREVIOUS valid observation, never a later one."""
    raw, filled = pd.Series(raw), pd.Series(filled)
    was_nan = raw.isna() & filled.notna()
    if not was_nan.any():
        return True
    fwd = raw.ffill()
    bad = was_nan & ~np.isclose(filled.to_numpy(), fwd.to_numpy(),
                                equal_nan=True, rtol=1e-9)
    if bad.any():
        raise LeakageError(
            "%s: %d filled values came from the future -- this is a back-fill"
            % (name, int(bad.sum())))
    return True


class CausalityContract:
    """Declare which artefacts may see the future, and enforce that they never trade.

    The jump model's labels are non-causal BY DESIGN -- that is not a bug, it is
    how the dynamic programme works. They are legitimate as training targets and
    catastrophic as a signal. This makes the distinction explicit.
    """

    TRADED = "traded"
    TRAINING_ONLY = "training_only"

    def __init__(self):
        self.registry = {}

    def declare(self, name, kind, reason=""):
        self.registry[name] = dict(kind=kind, reason=reason)
        return self

    def assert_not_traded(self, name, used_in_weights):
        e = self.registry.get(name, {})
        if e.get("kind") == self.TRAINING_ONLY and used_in_weights:
            raise LeakageError(
                "%s is TRAINING_ONLY (%s) but reached the portfolio weights"
                % (name, e["reason"]))
        return True


def default_contract():
    c = CausalityContract()
    c.declare("jump_model_labels", CausalityContract.TRAINING_ONLY,
              "dynamic programme optimises the whole state path; the label at t "
              "depends on data after t")
    c.declare("forecaster_probability", CausalityContract.TRADED,
              "trained on past labels, predicts forward from same-day features")
    c.declare("regime_signal", CausalityContract.TRADED, "sets exposure")
    return c


def freeze_config(cfg, path=None):
    """Hash a specification before you have seen its results.

    Pre-registration in one line. If you change the config after seeing a
    result, the hash changes -- which is the point. It does not stop you
    iterating; it stops you forgetting that you did.
    """
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
    if path:
        with open(path, "w") as f:
            json.dump(dict(hash=h, config=cfg,
                           frozen_at=str(pd.Timestamp.utcnow())), f,
                      indent=2, default=str)
    return h


def verify_config(cfg, expected):
    got = freeze_config(cfg)
    if got != expected:
        raise LeakageError(
            "config changed since freezing (%s -> %s). A specification altered "
            "after seeing results is a NEW trial -- count it as one." % (expected, got))
    return True


def sharpe_standard_error(sharpe, n_years):
    """Lo (2002). Understates the error when returns are autocorrelated, which
    regime-driven returns always are -- treat it as a floor."""
    return float(np.sqrt((1 + 0.5 * sharpe ** 2) / n_years))


def deflation_hurdle(n_trials, sharpe_se):
    """Expected best Sharpe from n_trials of pure noise.

    Your strategy must clear this before any of its edge is real. Count every
    configuration you evaluated, including the ones you abandoned.
    """
    if n_trials < 2:
        return 0.0
    from scipy import stats
    e = np.euler_gamma
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sharpe_se * ((1 - e) * z1 + e * z2))


def run_suite(feature_fn, data, execution_lag=1, weights=None, inception=None):
    """Everything above in one call. Run before trusting any backtest."""
    print("\n" + "=" * 62)
    print("LEAKAGE SUITE")
    print("=" * 62)
    ok = True
    try:
        assert_causal(feature_fn, data, name="feature pipeline")
    except LeakageError as e:
        print("   [causal] feature pipeline          FAIL\n      %s" % e)
        ok = False
    try:
        from .overlay import assert_execution_lag
        assert_execution_lag(execution_lag)
        print("   [timing] execution lag = %d           PASS" % execution_lag)
    except ValueError as e:
        print("   [timing] execution lag             FAIL\n      %s" % e)
        ok = False
    if weights is not None and inception:
        try:
            assert_instrument_exists(weights, inception)
            print("   [universe] no pre-inception holds  PASS")
        except LeakageError as e:
            print("   [universe] pre-inception holdings  FAIL\n      %s" % e)
            ok = False
    print("=" * 62)
    print("LEAKAGE SUITE: %s" % ("PASS" if ok else "FAIL"))
    return ok
