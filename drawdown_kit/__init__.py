"""drawdown_kit -- drawdown-preservation machinery, signal-agnostic.

Everything here is about HOW MUCH to hold, never WHAT to hold. Bring your own
selection signal, your own book, your own return stream; this layer sits on top.

Two things in here survived contact with controls over ~18 years of testing.
Everything else we tried did not. See FINDINGS.md.

    1. A walk-forward bull/bear regime overlay (jump model -> classifier ->
       hysteresis gate) applied to the equity exposure.
    2. Static multi-asset sleeves at fixed weights, rebalanced on a schedule.

Quick start
-----------
    import pandas as pd
    from drawdown_kit import RegimeDetector, apply_overlay, evaluate, print_evaluation

    returns = prices.pct_change().dropna()        # daily pd.Series

    det = RegimeDetector()                        # defaults are the tuned ones
    res = det.fit_predict(returns, riskfree=rf)
    res.print_summary()

    ov = apply_overlay(returns, res.regime, riskfree=rf, cost_bp=5)
    print_evaluation(evaluate(ov, returns))

Already have a signal of your own? Skip the detector entirely -- any 0/1 Series
indexed like your returns works:

    ov = apply_overlay(returns, my_own_regime, riskfree=rf, cost_bp=5)

Do not trust any result until it clears the control battery:

    from drawdown_kit import full_battery
    print(full_battery(returns, my_regime, riskfree=rf, cost_bp=5))
"""
from .detector import (RegimeDetector, DetectorResult, fit_jump_model,
                       fit_forecaster, predict_bull_prob, apply_hysteresis)
from .features import build_features, add_macro, TrainScaler, HALF_LIVES
from .overlay import (apply_overlay, evaluate, print_evaluation, crisis_report,
                      max_drawdown, assert_execution_lag)
from .benchmarks import (always_bull, moving_average_regime,
                         dual_moving_average_regime, momentum_regime,
                         volatility_regime, random_regime, random_ensemble,
                         build_benchmarks, ALL_BENCHMARKS)
from .compare import compare_signals, print_comparison, score_signal, verdict
from .validate import (assert_causal, assert_no_overlap, assert_instrument_exists,
                       detect_backfill, CausalityContract, default_contract,
                       freeze_config, verify_config, sharpe_standard_error,
                       deflation_hurdle, run_suite, LeakageError)
from .sleeves import (rebalanced, overlay_sleeve, build_portfolio, load_sleeves,
                      stats, DEFAULT_COST_BP, DEFAULT_WEIGHTS)
from .controls import (exposure_matched_control, vol_matched, regime_split,
                       weight_perturbation, episode_table, full_battery,
                       CRISES)

__version__ = "1.0.0"

__all__ = [
    # detector
    "RegimeDetector", "DetectorResult", "fit_jump_model", "fit_forecaster",
    "predict_bull_prob", "apply_hysteresis",
    "build_features", "add_macro", "TrainScaler", "HALF_LIVES",
    # overlay
    "apply_overlay", "evaluate", "print_evaluation", "crisis_report",
    "max_drawdown", "assert_execution_lag",
    # benchmarks and comparison
    "always_bull", "moving_average_regime", "dual_moving_average_regime",
    "momentum_regime", "volatility_regime", "random_regime", "random_ensemble",
    "build_benchmarks", "ALL_BENCHMARKS",
    "compare_signals", "print_comparison", "score_signal", "verdict",
    # leakage guards
    "assert_causal", "assert_no_overlap", "assert_instrument_exists",
    "detect_backfill", "CausalityContract", "default_contract",
    "freeze_config", "verify_config", "sharpe_standard_error",
    "deflation_hurdle", "run_suite", "LeakageError",
    # sleeves
    "rebalanced", "overlay_sleeve", "build_portfolio", "load_sleeves",
    "stats", "DEFAULT_COST_BP", "DEFAULT_WEIGHTS",
    # controls
    "exposure_matched_control", "vol_matched", "regime_split",
    "weight_perturbation", "episode_table", "full_battery", "CRISES",
]
