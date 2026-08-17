from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ============================================================
# v3.5 VALIDATION ENGINE
# ============================================================
#
# Goals:
# - Preserve backward compatibility with chronological_validation()
# - Add true multi-fold walk-forward testing
# - Require enough trades before drawing conclusions
# - Penalize unstable / overfit performance
# - Require positive OOS expectancy
# - Require acceptable OOS profit factor
# - Require acceptable drawdown
# - Require consistency across multiple unseen periods
# ============================================================


# ============================================================
# BASIC PERFORMANCE HELPERS
# ============================================================

def _safe_numeric_series(
    frame,
    column,
):
    if (
        frame is None
        or frame.empty
        or column not in frame.columns
    ):
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()


def _profit_factor(
    frame,
):
    pnl = _safe_numeric_series(
        frame,
        "pnl",
    )

    if pnl.empty:
        return 0.0

    gross_profit = float(
        pnl[
            pnl > 0
        ].sum()
    )

    gross_loss = abs(
        float(
            pnl[
                pnl < 0
            ].sum()
        )
    )

    if gross_loss == 0:
        return (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    return (
        gross_profit
        / gross_loss
    )


def _win_rate(
    frame,
):
    pnl = _safe_numeric_series(
        frame,
        "pnl",
    )

    if pnl.empty:
        return 0.0

    return float(
        (
            pnl > 0
        ).mean()
        * 100
    )


def _expectancy_r(
    frame,
):
    r = _safe_numeric_series(
        frame,
        "r_multiple",
    )

    if r.empty:
        return 0.0

    return float(
        r.mean()
    )


def _median_r(
    frame,
):
    r = _safe_numeric_series(
        frame,
        "r_multiple",
    )

    if r.empty:
        return 0.0

    return float(
        r.median()
    )


def _avg_trade_dollars(
    frame,
):
    pnl = _safe_numeric_series(
        frame,
        "pnl",
    )

    if pnl.empty:
        return 0.0

    return float(
        pnl.mean()
    )


def _max_drawdown_from_trades(
    frame,
):
    """
    Approximate trade-sequence drawdown using cumulative realized PnL.

    This is not a replacement for the portfolio equity curve drawdown
    calculated by the backtester. It is used for fold-level validation.
    """

    pnl = _safe_numeric_series(
        frame,
        "pnl",
    )

    if pnl.empty:
        return 0.0

    cumulative = pnl.cumsum()

    peak = cumulative.cummax()

    drawdown = (
        cumulative
        - peak
    )

    return float(
        drawdown.min()
    )


def _bootstrap_mean_ci(
    values,
    iterations=3000,
    confidence=0.95,
    seed=1337,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) < 10:
        return (
            None,
            None,
        )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    means = np.empty(
        iterations,
        dtype=float,
    )

    for index in range(
        iterations
    ):
        sample = rng.choice(
            values,
            size=len(
                values
            ),
            replace=True,
        )

        means[index] = float(
            sample.mean()
        )

    alpha = (
        1.0
        - confidence
    ) / 2.0

    low = float(
        np.quantile(
            means,
            alpha,
        )
    )

    high = float(
        np.quantile(
            means,
            1.0 - alpha,
        )
    )

    return (
        low,
        high,
    )


def _sort_trades(
    trades,
):
    if (
        trades is None
        or trades.empty
    ):
        return pd.DataFrame()

    frame = trades.copy()

    sort_column = None

    for candidate in (
        "exit_time",
        "entry_time",
        "signal_time",
        "signal_date",
    ):
        if candidate in frame.columns:
            sort_column = candidate
            break

    if sort_column is not None:
        frame[
            sort_column
        ] = pd.to_datetime(
            frame[
                sort_column
            ],
            utc=True,
            errors="coerce",
        )

        frame = (
            frame
            .sort_values(
                sort_column
            )
            .reset_index(
                drop=True
            )
        )

    else:
        frame = frame.reset_index(
            drop=True
        )

    return frame


# ============================================================
# LIVE CANDIDATE VALIDATION
# ============================================================

def validate_candidate(
    row,
):
    """
    Validate one live scanner candidate.

    This remains intentionally additive:
    it does not replace the strategy engine.
    """

    if row is None:
        return {
            "validation_pass": False,
            "validation_score": 0,
            "validation_grade": "FAIL",
            "validation_reasons": [
                "No candidate data was provided."
            ],
        }

    if hasattr(
        row,
        "to_dict",
    ):
        data = row.to_dict()
    else:
        data = dict(
            row
        )

    reasons = []

    score = 0

    signal = str(
        data.get(
            "signal",
            "",
        )
    ).upper()

    def _number(
        key,
        default=0.0,
    ):
        try:
            value = data.get(
                key,
                default,
            )

            if value is None:
                return float(
                    default
                )

            return float(
                value
            )

        except Exception:
            return float(
                default
            )

    swing_score = _number(
        "swing_score"
    )

    entry_quality = _number(
        "entry_quality"
    )

    reward_risk = _number(
        "reward_risk"
    )

    intraday_score = _number(
        "intraday_score"
    )

    leadership = _number(
        "leadership_percentile"
    )

    market_score = _number(
        "market_score"
    )

    try:
        distribution_days = int(
            float(
                data.get(
                    "distribution_days",
                    99,
                )
            )
        )
    except Exception:
        distribution_days = 99

    risk_flag = bool(
        data.get(
            "risk_flag",
            False,
        )
    )

    trend_health = bool(
        data.get(
            "trend_health",
            False,
        )
    )

    intraday_confirmed = bool(
        data.get(
            "intraday_confirmed",
            False,
        )
    )

    # --------------------------------------------------------
    # SIGNAL QUALITY
    # --------------------------------------------------------

    if signal == "A+ SWING BUY":

        score += 20

        reasons.append(
            "A+ swing signal passed."
        )

    elif signal == "BUY":

        score += 18

        reasons.append(
            "BUY signal passed."
        )

    elif signal == "WATCH":

        score += 8

        reasons.append(
            "Candidate remains WATCH rather than confirmed BUY."
        )

    else:

        reasons.append(
            f"Signal is {signal or 'N/A'}, not a confirmed BUY."
        )

    # --------------------------------------------------------
    # SWING SCORE
    # --------------------------------------------------------

    if swing_score >= 92:

        score += 20

        reasons.append(
            f"Elite Swing Score: {swing_score:.1f}."
        )

    elif swing_score >= 85:

        score += 18

        reasons.append(
            f"BUY-level Swing Score: {swing_score:.1f}."
        )

    elif swing_score >= 80:

        score += 12

        reasons.append(
            f"Strong but sub-threshold Swing Score: {swing_score:.1f}."
        )

    elif swing_score >= 72:

        score += 6

        reasons.append(
            f"Moderate Swing Score: {swing_score:.1f}."
        )

    else:

        reasons.append(
            f"Swing Score {swing_score:.1f} is weak."
        )

    # --------------------------------------------------------
    # ENTRY QUALITY
    # --------------------------------------------------------

    if entry_quality >= 13:

        score += 15

        reasons.append(
            f"Excellent Entry Quality: {entry_quality:.1f}/15."
        )

    elif entry_quality >= 10:

        score += 12

        reasons.append(
            f"Entry Quality passes BUY threshold: {entry_quality:.1f}/15."
        )

    else:

        reasons.append(
            f"Entry Quality {entry_quality:.1f}/15 is below BUY threshold."
        )

    # --------------------------------------------------------
    # REWARD / RISK
    # --------------------------------------------------------

    if reward_risk >= 2.5:

        score += 12

        reasons.append(
            f"Strong reward/risk: {reward_risk:.2f}:1."
        )

    elif reward_risk >= 2.0:

        score += 10

        reasons.append(
            f"Reward/risk passes minimum: {reward_risk:.2f}:1."
        )

    elif reward_risk >= 1.5:

        score += 4

        reasons.append(
            f"Marginal reward/risk: {reward_risk:.2f}:1."
        )

    else:

        reasons.append(
            f"Reward/risk {reward_risk:.2f}:1 is below 1.50:1."
        )

    # --------------------------------------------------------
    # INTRADAY CONFIRMATION
    # --------------------------------------------------------

    if (
        intraday_confirmed
        and intraday_score >= 90
    ):

        score += 15

        reasons.append(
            f"Excellent intraday confirmation: {intraday_score:.0f}/100."
        )

    elif (
        intraday_confirmed
        and intraday_score >= 85
    ):

        score += 13

        reasons.append(
            f"Intraday BUY confirmation passed: {intraday_score:.0f}/100."
        )

    elif intraday_score >= 70:

        score += 5

        reasons.append(
            f"Intraday score is promising but not confirmed: {intraday_score:.0f}/100."
        )

    else:

        reasons.append(
            f"Intraday confirmation has not passed; score is {intraday_score:.0f}/100."
        )

    # --------------------------------------------------------
    # LEADERSHIP
    # --------------------------------------------------------

    if leadership >= 90:

        score += 8

        reasons.append(
            f"Elite leadership: {leadership:.0f}th percentile."
        )

    elif leadership >= 70:

        score += 6

        reasons.append(
            f"Leadership passes threshold: {leadership:.0f}th percentile."
        )

    else:

        reasons.append(
            f"Leadership rank {leadership:.0f}th percentile is below 70."
        )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    if market_score >= 8:

        score += 5

        reasons.append(
            f"Strong market regime: {market_score:.1f}/10."
        )

    elif market_score >= 5:

        score += 3

        reasons.append(
            f"Acceptable market regime: {market_score:.1f}/10."
        )

    else:

        reasons.append(
            f"Market regime score {market_score:.1f}/10 is weak."
        )

    # --------------------------------------------------------
    # TREND HEALTH
    # --------------------------------------------------------

    if trend_health:

        score += 3

        reasons.append(
            "Trend health passed."
        )

    else:

        reasons.append(
            "Trend health did not pass."
        )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    if distribution_days <= 2:

        score += 2

        reasons.append(
            f"Low distribution pressure: {distribution_days} days."
        )

    elif distribution_days <= 4:

        score += 1

        reasons.append(
            f"Distribution remains acceptable: {distribution_days} days."
        )

    else:

        reasons.append(
            f"Distribution pressure is excessive: {distribution_days} days."
        )

    # --------------------------------------------------------
    # HARD RISK GATE
    # --------------------------------------------------------

    if risk_flag:

        score = min(
            score,
            40,
        )

        reasons.append(
            "Hard catalyst/risk gate is active."
        )

    score = int(
        max(
            0,
            min(
                score,
                100,
            ),
        )
    )

    validation_pass = bool(
        not risk_flag
        and signal in {
            "BUY",
            "A+ SWING BUY",
        }
        and swing_score >= 85
        and entry_quality >= 10
        and reward_risk >= 2.0
        and intraday_confirmed
        and intraday_score >= 85
        and leadership >= 70
        and market_score >= 5
        and trend_health
        and distribution_days <= 4
    )

    if (
        validation_pass
        and score >= 92
    ):

        grade = "A+"

    elif validation_pass:

        grade = "A"

    elif score >= 80:

        grade = "WATCH+"

    elif score >= 65:

        grade = "WATCH"

    else:

        grade = "FAIL"

    return {
        "validation_pass": validation_pass,
        "validation_score": score,
        "validation_grade": grade,
        "validation_reasons": reasons,
    }


# ============================================================
# LIVE VALIDATION SUMMARY
# ============================================================

def validation_summary(
    frame,
):
    if (
        frame is None
        or frame.empty
    ):

        return {
            "candidates": 0,
            "validated_buys": 0,
            "validation_rate_pct": 0.0,
            "average_validation_score": 0.0,
            "top_validation_grade": "N/A",
        }

    results = []

    for _, row in frame.iterrows():

        result = validate_candidate(
            row
        )

        results.append(
            result
        )

    result_frame = pd.DataFrame(
        results
    )

    validated_buys = int(
        result_frame[
            "validation_pass"
        ].sum()
    )

    candidates = len(
        result_frame
    )

    average_score = float(
        result_frame[
            "validation_score"
        ].mean()
    )

    grade_order = {
        "A+": 5,
        "A": 4,
        "WATCH+": 3,
        "WATCH": 2,
        "FAIL": 1,
    }

    grade_scores = (
        result_frame[
            "validation_grade"
        ]
        .map(
            grade_order
        )
        .fillna(0)
    )

    top_index = (
        grade_scores.idxmax()
    )

    top_validation_grade = (
        result_frame.loc[
            top_index,
            "validation_grade",
        ]
    )

    return {
        "candidates": candidates,
        "validated_buys": validated_buys,
        "validation_rate_pct": round(
            validated_buys
            / max(
                candidates,
                1,
            )
            * 100,
            1,
        ),
        "average_validation_score": round(
            average_score,
            1,
        ),
        "top_validation_grade": (
            top_validation_grade
        ),
    }


# ============================================================
# SINGLE HOLDOUT VALIDATION
# ============================================================

def chronological_validation(
    trades: pd.DataFrame,
    train_fraction: float = 0.70,
    min_total_trades: int = 30,
    min_oos_trades: int = 10,
):
    """
    Backward-compatible chronological holdout validation.

    v3.5 keeps this function because existing backtest.py code
    already depends on it.
    """

    frame = _sort_trades(
        trades
    )

    if frame.empty:

        return {
            "sample_trades": 0,
            "in_sample_trades": 0,
            "out_of_sample_trades": 0,
            "in_sample_win_rate_pct": 0.0,
            "out_of_sample_win_rate_pct": 0.0,
            "in_sample_expectancy_r": 0.0,
            "out_of_sample_expectancy_r": 0.0,
            "out_of_sample_profit_factor": 0.0,
            "bootstrap_expectancy_low_r": None,
            "bootstrap_expectancy_high_r": None,
            "confidence_grade": "INSUFFICIENT",
            "validation_pass": False,
            "validation_verdict": "INSUFFICIENT EVIDENCE",
            "notes": [
                "No completed trades were available for holdout validation."
            ],
        }

    n = len(
        frame
    )

    split = min(
        max(
            int(
                math.floor(
                    n
                    * train_fraction
                )
            ),
            1,
        ),
        max(
            n - 1,
            1,
        ),
    )

    train = (
        frame.iloc[
            :split
        ].copy()
    )

    test = (
        frame.iloc[
            split:
        ].copy()
    )

    oos_pf = _profit_factor(
        test
    )

    oos_expectancy = _expectancy_r(
        test
    )

    in_sample_expectancy = (
        _expectancy_r(
            train
        )
    )

    oos_r = _safe_numeric_series(
        test,
        "r_multiple",
    )

    ci_low, ci_high = (
        _bootstrap_mean_ci(
            oos_r.values
        )
    )

    notes = []

    enough_total = (
        n
        >= min_total_trades
    )

    enough_oos = (
        len(
            test
        )
        >= min_oos_trades
    )

    positive_expectancy = (
        oos_expectancy
        > 0
    )

    acceptable_pf = bool(
        (
            not np.isfinite(
                oos_pf
            )
        )
        or oos_pf
        >= 1.15
    )

    robust_ci = bool(
        ci_low is not None
        and ci_low > 0
    )

    # Detect severe train-to-test decay.
    if (
        in_sample_expectancy
        > 0
    ):
        degradation_ratio = (
            oos_expectancy
            / in_sample_expectancy
        )
    else:
        degradation_ratio = None

    severe_degradation = bool(
        degradation_ratio
        is not None
        and degradation_ratio
        < 0.25
    )

    if not enough_total:
        notes.append(
            f"Only {n} completed trades; "
            f"at least {min_total_trades} is preferred."
        )

    if not enough_oos:
        notes.append(
            f"Only {len(test)} out-of-sample trades; "
            f"at least {min_oos_trades} is preferred."
        )

    if not positive_expectancy:
        notes.append(
            "Out-of-sample expectancy is not positive."
        )

    if (
        np.isfinite(
            oos_pf
        )
        and oos_pf < 1.15
    ):
        notes.append(
            "Out-of-sample profit factor is below 1.15."
        )

    if (
        ci_low is not None
        and ci_low <= 0
    ):
        notes.append(
            "The 95% bootstrap interval for out-of-sample expectancy includes zero."
        )

    if severe_degradation:
        notes.append(
            "Out-of-sample expectancy deteriorated sharply relative to in-sample performance."
        )

    passed = bool(
        enough_total
        and enough_oos
        and positive_expectancy
        and acceptable_pf
        and robust_ci
        and not severe_degradation
    )

    if passed:

        if (
            len(
                test
            ) >= 20
            and (
                not np.isfinite(
                    oos_pf
                )
                or oos_pf
                >= 1.35
            )
            and oos_expectancy
            >= 0.15
        ):
            grade = "A"

        else:
            grade = "B"

        verdict = "PASS"

    elif (
        not enough_total
        or not enough_oos
    ):

        grade = (
            "INSUFFICIENT"
        )

        verdict = (
            "INSUFFICIENT EVIDENCE"
        )

    elif (
        positive_expectancy
        and acceptable_pf
    ):

        grade = "C"

        verdict = "FAIL"

    else:

        grade = "D"

        verdict = "FAIL"

    if not notes:
        notes.append(
            "Chronological holdout and bootstrap checks passed."
        )

    return {
        "sample_trades": n,

        "in_sample_trades": len(
            train
        ),

        "out_of_sample_trades": len(
            test
        ),

        "in_sample_win_rate_pct": round(
            _win_rate(
                train
            ),
            2,
        ),

        "out_of_sample_win_rate_pct": round(
            _win_rate(
                test
            ),
            2,
        ),

        "in_sample_expectancy_r": round(
            in_sample_expectancy,
            3,
        ),

        "out_of_sample_expectancy_r": round(
            oos_expectancy,
            3,
        ),

        "out_of_sample_profit_factor": (
            "inf"
            if not np.isfinite(
                oos_pf
            )
            else round(
                oos_pf,
                2,
            )
        ),

        "bootstrap_expectancy_low_r": (
            None
            if ci_low is None
            else round(
                ci_low,
                3,
            )
        ),

        "bootstrap_expectancy_high_r": (
            None
            if ci_high is None
            else round(
                ci_high,
                3,
            )
        ),

        "train_to_test_expectancy_ratio": (
            None
            if degradation_ratio
            is None
            else round(
                degradation_ratio,
                3,
            )
        ),

        "confidence_grade": grade,

        "validation_pass": (
            passed
        ),

        "validation_verdict": (
            verdict
        ),

        "notes": notes,
    }


# ============================================================
# MULTI-FOLD WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validation(
    trades: pd.DataFrame,
    folds: int = 4,
    min_total_trades: int = 40,
    min_fold_trades: int = 5,
    min_positive_fold_ratio: float = 0.60,
    min_profit_factor: float = 1.15,
    max_expectancy_degradation: float = 0.75,
):
    """
    Expanding-window walk-forward validation.

    Example with 4 folds:
        Train early history -> test next block
        Train more history  -> test next block
        Train more history  -> test next block
        Train most history  -> test final block

    This is intentionally stricter than a single 70/30 split.
    """

    frame = _sort_trades(
        trades
    )

    if frame.empty:

        return {
            "validation_verdict": "INSUFFICIENT EVIDENCE",
            "validation_pass": False,
            "promotion_candidate": False,
            "confidence_grade": "INSUFFICIENT",
            "sample_trades": 0,
            "folds_requested": int(
                folds
            ),
            "folds_completed": 0,
            "positive_folds": 0,
            "positive_fold_ratio": 0.0,
            "worst_fold_expectancy_r": 0.0,
            "median_fold_expectancy_r": 0.0,
            "worst_fold_profit_factor": 0.0,
            "median_fold_profit_factor": 0.0,
            "aggregate_oos_expectancy_r": 0.0,
            "aggregate_oos_profit_factor": 0.0,
            "aggregate_oos_win_rate_pct": 0.0,
            "bootstrap_expectancy_low_r": None,
            "bootstrap_expectancy_high_r": None,
            "fold_results": pd.DataFrame(),
            "notes": [
                "No completed trades were available for walk-forward validation."
            ],
        }

    total_trades = len(
        frame
    )

    folds = max(
        2,
        int(
            folds
        ),
    )

    if total_trades < max(
        min_total_trades,
        folds * min_fold_trades,
    ):

        return {
            "validation_verdict": "INSUFFICIENT EVIDENCE",
            "validation_pass": False,
            "promotion_candidate": False,
            "confidence_grade": "INSUFFICIENT",
            "sample_trades": total_trades,
            "folds_requested": folds,
            "folds_completed": 0,
            "positive_folds": 0,
            "positive_fold_ratio": 0.0,
            "worst_fold_expectancy_r": 0.0,
            "median_fold_expectancy_r": 0.0,
            "worst_fold_profit_factor": 0.0,
            "median_fold_profit_factor": 0.0,
            "aggregate_oos_expectancy_r": 0.0,
            "aggregate_oos_profit_factor": 0.0,
            "aggregate_oos_win_rate_pct": 0.0,
            "bootstrap_expectancy_low_r": None,
            "bootstrap_expectancy_high_r": None,
            "fold_results": pd.DataFrame(),
            "notes": [
                (
                    f"Only {total_trades} completed trades. "
                    f"At least {max(min_total_trades, folds * min_fold_trades)} "
                    f"are required for the requested walk-forward test."
                )
            ],
        }

    # Initial training period = approximately 40% of sample.
    initial_train = max(
        min_total_trades // 2,
        int(
            math.floor(
                total_trades
                * 0.40
            )
        ),
    )

    remaining = (
        total_trades
        - initial_train
    )

    test_size = max(
        min_fold_trades,
        int(
            math.floor(
                remaining
                / folds
            )
        ),
    )

    fold_rows = []

    oos_segments = []

    train_end = (
        initial_train
    )

    for fold_index in range(
        folds
    ):

        test_start = (
            train_end
        )

        if fold_index == folds - 1:
            test_end = (
                total_trades
            )
        else:
            test_end = min(
                total_trades,
                test_start
                + test_size,
            )

        if (
            test_start
            >= total_trades
            or test_end
            <= test_start
        ):
            break

        train = (
            frame.iloc[
                :test_start
            ].copy()
        )

        test = (
            frame.iloc[
                test_start:test_end
            ].copy()
        )

        if len(
            test
        ) < min_fold_trades:
            break

        train_expectancy = (
            _expectancy_r(
                train
            )
        )

        test_expectancy = (
            _expectancy_r(
                test
            )
        )

        test_pf = (
            _profit_factor(
                test
            )
        )

        test_win_rate = (
            _win_rate(
                test
            )
        )

        test_median_r = (
            _median_r(
                test
            )
        )

        if train_expectancy > 0:
            degradation = (
                (
                    train_expectancy
                    - test_expectancy
                )
                / abs(
                    train_expectancy
                )
            )
        else:
            degradation = None

        positive_fold = bool(
            test_expectancy > 0
            and (
                not np.isfinite(
                    test_pf
                )
                or test_pf
                >= min_profit_factor
            )
        )

        fold_rows.append(
            {
                "fold": (
                    fold_index
                    + 1
                ),

                "train_trades": len(
                    train
                ),

                "test_trades": len(
                    test
                ),

                "train_expectancy_r": round(
                    train_expectancy,
                    3,
                ),

                "test_expectancy_r": round(
                    test_expectancy,
                    3,
                ),

                "test_median_r": round(
                    test_median_r,
                    3,
                ),

                "test_win_rate_pct": round(
                    test_win_rate,
                    2,
                ),

                "test_profit_factor": (
                    "inf"
                    if not np.isfinite(
                        test_pf
                    )
                    else round(
                        test_pf,
                        2,
                    )
                ),

                "expectancy_degradation": (
                    None
                    if degradation
                    is None
                    else round(
                        degradation,
                        3,
                    )
                ),

                "positive_fold": (
                    positive_fold
                ),
            }
        )

        oos_segments.append(
            test
        )

        train_end = (
            test_end
        )

        if train_end >= total_trades:
            break

    fold_frame = pd.DataFrame(
        fold_rows
    )

    completed_folds = len(
        fold_frame
    )

    if completed_folds == 0:

        return {
            "validation_verdict": "INSUFFICIENT EVIDENCE",
            "validation_pass": False,
            "promotion_candidate": False,
            "confidence_grade": "INSUFFICIENT",
            "sample_trades": total_trades,
            "folds_requested": folds,
            "folds_completed": 0,
            "positive_folds": 0,
            "positive_fold_ratio": 0.0,
            "worst_fold_expectancy_r": 0.0,
            "median_fold_expectancy_r": 0.0,
            "worst_fold_profit_factor": 0.0,
            "median_fold_profit_factor": 0.0,
            "aggregate_oos_expectancy_r": 0.0,
            "aggregate_oos_profit_factor": 0.0,
            "aggregate_oos_win_rate_pct": 0.0,
            "bootstrap_expectancy_low_r": None,
            "bootstrap_expectancy_high_r": None,
            "fold_results": fold_frame,
            "notes": [
                "No valid walk-forward folds could be created."
            ],
        }

    aggregate_oos = pd.concat(
        oos_segments,
        ignore_index=True,
    )

    aggregate_expectancy = (
        _expectancy_r(
            aggregate_oos
        )
    )

    aggregate_pf = (
        _profit_factor(
            aggregate_oos
        )
    )

    aggregate_win_rate = (
        _win_rate(
            aggregate_oos
        )
    )

    aggregate_r = (
        _safe_numeric_series(
            aggregate_oos,
            "r_multiple",
        )
    )

    ci_low, ci_high = (
        _bootstrap_mean_ci(
            aggregate_r.values
        )
    )

    positive_folds = int(
        fold_frame[
            "positive_fold"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    positive_fold_ratio = (
        positive_folds
        / max(
            completed_folds,
            1,
        )
    )

    expectancy_values = pd.to_numeric(
        fold_frame[
            "test_expectancy_r"
        ],
        errors="coerce",
    ).dropna()

    pf_values = pd.to_numeric(
        fold_frame[
            "test_profit_factor"
        ],
        errors="coerce",
    )

    finite_pf_values = (
        pf_values[
            np.isfinite(
                pf_values
            )
        ]
        .dropna()
    )

    worst_expectancy = (
        float(
            expectancy_values.min()
        )
        if not expectancy_values.empty
        else 0.0
    )

    median_expectancy = (
        float(
            expectancy_values.median()
        )
        if not expectancy_values.empty
        else 0.0
    )

    if not finite_pf_values.empty:
        worst_pf = float(
            finite_pf_values.min()
        )

        median_pf = float(
            finite_pf_values.median()
        )

    elif (
        fold_frame[
            "test_profit_factor"
        ]
        .astype(str)
        .eq(
            "inf"
        )
        .any()
    ):
        worst_pf = float(
            "inf"
        )

        median_pf = float(
            "inf"
        )

    else:
        worst_pf = 0.0
        median_pf = 0.0

    degradation_values = pd.to_numeric(
        fold_frame[
            "expectancy_degradation"
        ],
        errors="coerce",
    ).dropna()

    max_degradation = (
        float(
            degradation_values.max()
        )
        if not degradation_values.empty
        else None
    )

    notes = []

    enough_folds = (
        completed_folds
        >= max(
            3,
            folds - 1,
        )
    )

    enough_trades = (
        total_trades
        >= min_total_trades
    )

    fold_consistency = (
        positive_fold_ratio
        >= min_positive_fold_ratio
    )

    positive_aggregate = (
        aggregate_expectancy
        > 0
    )

    acceptable_aggregate_pf = bool(
        not np.isfinite(
            aggregate_pf
        )
        or aggregate_pf
        >= min_profit_factor
    )

    bootstrap_robust = bool(
        ci_low is not None
        and ci_low > 0
    )

    degradation_ok = bool(
        max_degradation
        is None
        or max_degradation
        <= max_expectancy_degradation
    )

    # We do not require every single fold to be positive,
    # but the worst fold cannot be catastrophically negative.
    worst_fold_ok = bool(
        worst_expectancy
        >= -0.50
    )

    if not enough_trades:
        notes.append(
            f"Only {total_trades} trades; at least {min_total_trades} are required."
        )

    if not enough_folds:
        notes.append(
            f"Only {completed_folds} usable walk-forward folds were completed."
        )

    if not fold_consistency:
        notes.append(
            f"Only {positive_fold_ratio * 100:.0f}% of walk-forward folds were positive."
        )

    if not positive_aggregate:
        notes.append(
            "Aggregate out-of-sample expectancy is not positive."
        )

    if (
        np.isfinite(
            aggregate_pf
        )
        and aggregate_pf
        < min_profit_factor
    ):
        notes.append(
            f"Aggregate out-of-sample profit factor is below {min_profit_factor:.2f}."
        )

    if (
        ci_low is not None
        and ci_low <= 0
    ):
        notes.append(
            "The 95% bootstrap interval for aggregate OOS expectancy includes zero."
        )

    if not degradation_ok:
        notes.append(
            "Train-to-test expectancy degradation is too severe in at least one fold."
        )

    if not worst_fold_ok:
        notes.append(
            "The worst unseen fold had severely negative expectancy."
        )

    validation_pass = bool(
        enough_trades
        and enough_folds
        and fold_consistency
        and positive_aggregate
        and acceptable_aggregate_pf
        and bootstrap_robust
        and degradation_ok
        and worst_fold_ok
    )

    # Stronger requirement before even considering production promotion.
    promotion_candidate = bool(
        validation_pass
        and positive_fold_ratio
        >= 0.75
        and aggregate_expectancy
        >= 0.10
        and (
            not np.isfinite(
                aggregate_pf
            )
            or aggregate_pf
            >= 1.30
        )
        and median_expectancy
        > 0
        and worst_expectancy
        >= -0.25
        and ci_low is not None
        and ci_low > 0
    )

    if promotion_candidate:

        grade = "A"

        verdict = "PASS"

    elif validation_pass:

        grade = "B"

        verdict = "PASS"

    elif (
        not enough_trades
        or not enough_folds
    ):

        grade = "INSUFFICIENT"

        verdict = "INSUFFICIENT EVIDENCE"

    else:

        grade = "FAIL"

        verdict = "FAIL"

    if not notes:
        notes.append(
            "Multi-fold walk-forward validation passed all configured robustness checks."
        )

    return {
        "validation_verdict": (
            verdict
        ),

        "validation_pass": (
            validation_pass
        ),

        "promotion_candidate": (
            promotion_candidate
        ),

        "confidence_grade": (
            grade
        ),

        "sample_trades": (
            total_trades
        ),

        "folds_requested": (
            folds
        ),

        "folds_completed": (
            completed_folds
        ),

        "positive_folds": (
            positive_folds
        ),

        "positive_fold_ratio": round(
            positive_fold_ratio,
            3,
        ),

        "worst_fold_expectancy_r": round(
            worst_expectancy,
            3,
        ),

        "median_fold_expectancy_r": round(
            median_expectancy,
            3,
        ),

        "worst_fold_profit_factor": (
            "inf"
            if not np.isfinite(
                worst_pf
            )
            else round(
                worst_pf,
                2,
            )
        ),

        "median_fold_profit_factor": (
            "inf"
            if not np.isfinite(
                median_pf
            )
            else round(
                median_pf,
                2,
            )
        ),

        "aggregate_oos_expectancy_r": round(
            aggregate_expectancy,
            3,
        ),

        "aggregate_oos_profit_factor": (
            "inf"
            if not np.isfinite(
                aggregate_pf
            )
            else round(
                aggregate_pf,
                2,
            )
        ),

        "aggregate_oos_win_rate_pct": round(
            aggregate_win_rate,
            2,
        ),

        "bootstrap_expectancy_low_r": (
            None
            if ci_low is None
            else round(
                ci_low,
                3,
            )
        ),

        "bootstrap_expectancy_high_r": (
            None
            if ci_high is None
            else round(
                ci_high,
                3,
            )
        ),

        "max_expectancy_degradation": (
            None
            if max_degradation is None
            else round(
                max_degradation,
                3,
            )
        ),

        "fold_results": (
            fold_frame
        ),

        "notes": (
            notes
        ),
    }


# ============================================================
# COMBINED v3.5 STRATEGY VALIDATION
# ============================================================

def full_strategy_validation(
    trades: pd.DataFrame,
    holdout_train_fraction: float = 0.70,
    walk_forward_folds: int = 4,
    min_total_trades: int = 40,
):
    """
    Run both:
        1. classic chronological holdout
        2. multi-fold walk-forward validation

    The walk-forward result is the primary v3.5 verdict.
    """

    holdout = chronological_validation(
        trades,
        train_fraction=holdout_train_fraction,
        min_total_trades=max(
            30,
            min_total_trades,
        ),
        min_oos_trades=10,
    )

    walk_forward = walk_forward_validation(
        trades,
        folds=walk_forward_folds,
        min_total_trades=min_total_trades,
        min_fold_trades=5,
        min_positive_fold_ratio=0.60,
        min_profit_factor=1.15,
        max_expectancy_degradation=0.75,
    )

    final_verdict = (
        walk_forward.get(
            "validation_verdict",
            "INSUFFICIENT EVIDENCE",
        )
    )

    validation_pass = bool(
        walk_forward.get(
            "validation_pass",
            False,
        )
    )

    promotion_candidate = bool(
        walk_forward.get(
            "promotion_candidate",
            False,
        )
        and holdout.get(
            "validation_pass",
            False,
        )
    )

    if promotion_candidate:
        grade = "A"

    elif validation_pass:
        grade = (
            walk_forward.get(
                "confidence_grade",
                "B",
            )
        )

    elif final_verdict == "INSUFFICIENT EVIDENCE":
        grade = "INSUFFICIENT"

    else:
        grade = "FAIL"

    notes = []

    notes.extend(
        holdout.get(
            "notes",
            [],
        )
    )

    notes.extend(
        walk_forward.get(
            "notes",
            [],
        )
    )

    return {
        "validation_verdict": final_verdict,

        "validation_pass": validation_pass,

        "promotion_candidate": promotion_candidate,

        "confidence_grade": grade,

        "holdout": holdout,

        "walk_forward": walk_forward,

        "sample_trades": (
            walk_forward.get(
                "sample_trades",
                0,
            )
        ),

        "aggregate_oos_expectancy_r": (
            walk_forward.get(
                "aggregate_oos_expectancy_r",
                0.0,
            )
        ),

        "aggregate_oos_profit_factor": (
            walk_forward.get(
                "aggregate_oos_profit_factor",
                0.0,
            )
        ),

        "positive_fold_ratio": (
            walk_forward.get(
                "positive_fold_ratio",
                0.0,
            )
        ),

        "worst_fold_expectancy_r": (
            walk_forward.get(
                "worst_fold_expectancy_r",
                0.0,
            )
        ),

        "bootstrap_expectancy_low_r": (
            walk_forward.get(
                "bootstrap_expectancy_low_r"
            )
        ),

        "bootstrap_expectancy_high_r": (
            walk_forward.get(
                "bootstrap_expectancy_high_r"
            )
        ),

        "notes": notes,
    }
