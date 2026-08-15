from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ============================================================
# BASIC PERFORMANCE HELPERS
# ============================================================

def _profit_factor(frame):
    if frame is None or frame.empty or "pnl" not in frame:
        return 0.0

    pnl = pd.to_numeric(
        frame["pnl"],
        errors="coerce",
    ).dropna()

    gross_profit = float(
        pnl[pnl > 0].sum()
    )

    gross_loss = abs(
        float(
            pnl[pnl < 0].sum()
        )
    )

    if gross_loss == 0:
        return (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    return gross_profit / gross_loss


def _win_rate(frame):
    if frame is None or frame.empty or "pnl" not in frame:
        return 0.0

    pnl = pd.to_numeric(
        frame["pnl"],
        errors="coerce",
    ).dropna()

    if len(pnl) == 0:
        return 0.0

    return float(
        (pnl > 0).mean() * 100
    )


def _expectancy_r(frame):
    if frame is None or frame.empty or "r_multiple" not in frame:
        return 0.0

    r = pd.to_numeric(
        frame["r_multiple"],
        errors="coerce",
    ).dropna()

    if len(r) == 0:
        return 0.0

    return float(
        r.mean()
    )


def _bootstrap_mean_ci(
    values,
    iterations=2500,
    confidence=0.95,
    seed=1337,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 10:
        return None, None

    rng = np.random.default_rng(
        seed
    )

    means = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        means[i] = float(
            sample.mean()
        )

    alpha = (
        1.0 - confidence
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

    return low, high


# ============================================================
# CANDIDATE VALIDATION
# ============================================================

def validate_candidate(row):
    """
    Validate one live scanner candidate.

    This function is intentionally additive:
    it does not replace the strategy engine.
    It simply provides a second validation layer.
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

    if hasattr(row, "to_dict"):
        data = row.to_dict()
    else:
        data = dict(row)

    reasons = []
    score = 0

    signal = str(
        data.get(
            "signal",
            "",
        )
    ).upper()

    try:
        swing_score = float(
            data.get(
                "swing_score",
                0,
            )
            or 0
        )
    except Exception:
        swing_score = 0.0

    try:
        entry_quality = float(
            data.get(
                "entry_quality",
                0,
            )
            or 0
        )
    except Exception:
        entry_quality = 0.0

    try:
        reward_risk = float(
            data.get(
                "reward_risk",
                0,
            )
            or 0
        )
    except Exception:
        reward_risk = 0.0

    try:
        intraday_score = float(
            data.get(
                "intraday_score",
                0,
            )
            or 0
        )
    except Exception:
        intraday_score = 0.0

    try:
        leadership = float(
            data.get(
                "leadership_percentile",
                0,
            )
            or 0
        )
    except Exception:
        leadership = 0.0

    try:
        market_score = float(
            data.get(
                "market_score",
                0,
            )
            or 0
        )
    except Exception:
        market_score = 0.0

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

    if swing_score >= 90:
        score += 20
        reasons.append(
            f"Excellent Swing Score: {swing_score:.1f}."
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

    else:
        reasons.append(
            f"Swing Score {swing_score:.1f} is below 80."
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

    else:
        reasons.append(
            f"Reward/risk {reward_risk:.2f}:1 is below 2.00:1."
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
            f"Supportive market regime: {market_score:.1f}/10."
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
        and market_score >= 7
        and trend_health
        and distribution_days <= 4
    )

    if (
        validation_pass
        and score >= 90
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
# VALIDATION SUMMARY
# ============================================================

def validation_summary(frame):
    """
    Summarize validation results for a scanner DataFrame.
    """

    if frame is None or frame.empty:

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

    top_grade = (
        result_frame[
            "validation_grade"
        ]
        .map(
            grade_order
        )
        .idxmax()
    )

    top_validation_grade = (
        result_frame.loc[
            top_grade,
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
        "top_validation_grade": top_validation_grade,
    }


# ============================================================
# HISTORICAL / OUT-OF-SAMPLE VALIDATION
# ============================================================

def chronological_validation(
    trades: pd.DataFrame,
    train_fraction: float = 0.70,
    min_total_trades: int = 30,
    min_oos_trades: int = 10,
):

    if trades is None or trades.empty:

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
            "notes": [
                "No completed trades were available for holdout validation."
            ],
        }

    frame = trades.copy()

    sort_col = (
        "exit_time"
        if "exit_time" in frame.columns
        else (
            "entry_time"
            if "entry_time" in frame.columns
            else None
        )
    )

    if sort_col:

        frame[
            sort_col
        ] = pd.to_datetime(
            frame[
                sort_col
            ],
            utc=True,
            errors="coerce",
        )

        frame = frame.sort_values(
            sort_col
        )

    frame = frame.reset_index(
        drop=True
    )

    n = len(
        frame
    )

    split = min(
        max(
            int(
                math.floor(
                    n * train_fraction
                )
            ),
            1,
        ),
        max(
            n - 1,
            1,
        ),
    )

    train = frame.iloc[
        :split
    ].copy()

    test = frame.iloc[
        split:
    ].copy()

    pf = _profit_factor(
        test
    )

    oos_exp = _expectancy_r(
        test
    )

    if "r_multiple" in test.columns:

        oos_r = pd.to_numeric(
            test[
                "r_multiple"
            ],
            errors="coerce",
        ).dropna()

    else:

        oos_r = pd.Series(
            dtype=float
        )

    ci_low, ci_high = _bootstrap_mean_ci(
        oos_r.values
    )

    notes = []

    if n < min_total_trades:

        notes.append(
            f"Only {n} completed trades; "
            f"at least {min_total_trades} is preferred."
        )

    if len(test) < min_oos_trades:

        notes.append(
            f"Only {len(test)} out-of-sample trades; "
            f"at least {min_oos_trades} is preferred."
        )

    if oos_exp <= 0:

        notes.append(
            "Out-of-sample expectancy is not positive."
        )

    if (
        np.isfinite(
            pf
        )
        and pf < 1.15
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

    enough = bool(
        n >= min_total_trades
        and len(test) >= min_oos_trades
    )

    positive = bool(
        oos_exp > 0
        and (
            pf >= 1.15
            or not np.isfinite(
                pf
            )
        )
    )

    robust = bool(
        ci_low is not None
        and ci_low > 0
    )

    passed = bool(
        enough
        and positive
        and robust
    )

    if (
        passed
        and len(test) >= 20
        and (
            not np.isfinite(
                pf
            )
            or pf >= 1.35
        )
    ):

        grade = "A"

    elif passed:

        grade = "B"

    elif (
        enough
        and positive
    ):

        grade = "C"

    elif enough:

        grade = "D"

    else:

        grade = "INSUFFICIENT"

    if not notes:

        notes.append(
            "Chronological holdout and bootstrap checks passed "
            "the configured thresholds."
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
            _expectancy_r(
                train
            ),
            3,
        ),
        "out_of_sample_expectancy_r": round(
            oos_exp,
            3,
        ),
        "out_of_sample_profit_factor": (
            "inf"
            if not np.isfinite(
                pf
            )
            else round(
                pf,
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
        "confidence_grade": grade,
        "validation_pass": passed,
        "notes": notes,
    }
