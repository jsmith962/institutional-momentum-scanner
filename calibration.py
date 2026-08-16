"""
v3.4.2 fast calibration engine.

Research only.

This module evaluates bounded alternate threshold profiles by replaying
the historical candidate log created by the production backtest.

IMPORTANT:
- It does NOT call Alpaca.
- It does NOT rebuild daily or intraday indicators.
- It does NOT modify live production BUY thresholds.
- It is designed to run quickly inside Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# PRODUCTION REFERENCE THRESHOLDS
# ============================================================

PRODUCTION_SWING_SCORE = 85.0
PRODUCTION_INTRADAY_SCORE = 85.0
PRODUCTION_ENTRY_QUALITY = 10.0
PRODUCTION_MARKET_SCORE = 5.0
PRODUCTION_LEADERSHIP = 70.0
PRODUCTION_MAX_DISTRIBUTION_DAYS = 4
PRODUCTION_MIN_REWARD_RISK = 2.0


# ============================================================
# PROFILE MODEL
# ============================================================

@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    swing_score: float
    intraday_score: float
    entry_quality: float
    market_score: float = PRODUCTION_MARKET_SCORE
    leadership: float = PRODUCTION_LEADERSHIP
    max_distribution_days: int = PRODUCTION_MAX_DISTRIBUTION_DAYS
    reward_risk: float = PRODUCTION_MIN_REWARD_RISK


# ============================================================
# DEFAULT BOUNDED RESEARCH PROFILES
# ============================================================

DEFAULT_PROFILES = [
    CalibrationProfile(
        name="Production 85/85",
        swing_score=85.0,
        intraday_score=85.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Strict 80/85",
        swing_score=80.0,
        intraday_score=85.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Balanced 77.5/80",
        swing_score=77.5,
        intraday_score=80.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Balanced 75/80",
        swing_score=75.0,
        intraday_score=80.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research 72.5/75",
        swing_score=72.5,
        intraday_score=75.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research 70/75",
        swing_score=70.0,
        intraday_score=75.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research 67.5/70",
        swing_score=67.5,
        intraday_score=70.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Exploratory 65/70",
        swing_score=65.0,
        intraday_score=70.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="High-quality 72.5/75 Q12",
        swing_score=72.5,
        intraday_score=75.0,
        entry_quality=12.0,
    ),

    CalibrationProfile(
        name="High-quality 70/70 Q12",
        swing_score=70.0,
        intraday_score=70.0,
        entry_quality=12.0,
    ),
]


# ============================================================
# HELPERS
# ============================================================

def _numeric(
    frame: pd.DataFrame,
    column: str,
    default=np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _boolean(
    frame: pd.DataFrame,
    column: str,
    default=False,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            bool(default),
            index=frame.index,
            dtype="bool",
        )

    values = frame[column]

    if values.dtype == bool:
        return values.fillna(default)

    return (
        values
        .fillna(default)
        .astype(bool)
    )


def _normalise_candidate_log(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the production signal audit into a compact,
    vector-friendly calibration frame.
    """

    if (
        signal_log is None
        or not isinstance(signal_log, pd.DataFrame)
        or signal_log.empty
    ):
        return pd.DataFrame()

    frame = signal_log.copy()

    frame["_swing"] = _numeric(
        frame,
        "swing_score",
    )

    frame["_intraday"] = _numeric(
        frame,
        "intraday_score",
    )

    frame["_quality"] = _numeric(
        frame,
        "entry_quality",
    )

    frame["_market"] = _numeric(
        frame,
        "market_score",
    )

    frame["_leadership"] = _numeric(
        frame,
        "leadership_percentile",
    )

    frame["_distribution"] = _numeric(
        frame,
        "distribution_days",
    )

    frame["_rr"] = _numeric(
        frame,
        "reward_risk",
    )

    frame["_risk"] = _boolean(
        frame,
        "risk_flag",
        False,
    )

    frame["_trend"] = _boolean(
        frame,
        "trend_health",
        True,
    )

    # If an explicit inside-entry-zone field exists, use it.
    if "inside_entry_zone" in frame.columns:

        frame["_inside_zone"] = _boolean(
            frame,
            "inside_entry_zone",
            False,
        )

    else:

        price = _numeric(
            frame,
            "price",
        )

        entry_low = _numeric(
            frame,
            "entry_low",
        )

        entry_high = _numeric(
            frame,
            "entry_high",
        )

        # If the backtest audit does not contain the price-zone fields,
        # do not automatically reject every row. Treat missing zone data
        # as unavailable rather than failed.
        zone_data_available = (
            price.notna()
            & entry_low.notna()
            & entry_high.notna()
        )

        frame["_inside_zone"] = True

        frame.loc[
            zone_data_available,
            "_inside_zone",
        ] = (
            price[zone_data_available]
            >= entry_low[zone_data_available]
        ) & (
            price[zone_data_available]
            <= entry_high[zone_data_available]
        )

    # Explicit intraday signal if available.
    if "intraday_signal" in frame.columns:

        frame["_intraday_buy"] = (
            frame["intraday_signal"]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("BUY")
        )

    else:

        # Do not silently eliminate rows solely because an older
        # audit lacks the text signal field.
        frame["_intraday_buy"] = True

    # Clean invalid score rows.
    frame = frame[
        frame["_swing"].notna()
        & frame["_intraday"].notna()
        & frame["_quality"].notna()
    ].copy()

    return frame.reset_index(drop=True)


def _profile_mask(
    frame: pd.DataFrame,
    profile: CalibrationProfile,
) -> pd.Series:
    """
    Fully vectorized gate evaluation.
    """

    mask = (
        (frame["_swing"] >= profile.swing_score)
        & (
            frame["_intraday"]
            >= profile.intraday_score
        )
        & (
            frame["_quality"]
            >= profile.entry_quality
        )
        & ~frame["_risk"]
        & frame["_trend"]
        & frame["_inside_zone"]
        & frame["_intraday_buy"]
    )

    if frame["_market"].notna().any():

        mask &= (
            frame["_market"].isna()
            | (
                frame["_market"]
                >= profile.market_score
            )
        )

    if frame["_leadership"].notna().any():

        mask &= (
            frame["_leadership"].isna()
            | (
                frame["_leadership"]
                >= profile.leadership
            )
        )

    if frame["_distribution"].notna().any():

        mask &= (
            frame["_distribution"].isna()
            | (
                frame["_distribution"]
                <= profile.max_distribution_days
            )
        )

    if frame["_rr"].notna().any():

        mask &= (
            frame["_rr"].isna()
            | (
                frame["_rr"]
                >= profile.reward_risk
            )
        )

    return mask.fillna(False)


def _chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
):
    """
    Split observations chronologically when possible.
    """

    if frame.empty:
        return frame, frame

    date_column = None

    for candidate in (
        "signal_time",
        "session",
        "signal_date",
        "date",
        "timestamp",
    ):
        if candidate in frame.columns:
            date_column = candidate
            break

    ordered = frame.copy()

    if date_column is not None:

        ordered["_cal_date"] = pd.to_datetime(
            ordered[date_column],
            errors="coerce",
            utc=True,
        )

        ordered = (
            ordered
            .sort_values(
                "_cal_date",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    cutoff = int(
        round(
            len(ordered)
            * float(train_fraction)
        )
    )

    cutoff = max(
        1,
        min(
            cutoff,
            len(ordered),
        ),
    )

    return (
        ordered.iloc[:cutoff].copy(),
        ordered.iloc[cutoff:].copy(),
    )


# ============================================================
# PROFILE SELECTION
# ============================================================

def bounded_profiles(
    max_profiles: int = 6,
) -> list[CalibrationProfile]:

    count = max(
        1,
        int(max_profiles),
    )

    return DEFAULT_PROFILES[:count]


# ============================================================
# FAST CALIBRATION
# ============================================================

def run_fast_calibration(
    signal_log: pd.DataFrame,
    max_profiles: int = 6,
    train_fraction: float = 0.70,
    progress_callback=None,
) -> dict:
    """
    Replay the cached historical candidate audit through
    bounded alternate profiles.

    This function performs no market-data requests.
    """

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:

        return {
            "status": "NO_DATA",
            "summary": pd.DataFrame(),
            "profile_results": {},
            "candidate_count": 0,
            "message": (
                "No usable historical candidate log "
                "was available for calibration."
            ),
        }

    profiles = bounded_profiles(
        max_profiles
    )

    train_frame, oos_frame = (
        _chronological_split(
            frame,
            train_fraction=train_fraction,
        )
    )

    rows = []
    profile_results = {}

    total_profiles = len(
        profiles
    )

    for index, profile in enumerate(
        profiles,
        start=1,
    ):

        if progress_callback is not None:

            progress_callback(
                index,
                total_profiles,
                profile.name,
            )

        full_mask = _profile_mask(
            frame,
            profile,
        )

        train_mask = _profile_mask(
            train_frame,
            profile,
        )

        oos_mask = _profile_mask(
            oos_frame,
            profile,
        )

        full_candidates = frame[
            full_mask
        ].copy()

        train_candidates = train_frame[
            train_mask
        ].copy()

        oos_candidates = oos_frame[
            oos_mask
        ].copy()

        total_count = int(
            full_mask.sum()
        )

        train_count = int(
            train_mask.sum()
        )

        oos_count = int(
            oos_mask.sum()
        )

        candidate_rate = (
            total_count
            / max(
                len(frame),
                1,
            )
            * 100
        )

        oos_candidate_rate = (
            oos_count
            / max(
                len(oos_frame),
                1,
            )
            * 100
            if len(oos_frame)
            else 0.0
        )

        # Stability ratio:
        # compares candidate frequency in later data to earlier data.
        train_rate = (
            train_count
            / max(
                len(train_frame),
                1,
            )
        )

        oos_rate = (
            oos_count
            / max(
                len(oos_frame),
                1,
            )
            if len(oos_frame)
            else 0.0
        )

        if train_rate > 0:

            stability_ratio = (
                oos_rate
                / train_rate
            )

        else:

            stability_ratio = 0.0

        row = {
            "profile": profile.name,
            "swing_threshold": profile.swing_score,
            "intraday_threshold": profile.intraday_score,
            "entry_quality": profile.entry_quality,
            "market_score": profile.market_score,
            "leadership": profile.leadership,
            "max_distribution_days": profile.max_distribution_days,
            "reward_risk": profile.reward_risk,
            "all_candidates": total_count,
            "candidate_rate_pct": round(
                candidate_rate,
                3,
            ),
            "in_sample_candidates": train_count,
            "out_of_sample_candidates": oos_count,
            "oos_candidate_rate_pct": round(
                oos_candidate_rate,
                3,
            ),
            "stability_ratio": round(
                stability_ratio,
                3,
            ),
        }

        rows.append(
            row
        )

        profile_results[
            profile.name
        ] = {
            "profile": asdict(
                profile
            ),
            "all_candidates": full_candidates,
            "in_sample_candidates": train_candidates,
            "out_of_sample_candidates": oos_candidates,
            "metrics": row,
        }

    summary = pd.DataFrame(
        rows
    )

    # Prefer profiles that:
    # 1. actually produce candidates,
    # 2. continue producing them out of sample,
    # 3. remain reasonably stable across time,
    # while still preferring stricter thresholds.
    if not summary.empty:

        summary["_has_oos"] = (
            summary[
                "out_of_sample_candidates"
            ]
            > 0
        ).astype(int)

        summary["_strictness"] = (
            summary[
                "swing_threshold"
            ]
            + summary[
                "intraday_threshold"
            ]
            + (
                summary[
                    "entry_quality"
                ]
                * 2
            )
        )

        summary = (
            summary
            .sort_values(
                [
                    "_has_oos",
                    "out_of_sample_candidates",
                    "stability_ratio",
                    "_strictness",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_has_oos",
                    "_strictness",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    production_reachable = bool(
        (
            frame["_swing"]
            >= PRODUCTION_SWING_SCORE
        ).any()
    )

    best_profile = None

    if not summary.empty:

        viable = summary[
            summary[
                "out_of_sample_candidates"
            ]
            > 0
        ]

        if not viable.empty:

            best_profile = (
                viable.iloc[
                    0
                ].to_dict()
            )

    return {
        "status": "COMPLETE",
        "summary": summary,
        "profile_results": profile_results,
        "candidate_count": len(
            frame
        ),
        "in_sample_count": len(
            train_frame
        ),
        "out_of_sample_count": len(
            oos_frame
        ),
        "production_reachable": production_reachable,
        "best_profile": best_profile,
        "message": (
            f"Completed {len(profiles)} calibration "
            f"profiles across {len(frame):,} cached "
            f"historical observations."
        ),
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def score_distribution(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:
        return pd.DataFrame()

    rows = []

    mapping = [
        (
            "Swing Score",
            "_swing",
        ),
        (
            "Intraday Score",
            "_intraday",
        ),
        (
            "Entry Quality",
            "_quality",
        ),
    ]

    percentiles = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    for label, column in mapping:

        values = (
            frame[column]
            .dropna()
            .astype(float)
        )

        if values.empty:
            continue

        for pct in percentiles:

            if pct == 1.0:
                percentile_label = "Maximum"
            else:
                percentile_label = (
                    f"{int(pct * 100)}th"
                )

            rows.append(
                {
                    "score": label,
                    "observations": len(
                        values
                    ),
                    "percentile": percentile_label,
                    "value": round(
                        float(
                            values.quantile(
                                pct
                            )
                        ),
                        2,
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def production_gate_bottlenecks(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:
        return pd.DataFrame()

    total = len(
        frame
    )

    gates = [
        (
            "Swing Score >= 85",
            frame["_swing"]
            >= PRODUCTION_SWING_SCORE,
        ),

        (
            "Intraday Score >= 85",
            frame["_intraday"]
            >= PRODUCTION_INTRADAY_SCORE,
        ),

        (
            "Entry Quality >= 10",
            frame["_quality"]
            >= PRODUCTION_ENTRY_QUALITY,
        ),
    ]

    if frame["_leadership"].notna().any():

        gates.append(
            (
                "Leadership >= 70th percentile",
                (
                    frame["_leadership"].isna()
                    | (
                        frame["_leadership"]
                        >= PRODUCTION_LEADERSHIP
                    )
                ),
            )
        )

    if frame["_distribution"].notna().any():

        gates.append(
            (
                "Distribution Days <= 4",
                (
                    frame["_distribution"].isna()
                    | (
                        frame["_distribution"]
                        <= PRODUCTION_MAX_DISTRIBUTION_DAYS
                    )
                ),
            )
        )

    if frame["_market"].notna().any():

        gates.append(
            (
                "Market Score >= 5",
                (
                    frame["_market"].isna()
                    | (
                        frame["_market"]
                        >= PRODUCTION_MARKET_SCORE
                    )
                ),
            )
        )

    if frame["_rr"].notna().any():

        gates.append(
            (
                "Reward/Risk >= 2.0",
                (
                    frame["_rr"].isna()
                    | (
                        frame["_rr"]
                        >= PRODUCTION_MIN_REWARD_RISK
                    )
                ),
            )
        )

    rows = []

    for name, mask in gates:

        passed = int(
            mask.fillna(
                False
            ).sum()
        )

        failed = (
            total
            - passed
        )

        rows.append(
            {
                "gate": name,
                "passed": passed,
                "failed": failed,
                "pass_rate_pct": round(
                    passed
                    / max(
                        total,
                        1,
                    )
                    * 100,
                    2,
                ),
                "failure_rate_pct": round(
                    failed
                    / max(
                        total,
                        1,
                    )
                    * 100,
                    2,
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "failed",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BACKWARD-COMPATIBILITY WRAPPER
# ============================================================

def calibration_summary(
    signal_log: pd.DataFrame,
    max_profiles: int = 6,
) -> pd.DataFrame:
    """
    Compatibility helper for older UI code.
    """

    result = run_fast_calibration(
        signal_log,
        max_profiles=max_profiles,
    )

    return result.get(
        "summary",
        pd.DataFrame(),
    )
