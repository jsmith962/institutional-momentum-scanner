"""
Institutional Swing Scanner v3.5.2
Fast Calibration Engine

RESEARCH ONLY.

This module analyzes the historical signal log produced by the
production backtester.

IMPORTANT:
- It does NOT import itself.
- It does NOT request Alpaca market data.
- It does NOT modify live production thresholds.
- Production control retains the original intraday BUY-label requirement.
- Research profiles use their own numeric intraday thresholds.
- Candidate stability is NOT the same thing as profitability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


# ============================================================
# PRODUCTION THRESHOLDS
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
    production_control: bool = False


# ============================================================
# DEFAULT PROFILES
# ============================================================

DEFAULT_PROFILES = [
    CalibrationProfile(
        name="Production 85/85",
        swing_score=85.0,
        intraday_score=85.0,
        entry_quality=10.0,
        production_control=True,
    ),

    CalibrationProfile(
        name="Research S82.5/I80",
        swing_score=82.5,
        intraday_score=80.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S80/I80",
        swing_score=80.0,
        intraday_score=80.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S79/I75",
        swing_score=79.0,
        intraday_score=75.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S77.5/I70",
        swing_score=77.5,
        intraday_score=70.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S75/I65",
        swing_score=75.0,
        intraday_score=65.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S72.5/I60",
        swing_score=72.5,
        intraday_score=60.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S70/I60",
        swing_score=70.0,
        intraday_score=60.0,
        entry_quality=10.0,
    ),

    CalibrationProfile(
        name="Research S72.5/I60 Q12",
        swing_score=72.5,
        intraday_score=60.0,
        entry_quality=12.0,
    ),

    CalibrationProfile(
        name="Research S70/I55",
        swing_score=70.0,
        intraday_score=55.0,
        entry_quality=10.0,
    ),
]


# ============================================================
# SAFE COLUMN HELPERS
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

    if pd.api.types.is_bool_dtype(
        values
    ):

        return values.fillna(
            default
        ).astype(bool)

    # Handle text values safely.
    text = (
        values
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "f",
        "",
        "none",
        "nan",
    }

    out = pd.Series(
        bool(default),
        index=frame.index,
        dtype="bool",
    )

    out.loc[
        text.isin(
            true_values
        )
    ] = True

    out.loc[
        text.isin(
            false_values
        )
    ] = False

    return out


# ============================================================
# NORMALIZE SIGNAL LOG
# ============================================================

def _normalise_candidate_log(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:

    if (
        signal_log is None
        or not isinstance(
            signal_log,
            pd.DataFrame,
        )
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

    frame["_too_extended"] = _boolean(
        frame,
        "too_extended",
        False,
    )

    frame["_trend"] = _boolean(
        frame,
        "trend_health",
        True,
    )

    # --------------------------------------------------------
    # ENTRY ZONE
    # --------------------------------------------------------

    if "inside_entry_zone" in frame.columns:

        frame["_inside_zone"] = (
            _boolean(
                frame,
                "inside_entry_zone",
                True,
            )
        )

    else:

        reference_price = _numeric(
            frame,
            "reference_price",
        )

        if reference_price.isna().all():

            reference_price = _numeric(
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

        available = (
            reference_price.notna()
            & entry_low.notna()
            & entry_high.notna()
        )

        frame["_inside_zone"] = True

        frame.loc[
            available,
            "_inside_zone",
        ] = (
            (
                reference_price[
                    available
                ]
                >= entry_low[
                    available
                ]
            )
            &
            (
                reference_price[
                    available
                ]
                <= entry_high[
                    available
                ]
            )
        )

    # --------------------------------------------------------
    # ORIGINAL PRODUCTION INTRADAY LABEL
    # --------------------------------------------------------

    if "intraday_signal" in frame.columns:

        frame[
            "_production_intraday_buy"
        ] = (
            frame[
                "intraday_signal"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("BUY")
        )

    else:

        frame[
            "_production_intraday_buy"
        ] = False

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_column = None

    for candidate in [
        "signal_time",
        "session",
        "signal_date",
        "date",
        "timestamp",
    ]:

        if candidate in frame.columns:

            date_column = candidate
            break

    if date_column is not None:

        frame["_calibration_date"] = (
            pd.to_datetime(
                frame[
                    date_column
                ],
                errors="coerce",
                utc=True,
            )
        )

    else:

        frame["_calibration_date"] = (
            pd.NaT
        )

    # Must contain the three core research fields.
    frame = frame[
        frame["_swing"].notna()
        & frame["_intraday"].notna()
        & frame["_quality"].notna()
    ].copy()

    if frame.empty:

        return frame

    frame = (
        frame
        .sort_values(
            "_calibration_date",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return frame


# ============================================================
# INDIVIDUAL PROFILE GATES
# ============================================================

def _profile_gate_series(
    frame: pd.DataFrame,
    profile: CalibrationProfile,
) -> list[tuple[str, pd.Series]]:

    if frame.empty:

        return []

    gates = []

    # 1. Catalyst risk
    gates.append(
        (
            "No active catalyst-risk flag",
            ~frame["_risk"],
        )
    )

    # 2. Extension
    gates.append(
        (
            "Not too extended",
            ~frame["_too_extended"],
        )
    )

    # 3. Swing Score
    gates.append(
        (
            (
                f"Swing Score >= "
                f"{profile.swing_score:g}"
            ),
            (
                frame["_swing"]
                >= profile.swing_score
            ),
        )
    )

    # 4. Entry Quality
    gates.append(
        (
            (
                f"Entry Quality >= "
                f"{profile.entry_quality:g}/15"
            ),
            (
                frame["_quality"]
                >= profile.entry_quality
            ),
        )
    )

    # 5. Reward / risk
    if frame["_rr"].notna().any():

        gates.append(
            (
                (
                    f"Reward/Risk >= "
                    f"{profile.reward_risk:g}:1"
                ),
                (
                    frame["_rr"].isna()
                    |
                    (
                        frame["_rr"]
                        >= profile.reward_risk
                    )
                ),
            )
        )

    # 6. Market regime
    if frame["_market"].notna().any():

        gates.append(
            (
                (
                    f"Market Score >= "
                    f"{profile.market_score:g}"
                ),
                (
                    frame["_market"].isna()
                    |
                    (
                        frame["_market"]
                        >= profile.market_score
                    )
                ),
            )
        )

    # 7. Entry zone
    gates.append(
        (
            "Inside preferred entry zone",
            frame["_inside_zone"],
        )
    )

    # 8. Trend
    gates.append(
        (
            "Trend health passed",
            frame["_trend"],
        )
    )

    # 9. Distribution
    if frame[
        "_distribution"
    ].notna().any():

        gates.append(
            (
                (
                    "Distribution Days <= "
                    f"{profile.max_distribution_days}"
                ),
                (
                    frame[
                        "_distribution"
                    ].isna()
                    |
                    (
                        frame[
                            "_distribution"
                        ]
                        <= profile.max_distribution_days
                    )
                ),
            )
        )

    # 10. Leadership
    if frame[
        "_leadership"
    ].notna().any():

        gates.append(
            (
                (
                    "Leadership >= "
                    f"{profile.leadership:g}th percentile"
                ),
                (
                    frame[
                        "_leadership"
                    ].isna()
                    |
                    (
                        frame[
                            "_leadership"
                        ]
                        >= profile.leadership
                    )
                ),
            )
        )

    # 11. Intraday numeric score
    gates.append(
        (
            (
                f"Intraday Score >= "
                f"{profile.intraday_score:g}"
            ),
            (
                frame["_intraday"]
                >= profile.intraday_score
            ),
        )
    )

    # --------------------------------------------------------
    # KEY v3.5 FIX
    #
    # Production control requires original classifier BUY.
    #
    # Research profiles DO NOT.
    # --------------------------------------------------------

    if profile.production_control:

        gates.append(
            (
                "Original production intraday signal = BUY",
                frame[
                    "_production_intraday_buy"
                ],
            )
        )

    return [
        (
            name,
            mask.fillna(
                False
            ).astype(bool),
        )
        for name, mask in gates
    ]


# ============================================================
# PROFILE MASK
# ============================================================

def _profile_mask(
    frame: pd.DataFrame,
    profile: CalibrationProfile,
) -> pd.Series:

    if frame.empty:

        return pd.Series(
            False,
            index=frame.index,
            dtype="bool",
        )

    mask = pd.Series(
        True,
        index=frame.index,
        dtype="bool",
    )

    for _, gate_mask in (
        _profile_gate_series(
            frame,
            profile,
        )
    ):

        mask &= gate_mask

    return mask.fillna(
        False
    )


# ============================================================
# PUBLIC GATE FUNNEL
# ============================================================

def profile_gate_funnel(
    signal_log: pd.DataFrame,
    profile: CalibrationProfile,
) -> pd.DataFrame:

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:

        return pd.DataFrame(
            columns=[
                "gate",
                "remaining",
                "removed_at_gate",
                "percent_remaining",
            ]
        )

    surviving = pd.Series(
        True,
        index=frame.index,
        dtype="bool",
    )

    rows = [
        {
            "gate": "Starting observations",
            "remaining": len(frame),
            "removed_at_gate": 0,
            "percent_remaining": 100.0,
        }
    ]

    prior_count = len(
        frame
    )

    for name, gate_mask in (
        _profile_gate_series(
            frame,
            profile,
        )
    ):

        surviving &= gate_mask

        remaining = int(
            surviving.sum()
        )

        removed = (
            prior_count
            - remaining
        )

        rows.append(
            {
                "gate": name,
                "remaining": remaining,
                "removed_at_gate": removed,
                "percent_remaining": round(
                    remaining
                    / max(
                        len(frame),
                        1,
                    )
                    * 100,
                    2,
                ),
            }
        )

        prior_count = remaining

    return pd.DataFrame(
        rows
    )


# ============================================================
# PUBLIC INDEPENDENT GATE FAILURES
# ============================================================

def profile_gate_failures(
    signal_log: pd.DataFrame,
    profile: CalibrationProfile,
) -> pd.DataFrame:

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:

        return pd.DataFrame(
            columns=[
                "gate",
                "passed",
                "failed",
                "pass_rate_pct",
                "failure_rate_pct",
            ]
        )

    total = len(
        frame
    )

    rows = []

    for name, mask in (
        _profile_gate_series(
            frame,
            profile,
        )
    ):

        passed = int(
            mask.sum()
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
# SCORE DISTRIBUTION
# ============================================================

def score_distribution(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:

    frame = _normalise_candidate_log(
        signal_log
    )

    if frame.empty:

        return pd.DataFrame()

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
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    rows = []

    for label, column in mapping:

        values = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .dropna()
        )

        if values.empty:

            continue

        for percentile in percentiles:

            if percentile == 0:

                pct_label = "Minimum"

            elif percentile == 1:

                pct_label = "Maximum"

            else:

                pct_label = (
                    f"{int(percentile * 100)}th"
                )

            rows.append(
                {
                    "score": label,
                    "observations": len(
                        values
                    ),
                    "percentile": pct_label,
                    "value": round(
                        float(
                            values.quantile(
                                percentile
                            )
                        ),
                        2,
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRODUCTION GATE BOTTLENECKS
# ============================================================

def production_gate_bottlenecks(
    signal_log: pd.DataFrame,
) -> pd.DataFrame:

    production = (
        DEFAULT_PROFILES[0]
    )

    return profile_gate_failures(
        signal_log,
        production,
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def _chronological_split(
    frame: pd.DataFrame,
    train_fraction=0.70,
):

    if frame.empty:

        return (
            frame.copy(),
            frame.copy(),
        )

    fraction = float(
        train_fraction
    )

    fraction = max(
        0.50,
        min(
            fraction,
            0.90,
        ),
    )

    ordered = frame.copy()

    if (
        "_calibration_date"
        in ordered.columns
    ):

        ordered = (
            ordered
            .sort_values(
                "_calibration_date",
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    cutoff = int(
        np.floor(
            len(ordered)
            * fraction
        )
    )

    cutoff = max(
        1,
        min(
            cutoff,
            len(ordered),
        ),
    )

    train = (
        ordered
        .iloc[
            :cutoff
        ]
        .copy()
    )

    later = (
        ordered
        .iloc[
            cutoff:
        ]
        .copy()
    )

    return (
        train,
        later,
    )


# ============================================================
# STABILITY FOLDS
# ============================================================

def _stability_metrics(
    frame: pd.DataFrame,
    profile: CalibrationProfile,
    folds=4,
):

    if frame.empty:

        return {
            "folds": 0,
            "positive_folds": 0,
            "positive_folds_pct": 0.0,
            "fold_counts": [],
        }

    folds = max(
        2,
        int(
            folds
        ),
    )

    folds = min(
        folds,
        len(
            frame
        ),
    )

    if folds < 1:

        return {
            "folds": 0,
            "positive_folds": 0,
            "positive_folds_pct": 0.0,
            "fold_counts": [],
        }

    indexes = np.array_split(
        np.arange(
            len(frame)
        ),
        folds,
    )

    counts = []

    for indexes_for_fold in indexes:

        if len(
            indexes_for_fold
        ) == 0:

            continue

        fold = (
            frame
            .iloc[
                indexes_for_fold
            ]
            .copy()
        )

        count = int(
            _profile_mask(
                fold,
                profile,
            ).sum()
        )

        counts.append(
            count
        )

    positive = sum(
        1
        for count in counts
        if count > 0
    )

    return {
        "folds": len(
            counts
        ),
        "positive_folds": positive,
        "positive_folds_pct": round(
            positive
            / max(
                len(
                    counts
                ),
                1,
            )
            * 100,
            1,
        ),
        "fold_counts": counts,
    }


# ============================================================
# PROFILE SELECTION
# ============================================================

def bounded_profiles(
    max_profiles=6,
):

    count = max(
        2,
        min(
            int(
                max_profiles
            ),
            len(
                DEFAULT_PROFILES
            ),
        ),
    )

    return (
        DEFAULT_PROFILES[
            :count
        ]
    )


# ============================================================
# FAST CALIBRATION ENGINE
# ============================================================

def run_fast_calibration(
    signal_log: pd.DataFrame,
    max_profiles=6,
    train_fraction=0.70,
    stability_folds=4,
    progress_callback=None,
):

    frame = (
        _normalise_candidate_log(
            signal_log
        )
    )

    if frame.empty:

        return {
            "status": "NO_DATA",
            "summary": pd.DataFrame(),
            "profile_results": {},
            "candidate_count": 0,
            "in_sample_count": 0,
            "later_period_count": 0,
            "production_swing_reachable": False,
            "production_intraday_reachable": False,
            "any_research_candidates": False,
            "best_profile": None,
            "message": (
                "No usable historical candidate observations "
                "were available."
            ),
        }

    profiles = bounded_profiles(
        max_profiles
    )

    train_frame, later_frame = (
        _chronological_split(
            frame,
            train_fraction=train_fraction,
        )
    )

    rows = []

    results = {}

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

        full_mask = (
            _profile_mask(
                frame,
                profile,
            )
        )

        train_mask = (
            _profile_mask(
                train_frame,
                profile,
            )
        )

        later_mask = (
            _profile_mask(
                later_frame,
                profile,
            )
        )

        all_candidates = (
            frame[
                full_mask
            ]
            .copy()
        )

        train_candidates = (
            train_frame[
                train_mask
            ]
            .copy()
        )

        later_candidates = (
            later_frame[
                later_mask
            ]
            .copy()
        )

        all_count = len(
            all_candidates
        )

        train_count = len(
            train_candidates
        )

        later_count = len(
            later_candidates
        )

        full_rate = (
            all_count
            / max(
                len(frame),
                1,
            )
        )

        train_rate = (
            train_count
            / max(
                len(train_frame),
                1,
            )
        )

        later_rate = (
            later_count
            / max(
                len(later_frame),
                1,
            )
            if len(
                later_frame
            )
            else 0.0
        )

        stability_ratio = (
            later_rate
            / train_rate
            if train_rate > 0
            else 0.0
        )

        fold_metrics = (
            _stability_metrics(
                frame,
                profile,
                folds=stability_folds,
            )
        )

        row = {
            "profile": profile.name,
            "production_control": (
                profile.production_control
            ),
            "swing_threshold": (
                profile.swing_score
            ),
            "intraday_threshold": (
                profile.intraday_score
            ),
            "entry_quality": (
                profile.entry_quality
            ),
            "market_score": (
                profile.market_score
            ),
            "leadership": (
                profile.leadership
            ),
            "max_distribution_days": (
                profile.max_distribution_days
            ),
            "reward_risk": (
                profile.reward_risk
            ),
            "all_candidates": (
                all_count
            ),
            "candidate_rate_pct": round(
                full_rate
                * 100,
                3,
            ),
            "in_sample_candidates": (
                train_count
            ),
            "later_period_candidates": (
                later_count
            ),
            "later_candidate_rate_pct": round(
                later_rate
                * 100,
                3,
            ),
            "stability_ratio": round(
                stability_ratio,
                3,
            ),
            "candidate_positive_folds": (
                fold_metrics[
                    "positive_folds"
                ]
            ),
            "candidate_positive_folds_pct": (
                fold_metrics[
                    "positive_folds_pct"
                ]
            ),
        }

        rows.append(
            row
        )

        results[
            profile.name
        ] = {
            "profile": asdict(
                profile
            ),
            "all_candidates": (
                all_candidates
            ),
            "in_sample_candidates": (
                train_candidates
            ),
            "later_period_candidates": (
                later_candidates
            ),
            "gate_funnel": (
                profile_gate_funnel(
                    signal_log,
                    profile,
                )
            ),
            "gate_failures": (
                profile_gate_failures(
                    signal_log,
                    profile,
                )
            ),
            "fold_counts": (
                fold_metrics[
                    "fold_counts"
                ]
            ),
            "metrics": row,
        }

    summary = pd.DataFrame(
        rows
    )

    # ========================================================
    # RANK RESEARCH PROFILES
    # ========================================================

    if not summary.empty:

        summary[
            "_has_later"
        ] = (
            summary[
                "later_period_candidates"
            ]
            > 0
        ).astype(int)

        summary[
            "_has_full"
        ] = (
            summary[
                "all_candidates"
            ]
            > 0
        ).astype(int)

        summary[
            "_strictness"
        ] = (
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
                    "_has_later",
                    "candidate_positive_folds_pct",
                    "later_period_candidates",
                    "_has_full",
                    "stability_ratio",
                    "_strictness",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_has_later",
                    "_has_full",
                    "_strictness",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # REACHABILITY
    # ========================================================

    production_swing_reachable = bool(
        (
            frame[
                "_swing"
            ]
            >= PRODUCTION_SWING_SCORE
        ).any()
    )

    production_intraday_reachable = bool(
        (
            frame[
                "_intraday"
            ]
            >= PRODUCTION_INTRADAY_SCORE
        ).any()
    )

    research_rows = (
        summary[
            ~summary[
                "production_control"
            ]
        ]
        if not summary.empty
        else pd.DataFrame()
    )

    any_research_candidates = bool(
        not research_rows.empty
        and (
            research_rows[
                "all_candidates"
            ]
            > 0
        ).any()
    )

    best_profile = None

    if not research_rows.empty:

        viable = (
            research_rows[
                (
                    research_rows[
                        "all_candidates"
                    ]
                    > 0
                )
                &
                (
                    research_rows[
                        "later_period_candidates"
                    ]
                    > 0
                )
            ]
        )

        if not viable.empty:

            best_profile = (
                viable
                .iloc[0]
                .to_dict()
            )

    return {
        "status": "COMPLETE",
        "summary": summary,
        "profile_results": results,
        "candidate_count": len(
            frame
        ),
        "in_sample_count": len(
            train_frame
        ),
        "later_period_count": len(
            later_frame
        ),
        "production_swing_reachable": (
            production_swing_reachable
        ),
        "production_intraday_reachable": (
            production_intraday_reachable
        ),
        "any_research_candidates": (
            any_research_candidates
        ),
        "best_profile": (
            best_profile
        ),
        "profiles_tested": len(
            profiles
        ),
        "message": (
            f"Completed {len(profiles)} bounded calibration "
            f"profiles across {len(frame):,} cached historical "
            f"observations."
        ),
    }


# ============================================================
# BACKWARD-COMPATIBILITY HELPERS
# ============================================================

def calibration_summary(
    signal_log: pd.DataFrame,
    max_profiles=6,
):

    result = run_fast_calibration(
        signal_log,
        max_profiles=max_profiles,
    )

    return result.get(
        "summary",
        pd.DataFrame(),
    )
