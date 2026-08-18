"""
Institutional Swing Scanner v3.7
Gate Bottleneck + Forward Return Research Engine

RESEARCH ONLY.

Purpose
-------
This module analyzes the historical signal audit produced by the production
backtester and determines:

1. What happened AFTER each historical scanner observation.
2. Which production BUY gates are eliminating the most opportunities.
3. Whether passing each gate was historically associated with better
   forward returns.
4. What happens when individual Swing Score and Intraday Score thresholds
   are varied.
5. Which gates should be studied further before changing production rules.

IMPORTANT
---------
This module does NOT change production BUY thresholds.
It does NOT generate live trades.
It does NOT claim causation or profitability.

Forward returns are used only for historical research and are never fed back
into the production scanner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# VERSION
# ============================================================

RESEARCH_VERSION = "v3.7"


# ============================================================
# PRODUCTION THRESHOLDS
# ============================================================

PRODUCTION_SWING_THRESHOLD = 85.0
PRODUCTION_INTRADAY_THRESHOLD = 85.0
PRODUCTION_ENTRY_QUALITY = 10.0
PRODUCTION_REWARD_RISK = 2.0
PRODUCTION_MARKET_SCORE = 5.0
PRODUCTION_LEADERSHIP = 70.0
PRODUCTION_MAX_DISTRIBUTION_DAYS = 4


# ============================================================
# FORWARD-RETURN HORIZONS
# ============================================================

DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


# ============================================================
# HELPERS
# ============================================================

def _safe_df(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    return pd.DataFrame()


def _numeric(
    series,
    default=np.nan,
):
    try:
        return pd.to_numeric(
            series,
            errors="coerce",
        )
    except Exception:
        if isinstance(series, pd.Series):
            return pd.Series(
                default,
                index=series.index,
                dtype=float,
            )

        return pd.Series(
            dtype=float
        )


def _safe_bool_series(
    values,
    index,
    default=False,
):
    if isinstance(values, pd.Series):

        if values.dtype == bool:
            return values.reindex(
                index
            ).fillna(
                default
            ).astype(
                bool
            )

        text = (
            values
            .reindex(
                index
            )
            .astype(
                str
            )
            .str
            .strip()
            .str
            .lower()
        )

        return text.isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "pass",
                "passed",
            }
        )

    return pd.Series(
        default,
        index=index,
        dtype=bool,
    )


def _date_series(
    frame: pd.DataFrame,
):
    for column in [
        "session",
        "signal_date",
        "date",
    ]:

        if column in frame.columns:

            values = pd.to_datetime(
                frame[
                    column
                ],
                errors="coerce",
            )

            try:
                return values.dt.date
            except Exception:
                pass

    if "signal_time" in frame.columns:

        values = pd.to_datetime(
            frame[
                "signal_time"
            ],
            errors="coerce",
            utc=True,
        )

        try:
            return values.dt.date
        except Exception:
            pass

    return pd.Series(
        pd.NaT,
        index=frame.index,
    )


def _pct(
    end,
    start,
):
    try:
        if (
            pd.isna(
                end
            )
            or pd.isna(
                start
            )
            or float(
                start
            )
            == 0
        ):
            return np.nan

        return (
            float(
                end
            )
            / float(
                start
            )
            - 1.0
        ) * 100.0

    except Exception:
        return np.nan


def _win_rate(
    values,
):
    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        return np.nan

    return float(
        (
            values
            > 0
        )
        .mean()
        * 100.0
    )


def _median(
    values,
):
    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        return np.nan

    return float(
        values.median()
    )


def _mean(
    values,
):
    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        return np.nan

    return float(
        values.mean()
    )


# ============================================================
# FORWARD RETURN ATTACHMENT
# ============================================================

def attach_forward_returns(
    signal_log: pd.DataFrame,
    daily_bars: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_path_sessions: int = 20,
) -> pd.DataFrame:
    """
    Attach future price behavior to historical scanner observations.

    This should be called AFTER the historical backtest has been completed.

    No future information is used by the production scanner itself.

    Added fields include:

        forward_1d_pct
        forward_3d_pct
        forward_5d_pct
        forward_10d_pct
        forward_20d_pct

        forward_20d_mfe_pct
        forward_20d_mae_pct

    MFE:
        Maximum favorable excursion.

    MAE:
        Maximum adverse excursion.
    """

    signals = _safe_df(
        signal_log
    )

    daily = _safe_df(
        daily_bars
    )

    if signals.empty:
        return signals

    if daily.empty:
        return signals

    required_daily = {
        "symbol",
        "timestamp",
        "close",
    }

    if not required_daily.issubset(
        daily.columns
    ):
        return signals

    signals = signals.copy()

    signals[
        "_research_session"
    ] = _date_series(
        signals
    )

    daily[
        "timestamp"
    ] = pd.to_datetime(
        daily[
            "timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    daily[
        "_research_session"
    ] = (
        daily[
            "timestamp"
        ]
        .dt
        .date
    )

    daily[
        "symbol"
    ] = (
        daily[
            "symbol"
        ]
        .astype(
            str
        )
        .str
        .upper()
    )

    if "symbol" not in signals.columns:
        return signals.drop(
            columns=[
                "_research_session"
            ],
            errors="ignore",
        )

    signals[
        "symbol"
    ] = (
        signals[
            "symbol"
        ]
        .astype(
            str
        )
        .str
        .upper()
    )

    horizons = sorted(
        {
            int(
                x
            )
            for x in horizons
            if int(
                x
            )
            > 0
        }
    )

    for horizon in horizons:

        signals[
            f"forward_{horizon}d_pct"
        ] = np.nan

    signals[
        "forward_20d_mfe_pct"
    ] = np.nan

    signals[
        "forward_20d_mae_pct"
    ] = np.nan

    signals[
        "research_reference_price"
    ] = np.nan

    daily_groups = {}

    for (
        symbol,
        group,
    ) in daily.groupby(
        "symbol",
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=[
                    "_research_session"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        session_to_position = {
            session: i
            for (
                i,
                session,
            ) in enumerate(
                group[
                    "_research_session"
                ].tolist()
            )
        }

        daily_groups[
            symbol
        ] = (
            group,
            session_to_position,
        )

    for idx, observation in signals.iterrows():

        symbol = str(
            observation.get(
                "symbol",
                "",
            )
        ).upper()

        session = observation.get(
            "_research_session"
        )

        if symbol not in daily_groups:
            continue

        group, position_map = daily_groups[
            symbol
        ]

        if session not in position_map:
            continue

        position = position_map[
            session
        ]

        current_bar = group.iloc[
            position
        ]

        reference_price = observation.get(
            "reference_price"
        )

        try:
            reference_price = float(
                reference_price
            )
        except Exception:
            reference_price = np.nan

        if (
            pd.isna(
                reference_price
            )
            or reference_price
            <= 0
        ):

            try:
                reference_price = float(
                    current_bar[
                        "close"
                    ]
                )
            except Exception:
                reference_price = np.nan

        if (
            pd.isna(
                reference_price
            )
            or reference_price
            <= 0
        ):
            continue

        signals.at[
            idx,
            "research_reference_price",
        ] = reference_price

        for horizon in horizons:

            future_position = (
                position
                + horizon
            )

            if future_position >= len(
                group
            ):
                continue

            future_close = group.iloc[
                future_position
            ][
                "close"
            ]

            signals.at[
                idx,
                f"forward_{horizon}d_pct",
            ] = _pct(
                future_close,
                reference_price,
            )

        path_end = min(
            position
            + int(
                max_path_sessions
            ),
            len(
                group
            )
            - 1,
        )

        if path_end <= position:
            continue

        path = group.iloc[
            position
            + 1:
            path_end
            + 1
        ]

        if path.empty:
            continue

        if "high" in path.columns:

            future_high = pd.to_numeric(
                path[
                    "high"
                ],
                errors="coerce",
            ).max()

        else:

            future_high = pd.to_numeric(
                path[
                    "close"
                ],
                errors="coerce",
            ).max()

        if "low" in path.columns:

            future_low = pd.to_numeric(
                path[
                    "low"
                ],
                errors="coerce",
            ).min()

        else:

            future_low = pd.to_numeric(
                path[
                    "close"
                ],
                errors="coerce",
            ).min()

        signals.at[
            idx,
            "forward_20d_mfe_pct",
        ] = _pct(
            future_high,
            reference_price,
        )

        signals.at[
            idx,
            "forward_20d_mae_pct",
        ] = _pct(
            future_low,
            reference_price,
        )

    return signals.drop(
        columns=[
            "_research_session"
        ],
        errors="ignore",
    )


# ============================================================
# GATE CONSTRUCTION
# ============================================================

def build_gate_matrix(
    signal_log: pd.DataFrame,
    swing_threshold: float = PRODUCTION_SWING_THRESHOLD,
    intraday_threshold: float = PRODUCTION_INTRADAY_THRESHOLD,
    entry_quality_threshold: float = PRODUCTION_ENTRY_QUALITY,
    reward_risk_threshold: float = PRODUCTION_REWARD_RISK,
    market_score_threshold: float = PRODUCTION_MARKET_SCORE,
    leadership_threshold: float = PRODUCTION_LEADERSHIP,
    max_distribution_days: int = PRODUCTION_MAX_DISTRIBUTION_DAYS,
    require_production_intraday_label: bool = True,
) -> pd.DataFrame:
    """
    Reconstruct a transparent boolean matrix of production BUY gates.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    index = df.index

    swing_score = _numeric(
        df.get(
            "swing_score",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    intraday_score = _numeric(
        df.get(
            "intraday_score",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    entry_quality = _numeric(
        df.get(
            "entry_quality",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    reward_risk = _numeric(
        df.get(
            "reward_risk",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    market_score = _numeric(
        df.get(
            "market_score",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    leadership = _numeric(
        df.get(
            "leadership_percentile",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    distribution_days = _numeric(
        df.get(
            "distribution_days",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    risk_flag = _safe_bool_series(
        df.get(
            "risk_flag"
        ),
        index,
        default=False,
    )

    trend_health = _safe_bool_series(
        df.get(
            "trend_health"
        ),
        index,
        default=False,
    )

    reference_price = _numeric(
        df.get(
            "reference_price",
            df.get(
                "price",
                pd.Series(
                    np.nan,
                    index=index,
                ),
            ),
        )
    )

    entry_low = _numeric(
        df.get(
            "entry_low",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    entry_high = _numeric(
        df.get(
            "entry_high",
            pd.Series(
                np.nan,
                index=index,
            ),
        )
    )

    inside_entry_zone = (
        reference_price.notna()
        & entry_low.notna()
        & entry_high.notna()
        & (
            reference_price
            >= entry_low
        )
        & (
            reference_price
            <= entry_high
        )
    )

    daily_signal = (
        df.get(
            "daily_signal",
            df.get(
                "signal",
                pd.Series(
                    "",
                    index=index,
                ),
            ),
        )
        .astype(
            str
        )
        .str
        .upper()
    )

    not_too_extended = (
        daily_signal
        != "TOO EXTENDED"
    )

    intraday_signal = (
        df.get(
            "intraday_signal",
            pd.Series(
                "",
                index=index,
            ),
        )
        .astype(
            str
        )
        .str
        .upper()
    )

    gates = pd.DataFrame(
        index=index
    )

    gates[
        "risk_event_clear"
    ] = ~risk_flag

    gates[
        "not_too_extended"
    ] = not_too_extended

    gates[
        "swing_score"
    ] = (
        swing_score
        >= float(
            swing_threshold
        )
    )

    gates[
        "entry_quality"
    ] = (
        entry_quality
        >= float(
            entry_quality_threshold
        )
    )

    gates[
        "reward_risk"
    ] = (
        reward_risk
        >= float(
            reward_risk_threshold
        )
    )

    gates[
        "market_regime"
    ] = (
        market_score
        >= float(
            market_score_threshold
        )
    )

    gates[
        "inside_entry_zone"
    ] = inside_entry_zone

    gates[
        "trend_health"
    ] = trend_health

    gates[
        "distribution"
    ] = (
        distribution_days
        <= float(
            max_distribution_days
        )
    )

    gates[
        "leadership"
    ] = (
        leadership
        >= float(
            leadership_threshold
        )
    )

    if require_production_intraday_label:

        gates[
            "intraday_signal"
        ] = (
            intraday_signal
            == "BUY"
        )

    gates[
        "intraday_score"
    ] = (
        intraday_score
        >= float(
            intraday_threshold
        )
    )

    return gates.fillna(
        False
    ).astype(
        bool
    )


# ============================================================
# GATE LABELS
# ============================================================

GATE_LABELS = {
    "risk_event_clear": "No active catalyst-risk flag",
    "not_too_extended": "Not too extended",
    "swing_score": "Swing Score threshold",
    "entry_quality": "Entry Quality threshold",
    "reward_risk": "Reward/Risk threshold",
    "market_regime": "Market Score threshold",
    "inside_entry_zone": "Inside preferred entry zone",
    "trend_health": "Trend health",
    "distribution": "Distribution-day limit",
    "leadership": "Leadership threshold",
    "intraday_signal": "Production intraday BUY label",
    "intraday_score": "Intraday Score threshold",
}


# ============================================================
# BOTTLENECK ANALYSIS
# ============================================================

def gate_bottleneck_analysis(
    signal_log: pd.DataFrame,
    gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Determine how strongly each gate restricts candidate production.

    Includes a leave-one-gate-out calculation:

    recovered_if_removed

    = number of observations that would pass every OTHER gate but fail
      this specific gate.

    This is particularly useful for identifying gates that are acting as
    absolute blockers.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    if (
        gates is None
        or gates.empty
    ):
        gates = build_gate_matrix(
            df
        )

    if gates.empty:
        return pd.DataFrame()

    rows = []

    total = len(
        gates
    )

    for gate in gates.columns:

        passed = int(
            gates[
                gate
            ].sum()
        )

        failed = int(
            total
            - passed
        )

        other_columns = [
            column
            for column in gates.columns
            if column
            != gate
        ]

        if other_columns:

            passed_all_others = (
                gates[
                    other_columns
                ]
                .all(
                    axis=1
                )
            )

        else:

            passed_all_others = pd.Series(
                True,
                index=gates.index,
            )

        recovered_if_removed = int(
            (
                passed_all_others
                & ~gates[
                    gate
                ]
            )
            .sum()
        )

        rows.append(
            {
                "gate": gate,
                "gate_label": GATE_LABELS.get(
                    gate,
                    gate,
                ),
                "passed": passed,
                "failed": failed,
                "pass_rate_pct": (
                    passed
                    / max(
                        total,
                        1,
                    )
                    * 100.0
                ),
                "failure_rate_pct": (
                    failed
                    / max(
                        total,
                        1,
                    )
                    * 100.0
                ),
                "recovered_if_removed": recovered_if_removed,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    return (
        result
        .sort_values(
            [
                "recovered_if_removed",
                "failed",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# GATE + FORWARD RETURN ANALYSIS
# ============================================================

def gate_forward_return_analysis(
    signal_log: pd.DataFrame,
    gates: pd.DataFrame | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Compare historical forward returns for observations that passed
    versus failed each gate.

    This is association research only.

    It does NOT prove the gate caused the difference.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    if (
        gates is None
        or gates.empty
    ):
        gates = build_gate_matrix(
            df
        )

    if gates.empty:
        return pd.DataFrame()

    rows = []

    horizons = sorted(
        {
            int(
                x
            )
            for x in horizons
            if int(
                x
            )
            > 0
        }
    )

    for gate in gates.columns:

        passed_mask = gates[
            gate
        ]

        failed_mask = ~passed_mask

        base = {
            "gate": gate,
            "gate_label": GATE_LABELS.get(
                gate,
                gate,
            ),
            "passed_n": int(
                passed_mask.sum()
            ),
            "failed_n": int(
                failed_mask.sum()
            ),
        }

        for horizon in horizons:

            column = (
                f"forward_{horizon}d_pct"
            )

            if column not in df.columns:
                continue

            passed_returns = pd.to_numeric(
                df.loc[
                    passed_mask,
                    column,
                ],
                errors="coerce",
            ).dropna()

            failed_returns = pd.to_numeric(
                df.loc[
                    failed_mask,
                    column,
                ],
                errors="coerce",
            ).dropna()

            passed_mean = _mean(
                passed_returns
            )

            failed_mean = _mean(
                failed_returns
            )

            passed_median = _median(
                passed_returns
            )

            failed_median = _median(
                failed_returns
            )

            passed_win = _win_rate(
                passed_returns
            )

            failed_win = _win_rate(
                failed_returns
            )

            base[
                f"{horizon}d_pass_n"
            ] = len(
                passed_returns
            )

            base[
                f"{horizon}d_fail_n"
            ] = len(
                failed_returns
            )

            base[
                f"{horizon}d_pass_mean_pct"
            ] = passed_mean

            base[
                f"{horizon}d_fail_mean_pct"
            ] = failed_mean

            base[
                f"{horizon}d_mean_edge_pct"
            ] = (
                passed_mean
                - failed_mean
                if (
                    pd.notna(
                        passed_mean
                    )
                    and pd.notna(
                        failed_mean
                    )
                )
                else np.nan
            )

            base[
                f"{horizon}d_pass_median_pct"
            ] = passed_median

            base[
                f"{horizon}d_fail_median_pct"
            ] = failed_median

            base[
                f"{horizon}d_pass_win_rate_pct"
            ] = passed_win

            base[
                f"{horizon}d_fail_win_rate_pct"
            ] = failed_win

            base[
                f"{horizon}d_win_rate_edge_pct"
            ] = (
                passed_win
                - failed_win
                if (
                    pd.notna(
                        passed_win
                    )
                    and pd.notna(
                        failed_win
                    )
                )
                else np.nan
            )

        if "forward_20d_mfe_pct" in df.columns:

            base[
                "20d_pass_mfe_pct"
            ] = _mean(
                df.loc[
                    passed_mask,
                    "forward_20d_mfe_pct",
                ]
            )

            base[
                "20d_fail_mfe_pct"
            ] = _mean(
                df.loc[
                    failed_mask,
                    "forward_20d_mfe_pct",
                ]
            )

        if "forward_20d_mae_pct" in df.columns:

            base[
                "20d_pass_mae_pct"
            ] = _mean(
                df.loc[
                    passed_mask,
                    "forward_20d_mae_pct",
                ]
            )

            base[
                "20d_fail_mae_pct"
            ] = _mean(
                df.loc[
                    failed_mask,
                    "forward_20d_mae_pct",
                ]
            )

        rows.append(
            base
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    sort_column = (
        "10d_mean_edge_pct"
        if "10d_mean_edge_pct"
        in result.columns
        else None
    )

    if sort_column:

        result = result.sort_values(
            sort_column,
            ascending=False,
            na_position="last",
        )

    return result.reset_index(
        drop=True
    )


# ============================================================
# THRESHOLD SWEEP
# ============================================================

@dataclass(frozen=True)
class ThresholdSweep:
    swing_values: tuple = (
        70.0,
        72.5,
        75.0,
        77.5,
        80.0,
        82.5,
        85.0,
        87.5,
        90.0,
    )

    intraday_values: tuple = (
        40.0,
        50.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
    )


def score_threshold_sweep(
    signal_log: pd.DataFrame,
    sweep: ThresholdSweep | None = None,
) -> pd.DataFrame:
    """
    Descriptive two-variable Swing Score / Intraday Score study.

    IMPORTANT:
    This deliberately does NOT optimize every gate simultaneously.

    It asks a simpler question:

    Among historical observations meeting a given Swing Score and
    Intraday Score combination, what happened afterward?

    This helps determine whether 85/85 is empirically special or merely
    restrictive.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    if sweep is None:
        sweep = ThresholdSweep()

    swing = _numeric(
        df.get(
            "swing_score",
            pd.Series(
                np.nan,
                index=df.index,
            ),
        )
    )

    intraday = _numeric(
        df.get(
            "intraday_score",
            pd.Series(
                np.nan,
                index=df.index,
            ),
        )
    )

    rows = []

    for swing_threshold in sweep.swing_values:

        for intraday_threshold in sweep.intraday_values:

            mask = (
                (
                    swing
                    >= float(
                        swing_threshold
                    )
                )
                & (
                    intraday
                    >= float(
                        intraday_threshold
                    )
                )
            )

            count = int(
                mask.sum()
            )

            row = {
                "swing_threshold": float(
                    swing_threshold
                ),
                "intraday_threshold": float(
                    intraday_threshold
                ),
                "observations": count,
                "observation_rate_pct": (
                    count
                    / max(
                        len(
                            df
                        ),
                        1,
                    )
                    * 100.0
                ),
            }

            for horizon in DEFAULT_HORIZONS:

                column = (
                    f"forward_{horizon}d_pct"
                )

                if column not in df.columns:
                    continue

                returns = pd.to_numeric(
                    df.loc[
                        mask,
                        column,
                    ],
                    errors="coerce",
                ).dropna()

                row[
                    f"{horizon}d_n"
                ] = len(
                    returns
                )

                row[
                    f"{horizon}d_mean_pct"
                ] = _mean(
                    returns
                )

                row[
                    f"{horizon}d_median_pct"
                ] = _median(
                    returns
                )

                row[
                    f"{horizon}d_win_rate_pct"
                ] = _win_rate(
                    returns
                )

            if "forward_20d_mfe_pct" in df.columns:

                row[
                    "20d_mfe_pct"
                ] = _mean(
                    df.loc[
                        mask,
                        "forward_20d_mfe_pct",
                    ]
                )

            if "forward_20d_mae_pct" in df.columns:

                row[
                    "20d_mae_pct"
                ] = _mean(
                    df.loc[
                        mask,
                        "forward_20d_mae_pct",
                    ]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SCORE BUCKET ANALYSIS
# ============================================================

def score_bucket_analysis(
    signal_log: pd.DataFrame,
) -> dict:
    """
    Group Swing Score and Intraday Score into broad buckets to determine
    whether higher scores correspond to progressively better forward results.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return {
            "swing": pd.DataFrame(),
            "intraday": pd.DataFrame(),
        }

    output = {}

    configurations = {
        "swing": (
            "swing_score",
            [
                -np.inf,
                50,
                60,
                70,
                75,
                80,
                85,
                np.inf,
            ],
            [
                "<50",
                "50-59",
                "60-69",
                "70-74",
                "75-79",
                "80-84",
                "85+",
            ],
        ),
        "intraday": (
            "intraday_score",
            [
                -np.inf,
                20,
                40,
                60,
                70,
                80,
                85,
                np.inf,
            ],
            [
                "<20",
                "20-39",
                "40-59",
                "60-69",
                "70-79",
                "80-84",
                "85+",
            ],
        ),
    }

    for (
        name,
        (
            score_column,
            bins,
            labels,
        ),
    ) in configurations.items():

        if score_column not in df.columns:

            output[
                name
            ] = pd.DataFrame()

            continue

        working = df.copy()

        scores = pd.to_numeric(
            working[
                score_column
            ],
            errors="coerce",
        )

        working[
            "_bucket"
        ] = pd.cut(
            scores,
            bins=bins,
            labels=labels,
            right=False,
        )

        rows = []

        for bucket in labels:

            subset = working[
                working[
                    "_bucket"
                ].astype(
                    str
                )
                == bucket
            ]

            if subset.empty:
                continue

            row = {
                "bucket": bucket,
                "observations": len(
                    subset
                ),
            }

            for horizon in DEFAULT_HORIZONS:

                column = (
                    f"forward_{horizon}d_pct"
                )

                if column not in subset.columns:
                    continue

                values = pd.to_numeric(
                    subset[
                        column
                    ],
                    errors="coerce",
                ).dropna()

                row[
                    f"{horizon}d_n"
                ] = len(
                    values
                )

                row[
                    f"{horizon}d_mean_pct"
                ] = _mean(
                    values
                )

                row[
                    f"{horizon}d_median_pct"
                ] = _median(
                    values
                )

                row[
                    f"{horizon}d_win_rate_pct"
                ] = _win_rate(
                    values
                )

            rows.append(
                row
            )

        output[
            name
        ] = pd.DataFrame(
            rows
        )

    return output


# ============================================================
# ALL-GATE SURVIVORS
# ============================================================

def all_gate_survivors(
    signal_log: pd.DataFrame,
    gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return historical observations passing every reconstructed gate.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    if (
        gates is None
        or gates.empty
    ):

        gates = build_gate_matrix(
            df
        )

    if gates.empty:
        return pd.DataFrame()

    mask = gates.all(
        axis=1
    )

    return df.loc[
        mask
    ].copy()


# ============================================================
# LEAVE-ONE-GATE-OUT CANDIDATES
# ============================================================

def leave_one_gate_out_candidates(
    signal_log: pd.DataFrame,
    gates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Show observations that fail exactly one gate while passing every
    other reconstructed production gate.

    These are the cleanest observations for researching whether a single
    gate may be unnecessarily restrictive.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:
        return pd.DataFrame()

    if (
        gates is None
        or gates.empty
    ):
        gates = build_gate_matrix(
            df
        )

    if gates.empty:
        return pd.DataFrame()

    failure_count = (
        ~gates
    ).sum(
        axis=1
    )

    mask = (
        failure_count
        == 1
    )

    if not mask.any():
        return pd.DataFrame()

    selected = df.loc[
        mask
    ].copy()

    selected_gates = gates.loc[
        mask
    ]

    failed_gate = []

    for idx in selected_gates.index:

        failed = selected_gates.columns[
            ~selected_gates.loc[
                idx
            ]
        ].tolist()

        failed_gate.append(
            failed[
                0
            ]
            if failed
            else ""
        )

    selected[
        "single_failed_gate"
    ] = failed_gate

    selected[
        "single_failed_gate_label"
    ] = selected[
        "single_failed_gate"
    ].map(
        GATE_LABELS
    )

    return selected


# ============================================================
# RESEARCH SUMMARY
# ============================================================

def _research_summary(
    signal_log: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    gate_returns: pd.DataFrame,
    single_gate_misses: pd.DataFrame,
) -> dict:

    summary = {
        "observations": len(
            signal_log
        ),
        "forward_returns_available": False,
        "largest_bottleneck": None,
        "largest_bottleneck_label": None,
        "largest_recovered_if_removed": 0,
        "single_gate_near_misses": len(
            single_gate_misses
        ),
    }

    forward_columns = [
        column
        for column in signal_log.columns
        if column.startswith(
            "forward_"
        )
        and column.endswith(
            "_pct"
        )
    ]

    if forward_columns:

        available = (
            signal_log[
                forward_columns
            ]
            .notna()
            .any(
                axis=1
            )
            .sum()
        )

        summary[
            "forward_returns_available"
        ] = bool(
            available
            > 0
        )

        summary[
            "forward_observations"
        ] = int(
            available
        )

    if not bottlenecks.empty:

        top = bottlenecks.iloc[
            0
        ]

        summary[
            "largest_bottleneck"
        ] = top.get(
            "gate"
        )

        summary[
            "largest_bottleneck_label"
        ] = top.get(
            "gate_label"
        )

        summary[
            "largest_recovered_if_removed"
        ] = int(
            top.get(
                "recovered_if_removed",
                0,
            )
        )

    if (
        not gate_returns.empty
        and "10d_mean_edge_pct"
        in gate_returns.columns
    ):

        valid = gate_returns.dropna(
            subset=[
                "10d_mean_edge_pct"
            ]
        )

        if not valid.empty:

            best = valid.sort_values(
                "10d_mean_edge_pct",
                ascending=False,
            ).iloc[
                0
            ]

            worst = valid.sort_values(
                "10d_mean_edge_pct",
                ascending=True,
            ).iloc[
                0
            ]

            summary[
                "best_10d_gate"
            ] = best.get(
                "gate"
            )

            summary[
                "best_10d_gate_label"
            ] = best.get(
                "gate_label"
            )

            summary[
                "best_10d_edge_pct"
            ] = float(
                best.get(
                    "10d_mean_edge_pct"
                )
            )

            summary[
                "weakest_10d_gate"
            ] = worst.get(
                "gate"
            )

            summary[
                "weakest_10d_gate_label"
            ] = worst.get(
                "gate_label"
            )

            summary[
                "weakest_10d_edge_pct"
            ] = float(
                worst.get(
                    "10d_mean_edge_pct"
                )
            )

    return summary


# ============================================================
# MAIN ENGINE
# ============================================================

def run_forward_gate_research(
    signal_log: pd.DataFrame,
) -> dict:
    """
    Run the complete v3.7 Gate Bottleneck + Forward Return Analyzer.
    """

    df = _safe_df(
        signal_log
    )

    if df.empty:

        return {
            "status": "NO_DATA",
            "version": RESEARCH_VERSION,
            "message": (
                "No historical signal observations were supplied."
            ),
        }

    gates = build_gate_matrix(
        df
    )

    bottlenecks = gate_bottleneck_analysis(
        df,
        gates,
    )

    gate_returns = gate_forward_return_analysis(
        df,
        gates,
    )

    threshold_sweep = score_threshold_sweep(
        df
    )

    score_buckets = score_bucket_analysis(
        df
    )

    production_survivors = all_gate_survivors(
        df,
        gates,
    )

    single_gate_misses = (
        leave_one_gate_out_candidates(
            df,
            gates,
        )
    )

    summary = _research_summary(
        df,
        bottlenecks,
        gate_returns,
        single_gate_misses,
    )

    forward_available = bool(
        summary.get(
            "forward_returns_available",
            False,
        )
    )

    if forward_available:

        message = (
            f"v3.7 research completed across {len(df):,} historical "
            "scanner observations with forward-return data."
        )

    else:

        message = (
            f"v3.7 gate research completed across {len(df):,} observations, "
            "but forward returns have not yet been attached. "
            "The app must call attach_forward_returns() after the backtest."
        )

    return {
        "status": "COMPLETE",
        "version": RESEARCH_VERSION,
        "message": message,
        "summary": summary,
        "gate_matrix": gates,
        "bottlenecks": bottlenecks,
        "gate_forward_returns": gate_returns,
        "threshold_sweep": threshold_sweep,
        "swing_score_buckets": score_buckets.get(
            "swing",
            pd.DataFrame(),
        ),
        "intraday_score_buckets": score_buckets.get(
            "intraday",
            pd.DataFrame(),
        ),
        "production_survivors": production_survivors,
        "single_gate_near_misses": single_gate_misses,
    }
