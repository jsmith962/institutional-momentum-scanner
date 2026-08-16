"""Production-equivalent historical simulator for the v3.2.2 swing scanner.

v3.2.2 adds a research-only calibration engine.

IMPORTANT:
- The normal swing_backtest() still defaults to the live/production rules.
- Calibration profiles do NOT automatically alter the live scanner.
- Alternate thresholds are tested historically and evaluated chronologically.
- A configuration should not be promoted to production simply because it
  produces more trades or a higher in-sample return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategy import (
    MAX_DISTRIBUTION_DAYS_BUY,
    MIN_ENTRY_QUALITY_BUY,
    MIN_INTRADAY_CONFIRMATION_SCORE,
    MIN_MARKET_SCORE_BUY,
    MIN_REWARD_RISK_BUY,
    MIN_RS_PERCENTILE_BUY,
    MIN_SWING_SCORE_BUY,
    classify,
    combine_daily_intraday_signal,
    prepare_intraday,
    relative_strength_percentiles,
    score_swing_daily,
)

from validation import chronological_validation


ET = ZoneInfo("America/New_York")

BUY_SIGNALS = {
    "BUY",
    "A+ SWING BUY",
}


# ============================================================
# PRODUCTION CONFIGURATION
# ============================================================

PRODUCTION_GATE_CONFIG = {
    "swing_score": float(MIN_SWING_SCORE_BUY),
    "entry_quality": float(MIN_ENTRY_QUALITY_BUY),
    "reward_risk": float(MIN_REWARD_RISK_BUY),
    "market_score": float(MIN_MARKET_SCORE_BUY),
    "leadership_percentile": float(MIN_RS_PERCENTILE_BUY * 100),
    "max_distribution_days": int(MAX_DISTRIBUTION_DAYS_BUY),
    "intraday_score": float(MIN_INTRADAY_CONFIRMATION_SCORE),
}


# ============================================================
# RESEARCH CALIBRATION PROFILES
# ============================================================

CALIBRATION_PROFILES = [
    {
        "name": "PRODUCTION_85_85",
        **PRODUCTION_GATE_CONFIG,
    },
    {
        "name": "SCORE_82_5",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 82.5,
    },
    {
        "name": "SCORE_80",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 80.0,
    },
    {
        "name": "SCORE_77_5",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 77.5,
    },
    {
        "name": "SCORE_75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 75.0,
    },
    {
        "name": "82_5_INTRADAY_80",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 82.5,
        "intraday_score": 80.0,
    },
    {
        "name": "80_INTRADAY_80",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 80.0,
        "intraday_score": 80.0,
    },
    {
        "name": "80_INTRADAY_75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 80.0,
        "intraday_score": 75.0,
    },
    {
        "name": "80_IQ12_INTRADAY75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 80.0,
        "entry_quality": 12.0,
        "intraday_score": 75.0,
    },
    {
        "name": "77_5_IQ12_INTRADAY75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 77.5,
        "entry_quality": 12.0,
        "intraday_score": 75.0,
    },
    {
        "name": "80_LEADER65_INTRADAY75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 80.0,
        "leadership_percentile": 65.0,
        "intraday_score": 75.0,
    },
    {
        "name": "77_5_LEADER65_INTRADAY75",
        **PRODUCTION_GATE_CONFIG,
        "swing_score": 77.5,
        "leadership_percentile": 65.0,
        "intraday_score": 75.0,
    },
]


# ============================================================
# TRADE STRUCTURE
# ============================================================

TRADE_COLUMNS = [
    "symbol",
    "signal_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "signal",
    "setup",
    "swing_score",
    "intraday_score",
    "entry",
    "initial_stop",
    "target1",
    "target2",
    "shares",
    "pnl",
    "return_pct",
    "r_multiple",
    "holding_days",
    "exit_reason",
]


@dataclass
class Position:
    symbol: str
    signal_date: object
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    signal: str
    setup: str
    swing_score: float
    intraday_score: float
    entry: float
    initial_stop: float
    stop: float
    target1: float
    target2: float
    shares: int
    remaining: int
    entry_fee: float
    realized_pnl: float = 0.0
    target1_hit: bool = False
    holding_days: int = 0
    last_session: object | None = None
    trend_exit_pending: bool = False


# ============================================================
# BASIC HELPERS
# ============================================================

def _number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalise_gate_config(gate_config=None):
    config = PRODUCTION_GATE_CONFIG.copy()

    if gate_config:
        for key in config:
            if key in gate_config:
                config[key] = gate_config[key]

    config["swing_score"] = float(config["swing_score"])
    config["entry_quality"] = float(config["entry_quality"])
    config["reward_risk"] = float(config["reward_risk"])
    config["market_score"] = float(config["market_score"])
    config["leadership_percentile"] = float(
        config["leadership_percentile"]
    )
    config["max_distribution_days"] = int(
        config["max_distribution_days"]
    )
    config["intraday_score"] = float(config["intraday_score"])

    return config


def _gate_labels(config):
    return {
        "risk_event_clear": "No active catalyst-gap risk",
        "not_too_extended": "Not too extended",
        "swing_score": (
            f"Swing Score at least {config['swing_score']:g}"
        ),
        "entry_quality": (
            f"Entry Quality at least "
            f"{config['entry_quality']:g}/15"
        ),
        "reward_risk": (
            f"Reward/risk at least "
            f"{config['reward_risk']:.1f}:1"
        ),
        "market_regime": (
            f"Market Score at least "
            f"{config['market_score']:g}/10"
        ),
        "inside_entry_zone": (
            "Price inside preferred entry zone"
        ),
        "trend_health": (
            "20EMA and 50SMA trends rising"
        ),
        "distribution": (
            "Distribution days no more than "
            f"{config['max_distribution_days']}"
        ),
        "leadership": (
            "Market leadership at least "
            f"{config['leadership_percentile']:g}th percentile"
        ),
        "intraday_signal": (
            "Intraday signal is BUY"
        ),
        "intraday_score": (
            "Intraday Score at least "
            f"{config['intraday_score']:g}"
        ),
    }


def _empty_result(
    starting_capital: float,
    warnings=None,
    gate_config=None,
):
    config = _normalise_gate_config(
        gate_config
    )

    return {
        "trades": pd.DataFrame(
            columns=TRADE_COLUMNS
        ),
        "equity": pd.DataFrame(
            columns=[
                "date",
                "equity",
                "cash",
                "open_positions",
            ]
        ),
        "stats": {
            "starting_capital": round(
                float(starting_capital),
                2,
            ),
            "ending_capital": round(
                float(starting_capital),
                2,
            ),
            "total_return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "avg_trade_dollars": 0.0,
            "max_drawdown_pct": 0.0,
        },
        "signal_log": pd.DataFrame(),
        "diagnostics": {
            "funnel": pd.DataFrame(),
            "gate_failures": pd.DataFrame(),
            "near_misses": pd.DataFrame(),
            "score_distribution": pd.DataFrame(),
        },
        "validation": chronological_validation(
            pd.DataFrame(
                columns=TRADE_COLUMNS
            )
        ),
        "gate_config": config,
        "warnings": list(
            warnings or []
        ),
    }


# ============================================================
# GATE DIAGNOSTICS
# ============================================================

def _candidate_gate_diagnostics(
    swing,
    intraday_signal,
    intraday_score,
    gate_config=None,
):
    """Evaluate BUY gates without changing the underlying score."""

    config = _normalise_gate_config(
        gate_config
    )

    leadership = swing.get(
        "leadership_percentile"
    )

    leadership_pass = (
        leadership is None
        or pd.isna(leadership)
        or _number(leadership)
        >= config["leadership_percentile"]
    )

    daily_gates = {
        "risk_event_clear": (
            not bool(
                swing.get(
                    "risk_flag",
                    False,
                )
            )
        ),
        "not_too_extended": (
            not bool(
                swing.get(
                    "too_extended",
                    False,
                )
            )
        ),
        "swing_score": (
            _number(
                swing.get(
                    "swing_score"
                )
            )
            >= config["swing_score"]
        ),
        "entry_quality": (
            _number(
                swing.get(
                    "entry_quality"
                )
            )
            >= config["entry_quality"]
        ),
        "reward_risk": (
            _number(
                swing.get(
                    "reward_risk"
                )
            )
            >= config["reward_risk"]
        ),
        "market_regime": (
            _number(
                swing.get(
                    "market_score"
                )
            )
            >= config["market_score"]
        ),
        "inside_entry_zone": bool(
            swing.get(
                "inside_entry_zone",
                False,
            )
        ),
        "trend_health": bool(
            swing.get(
                "trend_health",
                False,
            )
        ),
        "distribution": (
            _number(
                swing.get(
                    "distribution_days"
                ),
                float("inf"),
            )
            <= config[
                "max_distribution_days"
            ]
        ),
        "leadership": bool(
            leadership_pass
        ),
    }

    intraday_gates = {
        "intraday_signal": (
            intraday_signal == "BUY"
        ),
        "intraday_score": (
            _number(
                intraday_score
            )
            >= config[
                "intraday_score"
            ]
        ),
    }

    all_gates = {
        **daily_gates,
        **intraday_gates,
    }

    labels = _gate_labels(
        config
    )

    daily_failed = [
        labels[key]
        for key, passed
        in daily_gates.items()
        if not passed
    ]

    intraday_failed = [
        labels[key]
        for key, passed
        in intraday_gates.items()
        if not passed
    ]

    failed = (
        daily_failed
        + intraday_failed
    )

    diagnostics = {
        "daily_all_gates_passed": bool(
            all(
                daily_gates.values()
            )
        ),
        "intraday_all_gates_passed": bool(
            all(
                intraday_gates.values()
            )
        ),
        "all_buy_gates_passed": bool(
            all(
                all_gates.values()
            )
        ),
        "daily_gates_passed": int(
            sum(
                daily_gates.values()
            )
        ),
        "daily_gate_count": len(
            daily_gates
        ),
        "buy_gates_passed": int(
            sum(
                all_gates.values()
            )
        ),
        "buy_gate_count": len(
            all_gates
        ),
        "failed_gate_count": len(
            failed
        ),
        "daily_failed_gates": (
            "; ".join(
                daily_failed
            )
            or "None"
        ),
        "intraday_failed_gates": (
            "; ".join(
                intraday_failed
            )
            or "None"
        ),
        "failed_buy_gates": (
            "; ".join(
                failed
            )
            or "None"
        ),
    }

    diagnostics.update(
        {
            f"gate_{key}": bool(
                passed
            )
            for key, passed
            in all_gates.items()
        }
    )

    return diagnostics


def _research_signal(
    swing,
    intraday_signal,
    intraday_score,
    gate_config,
):
    """Create a research-only signal using alternate gate thresholds."""

    diagnostics = (
        _candidate_gate_diagnostics(
            swing,
            intraday_signal,
            intraday_score,
            gate_config,
        )
    )

    if diagnostics[
        "all_buy_gates_passed"
    ]:

        score = _number(
            swing.get(
                "swing_score"
            )
        )

        entry_quality = _number(
            swing.get(
                "entry_quality"
            )
        )

        config = _normalise_gate_config(
            gate_config
        )

        a_plus_threshold = max(
            config["swing_score"]
            + 5,
            87.5,
        )

        if (
            score >= a_plus_threshold
            and entry_quality >= 13
        ):
            return (
                "A+ SWING BUY",
                "All research BUY gates passed at A+ strength.",
                diagnostics,
            )

        return (
            "BUY",
            "All research BUY gates passed.",
            diagnostics,
        )

    if bool(
        swing.get(
            "risk_flag",
            False,
        )
    ):
        return (
            "AVOID",
            str(
                swing.get(
                    "risk_reason",
                    "Catalyst-risk gate active.",
                )
            ),
            diagnostics,
        )

    if bool(
        swing.get(
            "too_extended",
            False,
        )
    ):
        return (
            "TOO EXTENDED",
            "Setup is too extended for the configured entry rules.",
            diagnostics,
        )

    if diagnostics[
        "daily_all_gates_passed"
    ]:
        return (
            "WATCH",
            "Daily BUY gates passed; waiting for intraday confirmation.",
            diagnostics,
        )

    return (
        "WATCH"
        if swing.get(
            "signal"
        ) == "WATCH"
        else "AVOID",
        "The configured BUY gates did not all pass.",
        diagnostics,
    )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

def _score_distribution(
    signal_log,
):
    if (
        signal_log is None
        or signal_log.empty
        or "swing_score"
        not in signal_log
    ):
        return pd.DataFrame()

    scores = pd.to_numeric(
        signal_log[
            "swing_score"
        ],
        errors="coerce",
    ).dropna()

    if scores.empty:
        return pd.DataFrame()

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

    for percentile in percentiles:
        rows.append(
            {
                "percentile": (
                    f"{percentile * 100:.0f}%"
                ),
                "swing_score": round(
                    float(
                        scores.quantile(
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


def _build_signal_diagnostics(
    signal_log,
    trade_count=0,
    gate_config=None,
):
    if (
        signal_log is None
        or signal_log.empty
    ):
        return {
            "funnel": pd.DataFrame(),
            "gate_failures": pd.DataFrame(),
            "near_misses": pd.DataFrame(),
            "score_distribution": pd.DataFrame(),
        }

    config = _normalise_gate_config(
        gate_config
    )

    labels = _gate_labels(
        config
    )

    frame = signal_log.copy()

    total = len(
        frame
    )

    final_buy = frame[
        "signal"
    ].isin(
        BUY_SIGNALS
    )

    gate_keys = [
        "risk_event_clear",
        "not_too_extended",
        "swing_score",
        "entry_quality",
        "reward_risk",
        "market_regime",
        "inside_entry_zone",
        "trend_health",
        "distribution",
        "leadership",
        "intraday_signal",
        "intraday_score",
    ]

    daily_gate_keys = [
        "risk_event_clear",
        "not_too_extended",
        "swing_score",
        "entry_quality",
        "reward_risk",
        "market_regime",
        "inside_entry_zone",
        "trend_health",
        "distribution",
        "leadership",
    ]

    for key in gate_keys:
        column = f"gate_{key}"

        if column not in frame:
            frame[column] = False

    daily_pass = (
        frame[
            [
                f"gate_{key}"
                for key
                in daily_gate_keys
            ]
        ]
        .astype(bool)
        .all(axis=1)
    )

    intraday_signal_pass = (
        frame[
            "gate_intraday_signal"
        ].astype(bool)
    )

    intraday_score_pass = (
        frame[
            "gate_intraday_score"
        ].astype(bool)
    )

    fully_confirmed = (
        daily_pass
        & intraday_signal_pass
        & intraday_score_pass
        & final_buy
    )

    stages = [
        (
            "Candidates evaluated",
            total,
        ),
        (
            f"Daily Swing Score at least "
            f"{config['swing_score']:g}",
            int(
                frame[
                    "gate_swing_score"
                ]
                .astype(bool)
                .sum()
            ),
        ),
        (
            "All daily BUY gates passed",
            int(
                daily_pass.sum()
            ),
        ),
        (
            "Daily BUY plus intraday BUY signal",
            int(
                (
                    daily_pass
                    & intraday_signal_pass
                ).sum()
            ),
        ),
        (
            "Fully confirmed BUY",
            int(
                fully_confirmed.sum()
            ),
        ),
        (
            "Completed simulated trades",
            int(
                trade_count
            ),
        ),
    ]

    funnel = pd.DataFrame(
        [
            {
                "stage": stage,
                "count": count,
                "percent_of_candidates": round(
                    (
                        count
                        / max(
                            total,
                            1,
                        )
                        * 100
                    ),
                    1,
                ),
            }
            for stage, count
            in stages
        ]
    )

    failure_rows = []

    for key in gate_keys:
        column = f"gate_{key}"

        failures = int(
            (
                ~frame[
                    column
                ].astype(bool)
            ).sum()
        )

        failure_rows.append(
            {
                "gate": labels[key],
                "stage": (
                    "Intraday"
                    if key
                    in {
                        "intraday_signal",
                        "intraday_score",
                    }
                    else "Daily"
                ),
                "failed": failures,
                "failure_percent": round(
                    failures
                    / max(
                        total,
                        1,
                    )
                    * 100,
                    1,
                ),
            }
        )

    gate_failures = (
        pd.DataFrame(
            failure_rows
        )
        .sort_values(
            [
                "failed",
                "stage",
                "gate",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    near_misses = frame[
        ~final_buy
    ].copy()

    if not near_misses.empty:

        near_misses[
            "gates_passed"
        ] = (
            near_misses[
                "buy_gates_passed"
            ]
            .astype(int)
            .astype(str)
            + "/"
            + near_misses[
                "buy_gate_count"
            ]
            .astype(int)
            .astype(str)
        )

        near_misses = (
            near_misses
            .sort_values(
                [
                    "failed_gate_count",
                    "swing_score",
                    "intraday_score",
                    "session",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    False,
                ],
            )
            .drop_duplicates(
                "symbol"
            )
            .head(
                10
            )
        )

        near_misses = near_misses[
            [
                "symbol",
                "session",
                "signal",
                "swing_score",
                "intraday_score",
                "entry_quality",
                "gates_passed",
                "failed_buy_gates",
            ]
        ].reset_index(
            drop=True
        )

    return {
        "funnel": funnel,
        "gate_failures": gate_failures,
        "near_misses": near_misses,
        "score_distribution": _score_distribution(
            frame
        ),
    }


# ============================================================
# MARKET DATA NORMALIZATION
# ============================================================

def _normalise_bars(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if (
        frame is None
        or frame.empty
        or not required.issubset(
            frame.columns
        )
    ):
        return pd.DataFrame()

    out = frame.copy()

    out[
        "timestamp"
    ] = pd.to_datetime(
        out[
            "timestamp"
        ],
        utc=True,
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            "timestamp",
            "symbol",
        ]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    out[
        "symbol"
    ] = (
        out[
            "symbol"
        ]
        .astype(str)
        .str.upper()
    )

    out[
        "session"
    ] = (
        out[
            "timestamp"
        ]
        .dt
        .tz_convert(
            ET
        )
        .dt
        .date
    )

    out[
        "clock"
    ] = (
        out[
            "timestamp"
        ]
        .dt
        .tz_convert(
            ET
        )
        .dt
        .time
    )

    return (
        out
        .sort_values(
            [
                "timestamp",
                "symbol",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _normalise_daily(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    out = _normalise_bars(
        frame
    )

    return out.drop(
        columns=[
            "session",
            "clock",
        ],
        errors="ignore",
    )


def _regular_minutes(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.empty:
        return frame

    return frame[
        (
            frame[
                "clock"
            ]
            >= time(
                9,
                30,
            )
        )
        & (
            frame[
                "clock"
            ]
            < time(
                16,
                0,
            )
        )
    ].copy()


def _daily_from_minutes(
    minutes: pd.DataFrame,
) -> pd.DataFrame:

    if minutes.empty:
        return pd.DataFrame()

    regular = minutes[
        (
            minutes[
                "clock"
            ]
            >= time(
                9,
                30,
            )
        )
        & (
            minutes[
                "clock"
            ]
            <= time(
                16,
                0,
            )
        )
    ].copy()

    if regular.empty:
        return pd.DataFrame()

    return (
        regular
        .groupby(
            [
                "symbol",
                "session",
            ],
            sort=True,
        )
        .agg(
            timestamp=(
                "timestamp",
                "last",
            ),
            open=(
                "open",
                "first",
            ),
            high=(
                "high",
                "max",
            ),
            low=(
                "low",
                "min",
            ),
            close=(
                "close",
                "last",
            ),
            volume=(
                "volume",
                "sum",
            ),
        )
        .reset_index(
            drop=False
        )
        .drop(
            columns="session"
        )
        .sort_values(
            [
                "timestamp",
                "symbol",
            ]
        )
    )


def _partial_daily(
    minutes: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:

    if (
        minutes is None
        or minutes.empty
    ):
        return pd.DataFrame()

    ordered = minutes.sort_values(
        "timestamp"
    )

    return pd.DataFrame(
        [
            {
                "timestamp": ordered[
                    "timestamp"
                ].iloc[-1],
                "symbol": symbol,
                "open": float(
                    ordered[
                        "open"
                    ].iloc[0]
                ),
                "high": float(
                    ordered[
                        "high"
                    ].max()
                ),
                "low": float(
                    ordered[
                        "low"
                    ].min()
                ),
                "close": float(
                    ordered[
                        "close"
                    ].iloc[-1]
                ),
                "volume": float(
                    ordered[
                        "volume"
                    ].sum()
                ),
            }
        ]
    )


def _completed_daily(
    daily: pd.DataFrame,
    session_date,
) -> pd.DataFrame:

    if daily.empty:
        return daily

    sessions = (
        pd.to_datetime(
            daily[
                "timestamp"
            ],
            utc=True,
        )
        .dt
        .tz_convert(
            ET
        )
        .dt
        .date
    )

    return daily[
        sessions
        < session_date
    ].copy()


def _session_slice(
    frame,
    session_date,
    through: time | None = None,
):

    out = frame[
        frame[
            "session"
        ]
        == session_date
    ]

    if through is not None:
        out = out[
            out[
                "clock"
            ]
            <= through
        ]

    return out.copy()


# ============================================================
# EXECUTION HELPERS
# ============================================================

def _execution_price(
    raw_price: float,
    side: str,
    slippage_bps: float,
) -> float:

    adjustment = (
        max(
            float(
                slippage_bps
            ),
            0.0,
        )
        / 10_000
    )

    multiplier = (
        1
        + adjustment
        if side == "buy"
        else 1
        - adjustment
    )

    return (
        float(
            raw_price
        )
        * multiplier
    )


def _fee(
    notional: float,
    commission_bps: float,
) -> float:

    return (
        abs(
            float(
                notional
            )
        )
        * max(
            float(
                commission_bps
            ),
            0.0,
        )
        / 10_000
    )


def _account_equity(
    cash,
    positions,
    prices,
):

    return (
        cash
        + sum(
            position.remaining
            * prices.get(
                symbol,
                position.entry,
            )
            for symbol, position
            in positions.items()
        )
    )


def _trade_record(
    position,
    timestamp,
    reason,
    total_pnl,
):

    risk_dollars = max(
        (
            position.entry
            - position.initial_stop
        )
        * position.shares,
        0.01,
    )

    return {
        "symbol": position.symbol,
        "signal_date": (
            position.signal_date
        ),
        "signal_time": (
            position.signal_time
        ),
        "entry_time": (
            position.entry_time
        ),
        "exit_time": timestamp,
        "signal": (
            position.signal
        ),
        "setup": (
            position.setup
        ),
        "swing_score": round(
            position.swing_score,
            1,
        ),
        "intraday_score": round(
            position.intraday_score,
            1,
        ),
        "entry": round(
            position.entry,
            4,
        ),
        "initial_stop": round(
            position.initial_stop,
            4,
        ),
        "target1": round(
            position.target1,
            4,
        ),
        "target2": round(
            position.target2,
            4,
        ),
        "shares": (
            position.shares
        ),
        "pnl": round(
            total_pnl,
            2,
        ),
        "return_pct": round(
            total_pnl
            / max(
                position.entry
                * position.shares,
                0.01,
            )
            * 100,
            2,
        ),
        "r_multiple": round(
            total_pnl
            / risk_dollars,
            3,
        ),
        "holding_days": (
            position.holding_days
        ),
        "exit_reason": reason,
    }


def _close_position(
    position,
    raw_price,
    timestamp,
    reason,
    cash,
    commission_bps,
    slippage_bps,
):

    fill = _execution_price(
        raw_price,
        "sell",
        slippage_bps,
    )

    proceeds = (
        fill
        * position.remaining
    )

    exit_fee = _fee(
        proceeds,
        commission_bps,
    )

    leg_pnl = (
        (
            fill
            - position.entry
        )
        * position.remaining
        - exit_fee
    )

    cash += (
        proceeds
        - exit_fee
    )

    total_pnl = (
        position.realized_pnl
        + leg_pnl
        - position.entry_fee
    )

    return (
        cash,
        _trade_record(
            position,
            timestamp,
            reason,
            total_pnl,
        ),
    )


def _take_target1(
    position,
    raw_price,
    timestamp,
    cash,
    commission_bps,
    slippage_bps,
):

    exit_shares = min(
        max(
            position.shares
            // 2,
            1,
        ),
        position.remaining,
    )

    fill = _execution_price(
        raw_price,
        "sell",
        slippage_bps,
    )

    proceeds = (
        fill
        * exit_shares
    )

    exit_fee = _fee(
        proceeds,
        commission_bps,
    )

    position.realized_pnl += (
        (
            fill
            - position.entry
        )
        * exit_shares
        - exit_fee
    )

    position.remaining -= (
        exit_shares
    )

    cash += (
        proceeds
        - exit_fee
    )

    position.target1_hit = True

    position.stop = (
        position.entry
    )

    if (
        position.remaining
        <= 0
    ):

        total_pnl = (
            position.realized_pnl
            - position.entry_fee
        )

        return (
            cash,
            _trade_record(
                position,
                timestamp,
                "TARGET 1",
                total_pnl,
            ),
        )

    return (
        cash,
        None,
    )


def _manage_bar(
    position,
    bar,
    cash,
    commission_bps,
    slippage_bps,
):

    if (
        position.last_session
        != bar.session
    ):
        position.holding_days += 1
        position.last_session = (
            bar.session
        )

    if (
        float(
            bar.open
        )
        <= position.stop
    ):

        reason = (
            "GAP STOP"
            if float(
                bar.open
            )
            < position.stop
            else "STOP"
        )

        return _close_position(
            position,
            float(
                bar.open
            ),
            bar.timestamp,
            reason,
            cash,
            commission_bps,
            slippage_bps,
        )

    if (
        position.trend_exit_pending
    ):

        return _close_position(
            position,
            float(
                bar.open
            ),
            bar.timestamp,
            "TREND EXIT",
            cash,
            commission_bps,
            slippage_bps,
        )

    if (
        position.target1_hit
        and float(
            bar.open
        )
        >= position.target2
    ):

        return _close_position(
            position,
            float(
                bar.open
            ),
            bar.timestamp,
            "GAP TARGET 2",
            cash,
            commission_bps,
            slippage_bps,
        )

    if (
        not position.target1_hit
        and float(
            bar.open
        )
        >= position.target1
    ):

        cash, closed = _take_target1(
            position,
            float(
                bar.open
            ),
            bar.timestamp,
            cash,
            commission_bps,
            slippage_bps,
        )

        if (
            closed
            is not None
        ):
            return (
                cash,
                closed,
            )

        if (
            float(
                bar.open
            )
            >= position.target2
        ):

            return _close_position(
                position,
                float(
                    bar.open
                ),
                bar.timestamp,
                "GAP TARGET 2",
                cash,
                commission_bps,
                slippage_bps,
            )

    if (
        float(
            bar.low
        )
        <= position.stop
    ):

        reason = (
            "BREAKEVEN STOP"
            if position.target1_hit
            else "STOP"
        )

        return _close_position(
            position,
            position.stop,
            bar.timestamp,
            reason,
            cash,
            commission_bps,
            slippage_bps,
        )

    if (
        not position.target1_hit
        and float(
            bar.high
        )
        >= position.target1
    ):

        cash, closed = _take_target1(
            position,
            position.target1,
            bar.timestamp,
            cash,
            commission_bps,
            slippage_bps,
        )

        if (
            closed
            is not None
        ):
            return (
                cash,
                closed,
            )

        if (
            float(
                bar.low
            )
            <= position.stop
        ):

            return _close_position(
                position,
                position.stop,
                bar.timestamp,
                "TARGET 1 + BREAKEVEN STOP",
                cash,
                commission_bps,
                slippage_bps,
            )

    if (
        position.target1_hit
        and float(
            bar.high
        )
        >= position.target2
    ):

        return _close_position(
            position,
            position.target2,
            bar.timestamp,
            "TARGET 2",
            cash,
            commission_bps,
            slippage_bps,
        )

    return (
        cash,
        None,
    )


# ============================================================
# TREND EXIT
# ============================================================

def _mark_trend_exits(
    positions,
    daily,
    all_today,
    session_date,
):

    if not positions:
        return

    completed = _completed_daily(
        daily,
        session_date,
    )

    for (
        symbol,
        position,
    ) in positions.items():

        today = all_today[
            all_today[
                "symbol"
            ]
            == symbol
        ]

        if today.empty:
            continue

        history = completed[
            completed[
                "symbol"
            ]
            == symbol
        ]

        history = pd.concat(
            [
                history,
                _partial_daily(
                    today,
                    symbol,
                ),
            ],
            ignore_index=True,
        )

        if len(
            history
        ) < 20:
            continue

        close = pd.to_numeric(
            history[
                "close"
            ],
            errors="coerce",
        ).dropna()

        if len(
            close
        ) < 20:
            continue

        e20 = float(
            close
            .ewm(
                span=20,
                adjust=False,
            )
            .mean()
            .iloc[-1]
        )

        position.trend_exit_pending = bool(
            float(
                close.iloc[-1]
            )
            < e20
        )


# ============================================================
# SIGNAL GENERATION
# ============================================================

def _signal_candidates(
    session_date,
    scan_clock,
    bars,
    daily,
    spy_minutes,
    qqq_minutes,
    spy_daily,
    qqq_daily,
    gate_config=None,
    production_mode=True,
):

    config = _normalise_gate_config(
        gate_config
    )

    completed_stocks = (
        _completed_daily(
            daily,
            session_date,
        )
    )

    completed_spy = (
        _completed_daily(
            spy_daily,
            session_date,
        )
    )

    completed_qqq = (
        _completed_daily(
            qqq_daily,
            session_date,
        )
    )

    spy_today = _session_slice(
        spy_minutes,
        session_date,
        scan_clock,
    )

    qqq_today = _session_slice(
        qqq_minutes,
        session_date,
        scan_clock,
    )

    spy_history = pd.concat(
        [
            completed_spy,
            _partial_daily(
                spy_today,
                "SPY",
            ),
        ],
        ignore_index=True,
    )

    qqq_history = pd.concat(
        [
            completed_qqq,
            _partial_daily(
                qqq_today,
                "QQQ",
            ),
        ],
        ignore_index=True,
    )

    rows = []

    stock_histories = {}

    for (
        symbol,
        all_symbol_bars,
    ) in bars.groupby(
        "symbol"
    ):

        today = _session_slice(
            all_symbol_bars,
            session_date,
            scan_clock,
        )

        if len(
            today
        ) < 20:
            continue

        prior_history = (
            completed_stocks[
                completed_stocks[
                    "symbol"
                ]
                == symbol
            ]
            .copy()
        )

        history = pd.concat(
            [
                prior_history,
                _partial_daily(
                    today,
                    symbol,
                ),
            ],
            ignore_index=True,
        )

        if len(
            history
        ) < 61:
            continue

        stock_histories[
            symbol
        ] = (
            history,
            prior_history,
            today,
        )

    if not stock_histories:
        return []

    leadership = (
        relative_strength_percentiles(
            pd.concat(
                [
                    item[0]
                    for item
                    in stock_histories.values()
                ],
                ignore_index=True,
            )
        )
    )

    for (
        symbol,
        (
            history,
            prior_history,
            today,
        ),
    ) in stock_histories.items():

        avg_share_volume = float(
            prior_history[
                "volume"
            ]
            .tail(
                20
            )
            .mean()
        )

        avg_dollar_volume = float(
            (
                prior_history[
                    "close"
                ]
                * prior_history[
                    "volume"
                ]
            )
            .tail(
                20
            )
            .mean()
        )

        prepared = prepare_intraday(
            today,
            spy_today,
            avg_share_volume,
        )

        if prepared.empty:
            continue

        intraday_row = (
            prepared.iloc[-1]
        )

        (
            intraday_score,
            intraday_signal,
            intraday_reasons,
        ) = classify(
            intraday_row,
            avg_dollar_volume,
        )

        swing = score_swing_daily(
            history,
            spy_history,
            qqq_history,
            leadership.get(
                symbol
            ),
        )

        if not swing:
            continue

        if production_mode:

            (
                final_signal,
                decision,
            ) = (
                combine_daily_intraday_signal(
                    swing[
                        "signal"
                    ],
                    intraday_signal,
                    intraday_score,
                    risk_flag=bool(
                        swing.get(
                            "risk_flag",
                            False,
                        )
                    ),
                )
            )

            diagnostics = (
                _candidate_gate_diagnostics(
                    swing,
                    intraday_signal,
                    intraday_score,
                    config,
                )
            )

        else:

            (
                final_signal,
                decision,
                diagnostics,
            ) = _research_signal(
                swing,
                intraday_signal,
                intraday_score,
                config,
            )

        row = {
            "symbol": symbol,
            "session": session_date,
            "signal_time": (
                pd.Timestamp(
                    intraday_row[
                        "timestamp"
                    ]
                )
            ),
            "signal": (
                final_signal
            ),
            "daily_signal": (
                swing[
                    "signal"
                ]
            ),
            "intraday_signal": (
                intraday_signal
            ),
            "swing_score": float(
                swing[
                    "swing_score"
                ]
            ),
            "intraday_score": float(
                intraday_score
            ),
            "entry_quality": float(
                swing[
                    "entry_quality"
                ]
            ),
            "setup": (
                swing[
                    "setup"
                ]
            ),
            "reference_price": float(
                intraday_row[
                    "close"
                ]
            ),
            "entry_low": _number(
                swing.get(
                    "entry_low"
                )
            ),
            "entry_high": _number(
                swing.get(
                    "entry_high"
                )
            ),
            "planned_stop": float(
                swing[
                    "stop"
                ]
            ),
            "reward_risk": _number(
                swing.get(
                    "reward_risk"
                )
            ),
            "market_score": _number(
                swing.get(
                    "market_score"
                )
            ),
            "leadership_percentile": (
                swing.get(
                    "leadership_percentile"
                )
            ),
            "distribution_days": int(
                _number(
                    swing.get(
                        "distribution_days"
                    )
                )
            ),
            "trend_health": bool(
                swing.get(
                    "trend_health",
                    False,
                )
            ),
            "inside_entry_zone": bool(
                swing.get(
                    "inside_entry_zone",
                    False,
                )
            ),
            "too_extended": bool(
                swing.get(
                    "too_extended",
                    False,
                )
            ),
            "risk_flag": bool(
                swing.get(
                    "risk_flag",
                    False,
                )
            ),
            "risk_reason": (
                swing.get(
                    "risk_reason",
                    "",
                )
            ),
            "decision": decision,
            "intraday_reasons": (
                "; ".join(
                    intraday_reasons
                )
            ),
            "configured_swing_score": (
                config[
                    "swing_score"
                ]
            ),
            "configured_intraday_score": (
                config[
                    "intraday_score"
                ]
            ),
        }

        row.update(
            diagnostics
        )

        rows.append(
            row
        )

    return sorted(
        rows,
        key=lambda row: (
            row[
                "signal"
            ]
            not in BUY_SIGNALS,
            -row[
                "swing_score"
            ],
            -row[
                "intraday_score"
            ],
        ),
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def swing_backtest(
    bars: pd.DataFrame,
    spy_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame | None = None,
    daily_bars: pd.DataFrame | None = None,
    market_daily_bars: pd.DataFrame | None = None,
    starting_capital: float = 2_000,
    risk_pct: float = 0.005,
    max_positions: int = 3,
    max_holding_days: int = 20,
    scan_time: str = "11:30",
    slippage_bps: float = 5.0,
    commission_bps: float = 0.0,
    gate_config=None,
    production_mode=True,
):
    """Backtest the complete daily-plus-intraday decision chain.

    production_mode=True:
        Uses the live production confluence logic.

    production_mode=False:
        Uses the supplied gate_config for research/calibration only.
    """

    config = _normalise_gate_config(
        gate_config
    )

    if starting_capital <= 0:
        raise ValueError(
            "Starting capital must be positive."
        )

    if not 0 < risk_pct <= 0.05:
        raise ValueError(
            "Risk per trade must be greater than 0% and at most 5%."
        )

    if max_positions < 1:
        raise ValueError(
            "Maximum positions must be at least 1."
        )

    if max_holding_days < 1:
        raise ValueError(
            "Maximum holding period must be at least 1 session."
        )

    try:
        hour, minute = (
            int(
                piece
            )
            for piece
            in scan_time.split(
                ":"
            )
        )

        scan_clock = time(
            hour,
            minute,
        )

    except Exception as exc:

        raise ValueError(
            "Scan time must use HH:MM, for example 11:30."
        ) from exc

    minutes = _regular_minutes(
        _normalise_bars(
            bars
        )
    )

    spy_minutes = _regular_minutes(
        _normalise_bars(
            spy_bars
        )
    )

    qqq_minutes = _regular_minutes(
        _normalise_bars(
            qqq_bars
            if qqq_bars is not None
            else pd.DataFrame()
        )
    )

    if (
        minutes.empty
        or spy_minutes.empty
    ):

        return _empty_result(
            starting_capital,
            [
                "Stock or SPY minute data was unavailable."
            ],
            gate_config=config,
        )

    daily = (
        _daily_from_minutes(
            minutes
        )
        if (
            daily_bars is None
            or daily_bars.empty
        )
        else _normalise_daily(
            daily_bars
        )
    )

    if (
        market_daily_bars is None
        or market_daily_bars.empty
    ):

        spy_daily = (
            _daily_from_minutes(
                spy_minutes
            )
        )

        qqq_daily = (
            _daily_from_minutes(
                qqq_minutes
            )
        )

    else:

        market_daily = (
            _normalise_daily(
                market_daily_bars
            )
        )

        spy_daily = market_daily[
            market_daily[
                "symbol"
            ]
            == "SPY"
        ]

        qqq_daily = market_daily[
            market_daily[
                "symbol"
            ]
            == "QQQ"
        ]

    if (
        daily.empty
        or spy_daily.empty
    ):

        return _empty_result(
            starting_capital,
            [
                "At least 60 completed daily bars are required."
            ],
            gate_config=config,
        )

    session_dates = sorted(
        set(
            minutes[
                "session"
            ]
        )
    )

    cash = float(
        starting_capital
    )

    positions = {}

    pending = []

    trades = []

    signal_log = []

    equity_rows = []

    latest_prices = {}

    for session_date in session_dates:

        all_today = minutes[
            minutes[
                "session"
            ]
            == session_date
        ]

        session_times = sorted(
            all_today[
                "timestamp"
            ].unique()
        )

        evaluated = False

        for raw_timestamp in session_times:

            timestamp = pd.Timestamp(
                raw_timestamp
            )

            current_rows = all_today[
                all_today[
                    "timestamp"
                ]
                == timestamp
            ]

            current_clock = (
                timestamp
                .tz_convert(
                    ET
                )
                .time()
            )

            for bar in current_rows.itertuples(
                index=False
            ):

                latest_prices[
                    bar.symbol
                ] = float(
                    bar.close
                )

                if (
                    bar.symbol
                    in positions
                ):

                    (
                        cash,
                        closed,
                    ) = _manage_bar(
                        positions[
                            bar.symbol
                        ],
                        bar,
                        cash,
                        commission_bps,
                        slippage_bps,
                    )

                    if (
                        closed
                        is not None
                    ):

                        trades.append(
                            closed
                        )

                        del positions[
                            bar.symbol
                        ]

            # ----------------------------------------
            # FILL PENDING ENTRIES
            # ----------------------------------------

            still_pending = []

            for order in pending:

                if (
                    order[
                        "signal_time"
                    ]
                    >= timestamp
                    or order[
                        "symbol"
                    ]
                    in positions
                ):

                    still_pending.append(
                        order
                    )

                    continue

                match = current_rows[
                    current_rows[
                        "symbol"
                    ]
                    == order[
                        "symbol"
                    ]
                ]

                if (
                    match.empty
                    or len(
                        positions
                    )
                    >= max_positions
                ):

                    still_pending.append(
                        order
                    )

                    continue

                bar = match.iloc[
                    0
                ]

                entry = _execution_price(
                    float(
                        bar[
                            "open"
                        ]
                    ),
                    "buy",
                    slippage_bps,
                )

                initial_stop = float(
                    order[
                        "planned_stop"
                    ]
                )

                if (
                    initial_stop
                    >= entry
                ):

                    initial_stop = (
                        entry
                        - max(
                            entry
                            * 0.025,
                            0.01,
                        )
                    )

                per_share_risk = (
                    entry
                    - initial_stop
                )

                equity = _account_equity(
                    cash,
                    positions,
                    latest_prices,
                )

                risk_budget = (
                    equity
                    * risk_pct
                )

                shares_by_risk = int(
                    risk_budget
                    / max(
                        per_share_risk,
                        0.01,
                    )
                )

                cost_per_share = (
                    entry
                    * (
                        1
                        + max(
                            commission_bps,
                            0,
                        )
                        / 10_000
                    )
                )

                shares_by_cash = int(
                    cash
                    / max(
                        cost_per_share,
                        0.01,
                    )
                )

                shares = min(
                    shares_by_risk,
                    shares_by_cash,
                )

                if shares < 1:
                    continue

                notional = (
                    shares
                    * entry
                )

                entry_fee = _fee(
                    notional,
                    commission_bps,
                )

                cash -= (
                    notional
                    + entry_fee
                )

                positions[
                    order[
                        "symbol"
                    ]
                ] = Position(
                    symbol=order[
                        "symbol"
                    ],
                    signal_date=order[
                        "session"
                    ],
                    signal_time=order[
                        "signal_time"
                    ],
                    entry_time=timestamp,
                    signal=order[
                        "signal"
                    ],
                    setup=order[
                        "setup"
                    ],
                    swing_score=order[
                        "swing_score"
                    ],
                    intraday_score=order[
                        "intraday_score"
                    ],
                    entry=entry,
                    initial_stop=initial_stop,
                    stop=initial_stop,
                    target1=(
                        entry
                        + 2
                        * per_share_risk
                    ),
                    target2=(
                        entry
                        + 3
                        * per_share_risk
                    ),
                    shares=shares,
                    remaining=shares,
                    entry_fee=entry_fee,
                )

                entry_bar = next(
                    match.itertuples(
                        index=False
                    )
                )

                (
                    cash,
                    closed,
                ) = _manage_bar(
                    positions[
                        order[
                            "symbol"
                        ]
                    ],
                    entry_bar,
                    cash,
                    commission_bps,
                    slippage_bps,
                )

                if (
                    closed
                    is not None
                ):

                    trades.append(
                        closed
                    )

                    del positions[
                        order[
                            "symbol"
                        ]
                    ]

            pending = (
                still_pending
            )

            # ----------------------------------------
            # SIGNAL EVALUATION
            # ----------------------------------------

            if (
                not evaluated
                and current_clock
                >= scan_clock
            ):

                candidates = (
                    _signal_candidates(
                        session_date,
                        current_clock,
                        all_today,
                        daily,
                        spy_minutes,
                        qqq_minutes,
                        spy_daily,
                        qqq_daily,
                        gate_config=config,
                        production_mode=production_mode,
                    )
                )

                signal_log.extend(
                    candidates
                )

                open_slots = (
                    max_positions
                    - len(
                        positions
                    )
                    - len(
                        pending
                    )
                )

                for candidate in candidates:

                    if (
                        open_slots
                        <= 0
                    ):
                        break

                    if (
                        candidate[
                            "signal"
                        ]
                        in BUY_SIGNALS
                        and candidate[
                            "symbol"
                        ]
                        not in positions
                        and all(
                            candidate[
                                "symbol"
                            ]
                            != order[
                                "symbol"
                            ]
                            for order
                            in pending
                        )
                    ):

                        candidate[
                            "signal_time"
                        ] = timestamp

                        pending.append(
                            candidate
                        )

                        open_slots -= 1

                evaluated = True

        pending.clear()

        # ----------------------------------------
        # TIME EXITS
        # ----------------------------------------

        final_rows = all_today[
            all_today[
                "clock"
            ]
            <= time(
                16,
                0,
            )
        ]

        for symbol in list(
            positions
        ):

            position = positions[
                symbol
            ]

            if (
                position.holding_days
                < max_holding_days
            ):
                continue

            match = final_rows[
                final_rows[
                    "symbol"
                ]
                == symbol
            ]

            if match.empty:
                continue

            final_bar = match.iloc[
                -1
            ]

            (
                cash,
                closed,
            ) = _close_position(
                position,
                float(
                    final_bar[
                        "close"
                    ]
                ),
                pd.Timestamp(
                    final_bar[
                        "timestamp"
                    ]
                ),
                "TIME EXIT",
                cash,
                commission_bps,
                slippage_bps,
            )

            trades.append(
                closed
            )

            del positions[
                symbol
            ]

        _mark_trend_exits(
            positions,
            daily,
            final_rows,
            session_date,
        )

        equity_rows.append(
            {
                "date": (
                    session_date
                ),
                "equity": round(
                    _account_equity(
                        cash,
                        positions,
                        latest_prices,
                    ),
                    2,
                ),
                "cash": round(
                    cash,
                    2,
                ),
                "open_positions": len(
                    positions
                ),
            }
        )

    # --------------------------------------------
    # CLOSE REMAINING POSITIONS
    # --------------------------------------------

    final_timestamp = (
        minutes[
            "timestamp"
        ].max()
    )

    for symbol in list(
        positions
    ):

        position = positions[
            symbol
        ]

        (
            cash,
            closed,
        ) = _close_position(
            position,
            latest_prices.get(
                symbol,
                position.entry,
            ),
            final_timestamp,
            "END OF TEST",
            cash,
            commission_bps,
            slippage_bps,
        )

        trades.append(
            closed
        )

        del positions[
            symbol
        ]

    # ========================================================
    # RESULTS
    # ========================================================

    trades_frame = pd.DataFrame(
        trades,
        columns=TRADE_COLUMNS,
    )

    equity_frame = pd.DataFrame(
        equity_rows
    )

    if not equity_frame.empty:

        equity_frame.loc[
            equity_frame.index[-1],
            [
                "equity",
                "cash",
                "open_positions",
            ],
        ] = [
            round(
                cash,
                2,
            ),
            round(
                cash,
                2,
            ),
            0,
        ]

    trade_count = len(
        trades_frame
    )

    wins = (
        trades_frame[
            trades_frame[
                "pnl"
            ]
            > 0
        ]
        if trade_count
        else trades_frame
    )

    losses = (
        trades_frame[
            trades_frame[
                "pnl"
            ]
            < 0
        ]
        if trade_count
        else trades_frame
    )

    gross_profit = (
        float(
            wins[
                "pnl"
            ].sum()
        )
        if len(
            wins
        )
        else 0.0
    )

    gross_loss = (
        abs(
            float(
                losses[
                    "pnl"
                ].sum()
            )
        )
        if len(
            losses
        )
        else 0.0
    )

    profit_factor = (
        gross_profit
        / gross_loss
        if gross_loss
        > 0
        else (
            np.inf
            if gross_profit
            > 0
            else 0.0
        )
    )

    if not equity_frame.empty:

        equity_values = pd.Series(
            [
                float(
                    starting_capital
                )
            ]
            + equity_frame[
                "equity"
            ]
            .astype(float)
            .tolist()
        )

        peak = (
            equity_values
            .cummax()
        )

        drawdown = (
            equity_values
            / peak.replace(
                0,
                np.nan,
            )
            - 1
        )

        max_drawdown = float(
            drawdown.min()
            * 100
        )

    else:

        max_drawdown = 0.0

    stats = {
        "starting_capital": round(
            float(
                starting_capital
            ),
            2,
        ),
        "ending_capital": round(
            cash,
            2,
        ),
        "total_return_pct": round(
            (
                cash
                / starting_capital
                - 1
            )
            * 100,
            2,
        ),
        "trades": (
            trade_count
        ),
        "win_rate_pct": (
            round(
                len(
                    wins
                )
                / trade_count
                * 100,
                2,
            )
            if trade_count
            else 0.0
        ),
        "profit_factor": (
            round(
                profit_factor,
                2,
            )
            if np.isfinite(
                profit_factor
            )
            else "inf"
        ),
        "expectancy_r": (
            round(
                float(
                    trades_frame[
                        "r_multiple"
                    ].mean()
                ),
                3,
            )
            if trade_count
            else 0.0
        ),
        "avg_trade_dollars": (
            round(
                float(
                    trades_frame[
                        "pnl"
                    ].mean()
                ),
                2,
            )
            if trade_count
            else 0.0
        ),
        "max_drawdown_pct": round(
            max_drawdown,
            2,
        ),
    }

    warnings = []

    if (
        trade_count
        < 30
    ):

        warnings.append(
            "Fewer than 30 completed trades: the sample is too small "
            "for a reliable conclusion."
        )

    signal_frame = pd.DataFrame(
        signal_log
    )

    validation = (
        chronological_validation(
            trades_frame
        )
    )

    return {
        "trades": (
            trades_frame
        ),
        "equity": (
            equity_frame
        ),
        "stats": stats,
        "signal_log": (
            signal_frame
        ),
        "diagnostics": (
            _build_signal_diagnostics(
                signal_frame,
                trade_count=trade_count,
                gate_config=config,
            )
        ),
        "validation": validation,
        "gate_config": config,
        "warnings": warnings,
    }


# ============================================================
# CALIBRATION ENGINE
# ============================================================

def calibrate_thresholds(
    bars: pd.DataFrame,
    spy_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame | None = None,
    daily_bars: pd.DataFrame | None = None,
    market_daily_bars: pd.DataFrame | None = None,
    starting_capital: float = 2_000,
    risk_pct: float = 0.005,
    max_positions: int = 3,
    max_holding_days: int = 20,
    scan_time: str = "11:30",
    slippage_bps: float = 5.0,
    commission_bps: float = 0.0,
    profiles=None,
):
    """Compare a small set of alternate thresholds.

    The calibration engine intentionally uses a limited number of profiles.
    It is designed to diagnose whether production gates are realistic, not
    brute-force thousands of combinations until something looks profitable.
    """

    profiles = (
        CALIBRATION_PROFILES
        if profiles is None
        else profiles
    )

    rows = []

    detailed_results = {}

    for profile in profiles:

        name = profile.get(
            "name",
            "UNNAMED",
        )

        config = {
            key: value
            for key, value
            in profile.items()
            if key
            != "name"
        }

        is_production = (
            name
            == "PRODUCTION_85_85"
        )

        result = swing_backtest(
            bars,
            spy_bars,
            qqq_bars=qqq_bars,
            daily_bars=daily_bars,
            market_daily_bars=market_daily_bars,
            starting_capital=starting_capital,
            risk_pct=risk_pct,
            max_positions=max_positions,
            max_holding_days=max_holding_days,
            scan_time=scan_time,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            gate_config=config,
            production_mode=is_production,
        )

        detailed_results[
            name
        ] = result

        stats = result[
            "stats"
        ]

        validation = result[
            "validation"
        ]

        pf = stats.get(
            "profit_factor",
            0,
        )

        try:
            pf_numeric = float(
                pf
            )
        except Exception:
            pf_numeric = np.inf

        oos_pf = validation.get(
            "out_of_sample_profit_factor",
            0,
        )

        try:
            oos_pf_numeric = float(
                oos_pf
            )
        except Exception:
            oos_pf_numeric = np.inf

        rows.append(
            {
                "profile": name,
                "production_rules": (
                    is_production
                ),
                "swing_score_gate": (
                    config[
                        "swing_score"
                    ]
                ),
                "intraday_score_gate": (
                    config[
                        "intraday_score"
                    ]
                ),
                "entry_quality_gate": (
                    config[
                        "entry_quality"
                    ]
                ),
                "leadership_gate": (
                    config[
                        "leadership_percentile"
                    ]
                ),
                "market_score_gate": (
                    config[
                        "market_score"
                    ]
                ),
                "reward_risk_gate": (
                    config[
                        "reward_risk"
                    ]
                ),
                "max_distribution_days": (
                    config[
                        "max_distribution_days"
                    ]
                ),
                "trades": (
                    stats.get(
                        "trades",
                        0,
                    )
                ),
                "return_pct": (
                    stats.get(
                        "total_return_pct",
                        0,
                    )
                ),
                "win_rate_pct": (
                    stats.get(
                        "win_rate_pct",
                        0,
                    )
                ),
                "profit_factor": (
                    pf
                ),
                "expectancy_r": (
                    stats.get(
                        "expectancy_r",
                        0,
                    )
                ),
                "max_drawdown_pct": (
                    stats.get(
                        "max_drawdown_pct",
                        0,
                    )
                ),
                "out_of_sample_trades": (
                    validation.get(
                        "out_of_sample_trades",
                        0,
                    )
                ),
                "out_of_sample_win_rate_pct": (
                    validation.get(
                        "out_of_sample_win_rate_pct",
                        0,
                    )
                ),
                "out_of_sample_expectancy_r": (
                    validation.get(
                        "out_of_sample_expectancy_r",
                        0,
                    )
                ),
                "out_of_sample_profit_factor": (
                    oos_pf
                ),
                "bootstrap_expectancy_low_r": (
                    validation.get(
                        "bootstrap_expectancy_low_r"
                    )
                ),
                "bootstrap_expectancy_high_r": (
                    validation.get(
                        "bootstrap_expectancy_high_r"
                    )
                ),
                "confidence_grade": (
                    validation.get(
                        "confidence_grade",
                        "INSUFFICIENT",
                    )
                ),
                "validation_pass": bool(
                    validation.get(
                        "validation_pass",
                        False,
                    )
                ),
                "_pf_numeric": (
                    pf_numeric
                ),
                "_oos_pf_numeric": (
                    oos_pf_numeric
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    if not comparison.empty:

        comparison = (
            comparison
            .sort_values(
                [
                    "validation_pass",
                    "out_of_sample_expectancy_r",
                    "_oos_pf_numeric",
                    "trades",
                    "max_drawdown_pct",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_pf_numeric",
                    "_oos_pf_numeric",
                ],
                errors="ignore",
            )
            .reset_index(
                drop=True
            )
        )

    production_result = (
        detailed_results.get(
            "PRODUCTION_85_85"
        )
    )

    score_distribution = (
        pd.DataFrame()
    )

    if (
        production_result
        is not None
    ):

        score_distribution = (
            production_result
            .get(
                "diagnostics",
                {},
            )
            .get(
                "score_distribution",
                pd.DataFrame(),
            )
        )

    return {
        "comparison": comparison,
        "results": detailed_results,
        "score_distribution": score_distribution,
        "profiles_tested": len(
            profiles
        ),
    }


# ============================================================
# BACKWARDS-COMPATIBLE IMPORT
# ============================================================

backtest = swing_backtest

# ============================================================
# v3.4 ADAPTIVE WALK-FORWARD CALIBRATION
# ============================================================

def _quantile_or_default(series, q, default):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float(default)
    return float(s.quantile(q))


def build_adaptive_calibration_profiles(production_result, max_profiles=16):
    """Build a bounded, data-aware research grid without changing production rules.

    Thresholds are anchored to reachable historical score percentiles, then clipped to
    conservative floors. This avoids wasting calibration runs on impossible 85/85 gates
    while also avoiding a brute-force search that would invite overfitting.
    """
    signal_log = (production_result or {}).get("signal_log", pd.DataFrame())
    if signal_log is None or signal_log.empty:
        return CALIBRATION_PROFILES[:max_profiles]

    swing = pd.to_numeric(signal_log.get("swing_score"), errors="coerce")
    intra = pd.to_numeric(signal_log.get("intraday_score"), errors="coerce")

    swing_candidates = {
        85.0,
        round(_quantile_or_default(swing, .99, 82.5) * 2) / 2,
        round(_quantile_or_default(swing, .95, 80.0) * 2) / 2,
        round(_quantile_or_default(swing, .90, 77.5) * 2) / 2,
        round(_quantile_or_default(swing, .80, 75.0) * 2) / 2,
    }
    intra_candidates = {
        85.0,
        round(_quantile_or_default(intra, .99, 80.0) / 5) * 5,
        round(_quantile_or_default(intra, .95, 75.0) / 5) * 5,
        round(_quantile_or_default(intra, .90, 70.0) / 5) * 5,
    }

    # Research floors only. These are NOT live-rule changes.
    swing_candidates = sorted({min(85.0, max(70.0, x)) for x in swing_candidates}, reverse=True)
    intra_candidates = sorted({min(85.0, max(55.0, x)) for x in intra_candidates}, reverse=True)

    profiles = [{"name": "PRODUCTION_85_85", **PRODUCTION_GATE_CONFIG}]
    seen = {(85.0, 85.0, 10.0, 70.0)}

    for s in swing_candidates:
        for i in intra_candidates:
            key = (float(s), float(i), 10.0, 70.0)
            if key in seen:
                continue
            seen.add(key)
            profiles.append({
                "name": f"ADAPT_S{s:g}_I{i:g}".replace(".", "_"),
                **PRODUCTION_GATE_CONFIG,
                "swing_score": float(s),
                "intraday_score": float(i),
            })
            if len(profiles) >= max_profiles:
                return profiles

    return profiles


def _add_v34_research_decision(comparison):
    if comparison is None or comparison.empty:
        return pd.DataFrame()
    out = comparison.copy()
    for c in ["trades", "out_of_sample_trades", "out_of_sample_expectancy_r",
              "out_of_sample_profit_factor", "max_drawdown_pct", "bootstrap_expectancy_low_r"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["research_eligible"] = (
        out["trades"].fillna(0).ge(30)
        & out["out_of_sample_trades"].fillna(0).ge(10)
        & out["out_of_sample_expectancy_r"].fillna(-999).gt(0)
        & out["out_of_sample_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).ge(1.15)
        & out["max_drawdown_pct"].abs().fillna(999).le(20)
    )
    # Stronger promotion flag: requires a positive lower bootstrap bound.
    out["promotion_candidate"] = (
        out["research_eligible"]
        & out["bootstrap_expectancy_low_r"].fillna(-999).gt(0)
        & out["out_of_sample_expectancy_r"].fillna(-999).ge(0.10)
        & out["out_of_sample_profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).ge(1.20)
    )

    # Rank for research only. Lower drawdown is better.
    out["research_score"] = (
        100 * out["promotion_candidate"].astype(int)
        + 30 * out["research_eligible"].astype(int)
        + 20 * out["validation_pass"].fillna(False).astype(bool).astype(int)
        + 12 * out["out_of_sample_expectancy_r"].clip(-1, 1).fillna(-1)
        + 4 * out["out_of_sample_profit_factor"].replace([np.inf, -np.inf], np.nan).clip(0, 3).fillna(0)
        + np.minimum(out["out_of_sample_trades"].fillna(0), 50) / 10
        - out["max_drawdown_pct"].abs().fillna(50) / 10
    ).round(3)
    return out.sort_values(
        ["promotion_candidate", "research_eligible", "research_score", "out_of_sample_expectancy_r"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def calibrate_thresholds_adaptive(
    bars,
    spy_bars,
    qqq_bars=None,
    daily_bars=None,
    market_daily_bars=None,
    starting_capital=2_000,
    risk_pct=0.005,
    max_positions=3,
    max_holding_days=20,
    scan_time="11:30",
    slippage_bps=5.0,
    commission_bps=0.0,
    production_result=None,
    max_profiles=16,
):
    """v3.4 bounded adaptive calibration using the real portfolio simulator.

    The live rules remain unchanged. Alternate settings are research-only and must pass
    chronological OOS validation before they can even be considered for promotion.
    """
    if production_result is None:
        production_result = swing_backtest(
            bars, spy_bars, qqq_bars=qqq_bars, daily_bars=daily_bars,
            market_daily_bars=market_daily_bars, starting_capital=starting_capital,
            risk_pct=risk_pct, max_positions=max_positions,
            max_holding_days=max_holding_days, scan_time=scan_time,
            slippage_bps=slippage_bps, commission_bps=commission_bps,
        )

    profiles = build_adaptive_calibration_profiles(production_result, max_profiles=max_profiles)
    # Reuse production result and only rerun alternate research profiles.
    alternate = [p for p in profiles if p.get("name") != "PRODUCTION_85_85"]
    cal = calibrate_thresholds(
        bars, spy_bars, qqq_bars=qqq_bars, daily_bars=daily_bars,
        market_daily_bars=market_daily_bars, starting_capital=starting_capital,
        risk_pct=risk_pct, max_positions=max_positions,
        max_holding_days=max_holding_days, scan_time=scan_time,
        slippage_bps=slippage_bps, commission_bps=commission_bps,
        profiles=alternate,
    )

    # Add the already-computed production row without rerunning it.
    prod_stats = production_result.get("stats", {})
    prod_val = production_result.get("validation", {})
    prod_row = {
        "profile": "PRODUCTION_85_85", "production_rules": True,
        "swing_score_gate": PRODUCTION_GATE_CONFIG["swing_score"],
        "intraday_score_gate": PRODUCTION_GATE_CONFIG["intraday_score"],
        "entry_quality_gate": PRODUCTION_GATE_CONFIG["entry_quality"],
        "leadership_gate": PRODUCTION_GATE_CONFIG["leadership_percentile"],
        "market_score_gate": PRODUCTION_GATE_CONFIG["market_score"],
        "reward_risk_gate": PRODUCTION_GATE_CONFIG["reward_risk"],
        "max_distribution_days": PRODUCTION_GATE_CONFIG["max_distribution_days"],
        "trades": prod_stats.get("trades", 0),
        "return_pct": prod_stats.get("total_return_pct", 0),
        "win_rate_pct": prod_stats.get("win_rate_pct", 0),
        "profit_factor": prod_stats.get("profit_factor", 0),
        "expectancy_r": prod_stats.get("expectancy_r", 0),
        "max_drawdown_pct": prod_stats.get("max_drawdown_pct", 0),
        "out_of_sample_trades": prod_val.get("out_of_sample_trades", 0),
        "out_of_sample_win_rate_pct": prod_val.get("out_of_sample_win_rate_pct", 0),
        "out_of_sample_expectancy_r": prod_val.get("out_of_sample_expectancy_r", 0),
        "out_of_sample_profit_factor": prod_val.get("out_of_sample_profit_factor", 0),
        "bootstrap_expectancy_low_r": prod_val.get("bootstrap_expectancy_low_r"),
        "bootstrap_expectancy_high_r": prod_val.get("bootstrap_expectancy_high_r"),
        "confidence_grade": prod_val.get("confidence_grade", "INSUFFICIENT"),
        "validation_pass": bool(prod_val.get("validation_pass", False)),
    }
    comparison = pd.concat([pd.DataFrame([prod_row]), cal.get("comparison", pd.DataFrame())], ignore_index=True)
    comparison = _add_v34_research_decision(comparison)
    cal["comparison"] = comparison
    cal["profiles_tested"] = len(profiles)
    cal["production_result"] = production_result
    cal["adaptive_profiles"] = pd.DataFrame(profiles)
    cal["recommendation"] = (
        "KEEP_PRODUCTION_RULES" if not comparison.get("promotion_candidate", pd.Series(dtype=bool)).any()
        else "REVIEW_PROMOTION_CANDIDATES"
    )
    return cal
