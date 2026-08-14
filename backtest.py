"""Production-equivalent historical simulator for the v3.2 swing scanner.

The simulator calls the same daily scorer, intraday feature builder,
classifier and confluence gate used by the live scanner. Signals are evaluated
at a fixed time with only information available then, and entries occur no
earlier than the following minute bar.
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


ET = ZoneInfo("America/New_York")
BUY_SIGNALS = {"BUY", "A+ SWING BUY"}
DAILY_BUY_GATE_LABELS = {
    "risk_event_clear": "No active catalyst-gap risk",
    "not_too_extended": "Not too extended",
    "swing_score": f"Swing Score at least {MIN_SWING_SCORE_BUY}",
    "entry_quality": f"Entry Quality at least {MIN_ENTRY_QUALITY_BUY}/15",
    "reward_risk": f"Reward/risk at least {MIN_REWARD_RISK_BUY:.1f}:1",
    "market_regime": f"Market Score at least {MIN_MARKET_SCORE_BUY:.0f}/10",
    "inside_entry_zone": "Price inside preferred entry zone",
    "trend_health": "20EMA and 50SMA trends rising",
    "distribution": (
        f"Distribution days no more than {MAX_DISTRIBUTION_DAYS_BUY}"
    ),
    "leadership": (
        f"Market leadership at least {MIN_RS_PERCENTILE_BUY * 100:.0f}th percentile"
    ),
}
INTRADAY_BUY_GATE_LABELS = {
    "intraday_signal": "Intraday signal is BUY",
    "intraday_score": (
        f"Intraday Score at least {MIN_INTRADAY_CONFIRMATION_SCORE}"
    ),
}
BUY_GATE_LABELS = {
    **DAILY_BUY_GATE_LABELS,
    **INTRADAY_BUY_GATE_LABELS,
}
TRADE_COLUMNS = [
    "symbol", "signal_date", "signal_time", "entry_time", "exit_time",
    "signal", "setup", "swing_score", "intraday_score", "entry",
    "initial_stop", "target1", "target2", "shares", "pnl",
    "return_pct", "r_multiple", "holding_days", "exit_reason",
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


def _empty_result(starting_capital: float, warnings=None):
    return {
        "trades": pd.DataFrame(columns=TRADE_COLUMNS),
        "equity": pd.DataFrame(
            columns=["date", "equity", "cash", "open_positions"]
        ),
        "stats": {
            "starting_capital": round(float(starting_capital), 2),
            "ending_capital": round(float(starting_capital), 2),
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
        },
        "warnings": list(warnings or []),
    }


def _number(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_gate_diagnostics(swing, intraday_signal, intraday_score):
    """Return additive BUY-gate diagnostics without changing a signal."""

    leadership = swing.get("leadership_percentile")
    leadership_pass = (
        leadership is None
        or pd.isna(leadership)
        or _number(leadership) >= MIN_RS_PERCENTILE_BUY * 100
    )
    daily_gates = {
        "risk_event_clear": not bool(swing.get("risk_flag", False)),
        "not_too_extended": not bool(swing.get("too_extended", False)),
        "swing_score": (
            _number(swing.get("swing_score")) >= MIN_SWING_SCORE_BUY
        ),
        "entry_quality": (
            _number(swing.get("entry_quality")) >= MIN_ENTRY_QUALITY_BUY
        ),
        "reward_risk": (
            _number(swing.get("reward_risk")) >= MIN_REWARD_RISK_BUY
        ),
        "market_regime": (
            _number(swing.get("market_score")) >= MIN_MARKET_SCORE_BUY
        ),
        "inside_entry_zone": bool(swing.get("inside_entry_zone", False)),
        "trend_health": bool(swing.get("trend_health", False)),
        "distribution": (
            _number(swing.get("distribution_days"), float("inf"))
            <= MAX_DISTRIBUTION_DAYS_BUY
        ),
        "leadership": bool(leadership_pass),
    }
    intraday_gates = {
        "intraday_signal": intraday_signal == "BUY",
        "intraday_score": (
            _number(intraday_score) >= MIN_INTRADAY_CONFIRMATION_SCORE
        ),
    }
    all_gates = {**daily_gates, **intraday_gates}
    daily_failed = [
        DAILY_BUY_GATE_LABELS[key]
        for key, passed in daily_gates.items()
        if not passed
    ]
    intraday_failed = [
        INTRADAY_BUY_GATE_LABELS[key]
        for key, passed in intraday_gates.items()
        if not passed
    ]
    failed = daily_failed + intraday_failed

    diagnostics = {
        "daily_gates_passed": sum(daily_gates.values()),
        "daily_gate_count": len(daily_gates),
        "buy_gates_passed": sum(all_gates.values()),
        "buy_gate_count": len(all_gates),
        "failed_gate_count": len(failed),
        "daily_failed_gates": "; ".join(daily_failed) or "None",
        "intraday_failed_gates": "; ".join(intraday_failed) or "None",
        "failed_buy_gates": "; ".join(failed) or "None",
    }
    diagnostics.update(
        {f"gate_{key}": bool(passed) for key, passed in all_gates.items()}
    )
    return diagnostics


def _build_signal_diagnostics(signal_log, trade_count=0):
    """Build a funnel, failure counts and one best near miss per symbol."""

    if signal_log is None or signal_log.empty:
        return {
            "funnel": pd.DataFrame(),
            "gate_failures": pd.DataFrame(),
            "near_misses": pd.DataFrame(),
        }

    frame = signal_log.copy()
    total = len(frame)
    final_buy = frame["signal"].isin(BUY_SIGNALS)
    for key in BUY_GATE_LABELS:
        column = f"gate_{key}"
        if column not in frame:
            # A legacy or externally supplied confirmed row necessarily
            # represents a pass; other legacy rows remain conservative.
            frame[column] = final_buy
    gate_columns = [f"gate_{key}" for key in BUY_GATE_LABELS]
    if "buy_gates_passed" not in frame:
        frame["buy_gates_passed"] = frame[gate_columns].astype(bool).sum(axis=1)
    if "buy_gate_count" not in frame:
        frame["buy_gate_count"] = len(gate_columns)
    if "failed_gate_count" not in frame:
        frame["failed_gate_count"] = (
            len(gate_columns) - frame["buy_gates_passed"].astype(int)
        )
    if "failed_buy_gates" not in frame:
        frame["failed_buy_gates"] = frame.apply(
            lambda row: "; ".join(
                label
                for key, label in BUY_GATE_LABELS.items()
                if not bool(row[f"gate_{key}"])
            )
            or "None",
            axis=1,
        )
    daily_buy = frame["daily_signal"].isin(BUY_SIGNALS)
    intraday_buy = frame["intraday_signal"].eq("BUY")
    intraday_score_pass = frame["gate_intraday_score"].astype(bool)

    stages = [
        ("Candidates evaluated", total),
        (
            f"Daily Swing Score at least {MIN_SWING_SCORE_BUY}",
            int(frame["gate_swing_score"].astype(bool).sum()),
        ),
        ("All daily BUY gates passed", int(daily_buy.sum())),
        (
            "Daily BUY plus intraday BUY signal",
            int((daily_buy & intraday_buy).sum()),
        ),
        (
            "Fully confirmed BUY",
            int((daily_buy & intraday_buy & intraday_score_pass & final_buy).sum()),
        ),
        ("Completed simulated trades", int(trade_count)),
    ]
    funnel = pd.DataFrame(
        [
            {
                "stage": stage,
                "count": count,
                "percent_of_candidates": round(count / total * 100, 1),
            }
            for stage, count in stages
        ]
    )

    failure_rows = []
    for key, label in BUY_GATE_LABELS.items():
        column = f"gate_{key}"
        failures = int((~frame[column].astype(bool)).sum())
        failure_rows.append(
            {
                "gate": label,
                "stage": "Intraday" if key in INTRADAY_BUY_GATE_LABELS else "Daily",
                "failed": failures,
                "failure_percent": round(failures / total * 100, 1),
            }
        )
    gate_failures = (
        pd.DataFrame(failure_rows)
        .sort_values(["failed", "stage", "gate"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    near_misses = frame[~final_buy].copy()
    if not near_misses.empty:
        near_misses["gates_passed"] = (
            near_misses["buy_gates_passed"].astype(int).astype(str)
            + "/"
            + near_misses["buy_gate_count"].astype(int).astype(str)
        )
        near_misses = (
            near_misses.sort_values(
                [
                    "failed_gate_count",
                    "swing_score",
                    "intraday_score",
                    "session",
                ],
                ascending=[True, False, False, False],
            )
            .drop_duplicates("symbol")
            .head(10)
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
        ].reset_index(drop=True)

    return {
        "funnel": funnel,
        "gate_failures": gate_failures,
        "near_misses": near_misses,
    }


def _normalise_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "symbol", "open", "high", "low", "close", "volume"
    }
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    out = frame.copy()
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], utc=True, errors="coerce"
    )
    out = out.dropna(subset=["timestamp", "symbol"])
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["session"] = out["timestamp"].dt.tz_convert(ET).dt.date
    out["clock"] = out["timestamp"].dt.tz_convert(ET).dt.time
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = _normalise_bars(frame)
    return out.drop(columns=["session", "clock"], errors="ignore")


def _regular_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[
        (frame["clock"] >= time(9, 30))
        & (frame["clock"] < time(16, 0))
    ].copy()


def _daily_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """Create completed daily bars when a separate daily feed is unavailable."""

    if minutes.empty:
        return pd.DataFrame()
    regular = minutes[
        (minutes["clock"] >= time(9, 30))
        & (minutes["clock"] <= time(16, 0))
    ].copy()
    if regular.empty:
        return pd.DataFrame()

    return (
        regular.groupby(["symbol", "session"], sort=True)
        .agg(
            timestamp=("timestamp", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=False)
        .drop(columns="session")
        .sort_values(["timestamp", "symbol"])
    )


def _partial_daily(minutes: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Aggregate only the minute bars visible at the scan timestamp."""

    if minutes is None or minutes.empty:
        return pd.DataFrame()
    ordered = minutes.sort_values("timestamp")
    return pd.DataFrame(
        [
            {
                "timestamp": ordered["timestamp"].iloc[-1],
                "symbol": symbol,
                "open": float(ordered["open"].iloc[0]),
                "high": float(ordered["high"].max()),
                "low": float(ordered["low"].min()),
                "close": float(ordered["close"].iloc[-1]),
                "volume": float(ordered["volume"].sum()),
            }
        ]
    )


def _completed_daily(daily: pd.DataFrame, session_date) -> pd.DataFrame:
    if daily.empty:
        return daily
    sessions = (
        pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert(ET).dt.date
    )
    return daily[sessions < session_date].copy()


def _session_slice(frame, session_date, through: time | None = None):
    out = frame[frame["session"] == session_date]
    if through is not None:
        out = out[out["clock"] <= through]
    return out.copy()


def _execution_price(raw_price: float, side: str, slippage_bps: float) -> float:
    adjustment = max(float(slippage_bps), 0.0) / 10_000
    multiplier = 1 + adjustment if side == "buy" else 1 - adjustment
    return float(raw_price) * multiplier


def _fee(notional: float, commission_bps: float) -> float:
    return abs(float(notional)) * max(float(commission_bps), 0.0) / 10_000


def _account_equity(cash, positions, prices):
    return cash + sum(
        position.remaining * prices.get(symbol, position.entry)
        for symbol, position in positions.items()
    )


def _trade_record(position, timestamp, reason, total_pnl):
    risk_dollars = max(
        (position.entry - position.initial_stop) * position.shares, 0.01
    )
    return {
        "symbol": position.symbol,
        "signal_date": position.signal_date,
        "signal_time": position.signal_time,
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "signal": position.signal,
        "setup": position.setup,
        "swing_score": round(position.swing_score, 1),
        "intraday_score": round(position.intraday_score, 1),
        "entry": round(position.entry, 4),
        "initial_stop": round(position.initial_stop, 4),
        "target1": round(position.target1, 4),
        "target2": round(position.target2, 4),
        "shares": position.shares,
        "pnl": round(total_pnl, 2),
        "return_pct": round(
            total_pnl / max(position.entry * position.shares, 0.01) * 100, 2
        ),
        "r_multiple": round(total_pnl / risk_dollars, 3),
        "holding_days": position.holding_days,
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
    fill = _execution_price(raw_price, "sell", slippage_bps)
    proceeds = fill * position.remaining
    exit_fee = _fee(proceeds, commission_bps)
    leg_pnl = (fill - position.entry) * position.remaining - exit_fee
    cash += proceeds - exit_fee
    total_pnl = position.realized_pnl + leg_pnl - position.entry_fee
    return cash, _trade_record(position, timestamp, reason, total_pnl)


def _take_target1(position, raw_price, timestamp, cash, commission_bps, slippage_bps):
    exit_shares = min(max(position.shares // 2, 1), position.remaining)
    fill = _execution_price(raw_price, "sell", slippage_bps)
    proceeds = fill * exit_shares
    exit_fee = _fee(proceeds, commission_bps)
    position.realized_pnl += (fill - position.entry) * exit_shares - exit_fee
    position.remaining -= exit_shares
    cash += proceeds - exit_fee
    position.target1_hit = True
    position.stop = position.entry

    if position.remaining <= 0:
        total_pnl = position.realized_pnl - position.entry_fee
        return cash, _trade_record(
            position, timestamp, "TARGET 1", total_pnl
        )
    return cash, None


def _manage_bar(
    position,
    bar,
    cash,
    commission_bps,
    slippage_bps,
):
    """Manage one minute conservatively; stop wins ambiguous intrabar ties."""

    if position.last_session != bar.session:
        position.holding_days += 1
        position.last_session = bar.session

    # An overnight gap through the stop fills at the worse opening price.
    if float(bar.open) <= position.stop:
        reason = "GAP STOP" if float(bar.open) < position.stop else "STOP"
        return _close_position(
            position,
            float(bar.open),
            bar.timestamp,
            reason,
            cash,
            commission_bps,
            slippage_bps,
        )

    if position.trend_exit_pending:
        return _close_position(
            position,
            float(bar.open),
            bar.timestamp,
            "TREND EXIT",
            cash,
            commission_bps,
            slippage_bps,
        )

    # A target crossed at the open occurs before the minute's high/low path.
    if position.target1_hit and float(bar.open) >= position.target2:
        return _close_position(
            position,
            float(bar.open),
            bar.timestamp,
            "GAP TARGET 2",
            cash,
            commission_bps,
            slippage_bps,
        )

    if not position.target1_hit and float(bar.open) >= position.target1:
        cash, closed = _take_target1(
            position,
            float(bar.open),
            bar.timestamp,
            cash,
            commission_bps,
            slippage_bps,
        )
        if closed is not None:
            return cash, closed
        if float(bar.open) >= position.target2:
            return _close_position(
                position,
                float(bar.open),
                bar.timestamp,
                "GAP TARGET 2",
                cash,
                commission_bps,
                slippage_bps,
            )

    if float(bar.low) <= position.stop:
        reason = "BREAKEVEN STOP" if position.target1_hit else "STOP"
        return _close_position(
            position,
            position.stop,
            bar.timestamp,
            reason,
            cash,
            commission_bps,
            slippage_bps,
        )

    if not position.target1_hit and float(bar.high) >= position.target1:
        cash, closed = _take_target1(
            position,
            position.target1,
            bar.timestamp,
            cash,
            commission_bps,
            slippage_bps,
        )
        if closed is not None:
            return cash, closed

        # The order inside one minute is unknown. After target 1, assume the
        # new breakeven stop was reached before target 2 if both are present.
        if float(bar.low) <= position.stop:
            return _close_position(
                position,
                position.stop,
                bar.timestamp,
                "TARGET 1 + BREAKEVEN STOP",
                cash,
                commission_bps,
                slippage_bps,
            )

    if position.target1_hit and float(bar.high) >= position.target2:
        return _close_position(
            position,
            position.target2,
            bar.timestamp,
            "TARGET 2",
            cash,
            commission_bps,
            slippage_bps,
        )

    return cash, None


def _mark_trend_exits(positions, daily, all_today, session_date):
    """Queue a next-session exit after a completed close below the 20EMA."""

    if not positions:
        return
    completed = _completed_daily(daily, session_date)
    for symbol, position in positions.items():
        today = all_today[all_today["symbol"] == symbol]
        if today.empty:
            continue
        history = completed[completed["symbol"] == symbol]
        history = pd.concat(
            [history, _partial_daily(today, symbol)],
            ignore_index=True,
        )
        if len(history) < 20:
            continue
        close = pd.to_numeric(history["close"], errors="coerce").dropna()
        if len(close) < 20:
            continue
        e20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        position.trend_exit_pending = bool(float(close.iloc[-1]) < e20)


def _signal_candidates(
    session_date,
    scan_clock,
    bars,
    daily,
    spy_minutes,
    qqq_minutes,
    spy_daily,
    qqq_daily,
):
    completed_stocks = _completed_daily(daily, session_date)
    completed_spy = _completed_daily(spy_daily, session_date)
    completed_qqq = _completed_daily(qqq_daily, session_date)
    spy_today = _session_slice(spy_minutes, session_date, scan_clock)
    qqq_today = _session_slice(qqq_minutes, session_date, scan_clock)
    spy_history = pd.concat(
        [completed_spy, _partial_daily(spy_today, "SPY")],
        ignore_index=True,
    )
    qqq_history = pd.concat(
        [completed_qqq, _partial_daily(qqq_today, "QQQ")],
        ignore_index=True,
    )
    rows = []
    stock_histories = {}

    for symbol, all_symbol_bars in bars.groupby("symbol"):
        today = _session_slice(all_symbol_bars, session_date, scan_clock)
        if len(today) < 20:
            continue
        prior_history = completed_stocks[
            completed_stocks["symbol"] == symbol
        ].copy()
        history = pd.concat(
            [prior_history, _partial_daily(today, symbol)],
            ignore_index=True,
        )
        if len(history) < 61:
            continue
        stock_histories[symbol] = (history, prior_history, today)

    if not stock_histories:
        return []

    leadership = relative_strength_percentiles(
        pd.concat(
            [item[0] for item in stock_histories.values()],
            ignore_index=True,
        )
    )

    for symbol, (history, prior_history, today) in stock_histories.items():

        avg_share_volume = float(prior_history["volume"].tail(20).mean())
        avg_dollar_volume = float(
            (prior_history["close"] * prior_history["volume"])
            .tail(20)
            .mean()
        )
        prepared = prepare_intraday(today, spy_today, avg_share_volume)
        if prepared.empty:
            continue
        intraday_row = prepared.iloc[-1]
        intraday_score, intraday_signal, intraday_reasons = classify(
            intraday_row, avg_dollar_volume
        )
        swing = score_swing_daily(
            history,
            spy_history,
            qqq_history,
            leadership.get(symbol),
        )
        if not swing:
            continue
        final_signal, decision = combine_daily_intraday_signal(
            swing["signal"],
            intraday_signal,
            intraday_score,
            risk_flag=bool(swing.get("risk_flag", False)),
        )
        row = {
            "symbol": symbol,
            "session": session_date,
            "signal_time": pd.Timestamp(intraday_row["timestamp"]),
            "signal": final_signal,
            "daily_signal": swing["signal"],
            "intraday_signal": intraday_signal,
            "swing_score": float(swing["swing_score"]),
            "intraday_score": float(intraday_score),
            "entry_quality": float(swing["entry_quality"]),
            "setup": swing["setup"],
            "reference_price": float(intraday_row["close"]),
            "entry_low": _number(swing.get("entry_low")),
            "entry_high": _number(swing.get("entry_high")),
            "planned_stop": float(swing["stop"]),
            "reward_risk": _number(swing.get("reward_risk")),
            "market_score": _number(swing.get("market_score")),
            "leadership_percentile": swing.get("leadership_percentile"),
            "distribution_days": int(
                _number(swing.get("distribution_days"))
            ),
            "trend_health": bool(swing.get("trend_health", False)),
            "inside_entry_zone": bool(
                swing.get("inside_entry_zone", False)
            ),
            "too_extended": bool(swing.get("too_extended", False)),
            "risk_flag": bool(swing.get("risk_flag", False)),
            "risk_reason": swing.get("risk_reason", ""),
            "decision": decision,
            "intraday_reasons": "; ".join(intraday_reasons),
        }
        row.update(
            _candidate_gate_diagnostics(
                swing,
                intraday_signal,
                intraday_score,
            )
        )
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            row["signal"] not in BUY_SIGNALS,
            -row["swing_score"],
            -row["intraday_score"],
        ),
    )


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
):
    """Backtest the complete daily-plus-intraday production decision chain."""

    if starting_capital <= 0:
        raise ValueError("Starting capital must be positive.")
    if not 0 < risk_pct <= 0.05:
        raise ValueError("Risk per trade must be greater than 0% and at most 5%.")
    if max_positions < 1:
        raise ValueError("Maximum positions must be at least 1.")
    if max_holding_days < 1:
        raise ValueError("Maximum holding period must be at least 1 session.")
    try:
        hour, minute = (int(piece) for piece in scan_time.split(":"))
        scan_clock = time(hour, minute)
    except Exception as exc:
        raise ValueError(
            "Scan time must use HH:MM, for example 11:30."
        ) from exc

    minutes = _regular_minutes(_normalise_bars(bars))
    spy_minutes = _regular_minutes(_normalise_bars(spy_bars))
    qqq_minutes = _regular_minutes(
        _normalise_bars(
            qqq_bars if qqq_bars is not None else pd.DataFrame()
        )
    )
    if minutes.empty or spy_minutes.empty:
        return _empty_result(
            starting_capital, ["Stock or SPY minute data was unavailable."]
        )

    daily = (
        _daily_from_minutes(minutes)
        if daily_bars is None or daily_bars.empty
        else _normalise_daily(daily_bars)
    )
    if market_daily_bars is None or market_daily_bars.empty:
        spy_daily = _daily_from_minutes(spy_minutes)
        qqq_daily = _daily_from_minutes(qqq_minutes)
    else:
        market_daily = _normalise_daily(market_daily_bars)
        spy_daily = market_daily[market_daily["symbol"] == "SPY"]
        qqq_daily = market_daily[market_daily["symbol"] == "QQQ"]

    if daily.empty or spy_daily.empty:
        return _empty_result(
            starting_capital, ["At least 60 completed daily bars are required."]
        )

    session_dates = sorted(set(minutes["session"]))
    cash = float(starting_capital)
    positions = {}
    pending = []
    trades = []
    signal_log = []
    equity_rows = []
    latest_prices = {}

    for session_date in session_dates:
        all_today = minutes[minutes["session"] == session_date]
        session_times = sorted(all_today["timestamp"].unique())
        evaluated = False

        for raw_timestamp in session_times:
            timestamp = pd.Timestamp(raw_timestamp)
            current_rows = all_today[all_today["timestamp"] == timestamp]
            current_clock = timestamp.tz_convert(ET).time()

            for bar in current_rows.itertuples(index=False):
                latest_prices[bar.symbol] = float(bar.close)
                if bar.symbol in positions:
                    cash, closed = _manage_bar(
                        positions[bar.symbol],
                        bar,
                        cash,
                        commission_bps,
                        slippage_bps,
                    )
                    if closed is not None:
                        trades.append(closed)
                        del positions[bar.symbol]

            # Fill only after the timestamp that generated the signal.
            still_pending = []
            for order in pending:
                if order["signal_time"] >= timestamp or order["symbol"] in positions:
                    still_pending.append(order)
                    continue
                match = current_rows[current_rows["symbol"] == order["symbol"]]
                if match.empty or len(positions) >= max_positions:
                    still_pending.append(order)
                    continue

                bar = match.iloc[0]
                entry = _execution_price(float(bar["open"]), "buy", slippage_bps)
                initial_stop = float(order["planned_stop"])
                if initial_stop >= entry:
                    initial_stop = entry - max(entry * 0.025, 0.01)
                per_share_risk = entry - initial_stop
                equity = _account_equity(cash, positions, latest_prices)
                risk_budget = equity * risk_pct
                shares_by_risk = int(risk_budget / max(per_share_risk, 0.01))
                cost_per_share = entry * (
                    1 + max(commission_bps, 0) / 10_000
                )
                shares_by_cash = int(cash / max(cost_per_share, 0.01))
                shares = min(shares_by_risk, shares_by_cash)
                if shares < 1:
                    continue

                notional = shares * entry
                entry_fee = _fee(notional, commission_bps)
                cash -= notional + entry_fee
                positions[order["symbol"]] = Position(
                    symbol=order["symbol"],
                    signal_date=order["session"],
                    signal_time=order["signal_time"],
                    entry_time=timestamp,
                    signal=order["signal"],
                    setup=order["setup"],
                    swing_score=order["swing_score"],
                    intraday_score=order["intraday_score"],
                    entry=entry,
                    initial_stop=initial_stop,
                    stop=initial_stop,
                    target1=entry + 2 * per_share_risk,
                    target2=entry + 3 * per_share_risk,
                    shares=shares,
                    remaining=shares,
                    entry_fee=entry_fee,
                )
                entry_bar = next(match.itertuples(index=False))
                cash, closed = _manage_bar(
                    positions[order["symbol"]],
                    entry_bar,
                    cash,
                    commission_bps,
                    slippage_bps,
                )
                if closed is not None:
                    trades.append(closed)
                    del positions[order["symbol"]]
            pending = still_pending

            # Evaluate at the first available bar at or after the selected time.
            if not evaluated and current_clock >= scan_clock:
                candidates = _signal_candidates(
                    session_date,
                    current_clock,
                    all_today,
                    daily,
                    spy_minutes,
                    qqq_minutes,
                    spy_daily,
                    qqq_daily,
                )
                signal_log.extend(candidates)
                open_slots = max_positions - len(positions) - len(pending)
                for candidate in candidates:
                    if open_slots <= 0:
                        break
                    if (
                        candidate["signal"] in BUY_SIGNALS
                        and candidate["symbol"] not in positions
                        and all(
                            candidate["symbol"] != order["symbol"]
                            for order in pending
                        )
                    ):
                        candidate["signal_time"] = timestamp
                        pending.append(candidate)
                        open_slots -= 1
                evaluated = True

        # Never carry an unfilled entry order into another day.
        pending.clear()

        # Time exits occur on the final regular-session close.
        final_rows = all_today[all_today["clock"] <= time(16, 0)]
        for symbol in list(positions):
            position = positions[symbol]
            if position.holding_days < max_holding_days:
                continue
            match = final_rows[final_rows["symbol"] == symbol]
            if match.empty:
                continue
            final_bar = match.iloc[-1]
            cash, closed = _close_position(
                position,
                float(final_bar["close"]),
                pd.Timestamp(final_bar["timestamp"]),
                "TIME EXIT",
                cash,
                commission_bps,
                slippage_bps,
            )
            trades.append(closed)
            del positions[symbol]

        _mark_trend_exits(
            positions,
            daily,
            final_rows,
            session_date,
        )

        equity_rows.append(
            {
                "date": session_date,
                "equity": round(
                    _account_equity(cash, positions, latest_prices), 2
                ),
                "cash": round(cash, 2),
                "open_positions": len(positions),
            }
        )

    final_timestamp = minutes["timestamp"].max()
    for symbol in list(positions):
        position = positions[symbol]
        cash, closed = _close_position(
            position,
            latest_prices.get(symbol, position.entry),
            final_timestamp,
            "END OF TEST",
            cash,
            commission_bps,
            slippage_bps,
        )
        trades.append(closed)
        del positions[symbol]

    trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_frame = pd.DataFrame(equity_rows)
    if not equity_frame.empty:
        equity_frame.loc[equity_frame.index[-1], [
            "equity", "cash", "open_positions"
        ]] = [round(cash, 2), round(cash, 2), 0]

    trade_count = len(trades_frame)
    wins = trades_frame[trades_frame["pnl"] > 0] if trade_count else trades_frame
    losses = (
        trades_frame[trades_frame["pnl"] < 0] if trade_count else trades_frame
    )
    gross_profit = float(wins["pnl"].sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if len(losses) else 0.0
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (np.inf if gross_profit > 0 else 0.0)
    )
    if not equity_frame.empty:
        equity_values = pd.Series(
            [float(starting_capital)]
            + equity_frame["equity"].astype(float).tolist()
        )
        peak = equity_values.cummax()
        drawdown = equity_values / peak.replace(0, np.nan) - 1
        max_drawdown = float(drawdown.min() * 100)
    else:
        max_drawdown = 0.0

    stats = {
        "starting_capital": round(float(starting_capital), 2),
        "ending_capital": round(cash, 2),
        "total_return_pct": round((cash / starting_capital - 1) * 100, 2),
        "trades": trade_count,
        "win_rate_pct": round(len(wins) / trade_count * 100, 2)
        if trade_count
        else 0.0,
        "profit_factor": round(profit_factor, 2)
        if np.isfinite(profit_factor)
        else "inf",
        "expectancy_r": round(float(trades_frame["r_multiple"].mean()), 3)
        if trade_count
        else 0.0,
        "avg_trade_dollars": round(float(trades_frame["pnl"].mean()), 2)
        if trade_count
        else 0.0,
        "max_drawdown_pct": round(max_drawdown, 2),
    }
    warnings = []
    if trade_count < 30:
        warnings.append(
            "Fewer than 30 completed trades: the sample is too small for a reliable conclusion."
        )

    signal_frame = pd.DataFrame(signal_log)
    return {
        "trades": trades_frame,
        "equity": equity_frame,
        "stats": stats,
        "signal_log": signal_frame,
        "diagnostics": _build_signal_diagnostics(
            signal_frame,
            trade_count=trade_count,
        ),
        "warnings": warnings,
    }


# Backwards-compatible import name.
backtest = swing_backtest
