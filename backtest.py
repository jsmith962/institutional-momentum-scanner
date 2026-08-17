"""Institutional Swing Scanner v3.5 backtest + validation engine.

Production rules remain unchanged by default. Research profiles are replayed
against the production signal log and must pass holdout + multi-fold validation
before they can be flagged for review.
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
from validation import chronological_validation, full_strategy_validation

ET = ZoneInfo("America/New_York")
BUY_SIGNALS = {"BUY", "A+ SWING BUY"}

PRODUCTION_GATE_CONFIG = {
    "swing_score": float(MIN_SWING_SCORE_BUY),
    "entry_quality": float(MIN_ENTRY_QUALITY_BUY),
    "reward_risk": float(MIN_REWARD_RISK_BUY),
    "market_score": float(MIN_MARKET_SCORE_BUY),
    "leadership_percentile": float(MIN_RS_PERCENTILE_BUY * 100),
    "max_distribution_days": int(MAX_DISTRIBUTION_DAYS_BUY),
    "intraday_score": float(MIN_INTRADAY_CONFIRMATION_SCORE),
}

CALIBRATION_PROFILES = [
    {"name": "PRODUCTION_85_85", **PRODUCTION_GATE_CONFIG},
    {"name": "SCORE_80_I80", **PRODUCTION_GATE_CONFIG, "swing_score": 80.0, "intraday_score": 80.0},
    {"name": "SCORE_77_5_I75", **PRODUCTION_GATE_CONFIG, "swing_score": 77.5, "intraday_score": 75.0},
    {"name": "SCORE_75_I70", **PRODUCTION_GATE_CONFIG, "swing_score": 75.0, "intraday_score": 70.0},
    {"name": "SCORE_72_5_I70", **PRODUCTION_GATE_CONFIG, "swing_score": 72.5, "intraday_score": 70.0},
]

TRADE_COLUMNS = [
    "symbol", "signal_date", "signal_time", "entry_time", "exit_time",
    "signal", "setup", "swing_score", "intraday_score", "entry",
    "initial_stop", "target1", "target2", "shares", "pnl", "return_pct",
    "r_multiple", "holding_days", "exit_reason",
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


def _number(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalise_gate_config(gate_config=None):
    cfg = PRODUCTION_GATE_CONFIG.copy()
    if gate_config:
        for k in cfg:
            if k in gate_config:
                cfg[k] = gate_config[k]
    cfg["swing_score"] = float(cfg["swing_score"])
    cfg["entry_quality"] = float(cfg["entry_quality"])
    cfg["reward_risk"] = float(cfg["reward_risk"])
    cfg["market_score"] = float(cfg["market_score"])
    cfg["leadership_percentile"] = float(cfg["leadership_percentile"])
    cfg["max_distribution_days"] = int(cfg["max_distribution_days"])
    cfg["intraday_score"] = float(cfg["intraday_score"])
    return cfg


def _empty_result(starting_capital, warnings=None, gate_config=None):
    cfg = _normalise_gate_config(gate_config)
    empty = pd.DataFrame(columns=TRADE_COLUMNS)
    return {
        "trades": empty,
        "equity": pd.DataFrame(columns=["date", "equity", "cash", "open_positions"]),
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
            "score_distribution": pd.DataFrame(),
        },
        "validation": chronological_validation(empty),
        "full_validation": full_strategy_validation(empty),
        "gate_config": cfg,
        "warnings": list(warnings or []),
    }


def _candidate_gate_diagnostics(swing, intraday_signal, intraday_score, gate_config=None):
    cfg = _normalise_gate_config(gate_config)
    leadership = swing.get("leadership_percentile")
    leadership_ok = leadership is None or pd.isna(leadership) or _number(leadership) >= cfg["leadership_percentile"]
    gates = {
        "risk_event_clear": not bool(swing.get("risk_flag", False)),
        "not_too_extended": not bool(swing.get("too_extended", False)),
        "swing_score": _number(swing.get("swing_score")) >= cfg["swing_score"],
        "entry_quality": _number(swing.get("entry_quality")) >= cfg["entry_quality"],
        "reward_risk": _number(swing.get("reward_risk")) >= cfg["reward_risk"],
        "market_regime": _number(swing.get("market_score")) >= cfg["market_score"],
        "inside_entry_zone": bool(swing.get("inside_entry_zone", False)),
        "trend_health": bool(swing.get("trend_health", False)),
        "distribution": _number(swing.get("distribution_days"), 999) <= cfg["max_distribution_days"],
        "leadership": bool(leadership_ok),
        "intraday_signal": intraday_signal == "BUY",
        "intraday_score": _number(intraday_score) >= cfg["intraday_score"],
    }
    daily_keys = list(gates)[:-2]
    failed = [k for k, passed in gates.items() if not passed]
    out = {
        "daily_all_gates_passed": all(gates[k] for k in daily_keys),
        "intraday_all_gates_passed": gates["intraday_signal"] and gates["intraday_score"],
        "all_buy_gates_passed": all(gates.values()),
        "daily_gates_passed": sum(gates[k] for k in daily_keys),
        "daily_gate_count": len(daily_keys),
        "buy_gates_passed": sum(gates.values()),
        "buy_gate_count": len(gates),
        "failed_gate_count": len(failed),
        "failed_buy_gates": "; ".join(failed) or "None",
    }
    out.update({f"gate_{k}": bool(v) for k, v in gates.items()})
    return out


def _research_signal(swing, intraday_signal, intraday_score, gate_config):
    diag = _candidate_gate_diagnostics(swing, intraday_signal, intraday_score, gate_config)
    if diag["all_buy_gates_passed"]:
        cfg = _normalise_gate_config(gate_config)
        if _number(swing.get("swing_score")) >= max(cfg["swing_score"] + 5, 87.5) and _number(swing.get("entry_quality")) >= 13:
            return "A+ SWING BUY", "All research BUY gates passed at A+ strength.", diag
        return "BUY", "All research BUY gates passed.", diag
    if bool(swing.get("risk_flag", False)):
        return "AVOID", str(swing.get("risk_reason", "Risk gate active.")), diag
    if bool(swing.get("too_extended", False)):
        return "TOO EXTENDED", "Setup is too extended.", diag
    return "WATCH", "Configured BUY gates did not all pass.", diag


def _normalise_bars(frame):
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "symbol"])
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out["symbol"] = out["symbol"].astype(str).str.upper()
    local = out["timestamp"].dt.tz_convert(ET)
    out["session"] = local.dt.date
    out["clock"] = local.dt.time
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _normalise_daily(frame):
    return _normalise_bars(frame).drop(columns=["session", "clock"], errors="ignore")


def _regular_minutes(frame):
    if frame.empty:
        return frame
    return frame[(frame["clock"] >= time(9, 30)) & (frame["clock"] < time(16, 0))].copy()


def _daily_from_minutes(minutes):
    if minutes.empty:
        return pd.DataFrame()
    regular = minutes[(minutes["clock"] >= time(9, 30)) & (minutes["clock"] <= time(16, 0))]
    if regular.empty:
        return pd.DataFrame()
    return (
        regular.groupby(["symbol", "session"], sort=True)
        .agg(timestamp=("timestamp", "last"), open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
        .reset_index(drop=False).drop(columns="session").sort_values(["timestamp", "symbol"])
    )


def _partial_daily(minutes, symbol):
    if minutes is None or minutes.empty:
        return pd.DataFrame()
    g = minutes.sort_values("timestamp")
    return pd.DataFrame([{
        "timestamp": g["timestamp"].iloc[-1], "symbol": symbol,
        "open": float(g["open"].iloc[0]), "high": float(g["high"].max()),
        "low": float(g["low"].min()), "close": float(g["close"].iloc[-1]),
        "volume": float(g["volume"].sum()),
    }])


def _completed_daily(daily, session_date):
    if daily.empty:
        return daily
    sessions = pd.to_datetime(daily["timestamp"], utc=True).dt.tz_convert(ET).dt.date
    return daily[sessions < session_date].copy()


def _session_slice(frame, session_date, through=None):
    out = frame[frame["session"] == session_date]
    if through is not None:
        out = out[out["clock"] <= through]
    return out.copy()


def _execution_price(raw_price, side, slippage_bps):
    adj = max(float(slippage_bps), 0.0) / 10000
    return float(raw_price) * (1 + adj if side == "buy" else 1 - adj)


def _fee(notional, commission_bps):
    return abs(float(notional)) * max(float(commission_bps), 0.0) / 10000


def _account_equity(cash, positions, prices):
    return cash + sum(p.remaining * prices.get(sym, p.entry) for sym, p in positions.items())


def _trade_record(position, timestamp, reason, total_pnl):
    risk_dollars = max((position.entry - position.initial_stop) * position.shares, 0.01)
    return {
        "symbol": position.symbol, "signal_date": position.signal_date,
        "signal_time": position.signal_time, "entry_time": position.entry_time,
        "exit_time": timestamp, "signal": position.signal, "setup": position.setup,
        "swing_score": round(position.swing_score, 1), "intraday_score": round(position.intraday_score, 1),
        "entry": round(position.entry, 4), "initial_stop": round(position.initial_stop, 4),
        "target1": round(position.target1, 4), "target2": round(position.target2, 4),
        "shares": position.shares, "pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / max(position.entry * position.shares, 0.01) * 100, 2),
        "r_multiple": round(total_pnl / risk_dollars, 3),
        "holding_days": position.holding_days, "exit_reason": reason,
    }


def _close_position(position, raw_price, timestamp, reason, cash, commission_bps, slippage_bps):
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
    fee = _fee(proceeds, commission_bps)
    position.realized_pnl += (fill - position.entry) * exit_shares - fee
    position.remaining -= exit_shares
    cash += proceeds - fee
    position.target1_hit = True
    position.stop = position.entry
    if position.remaining <= 0:
        return cash, _trade_record(position, timestamp, "TARGET 1", position.realized_pnl - position.entry_fee)
    return cash, None


def _manage_bar(position, bar, cash, commission_bps, slippage_bps):
    if position.last_session != bar.session:
        position.holding_days += 1
        position.last_session = bar.session
    if float(bar.open) <= position.stop:
        return _close_position(position, float(bar.open), bar.timestamp, "GAP STOP" if float(bar.open) < position.stop else "STOP", cash, commission_bps, slippage_bps)
    if position.trend_exit_pending:
        return _close_position(position, float(bar.open), bar.timestamp, "TREND EXIT", cash, commission_bps, slippage_bps)
    if position.target1_hit and float(bar.open) >= position.target2:
        return _close_position(position, float(bar.open), bar.timestamp, "GAP TARGET 2", cash, commission_bps, slippage_bps)
    if not position.target1_hit and float(bar.open) >= position.target1:
        cash, closed = _take_target1(position, float(bar.open), bar.timestamp, cash, commission_bps, slippage_bps)
        if closed is not None:
            return cash, closed
        if float(bar.open) >= position.target2:
            return _close_position(position, float(bar.open), bar.timestamp, "GAP TARGET 2", cash, commission_bps, slippage_bps)
    if float(bar.low) <= position.stop:
        return _close_position(position, position.stop, bar.timestamp, "BREAKEVEN STOP" if position.target1_hit else "STOP", cash, commission_bps, slippage_bps)
    if not position.target1_hit and float(bar.high) >= position.target1:
        cash, closed = _take_target1(position, position.target1, bar.timestamp, cash, commission_bps, slippage_bps)
        if closed is not None:
            return cash, closed
        if float(bar.low) <= position.stop:
            return _close_position(position, position.stop, bar.timestamp, "TARGET 1 + BREAKEVEN STOP", cash, commission_bps, slippage_bps)
    if position.target1_hit and float(bar.high) >= position.target2:
        return _close_position(position, position.target2, bar.timestamp, "TARGET 2", cash, commission_bps, slippage_bps)
    return cash, None


def _mark_trend_exits(positions, daily, all_today, session_date):
    if not positions:
        return
    completed = _completed_daily(daily, session_date)
    for symbol, position in positions.items():
        today = all_today[all_today["symbol"] == symbol]
        if today.empty:
            continue
        history = pd.concat([completed[completed["symbol"] == symbol], _partial_daily(today, symbol)], ignore_index=True)
        close = pd.to_numeric(history.get("close"), errors="coerce").dropna()
        if len(close) >= 20:
            e20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            position.trend_exit_pending = bool(float(close.iloc[-1]) < e20)


def _signal_candidates(session_date, scan_clock, bars, daily, spy_minutes, qqq_minutes, spy_daily, qqq_daily, gate_config=None, production_mode=True):
    cfg = _normalise_gate_config(gate_config)
    completed_stocks = _completed_daily(daily, session_date)
    completed_spy = _completed_daily(spy_daily, session_date)
    completed_qqq = _completed_daily(qqq_daily, session_date)
    spy_today = _session_slice(spy_minutes, session_date, scan_clock)
    qqq_today = _session_slice(qqq_minutes, session_date, scan_clock)
    spy_history = pd.concat([completed_spy, _partial_daily(spy_today, "SPY")], ignore_index=True)
    qqq_history = pd.concat([completed_qqq, _partial_daily(qqq_today, "QQQ")], ignore_index=True)

    histories = {}
    for symbol, bars_sym in bars.groupby("symbol"):
        today = _session_slice(bars_sym, session_date, scan_clock)
        prior = completed_stocks[completed_stocks["symbol"] == symbol].copy()
        history = pd.concat([prior, _partial_daily(today, symbol)], ignore_index=True)
        if len(today) >= 20 and len(history) >= 61:
            histories[symbol] = (history, prior, today)
    if not histories:
        return []
    leadership = relative_strength_percentiles(pd.concat([v[0] for v in histories.values()], ignore_index=True))

    rows = []
    for symbol, (history, prior, today) in histories.items():
        avg_share_vol = float(prior["volume"].tail(20).mean())
        avg_dollar_vol = float((prior["close"] * prior["volume"]).tail(20).mean())
        prepared = prepare_intraday(today, spy_today, avg_share_vol)
        if prepared.empty:
            continue
        ir = prepared.iloc[-1]
        intra_score, intra_signal, intra_reasons = classify(ir, avg_dollar_vol)
        swing = score_swing_daily(history, spy_history, qqq_history, leadership.get(symbol))
        if not swing:
            continue
        if production_mode:
            final_signal, decision = combine_daily_intraday_signal(swing["signal"], intra_signal, intra_score, risk_flag=bool(swing.get("risk_flag", False)))
            diag = _candidate_gate_diagnostics(swing, intra_signal, intra_score, cfg)
        else:
            final_signal, decision, diag = _research_signal(swing, intra_signal, intra_score, cfg)
        row = {
            "symbol": symbol, "session": session_date, "signal_time": pd.Timestamp(ir["timestamp"]),
            "signal": final_signal, "daily_signal": swing["signal"], "intraday_signal": intra_signal,
            "swing_score": float(swing["swing_score"]), "intraday_score": float(intra_score),
            "entry_quality": float(swing["entry_quality"]), "setup": swing["setup"],
            "reference_price": float(ir["close"]), "entry_low": _number(swing.get("entry_low")),
            "entry_high": _number(swing.get("entry_high")), "planned_stop": float(swing["stop"]),
            "reward_risk": _number(swing.get("reward_risk")), "market_score": _number(swing.get("market_score")),
            "leadership_percentile": swing.get("leadership_percentile"),
            "distribution_days": int(_number(swing.get("distribution_days"))),
            "trend_health": bool(swing.get("trend_health", False)), "inside_entry_zone": bool(swing.get("inside_entry_zone", False)),
            "too_extended": bool(swing.get("too_extended", False)), "risk_flag": bool(swing.get("risk_flag", False)),
            "risk_reason": swing.get("risk_reason", ""), "decision": decision,
            "intraday_reasons": "; ".join(intra_reasons),
            "configured_swing_score": cfg["swing_score"], "configured_intraday_score": cfg["intraday_score"],
        }
        row.update(diag)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["signal"] not in BUY_SIGNALS, -r["swing_score"], -r["intraday_score"]))


def _score_distribution(signal_log):
    if signal_log is None or signal_log.empty or "swing_score" not in signal_log.columns:
        return pd.DataFrame()
    s = pd.to_numeric(signal_log["swing_score"], errors="coerce").dropna()
    return pd.DataFrame([{"percentile": f"{p*100:.0f}%", "swing_score": round(float(s.quantile(p)), 2)} for p in [0, .1, .25, .5, .75, .9, .95, .99, 1]]) if not s.empty else pd.DataFrame()


def _build_signal_diagnostics(signal_log, trade_count=0, gate_config=None):
    if signal_log is None or signal_log.empty:
        return {"funnel": pd.DataFrame(), "gate_failures": pd.DataFrame(), "near_misses": pd.DataFrame(), "score_distribution": pd.DataFrame()}
    cfg = _normalise_gate_config(gate_config)
    frame = signal_log.copy()
    keys = ["risk_event_clear", "not_too_extended", "swing_score", "entry_quality", "reward_risk", "market_regime", "inside_entry_zone", "trend_health", "distribution", "leadership", "intraday_signal", "intraday_score"]
    for k in keys:
        if f"gate_{k}" not in frame.columns:
            frame[f"gate_{k}"] = False
    daily = frame[[f"gate_{k}" for k in keys[:-2]]].astype(bool).all(axis=1)
    intra_sig = frame["gate_intraday_signal"].astype(bool)
    intra_score = frame["gate_intraday_score"].astype(bool)
    final_buy = frame["signal"].isin(BUY_SIGNALS)
    total = len(frame)
    funnel = pd.DataFrame([
        {"stage": "Candidates evaluated", "count": total},
        {"stage": f"Daily Swing Score at least {cfg['swing_score']:g}", "count": int(frame["gate_swing_score"].astype(bool).sum())},
        {"stage": "All daily BUY gates passed", "count": int(daily.sum())},
        {"stage": "Daily BUY plus intraday BUY signal", "count": int((daily & intra_sig).sum())},
        {"stage": "Fully confirmed BUY", "count": int((daily & intra_sig & intra_score & final_buy).sum())},
        {"stage": "Completed simulated trades", "count": int(trade_count)},
    ])
    funnel["percent_of_candidates"] = (funnel["count"] / max(total, 1) * 100).round(1)
    failures = pd.DataFrame([{"gate": k, "failed": int((~frame[f"gate_{k}"].astype(bool)).sum())} for k in keys]).sort_values("failed", ascending=False)
    near = frame[~final_buy].copy()
    if not near.empty:
        near["gates_passed"] = near["buy_gates_passed"].astype(int).astype(str) + "/" + near["buy_gate_count"].astype(int).astype(str)
        near = near.sort_values(["failed_gate_count", "swing_score", "intraday_score"], ascending=[True, False, False]).drop_duplicates("symbol").head(10)
        cols = [c for c in ["symbol", "session", "signal", "swing_score", "intraday_score", "entry_quality", "gates_passed", "failed_buy_gates"] if c in near.columns]
        near = near[cols]
    return {"funnel": funnel, "gate_failures": failures.reset_index(drop=True), "near_misses": near.reset_index(drop=True), "score_distribution": _score_distribution(frame)}


def _portfolio_stats(trades, equity, starting_capital, ending_cash):
    n = len(trades)
    if n:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0)
        gp, gl = float(pnl[pnl > 0].sum()), abs(float(pnl[pnl < 0].sum()))
        pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
        exp = float(pd.to_numeric(trades["r_multiple"], errors="coerce").dropna().mean())
        avg = float(pnl.mean())
        wr = float((pnl > 0).mean() * 100)
    else:
        pf = exp = avg = wr = 0.0
    if not equity.empty:
        vals = pd.Series([float(starting_capital)] + equity["equity"].astype(float).tolist())
        peak = vals.cummax()
        dd = float((vals / peak.replace(0, np.nan) - 1).min() * 100)
    else:
        dd = 0.0
    return {
        "starting_capital": round(float(starting_capital), 2), "ending_capital": round(float(ending_cash), 2),
        "total_return_pct": round((ending_cash / starting_capital - 1) * 100, 2), "trades": n,
        "win_rate_pct": round(wr, 2), "profit_factor": round(pf, 2) if np.isfinite(pf) else "inf",
        "expectancy_r": round(exp, 3), "avg_trade_dollars": round(avg, 2), "max_drawdown_pct": round(dd, 2),
    }


def _finalize(trades_list, equity_rows, signal_log, starting_capital, ending_cash, gate_config, warnings=None):
    trades = pd.DataFrame(trades_list, columns=TRADE_COLUMNS)
    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity.loc[equity.index[-1], ["equity", "cash", "open_positions"]] = [round(ending_cash, 2), round(ending_cash, 2), 0]
    stats = _portfolio_stats(trades, equity, starting_capital, ending_cash)
    holdout = chronological_validation(trades)
    full = full_strategy_validation(trades, holdout_train_fraction=.70, walk_forward_folds=4, min_total_trades=40)
    warn = list(warnings or [])
    if stats["trades"] < 30:
        warn.append("Fewer than 30 completed trades: sample too small for a reliable conclusion.")
    if stats["trades"] < 40:
        warn.append("v3.5 walk-forward validation needs about 40 trades before it can produce meaningful evidence.")
    signal_frame = pd.DataFrame(signal_log)
    return {"trades": trades, "equity": equity, "stats": stats, "signal_log": signal_frame,
            "diagnostics": _build_signal_diagnostics(signal_frame, stats["trades"], gate_config),
            "validation": holdout, "full_validation": full, "gate_config": _normalise_gate_config(gate_config), "warnings": warn}


def swing_backtest(bars, spy_bars, qqq_bars=None, daily_bars=None, market_daily_bars=None, starting_capital=2000, risk_pct=.005, max_positions=3, max_holding_days=20, scan_time="11:30", slippage_bps=5.0, commission_bps=0.0, gate_config=None, production_mode=True):
    cfg = _normalise_gate_config(gate_config)
    if starting_capital <= 0 or not 0 < risk_pct <= .05 or max_positions < 1 or max_holding_days < 1:
        raise ValueError("Invalid capital, risk, positions, or holding-period setting.")
    try:
        hh, mm = [int(x) for x in scan_time.split(":")]
        scan_clock = time(hh, mm)
    except Exception as exc:
        raise ValueError("Scan time must use HH:MM, e.g. 11:30") from exc
    minutes = _regular_minutes(_normalise_bars(bars))
    spy_minutes = _regular_minutes(_normalise_bars(spy_bars))
    qqq_minutes = _regular_minutes(_normalise_bars(qqq_bars if qqq_bars is not None else pd.DataFrame()))
    if minutes.empty or spy_minutes.empty:
        return _empty_result(starting_capital, ["Stock or SPY minute data unavailable."], cfg)
    daily = _daily_from_minutes(minutes) if daily_bars is None or daily_bars.empty else _normalise_daily(daily_bars)
    if market_daily_bars is None or market_daily_bars.empty:
        spy_daily, qqq_daily = _daily_from_minutes(spy_minutes), _daily_from_minutes(qqq_minutes)
    else:
        md = _normalise_daily(market_daily_bars)
        spy_daily, qqq_daily = md[md["symbol"] == "SPY"], md[md["symbol"] == "QQQ"]
    if daily.empty or spy_daily.empty:
        return _empty_result(starting_capital, ["At least 60 completed daily bars are required."], cfg)

    cash = float(starting_capital); positions = {}; pending = []; trades = []; signal_log = []; equity_rows = []; latest = {}
    for session_date in sorted(set(minutes["session"])):
        day = minutes[minutes["session"] == session_date]
        evaluated = False
        for ts_raw in sorted(day["timestamp"].unique()):
            ts = pd.Timestamp(ts_raw); rows = day[day["timestamp"] == ts]; clock = ts.tz_convert(ET).time()
            for bar in rows.itertuples(index=False):
                latest[bar.symbol] = float(bar.close)
                if bar.symbol in positions:
                    cash, closed = _manage_bar(positions[bar.symbol], bar, cash, commission_bps, slippage_bps)
                    if closed is not None:
                        trades.append(closed); del positions[bar.symbol]
            still = []
            for order in pending:
                if order["signal_time"] >= ts or order["symbol"] in positions:
                    still.append(order); continue
                match = rows[rows["symbol"] == order["symbol"]]
                if match.empty or len(positions) >= max_positions:
                    still.append(order); continue
                b = match.iloc[0]; entry = _execution_price(float(b["open"]), "buy", slippage_bps)
                stop = float(order["planned_stop"])
                if stop >= entry: stop = entry - max(entry * .025, .01)
                risk_share = entry - stop; equity = _account_equity(cash, positions, latest); budget = equity * risk_pct
                shares = min(int(budget / max(risk_share, .01)), int(cash / max(entry * (1 + max(commission_bps, 0) / 10000), .01)))
                if shares < 1: continue
                fee = _fee(shares * entry, commission_bps); cash -= shares * entry + fee
                positions[order["symbol"]] = Position(order["symbol"], order["session"], order["signal_time"], ts, order["signal"], order["setup"], order["swing_score"], order["intraday_score"], entry, stop, stop, entry + 2*risk_share, entry + 3*risk_share, shares, shares, fee)
                entry_bar = next(match.itertuples(index=False)); cash, closed = _manage_bar(positions[order["symbol"]], entry_bar, cash, commission_bps, slippage_bps)
                if closed is not None: trades.append(closed); del positions[order["symbol"]]
            pending = still
            if not evaluated and clock >= scan_clock:
                candidates = _signal_candidates(session_date, clock, day, daily, spy_minutes, qqq_minutes, spy_daily, qqq_daily, cfg, production_mode)
                signal_log.extend(candidates)
                slots = max_positions - len(positions) - len(pending)
                for c in candidates:
                    if slots <= 0: break
                    if c["signal"] in BUY_SIGNALS and c["symbol"] not in positions and all(c["symbol"] != p["symbol"] for p in pending):
                        c["signal_time"] = ts; pending.append(c); slots -= 1
                evaluated = True
        pending.clear()
        final_rows = day[day["clock"] <= time(16, 0)]
        for symbol in list(positions):
            p = positions[symbol]
            if p.holding_days >= max_holding_days:
                match = final_rows[final_rows["symbol"] == symbol]
                if not match.empty:
                    b = match.iloc[-1]; cash, closed = _close_position(p, float(b["close"]), pd.Timestamp(b["timestamp"]), "TIME EXIT", cash, commission_bps, slippage_bps)
                    trades.append(closed); del positions[symbol]
        _mark_trend_exits(positions, daily, final_rows, session_date)
        equity_rows.append({"date": session_date, "equity": round(_account_equity(cash, positions, latest), 2), "cash": round(cash, 2), "open_positions": len(positions)})
    final_ts = minutes["timestamp"].max()
    for symbol in list(positions):
        p = positions[symbol]; cash, closed = _close_position(p, latest.get(symbol, p.entry), final_ts, "END OF TEST", cash, commission_bps, slippage_bps)
        trades.append(closed); del positions[symbol]
    return _finalize(trades, equity_rows, signal_log, starting_capital, cash, cfg)


backtest = swing_backtest


def _signal_row_to_swing(row):
    return {"signal": row.get("daily_signal", row.get("signal", "WATCH")), "swing_score": _number(row.get("swing_score")), "entry_quality": _number(row.get("entry_quality")), "entry_low": _number(row.get("entry_low")), "entry_high": _number(row.get("entry_high")), "stop": _number(row.get("planned_stop")), "reward_risk": _number(row.get("reward_risk")), "market_score": _number(row.get("market_score")), "leadership_percentile": row.get("leadership_percentile"), "distribution_days": int(_number(row.get("distribution_days"))), "trend_health": bool(row.get("trend_health", False)), "inside_entry_zone": bool(row.get("inside_entry_zone", False)), "too_extended": bool(row.get("too_extended", False)), "risk_flag": bool(row.get("risk_flag", False)), "risk_reason": str(row.get("risk_reason", "") or ""), "setup": row.get("setup", "")}


def replay_signal_log_backtest(bars, production_signal_log, daily_bars=None, starting_capital=2000, risk_pct=.005, max_positions=3, max_holding_days=20, slippage_bps=5.0, commission_bps=0.0, gate_config=None):
    cfg = _normalise_gate_config(gate_config)
    minutes = _regular_minutes(_normalise_bars(bars)); daily = _daily_from_minutes(minutes) if daily_bars is None or daily_bars.empty else _normalise_daily(daily_bars)
    if minutes.empty or production_signal_log is None or production_signal_log.empty:
        return _empty_result(starting_capital, ["No minute bars or production signal log available for replay."], cfg)
    signals = production_signal_log.copy()
    if "session" not in signals.columns:
        return _empty_result(starting_capital, ["Signal log missing session column."], cfg)
    signals["session"] = pd.to_datetime(signals["session"], errors="coerce").dt.date
    signals = signals[signals["session"].notna()].copy()
    if "signal_time" in signals.columns: signals["signal_time"] = pd.to_datetime(signals["signal_time"], errors="coerce", utc=True)
    cash = float(starting_capital); positions = {}; pending = []; trades = []; replay = []; equity_rows = []; latest = {}
    by_session = {d: g.copy() for d, g in signals.groupby("session", sort=False)}
    for session_date in sorted(set(minutes["session"])):
        day = minutes[minutes["session"] == session_date]; today_signals = by_session.get(session_date, pd.DataFrame()); evaluated = False
        eval_time = today_signals["signal_time"].dropna().min() if (not today_signals.empty and "signal_time" in today_signals.columns and not today_signals["signal_time"].dropna().empty) else None
        for ts_raw in sorted(day["timestamp"].unique()):
            ts = pd.Timestamp(ts_raw); rows = day[day["timestamp"] == ts]
            for bar in rows.itertuples(index=False):
                latest[bar.symbol] = float(bar.close)
                if bar.symbol in positions:
                    cash, closed = _manage_bar(positions[bar.symbol], bar, cash, commission_bps, slippage_bps)
                    if closed is not None: trades.append(closed); del positions[bar.symbol]
            still = []
            for order in pending:
                if order["signal_time"] >= ts or order["symbol"] in positions: still.append(order); continue
                match = rows[rows["symbol"] == order["symbol"]]
                if match.empty or len(positions) >= max_positions: still.append(order); continue
                b = match.iloc[0]; entry = _execution_price(float(b["open"]), "buy", slippage_bps); stop = float(order["planned_stop"])
                if stop >= entry: stop = entry - max(entry * .025, .01)
                risk_share = entry - stop; equity = _account_equity(cash, positions, latest); shares = min(int(equity * risk_pct / max(risk_share, .01)), int(cash / max(entry, .01)))
                if shares < 1: continue
                fee = _fee(shares * entry, commission_bps); cash -= shares * entry + fee
                positions[order["symbol"]] = Position(order["symbol"], order["session"], order["signal_time"], ts, order["signal"], order.get("setup", ""), _number(order.get("swing_score")), _number(order.get("intraday_score")), entry, stop, stop, entry + 2*risk_share, entry + 3*risk_share, shares, shares, fee)
                entry_bar = next(match.itertuples(index=False)); cash, closed = _manage_bar(positions[order["symbol"]], entry_bar, cash, commission_bps, slippage_bps)
                if closed is not None: trades.append(closed); del positions[order["symbol"]]
            pending = still
            if not evaluated and not today_signals.empty and (eval_time is None or ts >= eval_time):
                candidates = []
                for _, source in today_signals.iterrows():
                    row = source.to_dict(); swing = _signal_row_to_swing(row); intra_signal = str(row.get("intraday_signal", "NO BUY")); intra_score = _number(row.get("intraday_score"))
                    final_signal, decision, diag = _research_signal(swing, intra_signal, intra_score, cfg)
                    row.update(diag); row["signal"] = final_signal; row["decision"] = decision; row["signal_time"] = ts; row["planned_stop"] = _number(row.get("planned_stop", swing.get("stop")))
                    replay.append(row.copy()); candidates.append(row)
                candidates.sort(key=lambda r: (r.get("signal") not in BUY_SIGNALS, -_number(r.get("swing_score")), -_number(r.get("intraday_score"))))
                slots = max_positions - len(positions) - len(pending)
                for c in candidates:
                    if slots <= 0: break
                    if c.get("signal") in BUY_SIGNALS and c.get("symbol") not in positions and all(c.get("symbol") != p.get("symbol") for p in pending): pending.append(c); slots -= 1
                evaluated = True
        pending.clear(); final_rows = day[day["clock"] <= time(16, 0)]
        for symbol in list(positions):
            p = positions[symbol]
            if p.holding_days >= max_holding_days:
                match = final_rows[final_rows["symbol"] == symbol]
                if not match.empty:
                    b = match.iloc[-1]; cash, closed = _close_position(p, float(b["close"]), pd.Timestamp(b["timestamp"]), "TIME EXIT", cash, commission_bps, slippage_bps); trades.append(closed); del positions[symbol]
        _mark_trend_exits(positions, daily, final_rows, session_date)
        equity_rows.append({"date": session_date, "equity": round(_account_equity(cash, positions, latest), 2), "cash": round(cash, 2), "open_positions": len(positions)})
    final_ts = minutes["timestamp"].max()
    for symbol in list(positions):
        p = positions[symbol]; cash, closed = _close_position(p, latest.get(symbol, p.entry), final_ts, "END OF TEST", cash, commission_bps, slippage_bps); trades.append(closed); del positions[symbol]
    return _finalize(trades, equity_rows, replay, starting_capital, cash, cfg)


def build_adaptive_calibration_profiles(production_result, max_profiles=12):
    log = (production_result or {}).get("signal_log", pd.DataFrame())
    if log is None or log.empty: return CALIBRATION_PROFILES[:max_profiles]
    swing = pd.to_numeric(log.get("swing_score"), errors="coerce").dropna(); intra = pd.to_numeric(log.get("intraday_score"), errors="coerce").dropna()
    half = lambda x: round(float(x) * 2) / 2; five = lambda x: round(float(x) / 5) * 5
    s_levels = [85.0] + ([half(swing.quantile(q)) for q in [.99, .975, .95, .9, .8]] if not swing.empty else []) + [80, 77.5, 75, 72.5, 70, 67.5, 65]
    i_levels = [85.0] + ([five(intra.quantile(q)) for q in [.99, .975, .95, .9, .8]] if not intra.empty else []) + [80, 75, 70, 65, 60]
    s_levels = sorted({min(85., max(65., x)) for x in s_levels}, reverse=True); i_levels = sorted({min(85., max(60., x)) for x in i_levels}, reverse=True)
    profiles = [{"name": "PRODUCTION_85_85", **PRODUCTION_GATE_CONFIG}]; seen = {(85., 85., 10., 70.)}
    pairs = [(s_levels[min(i, len(s_levels)-1)], i_levels[min(i, len(i_levels)-1)], 10., 70.) for i in range(max(len(s_levels), len(i_levels)))]
    base_s = min(75., max(65., half(swing.quantile(.9)) if not swing.empty else 72.5)); base_i = min(75., max(60., five(intra.quantile(.9)) if not intra.empty else 70.))
    pairs += [(base_s, base_i, 12., 70.), (base_s, base_i, 10., 75.), (base_s, base_i, 10., 65.)]
    for s, i, q, l in pairs:
        key = (float(s), float(i), float(q), float(l))
        if key in seen: continue
        seen.add(key); profiles.append({"name": f"R_S{s:g}_I{i:g}_Q{q:g}_L{l:g}".replace(".", "_"), **PRODUCTION_GATE_CONFIG, "swing_score": float(s), "intraday_score": float(i), "entry_quality": float(q), "leadership_percentile": float(l)})
        if len(profiles) >= max_profiles: break
    return profiles


def _comparison_row(name, config, result, production=False):
    stats = result.get("stats", {}); hold = result.get("validation", {}); full = result.get("full_validation", {}); wf = full.get("walk_forward", {}) if isinstance(full, dict) else {}
    return {"profile": name, "production_rules": bool(production), "swing_score_gate": config.get("swing_score"), "intraday_score_gate": config.get("intraday_score"), "entry_quality_gate": config.get("entry_quality"), "leadership_gate": config.get("leadership_percentile"), "trades": stats.get("trades", 0), "return_pct": stats.get("total_return_pct", 0), "win_rate_pct": stats.get("win_rate_pct", 0), "profit_factor": stats.get("profit_factor", 0), "expectancy_r": stats.get("expectancy_r", 0), "max_drawdown_pct": stats.get("max_drawdown_pct", 0), "out_of_sample_trades": hold.get("out_of_sample_trades", 0), "out_of_sample_expectancy_r": hold.get("out_of_sample_expectancy_r", 0), "out_of_sample_profit_factor": hold.get("out_of_sample_profit_factor", 0), "validation_verdict": full.get("validation_verdict", "INSUFFICIENT EVIDENCE"), "confidence_grade": full.get("confidence_grade", "INSUFFICIENT"), "validation_pass": bool(full.get("validation_pass", False)), "promotion_candidate": bool(full.get("promotion_candidate", False)), "positive_fold_ratio": full.get("positive_fold_ratio", 0), "aggregate_oos_expectancy_r": full.get("aggregate_oos_expectancy_r", 0), "aggregate_oos_profit_factor": full.get("aggregate_oos_profit_factor", 0), "worst_fold_expectancy_r": full.get("worst_fold_expectancy_r", 0), "bootstrap_expectancy_low_r": full.get("bootstrap_expectancy_low_r"), "walk_forward_folds": wf.get("folds_completed", 0)}


def _pf(v):
    try: return np.inf if str(v).lower() == "inf" else float(v)
    except Exception: return 0.0


def _rank_profiles(df):
    if df is None or df.empty: return pd.DataFrame()
    out = df.copy(); out["_pf"] = out["aggregate_oos_profit_factor"].map(_pf)
    for c in ["trades", "positive_fold_ratio", "aggregate_oos_expectancy_r", "worst_fold_expectancy_r", "bootstrap_expectancy_low_r", "walk_forward_folds", "max_drawdown_pct"]: out[c] = pd.to_numeric(out[c], errors="coerce")
    out["research_eligible"] = out["trades"].fillna(0).ge(40) & out["walk_forward_folds"].fillna(0).ge(3) & out["positive_fold_ratio"].fillna(0).ge(.60) & out["aggregate_oos_expectancy_r"].fillna(-999).gt(0) & out["_pf"].ge(1.15) & out["max_drawdown_pct"].abs().fillna(999).le(20)
    out["promotion_candidate"] = out["promotion_candidate"].fillna(False).astype(bool) & out["research_eligible"] & out["bootstrap_expectancy_low_r"].fillna(-999).gt(0) & out["aggregate_oos_expectancy_r"].fillna(-999).ge(.10) & out["_pf"].ge(1.30) & out["positive_fold_ratio"].fillna(0).ge(.75) & out["worst_fold_expectancy_r"].fillna(-999).ge(-.25)
    out["research_score"] = (200*out["promotion_candidate"].astype(int) + 80*out["research_eligible"].astype(int) + 50*out["validation_pass"].fillna(False).astype(bool).astype(int) + 40*out["positive_fold_ratio"].fillna(0).clip(0,1) + 20*out["aggregate_oos_expectancy_r"].fillna(-1).clip(-1,1) + 8*out["_pf"].replace(np.inf,3).clip(0,3) + np.minimum(out["trades"].fillna(0),100)/10 - out["max_drawdown_pct"].abs().fillna(50)/10).round(3)
    return out.sort_values(["promotion_candidate", "research_eligible", "validation_pass", "research_score"], ascending=[False, False, False, False]).drop(columns="_pf").reset_index(drop=True)


def calibrate_thresholds_adaptive(bars, spy_bars, qqq_bars=None, daily_bars=None, market_daily_bars=None, starting_capital=2000, risk_pct=.005, max_positions=3, max_holding_days=20, scan_time="11:30", slippage_bps=5., commission_bps=0., production_result=None, max_profiles=12):
    if production_result is None:
        production_result = swing_backtest(bars, spy_bars, qqq_bars=qqq_bars, daily_bars=daily_bars, market_daily_bars=market_daily_bars, starting_capital=starting_capital, risk_pct=risk_pct, max_positions=max_positions, max_holding_days=max_holding_days, scan_time=scan_time, slippage_bps=slippage_bps, commission_bps=commission_bps)
    profiles = build_adaptive_calibration_profiles(production_result, max_profiles=max_profiles); log = production_result.get("signal_log", pd.DataFrame()); detailed = {"PRODUCTION_85_85": production_result}; rows = [_comparison_row("PRODUCTION_85_85", PRODUCTION_GATE_CONFIG, production_result, True)]
    for p in profiles:
        name = p.get("name", "UNNAMED")
        if name == "PRODUCTION_85_85": continue
        cfg = {k:v for k,v in p.items() if k != "name"}
        result = replay_signal_log_backtest(bars, log, daily_bars=daily_bars, starting_capital=starting_capital, risk_pct=risk_pct, max_positions=max_positions, max_holding_days=max_holding_days, slippage_bps=slippage_bps, commission_bps=commission_bps, gate_config=cfg)
        detailed[name] = result; rows.append(_comparison_row(name, cfg, result, False))
    comparison = _rank_profiles(pd.DataFrame(rows))
    return {"comparison": comparison, "results": detailed, "score_distribution": production_result.get("diagnostics", {}).get("score_distribution", pd.DataFrame()), "profiles_tested": len(profiles), "production_result": production_result, "adaptive_profiles": pd.DataFrame(profiles), "engine": "FAST_SIGNAL_REPLAY_V3_5", "recommendation": "REVIEW_PROMOTION_CANDIDATES" if (not comparison.empty and comparison["promotion_candidate"].fillna(False).any()) else "KEEP_PRODUCTION_RULES"}


def calibrate_thresholds(*args, **kwargs):
    """Backward-compatible calibration entry point; uses v3.5 adaptive replay."""
    return calibrate_thresholds_adaptive(*args, **kwargs)
