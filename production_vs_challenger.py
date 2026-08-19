"""
Institutional Swing Scanner v3.8
Production-vs-Challenger Portfolio Validation Engine

RESEARCH ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

from forward_research import build_gate_matrix

RESEARCH_VERSION = "v3.8"


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    swing_threshold: float
    intraday_threshold: float
    entry_quality: float = 10.0
    reward_risk: float = 2.0
    market_score: float = 5.0
    leadership: float = 70.0
    max_distribution_days: int = 4
    require_production_intraday_label: bool = True
    require_risk_event_clear: bool = True
    require_not_too_extended: bool = True
    require_inside_entry_zone: bool = True
    require_trend_health: bool = True

    def to_dict(self):
        return asdict(self)


PRODUCTION_PROFILE = ValidationProfile(
    name="Production 85/85",
    swing_threshold=85.0,
    intraday_threshold=85.0,
    require_production_intraday_label=True,
)

DEFAULT_CHALLENGERS = (
    ValidationProfile(
        name="Challenger 70/50",
        swing_threshold=70.0,
        intraday_threshold=50.0,
        require_production_intraday_label=False,
    ),
    ValidationProfile(
        name="Challenger 72.5/50",
        swing_threshold=72.5,
        intraday_threshold=50.0,
        require_production_intraday_label=False,
    ),
    ValidationProfile(
        name="Challenger 70/60",
        swing_threshold=70.0,
        intraday_threshold=60.0,
        require_production_intraday_label=False,
    ),
)


def _safe_df(value):
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _date_series(frame: pd.DataFrame):
    for column in ("session", "signal_date", "date"):
        if column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce")
            try:
                return values.dt.date
            except Exception:
                pass
    if "signal_time" in frame.columns:
        values = pd.to_datetime(frame["signal_time"], errors="coerce", utc=True)
        try:
            return values.dt.date
        except Exception:
            pass
    return pd.Series(pd.NaT, index=frame.index)


def _profile_gate_matrix(df: pd.DataFrame, profile: ValidationProfile):
    gates = build_gate_matrix(
        df,
        swing_threshold=profile.swing_threshold,
        intraday_threshold=profile.intraday_threshold,
        entry_quality_threshold=profile.entry_quality,
        reward_risk_threshold=profile.reward_risk,
        market_score_threshold=profile.market_score,
        leadership_threshold=profile.leadership,
        max_distribution_days=profile.max_distribution_days,
        require_production_intraday_label=profile.require_production_intraday_label,
    )
    if gates.empty:
        return gates
    optional = {
        "risk_event_clear": profile.require_risk_event_clear,
        "not_too_extended": profile.require_not_too_extended,
        "inside_entry_zone": profile.require_inside_entry_zone,
        "trend_health": profile.require_trend_health,
    }
    for gate, required in optional.items():
        if gate in gates.columns and not required:
            gates[gate] = True
    return gates


def select_profile_candidates(enriched_signal_log, profile):
    df = _safe_df(enriched_signal_log)
    if df.empty:
        return pd.DataFrame()
    gates = _profile_gate_matrix(df, profile)
    if gates.empty:
        return pd.DataFrame()
    selected = df.loc[gates.all(axis=1)].copy()
    selected["profile"] = profile.name
    selected["profile_swing_threshold"] = profile.swing_threshold
    selected["profile_intraday_threshold"] = profile.intraday_threshold
    selected["_session"] = _date_series(selected)
    if "symbol" in selected.columns:
        selected["symbol"] = selected["symbol"].astype(str).str.upper()
        sort_cols = ["_session"]
        ascending = [True]
        for c in ("swing_score", "entry_quality", "intraday_score"):
            if c in selected.columns:
                sort_cols.append(c)
                ascending.append(False)
        selected = (
            selected.sort_values(sort_cols, ascending=ascending)
            .drop_duplicates(subset=["symbol", "_session"], keep="first")
        )
    return selected.reset_index(drop=True)


def _trade_return_column(holding_sessions):
    valid = [1, 3, 5, 10, 20]
    target = min(valid, key=lambda x: abs(x - int(holding_sessions)))
    return f"forward_{target}d_pct"


def _max_drawdown(values):
    s = pd.Series(values, dtype=float)
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (s / peak - 1.0) * 100.0
    return float(dd.min())


def _profit_factor(pnls):
    s = pd.to_numeric(pnls, errors="coerce").dropna()
    if s.empty:
        return 0.0
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def replay_profile_portfolio(
    candidates,
    starting_capital=2000.0,
    max_positions=3,
    risk_pct=0.005,
    holding_sessions=10,
    slippage_bps=5.0,
    commission_bps=0.0,
):
    df = _safe_df(candidates)
    empty = {
        "trades": pd.DataFrame(),
        "equity": pd.DataFrame(columns=["session", "equity"]),
        "stats": {
            "starting_capital": float(starting_capital),
            "ending_capital": float(starting_capital),
            "total_return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pct": 0.0,
            "avg_trade_dollars": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy_pct": 0.0,
        },
    }
    if df.empty:
        return empty
    return_col = _trade_return_column(holding_sessions)
    if return_col not in df.columns:
        return empty

    df["_session"] = _date_series(df)
    df["_return_pct"] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna(subset=["_session", "_return_pct"]).copy()
    if df.empty:
        return empty

    sort_cols = ["_session"]
    ascending = [True]
    for c in ("swing_score", "entry_quality", "intraday_score"):
        if c in df.columns:
            sort_cols.append(c)
            ascending.append(False)
    df = df.sort_values(sort_cols, ascending=ascending)

    sessions = sorted(pd.Series(df["_session"].dropna().unique()).tolist())
    session_to_idx = {s: i for i, s in enumerate(sessions)}
    capital = float(starting_capital)
    active_until = {}
    trades = []
    equity_rows = []
    transaction_cost_pct = 2.0 * (float(slippage_bps) + float(commission_bps)) / 100.0

    for session in sessions:
        sidx = session_to_idx[session]
        active_until = {k: v for k, v in active_until.items() if v >= sidx}
        day = df[df["_session"] == session]
        available_slots = max(0, int(max_positions) - len(active_until))
        if available_slots > 0:
            for _, row in day.iterrows():
                if available_slots <= 0:
                    break
                symbol = str(row.get("symbol", "")).upper()
                if not symbol or symbol in active_until:
                    continue
                raw_return_pct = float(row["_return_pct"])
                net_return_pct = raw_return_pct - transaction_cost_pct
                equal_slot_capital = capital / max(int(max_positions), 1)
                risk_based_capital = capital * float(risk_pct) / 0.05
                allocation = min(equal_slot_capital, risk_based_capital)
                if allocation <= 0:
                    continue
                pnl = allocation * net_return_pct / 100.0
                capital += pnl
                trades.append({
                    "profile": row.get("profile", ""),
                    "symbol": symbol,
                    "session": session,
                    "holding_sessions": int(holding_sessions),
                    "return_column": return_col,
                    "gross_return_pct": raw_return_pct,
                    "net_return_pct": net_return_pct,
                    "allocation": allocation,
                    "pnl": pnl,
                    "ending_capital_after_trade": capital,
                    "swing_score": row.get("swing_score"),
                    "intraday_score": row.get("intraday_score"),
                    "entry_quality": row.get("entry_quality"),
                    "reward_risk": row.get("reward_risk"),
                    "market_score": row.get("market_score"),
                    "leadership_percentile": row.get("leadership_percentile"),
                    "distribution_days": row.get("distribution_days"),
                    "setup": row.get("setup"),
                })
                active_until[symbol] = sidx + max(int(holding_sessions), 1) - 1
                available_slots -= 1
        equity_rows.append({"session": session, "equity": capital})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if trades_df.empty:
        return {**empty, "equity": equity_df}

    pnls = pd.to_numeric(trades_df["pnl"], errors="coerce")
    net_returns = pd.to_numeric(trades_df["net_return_pct"], errors="coerce")
    stats = {
        "starting_capital": float(starting_capital),
        "ending_capital": float(capital),
        "total_return_pct": (capital / float(starting_capital) - 1.0) * 100.0,
        "trades": int(len(trades_df)),
        "win_rate_pct": float((pnls > 0).mean() * 100.0),
        "profit_factor": _profit_factor(pnls),
        "avg_trade_pct": float(net_returns.mean()),
        "avg_trade_dollars": float(pnls.mean()),
        "max_drawdown_pct": _max_drawdown(equity_df["equity"]),
        "expectancy_pct": float(net_returns.mean()),
    }
    return {"trades": trades_df, "equity": equity_df, "stats": stats}


def chronological_split(frame, development_fraction=0.70):
    df = _safe_df(frame)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["_session"] = _date_series(df)
    df = df.dropna(subset=["_session"]).sort_values("_session")
    sessions = sorted(df["_session"].unique())
    if len(sessions) < 2:
        return df.copy(), pd.DataFrame()
    split_idx = min(max(int(len(sessions) * float(development_fraction)), 1), len(sessions)-1)
    split_session = sessions[split_idx]
    return (
        df[df["_session"] < split_session].copy(),
        df[df["_session"] >= split_session].copy(),
    )


def _summary_row(profile, candidates, dev, oos, full):
    fs = full.get("stats", {})
    ds = dev.get("stats", {})
    os_ = oos.get("stats", {})
    return {
        "profile": profile.name,
        "production_control": profile.name == PRODUCTION_PROFILE.name,
        "swing_threshold": profile.swing_threshold,
        "intraday_threshold": profile.intraday_threshold,
        "require_production_intraday_label": profile.require_production_intraday_label,
        "candidate_observations": len(candidates),
        "full_trades": fs.get("trades", 0),
        "full_return_pct": fs.get("total_return_pct", 0.0),
        "full_win_rate_pct": fs.get("win_rate_pct", 0.0),
        "full_profit_factor": fs.get("profit_factor", 0.0),
        "full_expectancy_pct": fs.get("expectancy_pct", 0.0),
        "full_max_drawdown_pct": fs.get("max_drawdown_pct", 0.0),
        "dev_trades": ds.get("trades", 0),
        "dev_return_pct": ds.get("total_return_pct", 0.0),
        "dev_profit_factor": ds.get("profit_factor", 0.0),
        "oos_trades": os_.get("trades", 0),
        "oos_return_pct": os_.get("total_return_pct", 0.0),
        "oos_win_rate_pct": os_.get("win_rate_pct", 0.0),
        "oos_profit_factor": os_.get("profit_factor", 0.0),
        "oos_expectancy_pct": os_.get("expectancy_pct", 0.0),
        "oos_max_drawdown_pct": os_.get("max_drawdown_pct", 0.0),
    }


def _promotion_decision(summary, minimum_oos_trades=20):
    if summary.empty:
        return {"status": "INSUFFICIENT", "message": "No profile summary is available.", "best_challenger": None}
    production_rows = summary[summary["production_control"] == True]
    if production_rows.empty:
        return {"status": "INSUFFICIENT", "message": "Production control profile is missing.", "best_challenger": None}
    production = production_rows.iloc[0]
    challengers = summary[summary["production_control"] == False].copy()
    challengers = challengers[pd.to_numeric(challengers["oos_trades"], errors="coerce").fillna(0) >= int(minimum_oos_trades)]
    if challengers.empty:
        return {
            "status": "INSUFFICIENT",
            "message": f"No challenger reached the minimum of {minimum_oos_trades} out-of-sample replay trades.",
            "best_challenger": None,
        }
    challengers["_evidence_score"] = 0
    checks = [
        ("oos_return_pct", ">"),
        ("oos_profit_factor", ">"),
        ("oos_expectancy_pct", ">"),
        ("oos_win_rate_pct", ">"),
    ]
    for col, _ in checks:
        challengers.loc[challengers[col] > float(production[col]), "_evidence_score"] += 1
    challengers.loc[challengers["oos_max_drawdown_pct"] >= float(production["oos_max_drawdown_pct"]), "_evidence_score"] += 1
    best = challengers.sort_values(["_evidence_score", "oos_return_pct", "oos_profit_factor", "oos_trades"], ascending=[False, False, False, False]).iloc[0]
    score = int(best["_evidence_score"])
    name = str(best["profile"])
    if score >= 4 and float(best["oos_return_pct"]) > 0:
        status = "PROMISING"
        message = f"{name} beat production on {score}/5 out-of-sample dimensions. Keep production live; this challenger deserves paper trading."
    elif score >= 3:
        status = "MIXED"
        message = f"{name} showed mixed evidence ({score}/5 dimensions). Do not promote yet."
    else:
        status = "REJECT"
        message = "No challenger showed enough out-of-sample evidence to justify promotion over production."
    return {"status": status, "message": message, "best_challenger": name, "evidence_score": score, "minimum_oos_trades": int(minimum_oos_trades)}


def run_production_vs_challenger_validation(
    enriched_signal_log,
    challengers: Iterable[ValidationProfile] = DEFAULT_CHALLENGERS,
    starting_capital=2000.0,
    max_positions=3,
    risk_pct=0.005,
    holding_sessions=10,
    slippage_bps=5.0,
    commission_bps=0.0,
    development_fraction=0.70,
    minimum_oos_trades=20,
):
    df = _safe_df(enriched_signal_log)
    if df.empty:
        return {"status": "NO_DATA", "version": RESEARCH_VERSION, "message": "No enriched historical signal log was supplied."}
    required = _trade_return_column(holding_sessions)
    if required not in df.columns:
        return {"status": "NO_FORWARD_RETURNS", "version": RESEARCH_VERSION, "message": f"{required} is missing. Run v3.7 forward-return research first."}

    profiles = [PRODUCTION_PROFILE, *list(challengers)]
    profile_results = {}
    rows = []
    for profile in profiles:
        candidates = select_profile_candidates(df, profile)
        dev_cand, oos_cand = chronological_split(candidates, development_fraction)
        full = replay_profile_portfolio(candidates, starting_capital, max_positions, risk_pct, holding_sessions, slippage_bps, commission_bps)
        dev = replay_profile_portfolio(dev_cand, starting_capital, max_positions, risk_pct, holding_sessions, slippage_bps, commission_bps)
        oos = replay_profile_portfolio(oos_cand, starting_capital, max_positions, risk_pct, holding_sessions, slippage_bps, commission_bps)
        profile_results[profile.name] = {
            "profile": profile.to_dict(),
            "candidates": candidates,
            "development_candidates": dev_cand,
            "oos_candidates": oos_cand,
            "full_replay": full,
            "development_replay": dev,
            "oos_replay": oos,
        }
        rows.append(_summary_row(profile, candidates, dev, oos, full))

    summary = pd.DataFrame(rows)
    promotion = _promotion_decision(summary, minimum_oos_trades)
    return {
        "status": "COMPLETE",
        "version": RESEARCH_VERSION,
        "message": f"v3.8 production-vs-challenger validation completed across {len(df):,} enriched historical observations.",
        "settings": {
            "starting_capital": float(starting_capital),
            "max_positions": int(max_positions),
            "risk_pct": float(risk_pct),
            "holding_sessions": int(holding_sessions),
            "slippage_bps": float(slippage_bps),
            "commission_bps": float(commission_bps),
            "development_fraction": float(development_fraction),
            "minimum_oos_trades": int(minimum_oos_trades),
        },
        "summary": summary,
        "promotion": promotion,
        "profile_results": profile_results,
    }
