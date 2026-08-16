"""v3.4 calibration analytics for Institutional Swing Scanner.

Research only. Nothing in this module changes live production thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np
import pandas as pd

DEFAULT_SWING_THRESHOLDS = (65, 70, 72.5, 75, 77.5, 80, 82.5, 85)
DEFAULT_INTRADAY_THRESHOLDS = (40, 50, 60, 65, 70, 75, 80, 85)
OUTCOME_COLUMNS = ("forward_r", "hypothetical_r", "outcome_r", "r_multiple")

@dataclass(frozen=True)
class CalibrationConfig:
    oos_fraction: float = 0.30
    min_oos_observations: int = 20
    min_total_observations: int = 40
    stability_radius: int = 1


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((n for n in names if n in df.columns), None)


def _to_bool(series: pd.Series, default: bool = True) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(1 if default else 0).astype(float).ne(0)
    mapping = {"true": True, "yes": True, "y": True, "1": True, "pass": True, "passed": True,
               "false": False, "no": False, "n": False, "0": False, "fail": False, "failed": False}
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(default)


def normalize_candidate_log(signal_log: pd.DataFrame) -> pd.DataFrame:
    if signal_log is None or signal_log.empty:
        return pd.DataFrame()
    df = signal_log.copy()
    date_col = _first_existing(df, ("signal_time", "signal_date", "date", "session"))
    if date_col is None:
        raise ValueError("Candidate log needs signal_time, signal_date, date, or session.")
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df[df["_date"].notna()].copy()
    if "symbol" not in df.columns:
        df["symbol"] = df["ticker"] if "ticker" in df.columns else "UNKNOWN"
    for col in ("swing_score", "intraday_score", "entry_quality", "reward_risk",
                "market_score", "rs_percentile", "leadership_percentile", "distribution_days"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    outcome = _first_existing(df, OUTCOME_COLUMNS)
    if outcome is not None:
        df["_outcome_r"] = pd.to_numeric(df[outcome], errors="coerce")
    return df.sort_values(["_date", "symbol"]).reset_index(drop=True)


def score_distribution(signal_log: pd.DataFrame) -> pd.DataFrame:
    df = normalize_candidate_log(signal_log)
    rows = []
    for col, label in (("swing_score", "Swing Score"), ("intraday_score", "Intraday Score")):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        for p in (0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 1.0):
            if not s.empty:
                rows.append({"score": label, "percentile": "Max" if p == 1 else f"{100*p:g}th",
                             "value": round(float(s.quantile(p)), 2), "observations": len(s)})
    return pd.DataFrame(rows)


def threshold_reachability(signal_log: pd.DataFrame,
                           swing_thresholds: Sequence[float] = DEFAULT_SWING_THRESHOLDS,
                           intraday_thresholds: Sequence[float] = DEFAULT_INTRADAY_THRESHOLDS) -> pd.DataFrame:
    df = normalize_candidate_log(signal_log)
    rows, n = [], len(df)
    if n == 0:
        return pd.DataFrame()
    for col, label, thresholds in (("swing_score", "Swing", swing_thresholds),
                                   ("intraday_score", "Intraday", intraday_thresholds)):
        if col not in df.columns:
            continue
        for t in thresholds:
            passed = df[col].ge(float(t)).fillna(False)
            rows.append({"score": label, "threshold": float(t), "passed": int(passed.sum()),
                         "total": n, "pass_rate_pct": round(100 * passed.mean(), 2)})
    return pd.DataFrame(rows)


def bottleneck_report(signal_log: pd.DataFrame) -> pd.DataFrame:
    """Rank individual production-style gates by failure rate."""
    df = normalize_candidate_log(signal_log)
    if df.empty:
        return pd.DataFrame()
    checks = []
    def add(name, mask):
        mask = mask.fillna(False)
        checks.append({"gate": name, "passed": int(mask.sum()), "failed": int((~mask).sum()),
                       "pass_rate_pct": round(100 * mask.mean(), 2)})
    if "swing_score" in df: add("Swing Score >= 85", df["swing_score"] >= 85)
    if "intraday_score" in df: add("Intraday Score >= 85", df["intraday_score"] >= 85)
    if "entry_quality" in df: add("Entry Quality >= 10", df["entry_quality"] >= 10)
    if "reward_risk" in df: add("Reward/Risk >= 2.0", df["reward_risk"] >= 2)
    if "market_score" in df: add("Market Score >= 5", df["market_score"] >= 5)
    leader = "leadership_percentile" if "leadership_percentile" in df else "rs_percentile" if "rs_percentile" in df else None
    if leader:
        s = df[leader]
        cutoff = .70 if s.dropna().max() <= 1.5 else 70
        add("Leadership >= 70th percentile", s >= cutoff)
    if "distribution_days" in df: add("Distribution Days <= 4", df["distribution_days"] <= 4)
    return pd.DataFrame(checks).sort_values(["pass_rate_pct", "gate"]).reset_index(drop=True)


def calibration_summary(signal_log: pd.DataFrame, current_swing_threshold=85, current_intraday_threshold=85) -> dict:
    df = normalize_candidate_log(signal_log)
    if df.empty:
        return {"observations": 0, "message": "No candidate observations available."}
    out = {"observations": len(df)}
    for col, threshold, prefix in (("swing_score", current_swing_threshold, "swing"),
                                   ("intraday_score", current_intraday_threshold, "intraday")):
        if col in df:
            s = df[col].dropna()
            out[f"{prefix}_max"] = float(s.max()) if len(s) else np.nan
            out[f"{prefix}_p95"] = float(s.quantile(.95)) if len(s) else np.nan
            out[f"{prefix}_threshold_passes"] = int((s >= threshold).sum()) if len(s) else 0
            out[f"{prefix}_threshold_pass_rate_pct"] = round(100 * (s >= threshold).mean(), 2) if len(s) else 0
    if out.get("swing_threshold_passes", 0) == 0:
        out["warning"] = f"No historical candidate reached the current Swing threshold of {current_swing_threshold:g}."
    return out


def promotion_summary(calibration_result: dict | None) -> dict:
    """Summarize actual simulator calibration; never auto-promotes live rules."""
    if not calibration_result:
        return {"status": "NOT_RUN", "message": "Run v3.4 adaptive simulator calibration first."}
    comp = calibration_result.get("comparison", pd.DataFrame())
    if comp is None or comp.empty:
        return {"status": "NO_RESULTS", "message": "No calibration comparison is available."}
    candidates = comp[comp.get("promotion_candidate", False) == True] if "promotion_candidate" in comp else pd.DataFrame()
    if candidates.empty:
        return {"status": "KEEP_PRODUCTION", "message": "No alternate profile passed the promotion guardrails. Keep live production rules unchanged."}
    best = candidates.iloc[0]
    return {"status": "REVIEW", "message": "One or more profiles passed the research guardrails. Review before any production change.",
            "profile": best.get("profile"), "oos_expectancy_r": best.get("out_of_sample_expectancy_r"),
            "oos_profit_factor": best.get("out_of_sample_profit_factor"), "oos_trades": best.get("out_of_sample_trades")}
