
"""
v3.3 Score Calibration / Threshold Research Module

Purpose
-------
Analyze historical candidate observations without changing the live BUY rules.

This module:
1. Audits the actual distribution of swing and intraday scores.
2. Shows whether the current thresholds are realistically reachable.
3. Tests a matrix of candidate Swing/Intraday thresholds.
4. Uses chronological in-sample / out-of-sample splits when outcome data exists.
5. Searches for stable parameter regions rather than a single "best" backtest.
6. Never silently changes production thresholds.

Expected candidate log columns
------------------------------
Required:
    signal_date or signal_time
    symbol
    swing_score
    intraday_score

Strongly recommended:
    entry_quality
    reward_risk
    market_score
    rs_percentile
    inside_entry_zone
    trend_health
    distribution_days
    risk_event_clear
    not_too_extended
    intraday_signal

For full performance calibration, include ONE outcome column:
    forward_r
    hypothetical_r
    outcome_r
    r_multiple

If no outcome column exists, the module still performs threshold/distribution
calibration, but it will not invent win rate, expectancy or profit factor.
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
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_bool(series: pd.Series, default: bool = True) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(1 if default else 0).astype(float).ne(0)
    mapping = {
        "true": True, "yes": True, "y": True, "1": True, "pass": True, "passed": True,
        "false": False, "no": False, "n": False, "0": False, "fail": False, "failed": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna(default)


def normalize_candidate_log(signal_log: pd.DataFrame) -> pd.DataFrame:
    """Return a clean chronological candidate-level dataframe."""
    if signal_log is None or signal_log.empty:
        return pd.DataFrame()

    df = signal_log.copy()

    date_col = _first_existing(df, ("signal_time", "signal_date", "date", "session"))
    if date_col is None:
        raise ValueError("Candidate log needs signal_time, signal_date, date, or session.")
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df[df["_date"].notna()].copy()

    if "symbol" not in df.columns:
        if "ticker" in df.columns:
            df["symbol"] = df["ticker"]
        else:
            df["symbol"] = "UNKNOWN"

    for col in ("swing_score", "intraday_score", "entry_quality", "reward_risk",
                "market_score", "rs_percentile", "distribution_days"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outcome = _first_existing(df, OUTCOME_COLUMNS)
    if outcome is not None:
        df["_outcome_r"] = pd.to_numeric(df[outcome], errors="coerce")

    df = df.sort_values(["_date", "symbol"]).reset_index(drop=True)
    return df


def score_distribution(signal_log: pd.DataFrame) -> pd.DataFrame:
    """Percentile audit for the score scales."""
    df = normalize_candidate_log(signal_log)
    if df.empty:
        return pd.DataFrame()

    percentiles = [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 1.0]
    rows = []
    for col, label in (("swing_score", "Swing Score"), ("intraday_score", "Intraday Score")):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q = s.quantile(percentiles)
        for p, value in q.items():
            rows.append({
                "score": label,
                "percentile": "Max" if p == 1.0 else f"{100*p:g}th",
                "value": round(float(value), 2),
                "observations": int(len(s)),
            })
    return pd.DataFrame(rows)


def threshold_reachability(signal_log: pd.DataFrame,
                           swing_thresholds: Sequence[float] = DEFAULT_SWING_THRESHOLDS,
                           intraday_thresholds: Sequence[float] = DEFAULT_INTRADAY_THRESHOLDS) -> pd.DataFrame:
    """How often each individual score threshold is reachable."""
    df = normalize_candidate_log(signal_log)
    rows = []
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    if "swing_score" in df.columns:
        for t in swing_thresholds:
            passed = df["swing_score"].ge(float(t)).fillna(False)
            rows.append({"score": "Swing", "threshold": float(t), "passed": int(passed.sum()),
                         "total": n, "pass_rate_pct": round(100 * passed.mean(), 2)})
    if "intraday_score" in df.columns:
        for t in intraday_thresholds:
            passed = df["intraday_score"].ge(float(t)).fillna(False)
            rows.append({"score": "Intraday", "threshold": float(t), "passed": int(passed.sum()),
                         "total": n, "pass_rate_pct": round(100 * passed.mean(), 2)})
    return pd.DataFrame(rows)


def _base_non_score_gates(df: pd.DataFrame) -> pd.Series:
    """
    Preserve production-style non-score gates when the relevant columns exist.
    Missing optional columns are NOT treated as failures.
    """
    mask = pd.Series(True, index=df.index)

    if "entry_quality" in df.columns:
        mask &= df["entry_quality"].ge(10).fillna(False)
    if "reward_risk" in df.columns:
        mask &= df["reward_risk"].ge(2.0).fillna(False)
    if "market_score" in df.columns:
        mask &= df["market_score"].ge(5).fillna(False)
    if "rs_percentile" in df.columns:
        # supports 0-1 or 0-100 storage
        rs = df["rs_percentile"]
        cutoff = 0.70 if rs.dropna().max() <= 1.5 else 70.0
        mask &= rs.ge(cutoff).fillna(False)
    if "distribution_days" in df.columns:
        mask &= df["distribution_days"].le(4).fillna(False)

    for col in ("risk_event_clear", "not_too_extended", "inside_entry_zone", "trend_health"):
        if col in df.columns:
            mask &= _to_bool(df[col], default=False)

    return mask


def _metrics(frame: pd.DataFrame) -> dict:
    out = {
        "observations": int(len(frame)),
        "trades_with_outcomes": 0,
        "win_rate_pct": np.nan,
        "expectancy_r": np.nan,
        "profit_factor": np.nan,
        "avg_r": np.nan,
    }
    if "_outcome_r" not in frame.columns:
        return out
    r = frame["_outcome_r"].dropna().astype(float)
    out["trades_with_outcomes"] = int(len(r))
    if r.empty:
        return out
    wins = r[r > 0]
    losses = r[r < 0]
    out["win_rate_pct"] = round(100 * float((r > 0).mean()), 2)
    out["expectancy_r"] = round(float(r.mean()), 4)
    out["avg_r"] = round(float(r.mean()), 4)
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    out["profit_factor"] = round(gross_profit / gross_loss, 4) if gross_loss > 0 else np.inf
    return out


def _chronological_split(df: pd.DataFrame, oos_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    dates = pd.Series(df["_date"].dt.normalize().unique()).sort_values().reset_index(drop=True)
    if len(dates) < 2:
        return df.copy(), df.iloc[0:0].copy()
    split_idx = max(1, min(len(dates) - 1, int(math.floor(len(dates) * (1 - oos_fraction)))))
    split_date = dates.iloc[split_idx]
    ins = df[df["_date"].dt.normalize() < split_date].copy()
    oos = df[df["_date"].dt.normalize() >= split_date].copy()
    return ins, oos


def threshold_matrix(signal_log: pd.DataFrame,
                     swing_thresholds: Sequence[float] = DEFAULT_SWING_THRESHOLDS,
                     intraday_thresholds: Sequence[float] = DEFAULT_INTRADAY_THRESHOLDS,
                     config: CalibrationConfig = CalibrationConfig()) -> pd.DataFrame:
    """
    Test candidate score thresholds while keeping available non-score risk gates intact.

    NOTE: this is a calibration study, not permission to change production rules.
    """
    df = normalize_candidate_log(signal_log)
    if df.empty:
        return pd.DataFrame()
    if "swing_score" not in df.columns or "intraday_score" not in df.columns:
        raise ValueError("Candidate log must include swing_score and intraday_score.")

    base = _base_non_score_gates(df)
    rows = []

    for swing_t in swing_thresholds:
        for intra_t in intraday_thresholds:
            mask = (
                base
                & df["swing_score"].ge(float(swing_t)).fillna(False)
                & df["intraday_score"].ge(float(intra_t)).fillna(False)
            )
            selected = df[mask].copy()
            ins, oos = _chronological_split(selected, config.oos_fraction)
            all_m = _metrics(selected)
            ins_m = _metrics(ins)
            oos_m = _metrics(oos)

            rows.append({
                "swing_threshold": float(swing_t),
                "intraday_threshold": float(intra_t),
                "candidates": all_m["observations"],
                "outcomes": all_m["trades_with_outcomes"],
                "win_rate_pct": all_m["win_rate_pct"],
                "expectancy_r": all_m["expectancy_r"],
                "profit_factor": all_m["profit_factor"],
                "in_sample_n": ins_m["trades_with_outcomes"],
                "in_sample_win_rate_pct": ins_m["win_rate_pct"],
                "in_sample_expectancy_r": ins_m["expectancy_r"],
                "oos_n": oos_m["trades_with_outcomes"],
                "oos_win_rate_pct": oos_m["win_rate_pct"],
                "oos_expectancy_r": oos_m["expectancy_r"],
            })

    result = pd.DataFrame(rows)
    return add_stability_scores(result, config)


def add_stability_scores(matrix: pd.DataFrame,
                         config: CalibrationConfig = CalibrationConfig()) -> pd.DataFrame:
    """
    Reward broad, neighboring parameter regions with positive OOS behavior.
    Avoid choosing a single isolated historical optimum.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame()

    out = matrix.copy()
    swings = sorted(out["swing_threshold"].unique())
    intras = sorted(out["intraday_threshold"].unique())

    stability = []
    qualified = []

    for _, row in out.iterrows():
        si = swings.index(row["swing_threshold"])
        ii = intras.index(row["intraday_threshold"])
        neighbor_swings = swings[max(0, si-config.stability_radius): min(len(swings), si+config.stability_radius+1)]
        neighbor_intras = intras[max(0, ii-config.stability_radius): min(len(intras), ii+config.stability_radius+1)]
        neigh = out[
            out["swing_threshold"].isin(neighbor_swings)
            & out["intraday_threshold"].isin(neighbor_intras)
        ].copy()

        if "oos_expectancy_r" in neigh.columns and neigh["oos_expectancy_r"].notna().any():
            valid = neigh[neigh["oos_n"].fillna(0) >= config.min_oos_observations]
            if valid.empty:
                score = np.nan
            else:
                positive_fraction = float((valid["oos_expectancy_r"] > 0).mean())
                median_expectancy = float(valid["oos_expectancy_r"].median())
                score = 100 * positive_fraction + 10 * max(-2.0, min(2.0, median_expectancy))
        else:
            # No outcomes: stability is based only on sample persistence, not profitability.
            counts = neigh["candidates"].astype(float)
            score = float(counts.median()) if not counts.empty else np.nan

        stability.append(round(score, 3) if pd.notna(score) else np.nan)

        enough_total = int(row.get("outcomes", 0)) >= config.min_total_observations
        enough_oos = int(row.get("oos_n", 0)) >= config.min_oos_observations
        positive_oos = pd.notna(row.get("oos_expectancy_r")) and float(row["oos_expectancy_r"]) > 0
        qualified.append(bool(enough_total and enough_oos and positive_oos))

    out["stability_score"] = stability
    out["validation_ready"] = qualified
    return out


def calibration_summary(signal_log: pd.DataFrame,
                        current_swing_threshold: float = 85,
                        current_intraday_threshold: float = 85) -> dict:
    """Compact audit suitable for a Streamlit summary card."""
    df = normalize_candidate_log(signal_log)
    if df.empty:
        return {"observations": 0, "message": "No candidate observations available."}

    summary = {"observations": int(len(df))}
    for col, threshold, prefix in (
        ("swing_score", current_swing_threshold, "swing"),
        ("intraday_score", current_intraday_threshold, "intraday"),
    ):
        if col in df.columns:
            s = df[col].dropna()
            summary[f"{prefix}_max"] = float(s.max()) if not s.empty else np.nan
            summary[f"{prefix}_p95"] = float(s.quantile(.95)) if not s.empty else np.nan
            summary[f"{prefix}_threshold_passes"] = int((s >= threshold).sum()) if not s.empty else 0
            summary[f"{prefix}_threshold_pass_rate_pct"] = (
                round(100 * float((s >= threshold).mean()), 2) if not s.empty else 0.0
            )

    if summary.get("swing_threshold_passes", 0) == 0:
        summary["warning"] = (
            f"No historical candidate reached the current Swing threshold "
            f"of {current_swing_threshold:g}. Treat this as a calibration warning; "
            f"do not automatically lower the live threshold."
        )
    return summary


def best_stable_regions(matrix: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Return research candidates only. Does NOT declare a new production threshold.
    Prefers validated OOS rows; otherwise shows highest sample-persistence regions.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame()
    m = matrix.copy()

    ready = m[m["validation_ready"] == True] if "validation_ready" in m.columns else pd.DataFrame()
    if not ready.empty:
        sort_cols = ["stability_score", "oos_expectancy_r", "oos_n"]
        return ready.sort_values(sort_cols, ascending=[False, False, False]).head(limit)

    return m.sort_values(["stability_score", "candidates"], ascending=[False, False]).head(limit)
