from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _profit_factor(frame):
    if frame is None or frame.empty or "pnl" not in frame:
        return 0.0
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").dropna()
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _win_rate(frame):
    if frame is None or frame.empty or "pnl" not in frame:
        return 0.0
    pnl = pd.to_numeric(frame["pnl"], errors="coerce").dropna()
    return float((pnl > 0).mean() * 100) if len(pnl) else 0.0


def _expectancy_r(frame):
    if frame is None or frame.empty or "r_multiple" not in frame:
        return 0.0
    r = pd.to_numeric(frame["r_multiple"], errors="coerce").dropna()
    return float(r.mean()) if len(r) else 0.0


def _bootstrap_mean_ci(values, iterations=2500, confidence=0.95, seed=1337):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = float(rng.choice(values, size=len(values), replace=True).mean())
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


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
            "notes": ["No completed trades were available for holdout validation."],
        }

    frame = trades.copy()
    sort_col = "exit_time" if "exit_time" in frame.columns else (
        "entry_time" if "entry_time" in frame.columns else None
    )
    if sort_col:
        frame[sort_col] = pd.to_datetime(frame[sort_col], utc=True, errors="coerce")
        frame = frame.sort_values(sort_col)

    frame = frame.reset_index(drop=True)
    n = len(frame)
    split = min(max(int(math.floor(n * train_fraction)), 1), max(n - 1, 1))
    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()

    pf = _profit_factor(test)
    oos_exp = _expectancy_r(test)
    oos_r = pd.to_numeric(test.get("r_multiple"), errors="coerce").dropna()
    ci_low, ci_high = _bootstrap_mean_ci(oos_r.values)

    notes = []
    if n < min_total_trades:
        notes.append(f"Only {n} completed trades; at least {min_total_trades} is preferred.")
    if len(test) < min_oos_trades:
        notes.append(f"Only {len(test)} out-of-sample trades; at least {min_oos_trades} is preferred.")
    if oos_exp <= 0:
        notes.append("Out-of-sample expectancy is not positive.")
    if np.isfinite(pf) and pf < 1.15:
        notes.append("Out-of-sample profit factor is below 1.15.")
    if ci_low is not None and ci_low <= 0:
        notes.append("The 95% bootstrap interval for out-of-sample expectancy includes zero.")

    enough = n >= min_total_trades and len(test) >= min_oos_trades
    positive = oos_exp > 0 and (pf >= 1.15 or not np.isfinite(pf))
    robust = ci_low is not None and ci_low > 0
    passed = bool(enough and positive and robust)

    if passed and len(test) >= 20 and (not np.isfinite(pf) or pf >= 1.35):
        grade = "A"
    elif passed:
        grade = "B"
    elif enough and positive:
        grade = "C"
    elif enough:
        grade = "D"
    else:
        grade = "INSUFFICIENT"

    if not notes:
        notes.append("Chronological holdout and bootstrap checks passed the configured thresholds.")

    return {
        "sample_trades": n,
        "in_sample_trades": len(train),
        "out_of_sample_trades": len(test),
        "in_sample_win_rate_pct": round(_win_rate(train), 2),
        "out_of_sample_win_rate_pct": round(_win_rate(test), 2),
        "in_sample_expectancy_r": round(_expectancy_r(train), 3),
        "out_of_sample_expectancy_r": round(oos_exp, 3),
        "out_of_sample_profit_factor": "inf" if not np.isfinite(pf) else round(pf, 2),
        "bootstrap_expectancy_low_r": None if ci_low is None else round(ci_low, 3),
        "bootstrap_expectancy_high_r": None if ci_high is None else round(ci_high, 3),
        "confidence_grade": grade,
        "validation_pass": passed,
        "notes": notes,
    }
