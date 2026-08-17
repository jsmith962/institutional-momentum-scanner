"""
v3.5 calibration engine.

Research only. This module:
- analyzes cached signal logs without new market-data requests;
- measures threshold reachability and candidate stability;
- interprets actual portfolio-replay results from backtest.py;
- never changes live production thresholds automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


PRODUCTION_SWING_SCORE = 85.0
PRODUCTION_INTRADAY_SCORE = 85.0
PRODUCTION_ENTRY_QUALITY = 10.0
PRODUCTION_MARKET_SCORE = 5.0
PRODUCTION_LEADERSHIP = 70.0
PRODUCTION_MAX_DISTRIBUTION_DAYS = 4
PRODUCTION_MIN_REWARD_RISK = 2.0


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


DEFAULT_PROFILES = [
    CalibrationProfile("Production 85/85", 85.0, 85.0, 10.0),
    CalibrationProfile("Strict 82.5/80", 82.5, 80.0, 10.0),
    CalibrationProfile("Strict 80/80", 80.0, 80.0, 10.0),
    CalibrationProfile("Balanced 77.5/75", 77.5, 75.0, 10.0),
    CalibrationProfile("Balanced 75/70", 75.0, 70.0, 10.0),
    CalibrationProfile("Quality 75/70 Q12", 75.0, 70.0, 12.0),
    CalibrationProfile("Research 72.5/70", 72.5, 70.0, 10.0),
    CalibrationProfile("Research 70/65", 70.0, 65.0, 10.0),
]


def _numeric(frame, column, default=np.nan):
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _boolean(frame, column, default=False):
    if column not in frame.columns:
        return pd.Series(bool(default), index=frame.index, dtype="bool")
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default)
    text = values.fillna(default).astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


def profile_to_gate_config(profile):
    data = asdict(profile) if isinstance(profile, CalibrationProfile) else dict(profile)
    return {
        "swing_score": float(data.get("swing_score", PRODUCTION_SWING_SCORE)),
        "intraday_score": float(data.get("intraday_score", PRODUCTION_INTRADAY_SCORE)),
        "entry_quality": float(data.get("entry_quality", PRODUCTION_ENTRY_QUALITY)),
        "market_score": float(data.get("market_score", PRODUCTION_MARKET_SCORE)),
        "leadership_percentile": float(
            data.get("leadership", data.get("leadership_percentile", PRODUCTION_LEADERSHIP))
        ),
        "max_distribution_days": int(
            data.get("max_distribution_days", PRODUCTION_MAX_DISTRIBUTION_DAYS)
        ),
        "reward_risk": float(data.get("reward_risk", PRODUCTION_MIN_REWARD_RISK)),
    }


def _normalise_candidate_log(signal_log):
    if (
        signal_log is None
        or not isinstance(signal_log, pd.DataFrame)
        or signal_log.empty
    ):
        return pd.DataFrame()

    frame = signal_log.copy()
    frame["_swing"] = _numeric(frame, "swing_score")
    frame["_intraday"] = _numeric(frame, "intraday_score")
    frame["_quality"] = _numeric(frame, "entry_quality")
    frame["_market"] = _numeric(frame, "market_score")
    frame["_leadership"] = _numeric(frame, "leadership_percentile")
    frame["_distribution"] = _numeric(frame, "distribution_days")
    frame["_rr"] = _numeric(frame, "reward_risk")
    frame["_risk"] = _boolean(frame, "risk_flag", False)
    frame["_trend"] = _boolean(frame, "trend_health", True)
    frame["_too_extended"] = _boolean(frame, "too_extended", False)

    if "inside_entry_zone" in frame.columns:
        frame["_inside_zone"] = _boolean(frame, "inside_entry_zone", False)
    else:
        price = _numeric(frame, "reference_price")
        if price.isna().all():
            price = _numeric(frame, "price")
        low = _numeric(frame, "entry_low")
        high = _numeric(frame, "entry_high")
        available = price.notna() & low.notna() & high.notna()
        frame["_inside_zone"] = True
        frame.loc[available, "_inside_zone"] = (
            (price[available] >= low[available]) & (price[available] <= high[available])
        )

    if "intraday_signal" in frame.columns:
        frame["_intraday_buy"] = (
            frame["intraday_signal"].fillna("").astype(str).str.upper().eq("BUY")
        )
    else:
        frame["_intraday_buy"] = True

    date_column = next(
        (
            c
            for c in ("signal_time", "session", "signal_date", "date", "timestamp")
            if c in frame.columns
        ),
        None,
    )
    frame["_cal_date"] = (
        pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        if date_column
        else pd.NaT
    )

    frame = frame[
        frame["_swing"].notna()
        & frame["_intraday"].notna()
        & frame["_quality"].notna()
    ].copy()

    return frame.sort_values("_cal_date", na_position="last").reset_index(drop=True)


def _profile_mask(frame, profile):
    mask = (
        (frame["_swing"] >= profile.swing_score)
        & (frame["_intraday"] >= profile.intraday_score)
        & (frame["_quality"] >= profile.entry_quality)
        & ~frame["_risk"]
        & ~frame["_too_extended"]
        & frame["_trend"]
        & frame["_inside_zone"]
        & frame["_intraday_buy"]
    )

    optional = [
        ("_market", ">=", profile.market_score),
        ("_leadership", ">=", profile.leadership),
        ("_distribution", "<=", profile.max_distribution_days),
        ("_rr", ">=", profile.reward_risk),
    ]
    for column, op, threshold in optional:
        if frame[column].notna().any():
            if op == ">=":
                mask &= frame[column].isna() | (frame[column] >= threshold)
            else:
                mask &= frame[column].isna() | (frame[column] <= threshold)

    return mask.fillna(False)


def bounded_profiles(max_profiles=6):
    return DEFAULT_PROFILES[: max(1, int(max_profiles))]


def build_adaptive_profiles(signal_log, max_profiles=8):
    frame = _normalise_candidate_log(signal_log)
    if frame.empty:
        return bounded_profiles(max_profiles)

    swing = frame["_swing"].dropna()
    intra = frame["_intraday"].dropna()

    half = lambda x: round(float(x) * 2) / 2
    five = lambda x: round(float(x) / 5) * 5

    swing_levels = [85.0, 80.0, 77.5, 75.0, 72.5, 70.0, 67.5, 65.0]
    intra_levels = [85.0, 80.0, 75.0, 70.0, 65.0, 60.0]

    for q in (0.99, 0.95, 0.90, 0.80):
        if not swing.empty:
            swing_levels.append(half(swing.quantile(q)))
        if not intra.empty:
            intra_levels.append(five(intra.quantile(q)))

    swing_levels = sorted(
        {min(85.0, max(65.0, x)) for x in swing_levels}, reverse=True
    )
    intra_levels = sorted(
        {min(85.0, max(60.0, x)) for x in intra_levels}, reverse=True
    )

    profiles = [CalibrationProfile("Production 85/85", 85.0, 85.0, 10.0)]
    seen = {(85.0, 85.0, 10.0, 70.0)}

    for i in range(max(len(swing_levels), len(intra_levels))):
        s = swing_levels[min(i, len(swing_levels) - 1)]
        intr = intra_levels[min(i, len(intra_levels) - 1)]
        key = (float(s), float(intr), 10.0, 70.0)
        if key not in seen:
            seen.add(key)
            profiles.append(
                CalibrationProfile(
                    f"Research S{s:g}/I{intr:g}",
                    float(s),
                    float(intr),
                    10.0,
                )
            )
        if len(profiles) >= max(1, int(max_profiles)):
            break

    return profiles


def _chronological_split(frame, train_fraction=0.70):
    if frame.empty:
        return frame, frame
    ordered = frame.sort_values("_cal_date", na_position="last").reset_index(drop=True)
    cutoff = int(round(len(ordered) * float(train_fraction)))
    cutoff = max(1, min(cutoff, max(len(ordered) - 1, 1)))
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def _candidate_fold_stability(frame, profile, folds=4):
    if frame.empty:
        return {
            "folds_completed": 0,
            "positive_fold_ratio": 0.0,
            "median_fold_rate_ratio": 0.0,
            "worst_fold_rate_ratio": 0.0,
        }

    ordered = frame.sort_values("_cal_date", na_position="last").reset_index(drop=True)
    folds = max(2, int(folds))
    n = len(ordered)
    initial = max(1, int(n * 0.40))
    remaining = n - initial
    if remaining < folds:
        return {
            "folds_completed": 0,
            "positive_fold_ratio": 0.0,
            "median_fold_rate_ratio": 0.0,
            "worst_fold_rate_ratio": 0.0,
        }

    test_size = max(1, remaining // folds)
    ratios = []
    positive = 0
    completed = 0
    train_end = initial

    for fold in range(folds):
        start = train_end
        end = n if fold == folds - 1 else min(n, start + test_size)
        if end <= start:
            break

        train = ordered.iloc[:start]
        test = ordered.iloc[start:end]
        train_rate = float(_profile_mask(train, profile).mean()) if len(train) else 0.0
        test_mask = _profile_mask(test, profile)
        test_rate = float(test_mask.mean()) if len(test) else 0.0

        if int(test_mask.sum()) > 0:
            positive += 1
        completed += 1

        if train_rate > 0:
            ratios.append(test_rate / train_rate)

        train_end = end
        if train_end >= n:
            break

    clean = [x for x in ratios if np.isfinite(x)]
    return {
        "folds_completed": completed,
        "positive_fold_ratio": round(positive / max(completed, 1), 3),
        "median_fold_rate_ratio": round(float(np.median(clean)), 3) if clean else 0.0,
        "worst_fold_rate_ratio": round(float(np.min(clean)), 3) if clean else 0.0,
    }


def run_fast_calibration(
    signal_log,
    max_profiles=6,
    train_fraction=0.70,
    progress_callback=None,
    adaptive=True,
    folds=4,
):
    """
    Fast reachability/stability research only.
    This function does NOT simulate trades and does NOT rank profitability.
    """
    frame = _normalise_candidate_log(signal_log)
    if frame.empty:
        return {
            "status": "NO_DATA",
            "summary": pd.DataFrame(),
            "profile_results": {},
            "candidate_count": 0,
            "profitability_evidence": False,
            "message": "No usable historical candidate log was available.",
        }

    profiles = (
        build_adaptive_profiles(signal_log, max_profiles)
        if adaptive
        else bounded_profiles(max_profiles)
    )
    train, oos = _chronological_split(frame, train_fraction)
    rows = []
    profile_results = {}

    for i, profile in enumerate(profiles, start=1):
        if progress_callback:
            progress_callback(i, len(profiles), profile.name)

        full_mask = _profile_mask(frame, profile)
        train_mask = _profile_mask(train, profile)
        oos_mask = _profile_mask(oos, profile)

        full_count = int(full_mask.sum())
        train_count = int(train_mask.sum())
        oos_count = int(oos_mask.sum())

        train_rate = train_count / max(len(train), 1)
        oos_rate = oos_count / max(len(oos), 1) if len(oos) else 0.0
        stability = oos_rate / train_rate if train_rate > 0 else 0.0
        folds_result = _candidate_fold_stability(frame, profile, folds)

        row = {
            "profile": profile.name,
            "swing_threshold": profile.swing_score,
            "intraday_threshold": profile.intraday_score,
            "entry_quality": profile.entry_quality,
            "market_score": profile.market_score,
            "leadership": profile.leadership,
            "max_distribution_days": profile.max_distribution_days,
            "reward_risk": profile.reward_risk,
            "all_candidates": full_count,
            "candidate_rate_pct": round(full_count / max(len(frame), 1) * 100, 3),
            "in_sample_candidates": train_count,
            "out_of_sample_candidates": oos_count,
            "oos_candidate_rate_pct": round(oos_rate * 100, 3),
            "stability_ratio": round(stability, 3),
            "candidate_folds_completed": folds_result["folds_completed"],
            "candidate_positive_fold_ratio": folds_result["positive_fold_ratio"],
            "median_fold_rate_ratio": folds_result["median_fold_rate_ratio"],
            "worst_fold_rate_ratio": folds_result["worst_fold_rate_ratio"],
        }
        rows.append(row)
        profile_results[profile.name] = {
            "profile": asdict(profile),
            "gate_config": profile_to_gate_config(profile),
            "all_candidates": frame[full_mask].copy(),
            "in_sample_candidates": train[train_mask].copy(),
            "out_of_sample_candidates": oos[oos_mask].copy(),
            "metrics": row,
        }

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["_has_oos"] = (summary["out_of_sample_candidates"] > 0).astype(int)
        summary["_strictness"] = (
            summary["swing_threshold"]
            + summary["intraday_threshold"]
            + 2 * summary["entry_quality"]
        )
        summary = (
            summary.sort_values(
                [
                    "_has_oos",
                    "candidate_positive_fold_ratio",
                    "out_of_sample_candidates",
                    "_strictness",
                ],
                ascending=[False, False, False, False],
            )
            .drop(columns=["_has_oos", "_strictness"])
            .reset_index(drop=True)
        )

    production_reachable = bool((frame["_swing"] >= PRODUCTION_SWING_SCORE).any())
    production_intraday_reachable = bool(
        (frame["_intraday"] >= PRODUCTION_INTRADAY_SCORE).any()
    )

    viable = (
        summary[summary["out_of_sample_candidates"] > 0]
        if not summary.empty
        else pd.DataFrame()
    )
    best_profile = viable.iloc[0].to_dict() if not viable.empty else None

    return {
        "status": "COMPLETE",
        "summary": summary,
        "profile_results": profile_results,
        "candidate_count": len(frame),
        "in_sample_count": len(train),
        "out_of_sample_count": len(oos),
        "production_reachable": production_reachable,
        "production_intraday_reachable": production_intraday_reachable,
        "best_profile": best_profile,
        "profitability_evidence": False,
        "message": (
            f"Completed {len(profiles)} bounded profiles across "
            f"{len(frame):,} cached observations. "
            "This measures reachability/stability, not profitability."
        ),
    }


def score_distribution(signal_log):
    frame = _normalise_candidate_log(signal_log)
    if frame.empty:
        return pd.DataFrame()

    mapping = [
        ("Swing Score", "_swing"),
        ("Intraday Score", "_intraday"),
        ("Entry Quality", "_quality"),
        ("Market Score", "_market"),
        ("Leadership", "_leadership"),
        ("Reward/Risk", "_rr"),
    ]
    percentiles = [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    rows = []

    for label, column in mapping:
        values = frame[column].dropna().astype(float)
        if values.empty:
            continue
        for pct in percentiles:
            rows.append(
                {
                    "score": label,
                    "observations": len(values),
                    "percentile": "Maximum" if pct == 1.0 else f"{int(pct * 100)}th",
                    "value": round(float(values.quantile(pct)), 2),
                }
            )
    return pd.DataFrame(rows)


def production_gate_bottlenecks(signal_log):
    frame = _normalise_candidate_log(signal_log)
    if frame.empty:
        return pd.DataFrame()

    total = len(frame)
    gates = [
        ("Swing Score >= 85", frame["_swing"] >= PRODUCTION_SWING_SCORE),
        ("Intraday Score >= 85", frame["_intraday"] >= PRODUCTION_INTRADAY_SCORE),
        ("Entry Quality >= 10", frame["_quality"] >= PRODUCTION_ENTRY_QUALITY),
        ("No active risk event", ~frame["_risk"]),
        ("Not too extended", ~frame["_too_extended"]),
        ("Trend health passed", frame["_trend"]),
        ("Inside preferred entry zone", frame["_inside_zone"]),
        ("Intraday signal BUY", frame["_intraday_buy"]),
    ]

    if frame["_leadership"].notna().any():
        gates.append(
            (
                "Leadership >= 70th percentile",
                frame["_leadership"].isna()
                | (frame["_leadership"] >= PRODUCTION_LEADERSHIP),
            )
        )
    if frame["_distribution"].notna().any():
        gates.append(
            (
                "Distribution Days <= 4",
                frame["_distribution"].isna()
                | (frame["_distribution"] <= PRODUCTION_MAX_DISTRIBUTION_DAYS),
            )
        )
    if frame["_market"].notna().any():
        gates.append(
            (
                "Market Score >= 5",
                frame["_market"].isna()
                | (frame["_market"] >= PRODUCTION_MARKET_SCORE),
            )
        )
    if frame["_rr"].notna().any():
        gates.append(
            (
                "Reward/Risk >= 2.0",
                frame["_rr"].isna()
                | (frame["_rr"] >= PRODUCTION_MIN_REWARD_RISK),
            )
        )

    rows = []
    for name, mask in gates:
        passed = int(mask.fillna(False).sum())
        failed = total - passed
        rows.append(
            {
                "gate": name,
                "passed": passed,
                "failed": failed,
                "pass_rate_pct": round(passed / max(total, 1) * 100, 2),
                "failure_rate_pct": round(failed / max(total, 1) * 100, 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["failed", "gate"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _pf_number(value):
    try:
        if str(value).strip().lower() == "inf":
            return float("inf")
        return float(value)
    except Exception:
        return 0.0


def rank_portfolio_calibration(comparison):
    """
    Rank ACTUAL portfolio-replay results from backtest.py.
    This is profitability evidence; run_fast_calibration() is not.
    """
    if (
        comparison is None
        or not isinstance(comparison, pd.DataFrame)
        or comparison.empty
    ):
        return pd.DataFrame()

    frame = comparison.copy()
    numeric_columns = [
        "trades",
        "return_pct",
        "win_rate_pct",
        "expectancy_r",
        "max_drawdown_pct",
        "out_of_sample_trades",
        "out_of_sample_expectancy_r",
        "positive_fold_ratio",
        "aggregate_oos_expectancy_r",
        "worst_fold_expectancy_r",
        "bootstrap_expectancy_low_r",
        "bootstrap_expectancy_high_r",
        "walk_forward_folds",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["_aggregate_pf"] = (
        frame["aggregate_oos_profit_factor"].map(_pf_number)
        if "aggregate_oos_profit_factor" in frame.columns
        else 0.0
    )

    if "validation_pass" not in frame.columns:
        frame["validation_pass"] = False
    if "promotion_candidate" not in frame.columns:
        frame["promotion_candidate"] = False

    frame["research_eligible_v35"] = (
        frame["trades"].fillna(0).ge(40)
        & frame["walk_forward_folds"].fillna(0).ge(3)
        & frame["positive_fold_ratio"].fillna(0).ge(0.60)
        & frame["aggregate_oos_expectancy_r"].fillna(-999).gt(0)
        & frame["_aggregate_pf"].replace(np.inf, 999).fillna(0).ge(1.15)
        & frame["max_drawdown_pct"].abs().fillna(999).le(20)
    )

    frame["promotion_candidate_v35"] = (
        frame["promotion_candidate"].fillna(False).astype(bool)
        & frame["research_eligible_v35"]
        & frame["bootstrap_expectancy_low_r"].fillna(-999).gt(0)
        & frame["aggregate_oos_expectancy_r"].fillna(-999).ge(0.10)
        & frame["_aggregate_pf"].replace(np.inf, 999).fillna(0).ge(1.30)
        & frame["positive_fold_ratio"].fillna(0).ge(0.75)
        & frame["worst_fold_expectancy_r"].fillna(-999).ge(-0.25)
    )

    frame["v35_evidence_score"] = (
        200 * frame["promotion_candidate_v35"].astype(int)
        + 80 * frame["research_eligible_v35"].astype(int)
        + 50 * frame["validation_pass"].fillna(False).astype(bool).astype(int)
        + 40 * frame["positive_fold_ratio"].fillna(0).clip(0, 1)
        + 20 * frame["aggregate_oos_expectancy_r"].fillna(-1).clip(-1, 1)
        + 8 * frame["_aggregate_pf"].replace(np.inf, 3).clip(0, 3)
        + np.minimum(frame["trades"].fillna(0), 100) / 10
        + 5 * frame["worst_fold_expectancy_r"].fillna(-1).clip(-1, 1)
        - frame["max_drawdown_pct"].abs().fillna(50) / 10
    ).round(3)

    return (
        frame.sort_values(
            [
                "promotion_candidate_v35",
                "research_eligible_v35",
                "validation_pass",
                "v35_evidence_score",
                "aggregate_oos_expectancy_r",
            ],
            ascending=[False, False, False, False, False],
        )
        .drop(columns=["_aggregate_pf"], errors="ignore")
        .reset_index(drop=True)
    )


def portfolio_calibration_verdict(comparison):
    ranked = rank_portfolio_calibration(comparison)
    if ranked.empty:
        return {
            "verdict": "INSUFFICIENT EVIDENCE",
            "promotion_profile": None,
            "message": "No portfolio calibration results were available.",
        }

    promotion = ranked[ranked["promotion_candidate_v35"]]
    if not promotion.empty:
        top = promotion.iloc[0]
        return {
            "verdict": "REVIEW PROMOTION CANDIDATE",
            "promotion_profile": top.get("profile"),
            "message": (
                "A research profile passed the v3.5 portfolio/walk-forward screen. "
                "Do not change live rules automatically; verify it on additional "
                "non-overlapping periods and paper trading."
            ),
        }

    eligible = ranked[ranked["research_eligible_v35"]]
    if not eligible.empty:
        return {
            "verdict": "PROMISING BUT NOT PROMOTABLE",
            "promotion_profile": None,
            "best_research_profile": eligible.iloc[0].get("profile"),
            "message": (
                "At least one profile has meaningful out-of-sample evidence, "
                "but none met the stronger promotion standard."
            ),
        }

    max_trades = pd.to_numeric(ranked["trades"], errors="coerce").fillna(0).max()
    if max_trades < 40:
        return {
            "verdict": "INSUFFICIENT EVIDENCE",
            "promotion_profile": None,
            "message": (
                "No tested profile produced enough completed trades for dependable "
                "v3.5 walk-forward conclusions."
            ),
        }

    return {
        "verdict": "FAIL",
        "promotion_profile": None,
        "message": (
            "The tested profiles produced enough data to evaluate, but none "
            "demonstrated the required durable out-of-sample edge."
        ),
    }


def calibration_summary(signal_log, max_profiles=6):
    return run_fast_calibration(
        signal_log,
        max_profiles=max_profiles,
    ).get("summary", pd.DataFrame())
