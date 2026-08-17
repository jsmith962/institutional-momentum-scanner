
"""
v3.5 Streamlit calibration and validation UI.

Research only.

This UI separates three different kinds of evidence:

1. Production backtest validation
   Displays the actual completed-trade holdout and multi-fold walk-forward
   results already attached to the production backtest.

2. Fast threshold diagnostics
   Reuses the cached signal log to show threshold reachability, bottlenecks,
   candidate frequency, and candidate stability. This is NOT profitability
   evidence.

3. Portfolio calibration evidence
   If actual replay/calibration comparison results are present, displays the
   profitability-based v3.5 ranking and conservative promotion verdict.

Live production thresholds are never changed automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from calibration import (
    portfolio_calibration_verdict,
    production_gate_bottlenecks,
    rank_portfolio_calibration,
    run_fast_calibration,
    score_distribution,
)


# ============================================================
# HELPERS
# ============================================================

def _safe_df(value):
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _number(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str) and value.strip().lower() == "inf":
            return float("inf")
        return float(value)
    except Exception:
        return default


def _metric_value(value, digits=2, suffix=""):
    number = _number(value)

    if number is None:
        return "â"

    if np.isinf(number):
        return "â"

    if digits == 0:
        return f"{number:,.0f}{suffix}"

    return f"{number:,.{digits}f}{suffix}"


def _percent_ratio(value):
    number = _number(value)

    if number is None:
        return "â"

    return f"{number * 100:.0f}%"


def _verdict_box(verdict, message=""):
    verdict = str(
        verdict or "INSUFFICIENT EVIDENCE"
    ).upper()

    text = (
        f"**{verdict}**"
        + (
            f" â {message}"
            if message
            else ""
        )
    )

    if (
        "PASS" in verdict
        or "PROMOTION" in verdict
        or "PROMISING" in verdict
    ):
        st.success(text)

    elif "FAIL" in verdict:
        st.error(text)

    else:
        st.warning(text)


def _validation_notes(notes):
    if not notes:
        return

    with st.expander(
        "Validation notes"
    ):
        for note in notes:
            st.write(
                f"â¢ {note}"
            )


# ============================================================
# PRODUCTION VALIDATION
# ============================================================

def _render_production_validation(
    backtest_result,
):
    st.subheader(
        "Production strategy validation"
    )

    stats = (
        backtest_result.get(
            "stats",
            {},
        )
        if isinstance(
            backtest_result,
            dict,
        )
        else {}
    )

    full_validation = (
        backtest_result.get(
            "full_validation",
            {},
        )
        if isinstance(
            backtest_result,
            dict,
        )
        else {}
    )

    holdout = (
        backtest_result.get(
            "validation",
            {},
        )
        if isinstance(
            backtest_result,
            dict,
        )
        else {}
    )

    if not isinstance(
        full_validation,
        dict,
    ):
        full_validation = {}

    if not isinstance(
        holdout,
        dict,
    ):
        holdout = {}

    trade_count = stats.get(
        "trades",
        0,
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Completed trades",
        _metric_value(
            trade_count,
            0,
        ),
    )

    c2.metric(
        "Expectancy",
        _metric_value(
            stats.get(
                "expectancy_r"
            ),
            3,
            " R",
        ),
    )

    c3.metric(
        "Profit factor",
        _metric_value(
            stats.get(
                "profit_factor"
            ),
            2,
        ),
    )

    c4.metric(
        "Max drawdown",
        _metric_value(
            stats.get(
                "max_drawdown_pct"
            ),
            2,
            "%",
        ),
    )

    if not full_validation:
        st.warning(
            "This backtest does not contain the v3.5 multi-fold validation "
            "payload. Run a new backtest after the v3.5 backtest.py update."
        )

        if holdout:
            st.caption(
                "Legacy chronological holdout result"
            )

            h1, h2, h3 = st.columns(
                3
            )

            h1.metric(
                "OOS trades",
                _metric_value(
                    holdout.get(
                        "out_of_sample_trades"
                    ),
                    0,
                ),
            )

            h2.metric(
                "OOS expectancy",
                _metric_value(
                    holdout.get(
                        "out_of_sample_expectancy_r"
                    ),
                    3,
                    " R",
                ),
            )

            h3.metric(
                "OOS profit factor",
                _metric_value(
                    holdout.get(
                        "out_of_sample_profit_factor"
                    ),
                    2,
                ),
            )

        return

    verdict = full_validation.get(
        "validation_verdict",
        "INSUFFICIENT EVIDENCE",
    )

    promotion = bool(
        full_validation.get(
            "promotion_candidate",
            False,
        )
    )

    message = (
        "The production rule set passed the configured multi-fold validation."
        if full_validation.get(
            "validation_pass",
            False,
        )
        else (
            "There are not enough completed trades for a dependable conclusion."
            if str(
                verdict
            ).upper()
            == "INSUFFICIENT EVIDENCE"
            else "The production rule set did not pass all v3.5 robustness checks."
        )
    )

    _verdict_box(
        verdict,
        message,
    )

    if promotion:
        st.success(
            "The current production rules also meet the internal promotion-quality "
            "screen. This supports keeping/testing them; it does not guarantee "
            "future profitability."
        )

    v1, v2, v3, v4 = st.columns(
        4
    )

    v1.metric(
        "Positive WF folds",
        _percent_ratio(
            full_validation.get(
                "positive_fold_ratio"
            )
        ),
    )

    v2.metric(
        "Aggregate OOS expectancy",
        _metric_value(
            full_validation.get(
                "aggregate_oos_expectancy_r"
            ),
            3,
            " R",
        ),
    )

    v3.metric(
        "Aggregate OOS PF",
        _metric_value(
            full_validation.get(
                "aggregate_oos_profit_factor"
            ),
            2,
        ),
    )

    v4.metric(
        "Worst fold expectancy",
        _metric_value(
            full_validation.get(
                "worst_fold_expectancy_r"
            ),
            3,
            " R",
        ),
    )

    ci_low = full_validation.get(
        "bootstrap_expectancy_low_r"
    )

    ci_high = full_validation.get(
        "bootstrap_expectancy_high_r"
    )

    if (
        ci_low is not None
        or ci_high is not None
    ):
        st.caption(
            "95% bootstrap expectancy interval: "
            f"{_metric_value(ci_low, 3)} R to "
            f"{_metric_value(ci_high, 3)} R"
        )

    walk_forward = full_validation.get(
        "walk_forward",
        {},
    )

    if isinstance(
        walk_forward,
        dict,
    ):
        fold_results = _safe_df(
            walk_forward.get(
                "fold_results"
            )
        )

        if not fold_results.empty:
            with st.expander(
                "Show walk-forward folds"
            ):
                st.dataframe(
                    fold_results,
                    width="stretch",
                    hide_index=True,
                )

    _validation_notes(
        full_validation.get(
            "notes",
            [],
        )
    )


# ============================================================
# FAST PROFILE CARD
# ============================================================

def _render_fast_profile_card(
    row,
):
    with st.container(
        border=True
    ):
        st.markdown(
            f"### {row.get('profile', 'Research profile')}"
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Swing threshold",
            _metric_value(
                row.get(
                    "swing_threshold"
                ),
                1,
            ),
        )

        c2.metric(
            "Intraday threshold",
            _metric_value(
                row.get(
                    "intraday_threshold"
                ),
                1,
            ),
        )

        c3.metric(
            "Entry quality",
            _metric_value(
                row.get(
                    "entry_quality"
                ),
                1,
            ),
        )

        c4, c5, c6 = st.columns(
            3
        )

        c4.metric(
            "All candidates",
            _metric_value(
                row.get(
                    "all_candidates"
                ),
                0,
            ),
        )

        c5.metric(
            "Later-period candidates",
            _metric_value(
                row.get(
                    "out_of_sample_candidates"
                ),
                0,
            ),
        )

        c6.metric(
            "Candidate-positive folds",
            _percent_ratio(
                row.get(
                    "candidate_positive_fold_ratio"
                )
            ),
        )

        st.caption(
            "Candidate stability only â this profile has not been proven profitable "
            "by this fast diagnostic."
        )


# ============================================================
# PORTFOLIO CALIBRATION DISPLAY
# ============================================================

def _extract_portfolio_comparison(
    backtest_result,
):
    """
    Look for a portfolio-calibration comparison in several backward/future
    compatible locations.
    """

    if not isinstance(
        backtest_result,
        dict,
    ):
        return pd.DataFrame()

    direct_keys = [
        "calibration_comparison",
        "portfolio_calibration_comparison",
        "adaptive_calibration_comparison",
    ]

    for key in direct_keys:
        candidate = _safe_df(
            backtest_result.get(
                key
            )
        )

        if not candidate.empty:
            return candidate

    for parent_key in (
        "calibration",
        "adaptive_calibration",
        "portfolio_calibration",
    ):
        parent = backtest_result.get(
            parent_key
        )

        if isinstance(
            parent,
            dict,
        ):
            candidate = _safe_df(
                parent.get(
                    "comparison"
                )
            )

            if not candidate.empty:
                return candidate

    # Session-state route allows app.py to store a portfolio calibration
    # later without changing this UI.
    session_candidates = [
        "latest_portfolio_calibration",
        "latest_adaptive_calibration",
        "latest_v35_portfolio_calibration",
    ]

    for key in session_candidates:
        parent = st.session_state.get(
            key
        )

        if isinstance(
            parent,
            dict,
        ):
            candidate = _safe_df(
                parent.get(
                    "comparison"
                )
            )

            if not candidate.empty:
                return candidate

    return pd.DataFrame()


def _render_portfolio_calibration(
    comparison,
):
    st.subheader(
        "Portfolio calibration evidence"
    )

    st.caption(
        "This section is the profitability-based evidence layer. Profiles shown "
        "here must come from actual portfolio replay, including entries, exits, "
        "position sizing, slippage, fees, and v3.5 walk-forward validation."
    )

    if comparison.empty:
        st.info(
            "No portfolio-replay calibration comparison is attached yet. "
            "The fast profile diagnostic below can identify reachable thresholds, "
            "but it cannot prove profitability."
        )
        return

    ranked = rank_portfolio_calibration(
        comparison
    )

    verdict = portfolio_calibration_verdict(
        ranked
    )

    _verdict_box(
        verdict.get(
            "verdict"
        ),
        verdict.get(
            "message",
            "",
        ),
    )

    if ranked.empty:
        return

    top = ranked.iloc[
        0
    ]

    st.markdown(
        f"### Top evidence-ranked profile: {top.get('profile', 'Unknown')}"
    )

    p1, p2, p3, p4 = st.columns(
        4
    )

    p1.metric(
        "Trades",
        _metric_value(
            top.get(
                "trades"
            ),
            0,
        ),
    )

    p2.metric(
        "Aggregate OOS expectancy",
        _metric_value(
            top.get(
                "aggregate_oos_expectancy_r"
            ),
            3,
            " R",
        ),
    )

    p3.metric(
        "Aggregate OOS PF",
        _metric_value(
            top.get(
                "aggregate_oos_profit_factor"
            ),
            2,
        ),
    )

    p4.metric(
        "Positive folds",
        _percent_ratio(
            top.get(
                "positive_fold_ratio"
            )
        ),
    )

    q1, q2, q3 = st.columns(
        3
    )

    q1.metric(
        "Worst fold expectancy",
        _metric_value(
            top.get(
                "worst_fold_expectancy_r"
            ),
            3,
            " R",
        ),
    )

    q2.metric(
        "Max drawdown",
        _metric_value(
            top.get(
                "max_drawdown_pct"
            ),
            2,
            "%",
        ),
    )

    q3.metric(
        "Evidence score",
        _metric_value(
            top.get(
                "v35_evidence_score"
            ),
            2,
        ),
    )

    if bool(
        top.get(
            "promotion_candidate_v35",
            False,
        )
    ):
        st.success(
            "This profile passed the v3.5 promotion-review screen. "
            "It still should not replace production rules until it survives "
            "additional non-overlapping tests and paper trading."
        )

    elif bool(
        top.get(
            "research_eligible_v35",
            False,
        )
    ):
        st.warning(
            "This profile has meaningful research evidence but does not meet "
            "the stronger promotion-review standard."
        )

    else:
        st.warning(
            "This profile does not yet have enough robust evidence for promotion."
        )

    with st.expander(
        "Show full portfolio-calibration table"
    ):
        st.dataframe(
            ranked,
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "Download v3.5 portfolio calibration",
        data=(
            ranked
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),
        file_name="v3_5_portfolio_calibration.csv",
        mime="text/csv",
        width="stretch",
        key="v35_download_portfolio_calibration",
    )


# ============================================================
# MAIN UI
# ============================================================

def render_calibration_lab(
    backtest_result: dict,
    default_profiles: int = 6,
):
    """
    Render the complete v3.5 calibration and validation lab.

    Expected input:
        result returned by swing_backtest()
    """

    st.header(
        "v3.5 Calibration & Walk-Forward Validation"
    )

    st.caption(
        "Research only. Production validation is based on simulated completed "
        "trades. Fast threshold diagnostics reuse the cached signal audit and "
        "measure reachability/stability only. Neither section automatically "
        "changes the live scanner."
    )

    if not isinstance(
        backtest_result,
        dict,
    ):
        st.info(
            "Run the production backtest first."
        )
        return

    signal_log = _safe_df(
        backtest_result.get(
            "signal_log"
        )
    )

    # ========================================================
    # 1. ACTUAL PRODUCTION PERFORMANCE VALIDATION
    # ========================================================

    _render_production_validation(
        backtest_result
    )

    st.divider()

    # ========================================================
    # 2. ACTUAL PORTFOLIO CALIBRATION, IF PRESENT
    # ========================================================

    portfolio_comparison = (
        _extract_portfolio_comparison(
            backtest_result
        )
    )

    _render_portfolio_calibration(
        portfolio_comparison
    )

    st.divider()

    # ========================================================
    # 3. FAST SIGNAL-LOG DIAGNOSTICS
    # ========================================================

    st.subheader(
        "Fast threshold reachability diagnostics"
    )

    st.warning(
        "This section does NOT measure profit. It only tells us whether "
        "alternate thresholds would have produced historical candidates and "
        "whether candidate frequency persisted through time."
    )

    if signal_log.empty:
        st.info(
            "The most recent backtest does not contain a usable historical "
            "signal audit, so fast threshold diagnostics are unavailable."
        )
        return

    candidate_count = len(
        signal_log
    )

    swing_series = pd.to_numeric(
        signal_log.get(
            "swing_score",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    intraday_series = pd.to_numeric(
        signal_log.get(
            "intraday_score",
            pd.Series(
                dtype=float
            ),
        ),
        errors="coerce",
    )

    d1, d2, d3 = st.columns(
        3
    )

    d1.metric(
        "Historical observations",
        f"{candidate_count:,}",
    )

    d2.metric(
        "Maximum Swing Score",
        (
            f"{swing_series.max():.1f}"
            if not swing_series.dropna().empty
            else "â"
        ),
    )

    d3.metric(
        "Maximum Intraday Score",
        (
            f"{intraday_series.max():.1f}"
            if not intraday_series.dropna().empty
            else "â"
        ),
    )

    if (
        not swing_series.dropna().empty
        and swing_series.max()
        < 85
    ):
        st.warning(
            "No observation in this backtest reached the production Swing "
            "Score threshold of 85. Production BUYs therefore could not occur "
            "in this sample regardless of the remaining gates."
        )

    if (
        not intraday_series.dropna().empty
        and intraday_series.max()
        < 85
    ):
        st.warning(
            "No observation in this backtest reached the production Intraday "
            "Score threshold of 85."
        )

    st.markdown(
        "#### Production-gate bottlenecks"
    )

    bottlenecks = (
        production_gate_bottlenecks(
            signal_log
        )
    )

    if bottlenecks.empty:
        st.info(
            "No gate diagnostics are available."
        )
    else:
        st.dataframe(
            bottlenecks,
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        "Score distribution and threshold reachability"
    ):
        distribution = score_distribution(
            signal_log
        )

        if distribution.empty:
            st.write(
                "No score-distribution data are available."
            )
        else:
            st.dataframe(
                distribution,
                width="stretch",
                hide_index=True,
            )

    st.markdown(
        "#### Run bounded fast profiles"
    )

    profile_count = st.slider(
        "Fast diagnostic profiles",
        min_value=1,
        max_value=8,
        value=min(
            max(
                int(
                    default_profiles
                ),
                1,
            ),
            8,
        ),
        step=1,
        key="v35_fast_calibration_profiles",
        help=(
            "These are candidate-log diagnostics only. "
            "They do not rerun market data or simulate portfolio returns."
        ),
    )

    run_calibration = st.button(
        "RUN FAST THRESHOLD DIAGNOSTIC",
        type="primary",
        width="stretch",
        key="v35_run_fast_calibration",
    )

    if run_calibration:

        progress = st.progress(
            0
        )

        status = st.status(
            "Starting fast threshold diagnostic...",
            expanded=True,
        )

        def update_progress(
            done,
            total,
            profile_name,
        ):
            pct = int(
                done
                / max(
                    total,
                    1,
                )
                * 100
            )

            progress.progress(
                min(
                    pct,
                    100,
                )
            )

            status.write(
                f"{done}/{total} testing {profile_name}"
            )

        try:

            result = run_fast_calibration(
                signal_log,
                max_profiles=profile_count,
                train_fraction=0.70,
                progress_callback=update_progress,
                adaptive=True,
                folds=4,
            )

            st.session_state[
                "latest_fast_calibration"
            ] = result

            progress.progress(
                100
            )

            status.update(
                label="Fast threshold diagnostic complete.",
                state="complete",
                expanded=False,
            )

        except Exception as exc:

            status.update(
                label="Fast diagnostic stopped because of an error.",
                state="error",
            )

            st.error(
                str(
                    exc
                )
            )

    calibration_result = st.session_state.get(
        "latest_fast_calibration"
    )

    if not isinstance(
        calibration_result,
        dict,
    ):
        st.info(
            "Run the fast threshold diagnostic to compare bounded "
            "candidate-reachability profiles."
        )
        return

    if calibration_result.get(
        "status"
    ) != "COMPLETE":
        st.info(
            calibration_result.get(
                "message",
                "Fast calibration has not completed.",
            )
        )
        return

    summary = _safe_df(
        calibration_result.get(
            "summary"
        )
    )

    st.success(
        calibration_result.get(
            "message",
            "Fast diagnostic complete.",
        )
    )

    s1, s2, s3 = st.columns(
        3
    )

    s1.metric(
        "Candidate observations",
        calibration_result.get(
            "candidate_count",
            0,
        ),
    )

    s2.metric(
        "In-sample observations",
        calibration_result.get(
            "in_sample_count",
            0,
        ),
    )

    s3.metric(
        "Later-period observations",
        calibration_result.get(
            "out_of_sample_count",
            0,
        ),
    )

    if calibration_result.get(
        "production_reachable",
        False,
    ):
        st.success(
            "The production Swing Score threshold of 85 was reachable "
            "at least once in this sample."
        )
    else:
        st.warning(
            "The production Swing Score threshold of 85 was not reached "
            "in this sample."
        )

    if summary.empty:
        st.info(
            "No fast-profile results were generated."
        )
        return

    st.markdown(
        "### Candidate-stability profiles"
    )

    st.caption(
        "The ordering below favors profiles that continue producing candidates "
        "in later periods and across candidate-stability folds. It is deliberately "
        "not a profitability ranking."
    )

    for _, row in summary.head(
        6
    ).iterrows():

        _render_fast_profile_card(
            row
        )

    with st.expander(
        "Show full fast-diagnostic table"
    ):
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "Download fast threshold diagnostics",
        data=(
            summary
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),
        file_name="v3_5_fast_threshold_diagnostics.csv",
        mime="text/csv",
        width="stretch",
        key="v35_download_fast_calibration",
    )

    st.divider()

    st.caption(
        "v3.5 rule: fast candidate diagnostics may identify thresholds worth "
        "testing, but only actual portfolio replay plus multi-fold walk-forward "
        "validation can support a promotion review. Live production rules remain "
        "unchanged automatically."
    )
