"""
Institutional Swing Scanner v3.5.2
Calibration UI

Research only.

This version removes unnecessary calibration imports that could
cause ImportError failures when calibration.py changes.

PRODUCTION CONTROL
    Uses the original production thresholds.

RESEARCH PROFILES
    Use their own numeric Intraday Score thresholds.

The calibration system is diagnostic/research-only and does not
automatically alter the production scanner.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from calibration import (
    production_gate_bottlenecks,
    run_fast_calibration,
    score_distribution,
)


# ============================================================
# HELPERS
# ============================================================

def _safe_df(value):
    """Return a DataFrame or an empty DataFrame."""

    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def _number(value, digits=1):
    """Safely format numeric values for the UI."""

    try:
        if pd.isna(value):
            return "—"

        number = float(value)

        if digits == 0:
            return f"{number:,.0f}"

        return f"{number:,.{digits}f}"

    except Exception:
        return "—"


def _profile_card(row):
    """Render one calibration-profile summary card."""

    production = bool(
        row.get(
            "production_control",
            False,
        )
    )

    with st.container(border=True):

        st.markdown(
            f"### {row.get('profile', 'Profile')}"
        )

        if production:

            st.caption(
                "Production control profile."
            )

        else:

            st.caption(
                "Research profile using its own numeric thresholds."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Swing threshold",
            _number(
                row.get("swing_threshold"),
                1,
            ),
        )

        c2.metric(
            "Intraday threshold",
            _number(
                row.get("intraday_threshold"),
                1,
            ),
        )

        c3.metric(
            "Entry quality",
            _number(
                row.get("entry_quality"),
                1,
            ),
        )

        c4, c5 = st.columns(2)

        c4.metric(
            "All candidates",
            _number(
                row.get("all_candidates"),
                0,
            ),
        )

        c5.metric(
            "Later-period candidates",
            _number(
                row.get(
                    "later_period_candidates",
                    row.get(
                        "out_of_sample_candidates",
                        0,
                    ),
                ),
                0,
            ),
        )

        c6, c7 = st.columns(2)

        c6.metric(
            "Candidate-positive folds",
            (
                f"{_number(row.get('candidate_positive_folds_pct'), 0)}%"
            ),
        )

        c7.metric(
            "Stability ratio",
            _number(
                row.get("stability_ratio"),
                2,
            ),
        )

        st.caption(
            "Candidate stability only. This does not establish "
            "historical profitability."
        )


# ============================================================
# MAIN CALIBRATION UI
# ============================================================

def render_calibration_lab(
    backtest_result: dict,
    default_profiles=6,
):
    """
    Render the research calibration laboratory.

    Expected input:
        The result dictionary returned by swing_backtest().
    """

    st.header(
        "v3.5.2 Calibration Lab"
    )

    st.caption(
        "Research only. Production scanner thresholds remain unchanged."
    )

    st.info(
        "Calibration evaluates whether alternate research thresholds "
        "produce sufficiently stable historical candidates. "
        "It does not automatically change the live BUY rules."
    )

    # ========================================================
    # VALIDATE BACKTEST RESULT
    # ========================================================

    if not isinstance(
        backtest_result,
        dict,
    ):

        st.warning(
            "Run the production backtest first."
        )

        return

    signal_log = _safe_df(
        backtest_result.get(
            "signal_log"
        )
    )

    if signal_log.empty:

        st.warning(
            "The latest backtest does not contain a usable "
            "historical signal log."
        )

        return

    # ========================================================
    # HISTORICAL AUDIT
    # ========================================================

    st.subheader(
        "Historical audit"
    )

    swing = pd.to_numeric(
        signal_log.get(
            "swing_score",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    intraday = pd.to_numeric(
        signal_log.get(
            "intraday_score",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Observations",
        f"{len(signal_log):,}",
    )

    c2.metric(
        "Maximum Swing Score",
        (
            f"{swing.max():.1f}"
            if not swing.dropna().empty
            else "—"
        ),
    )

    c3.metric(
        "Maximum Intraday Score",
        (
            f"{intraday.max():.1f}"
            if not intraday.dropna().empty
            else "—"
        ),
    )

    # ========================================================
    # PRODUCTION BOTTLENECKS
    # ========================================================

    st.subheader(
        "Current production bottlenecks"
    )

    try:

        bottlenecks = (
            production_gate_bottlenecks(
                signal_log
            )
        )

    except Exception as exc:

        st.warning(
            "Production-gate diagnostics could not be generated."
        )

        st.caption(
            str(exc)
        )

        bottlenecks = pd.DataFrame()

    if bottlenecks.empty:

        st.info(
            "No production gate diagnostics are available."
        )

    else:

        st.dataframe(
            bottlenecks,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # SCORE DISTRIBUTIONS
    # ========================================================

    with st.expander(
        "Score distributions"
    ):

        try:

            distribution = (
                score_distribution(
                    signal_log
                )
            )

        except Exception as exc:

            distribution = pd.DataFrame()

            st.warning(
                "Score-distribution diagnostics could not be generated."
            )

            st.caption(
                str(exc)
            )

        if distribution.empty:

            st.write(
                "No score-distribution data available."
            )

        else:

            st.dataframe(
                distribution,
                width="stretch",
                hide_index=True,
            )

    st.divider()

    # ========================================================
    # CALIBRATION CONTROLS
    # ========================================================

    st.subheader(
        "Run threshold diagnostic"
    )

    st.write(
        "This test measures candidate reachability and stability "
        "across historical periods."
    )

    profile_count = st.slider(
        "Calibration profiles",
        min_value=2,
        max_value=10,
        value=min(
            max(
                int(default_profiles),
                2,
            ),
            10,
        ),
        step=1,
        key="v352_profile_count",
    )

    run = st.button(
        "RUN v3.5.2 CALIBRATION",
        type="primary",
        width="stretch",
        key="run_v352_calibration",
    )

    # ========================================================
    # RUN CALIBRATION
    # ========================================================

    if run:

        progress = st.progress(0)

        status = st.status(
            "Starting calibration...",
            expanded=True,
        )

        def update(
            completed,
            total,
            profile_name,
        ):

            pct = int(
                completed
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
                f"{completed}/{total}: {profile_name}"
            )

        try:

            # ------------------------------------------------
            # First try the v3.5 interface.
            # ------------------------------------------------

            try:

                result = run_fast_calibration(
                    signal_log,
                    max_profiles=profile_count,
                    train_fraction=0.70,
                    stability_folds=4,
                    progress_callback=update,
                )

            # ------------------------------------------------
            # Backward compatibility:
            # older calibration.py versions may not yet
            # accept stability_folds.
            # ------------------------------------------------

            except TypeError as exc:

                if "stability_folds" not in str(exc):
                    raise

                status.write(
                    "Using compatibility calibration mode."
                )

                result = run_fast_calibration(
                    signal_log,
                    max_profiles=profile_count,
                    train_fraction=0.70,
                    progress_callback=update,
                )

            st.session_state[
                "v352_calibration_result"
            ] = result

            progress.progress(100)

            status.update(
                label="Calibration complete.",
                state="complete",
                expanded=False,
            )

        except Exception as exc:

            status.update(
                label="Calibration failed.",
                state="error",
                expanded=True,
            )

            st.error(
                "The calibration engine returned an error."
            )

            st.exception(
                exc
            )

            return

    # ========================================================
    # LOAD SAVED RESULT
    # ========================================================

    result = st.session_state.get(
        "v352_calibration_result"
    )

    if not isinstance(
        result,
        dict,
    ):

        st.info(
            "Run the calibration above to create a research comparison."
        )

        return

    if result.get(
        "status"
    ) != "COMPLETE":

        st.warning(
            result.get(
                "message",
                "Calibration did not complete.",
            )
        )

        return

    # ========================================================
    # COMPLETION SUMMARY
    # ========================================================

    st.divider()

    st.success(
        result.get(
            "message",
            "Calibration complete.",
        )
    )

    candidate_count = result.get(
        "candidate_count",
        len(signal_log),
    )

    in_sample_count = result.get(
        "in_sample_count",
        0,
    )

    later_period_count = result.get(
        "later_period_count",
        result.get(
            "out_of_sample_count",
            0,
        ),
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Candidate observations",
        candidate_count,
    )

    r2.metric(
        "In-sample observations",
        in_sample_count,
    )

    r3.metric(
        "Later-period observations",
        later_period_count,
    )

    # ========================================================
    # REACHABILITY FLAGS
    # ========================================================

    production_swing_reachable = result.get(
        "production_swing_reachable",
        result.get(
            "production_reachable",
            False,
        ),
    )

    if production_swing_reachable:

        st.success(
            "The production Swing Score threshold of 85 "
            "was reached at least once."
        )

    else:

        st.warning(
            "The production Swing Score threshold of 85 "
            "was not reached in this historical sample."
        )

    production_intraday_reachable = result.get(
        "production_intraday_reachable"
    )

    if production_intraday_reachable is True:

        st.success(
            "The production Intraday Score threshold of 85 "
            "was reached at least once."
        )

    elif production_intraday_reachable is False:

        st.warning(
            "The production Intraday Score threshold of 85 "
            "was not reached in this historical sample."
        )

    any_research_candidates = result.get(
        "any_research_candidates"
    )

    if any_research_candidates is True:

        st.success(
            "At least one research profile produced historical candidates."
        )

    elif any_research_candidates is False:

        st.warning(
            "The tested research profiles produced no candidates. "
            "Review the gate diagnostics before changing any threshold."
        )

    # ========================================================
    # PROFILE SUMMARY
    # ========================================================

    summary = _safe_df(
        result.get(
            "summary"
        )
    )

    if summary.empty:

        st.warning(
            "No calibration profile summary was produced."
        )

        return

    st.subheader(
        "Candidate-stability profiles"
    )

    st.caption(
        "Profiles are research comparisons only. "
        "Candidate frequency is not the same as profitability."
    )

    for _, row in summary.iterrows():

        _profile_card(
            row
        )

    # ========================================================
    # FULL TABLE
    # ========================================================

    with st.expander(
        "Full profile comparison table"
    ):

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # SELECT PROFILE FOR DETAILED DIAGNOSTICS
    # ========================================================

    st.divider()

    st.subheader(
        "Profile diagnostics"
    )

    profile_results = result.get(
        "profile_results",
        result.get(
            "results",
            {},
        ),
    )

    if not isinstance(
        profile_results,
        dict,
    ):

        profile_results = {}

    available_profiles = [
        str(name)
        for name in summary[
            "profile"
        ].dropna().tolist()
        if str(name) in profile_results
    ]

    if not available_profiles:

        st.info(
            "Detailed gate diagnostics are not available "
            "for this calibration result."
        )

    else:

        best = result.get(
            "best_profile"
        )

        best_name = None

        if isinstance(
            best,
            dict,
        ):

            best_name = best.get(
                "profile"
            )

        default_index = 0

        if best_name in available_profiles:

            default_index = (
                available_profiles.index(
                    best_name
                )
            )

        selected_name = st.selectbox(
            "Profile to inspect",
            available_profiles,
            index=default_index,
            key="v352_profile_inspector",
        )

        selected = profile_results.get(
            selected_name,
            {},
        )

        if not isinstance(
            selected,
            dict,
        ):

            selected = {}

        # ====================================================
        # SEQUENTIAL GATE FUNNEL
        # ====================================================

        funnel = _safe_df(
            selected.get(
                "gate_funnel"
            )
        )

        if not funnel.empty:

            st.markdown(
                f"### Gate funnel — {selected_name}"
            )

            st.caption(
                "Shows where candidates disappear as gates "
                "are applied sequentially."
            )

            st.dataframe(
                funnel,
                width="stretch",
                hide_index=True,
            )

            if (
                "remaining"
                in funnel.columns
            ):

                zero_rows = funnel[
                    pd.to_numeric(
                        funnel[
                            "remaining"
                        ],
                        errors="coerce",
                    )
                    == 0
                ]

                if not zero_rows.empty:

                    first_zero = (
                        zero_rows.iloc[0]
                    )

                    st.warning(
                        "First gate reducing surviving candidates "
                        f"to zero: **{first_zero.get('gate', 'Unknown')}**"
                    )

        # ====================================================
        # INDEPENDENT GATE FAILURES
        # ====================================================

        failures = _safe_df(
            selected.get(
                "gate_failures"
            )
        )

        if not failures.empty:

            st.markdown(
                "### Independent gate failures"
            )

            st.dataframe(
                failures,
                width="stretch",
                hide_index=True,
            )

        # ====================================================
        # PROFILE CANDIDATES
        # ====================================================

        candidates = _safe_df(
            selected.get(
                "all_candidates"
            )
        )

        if not candidates.empty:

            st.markdown(
                "### Research candidates"
            )

            display_columns = [
                column
                for column in [
                    "symbol",
                    "session",
                    "signal_time",
                    "swing_score",
                    "intraday_score",
                    "entry_quality",
                    "market_score",
                    "leadership_percentile",
                    "distribution_days",
                    "reward_risk",
                    "setup",
                ]
                if column
                in candidates.columns
            ]

            if display_columns:

                st.dataframe(
                    candidates[
                        display_columns
                    ].head(100),
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.dataframe(
                    candidates.head(100),
                    width="stretch",
                    hide_index=True,
                )

    # ========================================================
    # DOWNLOAD SUMMARY
    # ========================================================

    st.download_button(
        "Download calibration summary",
        data=(
            summary
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),
        file_name=(
            "v3_5_2_calibration_summary.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    st.divider()

    st.warning(
        "Do not change live scanner thresholds from this diagnostic alone. "
        "A research profile should next pass portfolio replay, "
        "out-of-sample profitability validation, sufficient trade count, "
        "and paper trading before production consideration."
    )
