"""
Institutional Swing Scanner v3.5.1
Calibration UI.

Research only.

This version is designed to work with the corrected v3.5.1 calibration engine.

PRODUCTION CONTROL
    Uses the original production intraday BUY label.

RESEARCH PROFILES
    Use their own numeric Intraday Score threshold and do not remain
    artificially blocked by the production classifier's BUY label.

The UI also displays:
- Production bottlenecks
- Score distributions
- Corrected threshold calibration
- Candidate stability
- Sequential gate funnels
- Independent gate failures
- Research candidates
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
    if isinstance(value, pd.DataFrame):
        return value

    return pd.DataFrame()


def _number(value, digits=1):
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
                "Production control: requires the original production "
                "intraday BUY label."
            )

        else:
            st.caption(
                "Research profile: uses its own numeric Intraday Score "
                "threshold and does not require the production BUY label."
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Swing threshold",
            _number(
                row.get(
                    "swing_threshold"
                ),
                1,
            ),
        )

        c2.metric(
            "Intraday threshold",
            _number(
                row.get(
                    "intraday_threshold"
                ),
                1,
            ),
        )

        c3.metric(
            "Entry quality",
            _number(
                row.get(
                    "entry_quality"
                ),
                1,
            ),
        )

        c4, c5 = st.columns(2)

        c4.metric(
            "All candidates",
            _number(
                row.get(
                    "all_candidates"
                ),
                0,
            ),
        )

        c5.metric(
            "Later-period candidates",
            _number(
                row.get(
                    "later_period_candidates"
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
                row.get(
                    "stability_ratio"
                ),
                2,
            ),
        )

        st.caption(
            "Candidate stability only. This profile has not yet been "
            "proven profitable."
        )


# ============================================================
# MAIN CALIBRATION UI
# ============================================================

def render_calibration_lab(
    backtest_result: dict,
    default_profiles=6,
):
    st.header(
        "v3.5.1 Corrected Calibration Lab"
    )

    st.caption(
        "Research only. The live production scanner remains unchanged."
    )

    st.info(
        "Research profiles use their own numeric Intraday Score threshold "
        "instead of being forced to pass the production intraday BUY label."
    )

    # ========================================================
    # REQUIRE A BACKTEST RESULT
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
            "The latest backtest does not contain a usable historical "
            "signal log."
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
        bottlenecks = pd.DataFrame()

        st.warning(
            f"Production bottleneck analysis could not be generated: {exc}"
        )

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
            distribution = score_distribution(
                signal_log
            )
        except Exception as exc:
            distribution = pd.DataFrame()

            st.warning(
                f"Score distribution could not be generated: {exc}"
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
        "Run corrected threshold diagnostic"
    )

    st.write(
        "This test measures candidate reachability and stability. "
        "It does not yet claim profitability."
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
        key="v351_profile_count",
    )

    run = st.button(
        "RUN v3.5.1 CORRECTED CALIBRATION",
        type="primary",
        width="stretch",
        key="run_v351_calibration",
    )

    # ========================================================
    # RUN CALIBRATION
    # ========================================================

    if run:
        progress = st.progress(0)

        status = st.status(
            "Starting corrected calibration...",
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
            result = run_fast_calibration(
                signal_log,
                max_profiles=profile_count,
                train_fraction=0.70,
                stability_folds=4,
                progress_callback=update,
            )

            st.session_state[
                "v351_calibration_result"
            ] = result

            progress.progress(100)

            status.update(
                label=(
                    "v3.5.1 corrected calibration complete."
                ),
                state="complete",
                expanded=False,
            )

        except Exception as exc:
            status.update(
                label="Calibration failed.",
                state="error",
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
        "v351_calibration_result"
    )

    if not isinstance(
        result,
        dict,
    ):
        st.info(
            "Run the corrected calibration above."
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

    st.divider()

    st.success(
        result.get(
            "message",
            "Calibration complete.",
        )
    )

    # ========================================================
    # SAMPLE COUNTS
    # ========================================================

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Candidate observations",
        result.get(
            "candidate_count",
            0,
        ),
    )

    r2.metric(
        "In-sample observations",
        result.get(
            "in_sample_count",
            0,
        ),
    )

    later_period_count = result.get(
        "later_period_count",
        result.get(
            "out_of_sample_count",
            0,
        ),
    )

    r3.metric(
        "Later-period observations",
        later_period_count,
    )

    # ========================================================
    # PRODUCTION REACHABILITY
    # ========================================================

    if result.get(
        "production_swing_reachable",
        result.get(
            "production_reachable",
            False,
        ),
    ):
        st.success(
            "The production Swing Score threshold of 85 was reached "
            "at least once."
        )

    else:
        st.warning(
            "The production Swing Score threshold of 85 was never reached."
        )

    production_intraday_reachable = result.get(
        "production_intraday_reachable"
    )

    if production_intraday_reachable is not None:
        if production_intraday_reachable:
            st.success(
                "The production Intraday Score threshold of 85 was reached "
                "at least once."
            )

        else:
            st.warning(
                "The production Intraday Score threshold of 85 was never reached."
            )

    # ========================================================
    # RESEARCH CANDIDATE CHECK
    # ========================================================

    any_research_candidates = result.get(
        "any_research_candidates"
    )

    if any_research_candidates is not None:
        if any_research_candidates:
            st.success(
                "At least one corrected research profile produced candidates. "
                "The production intraday-label lock is no longer blocking "
                "research testing."
            )

        else:
            st.error(
                "The tested research profiles still produced zero candidates. "
                "The profile gate funnels below should identify the next "
                "actual bottleneck."
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
            "No profile summary was produced."
        )
        return

    st.subheader(
        "Candidate-stability profiles"
    )

    st.caption(
        "Ordering favors profiles that continue producing candidates "
        "in later periods and across chronological stability folds. "
        "This is not a profitability ranking."
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
    # CHOOSE PROFILE FOR DIAGNOSTICS
    # ========================================================

    best = result.get(
        "best_profile"
    )

    st.divider()

    st.subheader(
        "Next research profile"
    )

    selected_name = None

    if isinstance(
        best,
        dict,
    ):
        selected_name = best.get(
            "profile"
        )

        if selected_name:
            st.info(
                "Best candidate-stability profile in this test: "
                f"**{selected_name}**"
            )

    if not selected_name:
        st.warning(
            "No research profile clearly survived both the earlier and "
            "later chronological portions. Do not change live thresholds."
        )

        if "production_control" in summary.columns:
            research_rows = summary[
                ~summary[
                    "production_control"
                ].fillna(
                    False
                ).astype(
                    bool
                )
            ]

        else:
            research_rows = summary.copy()

        if research_rows.empty:
            return

        selected_name = (
            research_rows.iloc[
                -1
            ].get(
                "profile"
            )
        )

    if not selected_name:
        return

    # ========================================================
    # PROFILE-SPECIFIC DETAILS
    # ========================================================

    profile_results = result.get(
        "profile_results",
        {}
    )

    if not isinstance(
        profile_results,
        dict,
    ):
        profile_results = {}

    selected = profile_results.get(
        selected_name,
        {}
    )

    if not isinstance(
        selected,
        dict,
    ):
        selected = {}

    # ========================================================
    # SEQUENTIAL GATE FUNNEL
    # ========================================================

    st.markdown(
        f"### Gate funnel — {selected_name}"
    )

    st.caption(
        "This shows the exact stage at which candidates disappear."
    )

    funnel = _safe_df(
        selected.get(
            "gate_funnel"
        )
    )

    if funnel.empty:
        st.info(
            "No sequential gate funnel is available for this profile."
        )

    else:
        st.dataframe(
            funnel,
            width="stretch",
            hide_index=True,
        )

        if (
            "remaining"
            in funnel.columns
        ):
            remaining = pd.to_numeric(
                funnel[
                    "remaining"
                ],
                errors="coerce",
            )

            zero_rows = funnel[
                remaining.eq(
                    0
                )
            ]

            if not zero_rows.empty:
                first_zero = (
                    zero_rows.iloc[
                        0
                    ]
                )

                st.error(
                    "First gate reducing the surviving candidate pool "
                    "to zero: "
                    f"**{first_zero.get('gate', 'Unknown gate')}**"
                )

    # ========================================================
    # INDEPENDENT GATE FAILURES
    # ========================================================

    st.markdown(
        "### Independent gate failures"
    )

    failures = _safe_df(
        selected.get(
            "gate_failures"
        )
    )

    if failures.empty:
        st.info(
            "No independent gate-failure table is available."
        )

    else:
        st.dataframe(
            failures,
            width="stretch",
            hide_index=True,
        )

    # ========================================================
    # RESEARCH CANDIDATES
    # ========================================================

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
                ].head(
                    100
                ),
                width="stretch",
                hide_index=True,
            )

        else:
            st.dataframe(
                candidates.head(
                    100
                ),
                width="stretch",
                hide_index=True,
            )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.download_button(
        "Download v3.5.1 calibration summary",
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
            "v3_5_1_calibration_summary.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    st.divider()

    st.warning(
        "Do not change the live scanner thresholds based on candidate "
        "reachability alone. A research profile must next be tested through "
        "the portfolio simulator for trade count, expectancy, profit factor, "
        "drawdown and out-of-sample performance."
    )
