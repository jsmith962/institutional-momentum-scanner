"""
v3.4.2 Streamlit UI for fast adaptive calibration.

Research only.

This UI reuses the historical signal log already created by the
production backtest. It does not request market data and does not
change live production thresholds.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from calibration import (
    run_fast_calibration,
    production_gate_bottlenecks,
    score_distribution,
)


def _safe_df(value):
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _metric_value(value, digits=2):
    try:
        if pd.isna(value):
            return "—"

        number = float(value)

        if digits == 0:
            return f"{number:,.0f}"

        return f"{number:,.{digits}f}"

    except Exception:
        return "—"


def _render_profile_card(row):
    with st.container(border=True):

        st.markdown(
            f"### {row.get('profile', 'Calibration profile')}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Swing threshold",
            _metric_value(
                row.get("swing_threshold"),
                1,
            ),
        )

        c2.metric(
            "Intraday threshold",
            _metric_value(
                row.get("intraday_threshold"),
                1,
            ),
        )

        c3.metric(
            "Entry quality",
            _metric_value(
                row.get("entry_quality"),
                1,
            ),
        )

        c4, c5, c6 = st.columns(3)

        c4.metric(
            "Candidates",
            _metric_value(
                row.get("all_candidates"),
                0,
            ),
        )

        c5.metric(
            "OOS candidates",
            _metric_value(
                row.get("out_of_sample_candidates"),
                0,
            ),
        )

        c6.metric(
            "Stability ratio",
            _metric_value(
                row.get("stability_ratio"),
                2,
            ),
        )

        candidate_rate = row.get(
            "candidate_rate_pct",
            0,
        )

        oos_rate = row.get(
            "oos_candidate_rate_pct",
            0,
        )

        st.caption(
            f"Historical candidate rate: "
            f"{_metric_value(candidate_rate, 2)}% | "
            f"Later-period candidate rate: "
            f"{_metric_value(oos_rate, 2)}%"
        )


def render_calibration_lab(
    backtest_result: dict,
    default_profiles: int = 6,
):
    """
    Render the v3.4.2 calibration lab.

    Expected input:
        result returned by swing_backtest()
    """

    st.header(
        "v3.4.2 Fast Adaptive Calibration"
    )

    st.caption(
        "Research only. This lab reuses the historical candidate log "
        "already produced by the backtest. It does not rerun Alpaca data "
        "downloads or rebuild every indicator for every threshold profile."
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

    if signal_log.empty:
        st.info(
            "The most recent backtest does not contain a usable "
            "historical signal audit."
        )
        return

    # ========================================================
    # QUICK DIAGNOSTICS
    # ========================================================

    candidate_count = len(
        signal_log
    )

    swing_series = pd.to_numeric(
        signal_log.get(
            "swing_score",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    intraday_series = pd.to_numeric(
        signal_log.get(
            "intraday_score",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Historical observations",
        f"{candidate_count:,}",
    )

    c2.metric(
        "Maximum Swing Score",
        (
            f"{swing_series.max():.1f}"
            if not swing_series.dropna().empty
            else "—"
        ),
    )

    c3.metric(
        "Maximum Intraday Score",
        (
            f"{intraday_series.max():.1f}"
            if not intraday_series.dropna().empty
            else "—"
        ),
    )

    if (
        not swing_series.dropna().empty
        and swing_series.max() < 85
    ):
        st.warning(
            "No historical observation in this sample reached the "
            "production Swing Score threshold of 85. That means the "
            "production strategy could not create a BUY in this sample "
            "regardless of the other gates."
        )

    # ========================================================
    # PRODUCTION GATE BOTTLENECKS
    # ========================================================

    st.subheader(
        "Production-gate bottlenecks"
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
        "Run bounded research profiles"
    )

    st.write(
        "This run tests a small number of alternate threshold combinations "
        "against the cached historical candidate log."
    )

    profile_count = st.slider(
        "Calibration profiles",
        min_value=1,
        max_value=10,
        value=min(
            max(
                int(default_profiles),
                1,
            ),
            10,
        ),
        step=1,
        key="v342_calibration_profiles",
        help=(
            "Six profiles is recommended for a fast first pass. "
            "More profiles remain inexpensive because no market data "
            "is downloaded again."
        ),
    )

    run_calibration = st.button(
        "RUN FAST ADAPTIVE CALIBRATION",
        type="primary",
        width="stretch",
        key="v342_run_fast_calibration",
    )

    if run_calibration:

        progress = st.progress(
            0
        )

        status = st.status(
            "Starting fast calibration...",
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
                f"{done}/{total} testing "
                f"{profile_name}"
            )

        try:

            result = run_fast_calibration(
                signal_log,
                max_profiles=profile_count,
                train_fraction=0.70,
                progress_callback=update_progress,
            )

            st.session_state[
                "latest_fast_calibration"
            ] = result

            progress.progress(
                100
            )

            status.update(
                label="Fast calibration complete.",
                state="complete",
                expanded=False,
            )

        except Exception as exc:

            status.update(
                label="Calibration stopped because of an error.",
                state="error",
            )

            st.error(
                str(
                    exc
                )
            )

    # ========================================================
    # DISPLAY SAVED RESULT
    # ========================================================

    calibration_result = (
        st.session_state.get(
            "latest_fast_calibration"
        )
    )

    if not isinstance(
        calibration_result,
        dict,
    ):
        st.info(
            "Run Fast Adaptive Calibration to compare bounded "
            "research profiles."
        )
        return

    if calibration_result.get(
        "status"
    ) != "COMPLETE":
        st.info(
            calibration_result.get(
                "message",
                "Calibration has not completed.",
            )
        )
        return

    summary = _safe_df(
        calibration_result.get(
            "summary"
        )
    )

    st.divider()

    st.subheader(
        "Calibration results"
    )

    st.success(
        calibration_result.get(
            "message",
            "Calibration complete.",
        )
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Candidate observations",
        calibration_result.get(
            "candidate_count",
            0,
        ),
    )

    r2.metric(
        "In-sample observations",
        calibration_result.get(
            "in_sample_count",
            0,
        ),
    )

    r3.metric(
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
            "At least one historical observation reached the production "
            "Swing Score threshold of 85."
        )
    else:
        st.warning(
            "The production Swing Score threshold of 85 was not reached "
            "in this sample."
        )

    if summary.empty:
        st.info(
            "No profile results were generated."
        )
        return

    st.markdown(
        "### Ranked calibration profiles"
    )

    st.caption(
        "Ranking favors profiles that continue producing candidates in "
        "the later chronological holdout while remaining selective. "
        "This is still not a profitability ranking."
    )

    for _, row in summary.head(
        6
    ).iterrows():

        _render_profile_card(
            row
        )

    with st.expander(
        "Show full calibration table"
    ):
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

    st.download_button(
        "Download calibration results",
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
            "v3_4_2_fast_calibration.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    # ========================================================
    # BEST RESEARCH PROFILE
    # ========================================================

    best_profile = calibration_result.get(
        "best_profile"
    )

    st.divider()

    st.subheader(
        "Research interpretation"
    )

    if not best_profile:

        st.warning(
            "None of the tested profiles produced candidates in the "
            "later out-of-sample portion. Do not change the live thresholds "
            "based on this sample."
        )

    else:

        profile_name = best_profile.get(
            "profile",
            "Unknown profile",
        )

        st.info(
            f"Best later-period candidate stability in this bounded test: "
            f"**{profile_name}**."
        )

        st.write(
            f"Swing threshold: "
            f"**{best_profile.get('swing_threshold')}**"
        )

        st.write(
            f"Intraday threshold: "
            f"**{best_profile.get('intraday_threshold')}**"
        )

        st.write(
            f"Entry quality: "
            f"**{best_profile.get('entry_quality')}**"
        )

        st.write(
            f"Out-of-sample candidates: "
            f"**{best_profile.get('out_of_sample_candidates')}**"
        )

        st.write(
            f"Stability ratio: "
            f"**{best_profile.get('stability_ratio')}**"
        )

        st.warning(
            "Do not promote this profile to the live scanner yet. "
            "The next required step is to run the actual portfolio simulator "
            "on the strongest research profiles and compare expectancy, "
            "profit factor, drawdown and trade count."
        )

    st.divider()

    st.caption(
        "The production scanner remains unchanged. Calibration is "
        "research-only until a profile proves itself across multiple "
        "non-overlapping periods and paper trading."
    )
