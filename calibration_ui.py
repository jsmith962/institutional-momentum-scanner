
"""
Drop-in Streamlit renderer for v3.3 calibration results.

Usage inside your backtester UI after:
    result = backtest(...)

    from calibration_ui import render_calibration_lab
    render_calibration_lab(result)
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from calibration import (
    CalibrationConfig,
    calibration_summary,
    score_distribution,
    threshold_reachability,
    threshold_matrix,
    best_stable_regions,
)


def render_calibration_lab(result: dict) -> None:
    st.divider()
    st.header("v3.3 Score Calibration Lab")
    st.caption(
        "Research-only calibration. This section does not change the live BUY rules."
    )

    signal_log = result.get("signal_log")
    if signal_log is None or not isinstance(signal_log, pd.DataFrame) or signal_log.empty:
        st.info("Run a historical test that produces a candidate signal log first.")
        return

    summary = calibration_summary(signal_log, 85, 85)
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical candidates", summary.get("observations", 0))
    c2.metric("Max Swing Score", f'{summary.get("swing_max", float("nan")):.1f}')
    c3.metric("Max Intraday Score", f'{summary.get("intraday_max", float("nan")):.1f}')

    if summary.get("warning"):
        st.warning(summary["warning"])

    st.subheader("Score distribution")
    dist = score_distribution(signal_log)
    if not dist.empty:
        st.dataframe(dist, use_container_width=True, hide_index=True)

    st.subheader("Threshold reachability")
    reach = threshold_reachability(signal_log)
    if not reach.empty:
        st.dataframe(reach, use_container_width=True, hide_index=True)

    st.subheader("Threshold matrix")
    st.caption(
        "The matrix preserves available non-score safety gates. "
        "It compares score thresholds without silently changing production settings."
    )

    cfg = CalibrationConfig(
        oos_fraction=0.30,
        min_oos_observations=20,
        min_total_observations=40,
        stability_radius=1,
    )
    matrix = threshold_matrix(signal_log, config=cfg)
    if matrix.empty:
        st.info("No matrix results available.")
        return

    if matrix["outcomes"].max() == 0:
        st.warning(
            "Candidate-level forward R outcomes are not present yet. "
            "Counts and score calibration are valid, but win rate / expectancy / "
            "profit factor are intentionally left blank rather than guessed."
        )
    else:
        st.success(
            "Forward outcomes detected. The matrix includes chronological held-out validation."
        )

    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.subheader("Most stable research regions")
    stable = best_stable_regions(matrix, limit=10)
    st.dataframe(stable, use_container_width=True, hide_index=True)

    st.download_button(
        "Download calibration matrix CSV",
        data=matrix.to_csv(index=False).encode("utf-8"),
        file_name="v3_3_threshold_calibration.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Do not change the live BUY threshold from this screen alone. "
        "A production change should require enough completed candidate outcomes, "
        "positive held-out expectancy, and a stable neighboring parameter region."
    )
