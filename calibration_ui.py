"""Streamlit renderer for v3.4 calibration and walk-forward research."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from calibration import calibration_summary, score_distribution, threshold_reachability, bottleneck_report, promotion_summary


def render_calibration_lab(result: dict) -> None:
    st.divider()
    st.header("v3.4 Calibration & Walk-Forward Lab")
    st.caption("Research only. This lab never changes the live BUY thresholds automatically.")
    if not isinstance(result, dict):
        st.info("Run a historical backtest first.")
        return
    signal_log = result.get("signal_log")
    if signal_log is None or not isinstance(signal_log, pd.DataFrame) or signal_log.empty:
        st.info("Run a historical test that produces a candidate signal log first.")
        return

    summary = calibration_summary(signal_log, 85, 85)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates", summary.get("observations", 0))
    c2.metric("Max Swing", f'{summary.get("swing_max", float("nan")):.1f}')
    c3.metric("Swing 95th %ile", f'{summary.get("swing_p95", float("nan")):.1f}')
    c4.metric("Max Intraday", f'{summary.get("intraday_max", float("nan")):.1f}')
    if summary.get("warning"):
        st.warning(summary["warning"] + " This is a calibration warning, not permission to lower the live gate.")

    st.subheader("Production-gate bottlenecks")
    bottlenecks = bottleneck_report(signal_log)
    if not bottlenecks.empty:
        st.dataframe(bottlenecks, use_container_width=True, hide_index=True)

    with st.expander("Score distribution and threshold reachability"):
        dist = score_distribution(signal_log)
        if not dist.empty:
            st.dataframe(dist, use_container_width=True, hide_index=True)
        reach = threshold_reachability(signal_log)
        if not reach.empty:
            st.dataframe(reach, use_container_width=True, hide_index=True)

    cal = result.get("calibration_result")
    st.subheader("Actual portfolio-simulator calibration")
    if not cal:
        st.info("This backtest did not run the v3.4 adaptive calibration. Enable the v3.4 calibration run in the Backtester and rerun it.")
        return

    comp = cal.get("comparison", pd.DataFrame())
    if comp is None or comp.empty:
        st.warning("Calibration ran but returned no comparison rows.")
        return

    p = promotion_summary(cal)
    if p["status"] == "REVIEW":
        st.warning(p["message"])
    else:
        st.success(p["message"] if p["status"] == "KEEP_PRODUCTION" else p.get("message", "Calibration complete."))

    preferred = ["profile", "production_rules", "swing_score_gate", "intraday_score_gate", "entry_quality_gate",
                 "leadership_gate", "trades", "win_rate_pct", "expectancy_r", "profit_factor", "return_pct",
                 "max_drawdown_pct", "out_of_sample_trades", "out_of_sample_win_rate_pct",
                 "out_of_sample_expectancy_r", "out_of_sample_profit_factor", "bootstrap_expectancy_low_r",
                 "confidence_grade", "validation_pass", "research_eligible", "promotion_candidate", "research_score"]
    cols = [c for c in preferred if c in comp.columns]
    st.dataframe(comp[cols], use_container_width=True, hide_index=True)

    st.download_button("Download v3.4 calibration comparison", data=comp.to_csv(index=False).encode("utf-8"),
                       file_name="v3_4_actual_simulator_calibration.csv", mime="text/csv", use_container_width=True)
    st.info("A profile must have enough total and out-of-sample trades, positive OOS expectancy, acceptable drawdown, and a positive bootstrap lower bound before it is even flagged for review. Live rules remain unchanged until you deliberately change them.")
