"""Institutional Swing Scanner v3.8 Production-vs-Challenger UI."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from production_vs_challenger import (
    DEFAULT_CHALLENGERS,
    ValidationProfile,
    run_production_vs_challenger_validation,
)


def _safe_df(value):
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _number(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "—"
        if digits == 0:
            return f"{float(value):,.0f}"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"


def _pct(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}%"
    except Exception:
        return "—"


def _profile_card(row):
    production = bool(row.get("production_control", False))
    with st.container(border=True):
        st.markdown(f"### {row.get('profile', 'Profile')}")
        st.caption("Production control. Live rules are unchanged." if production else "Research challenger. Not a live BUY rule.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Swing threshold", _number(row.get("swing_threshold"), 1))
        c2.metric("Intraday threshold", _number(row.get("intraday_threshold"), 1))
        c3.metric("Candidates", _number(row.get("candidate_observations"), 0))
        c4, c5, c6 = st.columns(3)
        c4.metric("OOS trades", _number(row.get("oos_trades"), 0))
        c5.metric("OOS return", _pct(row.get("oos_return_pct")))
        c6.metric("OOS win rate", _pct(row.get("oos_win_rate_pct")))
        c7, c8, c9 = st.columns(3)
        c7.metric("OOS profit factor", _number(row.get("oos_profit_factor"), 2))
        c8.metric("OOS expectancy", _pct(row.get("oos_expectancy_pct")))
        c9.metric("OOS max drawdown", _pct(row.get("oos_max_drawdown_pct")))


def render_production_vs_challenger_lab(enriched_signal_log: pd.DataFrame):
    st.header("v3.8 Production-vs-Challenger Portfolio Validation")
    st.caption("Research only. Production remains unchanged until a challenger survives out-of-sample replay and paper trading.")
    st.info("v3.8 compares the current 85/85 production control with bounded challenger rules on the same historical observations. It evaluates portfolio-level results rather than only forward-return averages.")

    df = _safe_df(enriched_signal_log)
    if df.empty:
        st.warning("Run v3.7 Forward Research first. v3.8 requires the enriched historical signal audit with forward returns.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Enriched observations", f"{len(df):,}")
    c2.metric("Symbols", f"{df['symbol'].nunique():,}" if "symbol" in df.columns else "0")
    c3.metric("Forward 10-session rows", f"{pd.to_numeric(df['forward_10d_pct'], errors='coerce').notna().sum():,}" if "forward_10d_pct" in df.columns else "0")

    st.divider()
    st.subheader("Replay settings")
    a1, a2, a3 = st.columns(3)
    starting_capital = a1.number_input("Starting capital", 500.0, 1_000_000.0, 2000.0, 500.0, key="v38_starting_capital")
    max_positions = a2.slider("Maximum open positions", 1, 10, 3, 1, key="v38_max_positions")
    risk_pct = a3.slider("Approx. risk per trade", 0.25, 2.0, 0.50, 0.25, key="v38_risk_pct") / 100.0
    b1, b2, b3 = st.columns(3)
    holding_sessions = b1.selectbox("Holding horizon", [1, 3, 5, 10, 20], index=3, key="v38_holding_sessions")
    slippage_bps = b2.slider("Slippage (bps/order)", 0, 25, 5, 1, key="v38_slippage")
    commission_bps = b3.slider("Fees (bps/order)", 0, 10, 0, 1, key="v38_commission")
    c1, c2 = st.columns(2)
    development_fraction = c1.slider("Development sample %", 50, 85, 70, 5, key="v38_dev_fraction") / 100.0
    minimum_oos_trades = c2.slider("Minimum OOS trades for promotion review", 10, 100, 20, 5, key="v38_min_oos_trades")

    st.divider()
    st.subheader("Challenger profiles")
    st.caption("The defaults are bounded around the v3.7 research result instead of running an unrestricted optimizer.")
    challengers = list(DEFAULT_CHALLENGERS)
    use_custom = st.toggle("Add one custom challenger", value=False, key="v38_custom_toggle")
    if use_custom:
        d1, d2 = st.columns(2)
        custom_swing = d1.number_input("Custom Swing threshold", 50.0, 90.0, 70.0, 2.5, key="v38_custom_swing")
        custom_intraday = d2.number_input("Custom Intraday threshold", 20.0, 90.0, 50.0, 5.0, key="v38_custom_intraday")
        custom_require_label = st.checkbox("Require original production intraday BUY label", value=False, key="v38_custom_label")
        challengers.append(ValidationProfile(name=f"Custom S{custom_swing:g}/I{custom_intraday:g}", swing_threshold=float(custom_swing), intraday_threshold=float(custom_intraday), require_production_intraday_label=bool(custom_require_label)))

    if st.button("RUN v3.8 PRODUCTION-vs-CHALLENGER VALIDATION", type="primary", width="stretch", key="run_v38_validation"):
        status = st.status("Running v3.8 production-vs-challenger replay...", expanded=True)
        progress = st.progress(0)
        try:
            progress.progress(15)
            status.write("Building production and challenger candidate sets...")
            result = run_production_vs_challenger_validation(
                df,
                challengers=challengers,
                starting_capital=starting_capital,
                max_positions=max_positions,
                risk_pct=risk_pct,
                holding_sessions=holding_sessions,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
                development_fraction=development_fraction,
                minimum_oos_trades=minimum_oos_trades,
            )
            progress.progress(90)
            status.write("Ranking out-of-sample evidence...")
            st.session_state["v38_validation_result"] = result
            progress.progress(100)
            status.update(label="v3.8 validation complete.", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="v3.8 validation failed.", state="error", expanded=True)
            st.exception(exc)
            return

    result = st.session_state.get("v38_validation_result")
    if not isinstance(result, dict):
        st.info("Run the v3.8 validation above.")
        return
    if result.get("status") != "COMPLETE":
        st.warning(result.get("message", "v3.8 did not complete."))
        return

    st.divider()
    st.success(result.get("message", "v3.8 completed."))
    promotion = result.get("promotion", {})
    verdict = promotion.get("status", "INSUFFICIENT")
    message = promotion.get("message", "")
    if verdict == "PROMISING":
        st.success("### Research verdict: PROMISING")
        st.write(message)
        st.warning("Do not replace production yet. The next step is paper trading the challenger while production continues as the control.")
    elif verdict == "MIXED":
        st.warning("### Research verdict: MIXED")
        st.write(message)
    elif verdict == "REJECT":
        st.error("### Research verdict: REJECT")
        st.write(message)
    else:
        st.info("### Research verdict: INSUFFICIENT")
        st.write(message)

    summary = _safe_df(result.get("summary"))
    if summary.empty:
        st.warning("No profile summary was produced.")
        return

    st.subheader("Production vs challenger comparison")
    for _, row in summary.iterrows():
        _profile_card(row)

    with st.expander("Full v3.8 comparison table"):
        st.dataframe(summary, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Out-of-sample ranking")
    ranking = summary.sort_values(["oos_return_pct", "oos_profit_factor", "oos_expectancy_pct", "oos_trades"], ascending=[False, False, False, False])
    display_columns = ["profile", "production_control", "candidate_observations", "oos_trades", "oos_return_pct", "oos_win_rate_pct", "oos_profit_factor", "oos_expectancy_pct", "oos_max_drawdown_pct", "dev_return_pct", "full_return_pct"]
    st.dataframe(ranking[[c for c in display_columns if c in ranking.columns]], width="stretch", hide_index=True)

    st.divider()
    st.subheader("Profile drill-down")
    selected_profile = st.selectbox("Profile to inspect", summary["profile"].tolist(), key="v38_profile_drilldown")
    selected = result.get("profile_results", {}).get(selected_profile, {})
    if selected:
        oos_replay = selected.get("oos_replay", {})
        oos_trades = _safe_df(oos_replay.get("trades"))
        oos_equity = _safe_df(oos_replay.get("equity"))
        candidates = _safe_df(selected.get("oos_candidates"))
        st.markdown(f"### {selected_profile} — OOS replay")
        if not oos_equity.empty:
            st.line_chart(oos_equity.set_index("session")["equity"])
        if not oos_trades.empty:
            st.markdown("#### OOS replay trades")
            st.dataframe(oos_trades.head(250), width="stretch", hide_index=True)
            st.download_button("Download selected OOS replay trades", data=oos_trades.to_csv(index=False).encode("utf-8"), file_name="v3_8_selected_oos_trades.csv", mime="text/csv", width="stretch")
        if not candidates.empty:
            with st.expander("Show selected OOS candidate observations"):
                st.dataframe(candidates.head(250), width="stretch", hide_index=True)

    st.divider()
    st.download_button("Download v3.8 profile summary", data=summary.to_csv(index=False).encode("utf-8"), file_name="v3_8_production_vs_challenger_summary.csv", mime="text/csv", width="stretch")
    st.warning("v3.8 is still research. A challenger should not replace production until it shows durable out-of-sample improvement, adequate sample size, acceptable drawdown, and then survives paper trading.")
p
