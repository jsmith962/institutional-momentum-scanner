import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from alerts import build_buy_message, build_sell_message, send_sms, sms_configured
from backtest import swing_backtest
from calibration_ui import render_calibration_lab
from forward_research_ui import render_forward_research_lab
from market_data import eligible_us_equity_universe, get_bars, get_bars_batched
from strategy import (
    classify,
    combine_daily_intraday_signal,
    prefilter_daily,
    prepare_intraday,
    relative_strength_percentiles,
    score_swing_daily,
)

try:
    from threshold_discovery_ui import render_threshold_discovery_lab
    V36_RESEARCH_AVAILABLE = True
    V36_IMPORT_ERROR = None
except Exception as exc:
    render_threshold_discovery_lab = None
    V36_RESEARCH_AVAILABLE = False
    V36_IMPORT_ERROR = str(exc)

load_dotenv()
ET = ZoneInfo("America/New_York")

st.set_page_config(page_title="Institutional Swing Scanner v3.7", layout="wide")
st.title("Institutional Swing Scanner v3.7")
st.caption(
    "Full U.S. market | catalyst-gap protection | daily + intraday confirmation | "
    "SMS alerts | production-equivalent backtesting | walk-forward validation | "
    "fast calibration | empirical threshold discovery | gate bottleneck research | "
    "forward-return analysis | no live orders"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Swing Scanner",
    "$2,000 Swing Backtester",
    "Calibration & Validation",
    "v3.6 Threshold Research",
    "v3.7 Forward Research",
])

for key, default in {
    "latest_backtest_result": None,
    "latest_backtest_settings": None,
    "latest_backtest_daily_bars": None,
    "latest_backtest_market_daily": None,
    "v37_forward_research_result": None,
    "v37_enriched_signal_log": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def money(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def score_display(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except Exception:
        return "N/A"


def rr_display(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.2f}:1"
    except Exception:
        return "N/A"


def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def signal_icon(signal):
    if signal in {"A+ SWING BUY", "BUY"}:
        return "ð¢"
    if signal == "WATCH":
        return "ð¡"
    if signal == "TOO EXTENDED":
        return "ð´"
    return "âª"


def action_text(signal):
    return {
        "A+ SWING BUY": "BUY â top-tier setup confirmed",
        "BUY": "BUY â entry rules confirmed",
        "WATCH": "WAIT FOR BUY TRIGGER",
        "TOO EXTENDED": "WAIT FOR PULLBACK / RETEST",
        "AVOID": "PASS",
        "NO BUY": "WAIT",
    }.get(signal, "WAIT")


def why_not_buy(row):
    signal = row.get("signal", "")
    if bool(row.get("risk_flag", False)):
        return str(row.get("risk_reason", "A hard downside catalyst-risk gate is active."))
    if signal in {"A+ SWING BUY", "BUY"}:
        return "All required BUY gates passed."
    if signal == "TOO EXTENDED":
        return "The stock is too extended from its preferred entry. Wait for a pullback or retest."

    failures = []
    swing_score = safe_float(row.get("swing_score"))
    entry_quality = safe_float(row.get("entry_quality"))
    reward_risk = safe_float(row.get("reward_risk"))
    market_score = safe_float(row.get("market_score"))
    intraday_score = safe_float(row.get("intraday_score"))

    try:
        price = float(row.get("price", 0))
        entry_low = float(row.get("entry_low", 0))
        entry_high = float(row.get("entry_high", 0))
        inside_entry_zone = entry_low <= price <= entry_high
    except Exception:
        inside_entry_zone = False

    if swing_score < 85:
        failures.append(f"Swing Score {swing_score:.1f} is below the 85 BUY threshold")
    if entry_quality < 10:
        failures.append(f"Entry Quality {entry_quality:.1f}/15 is below the 10/15 BUY requirement")
    if reward_risk < 2:
        failures.append(f"Reward/Risk {reward_risk:.2f}:1 is below the required 2.00:1")
    if market_score < 5:
        failures.append(f"Market Score {market_score:.1f}/10 is below the minimum 5/10")
    if not inside_entry_zone:
        failures.append("Current price is outside the preferred entry zone")
    if not bool(row.get("trend_health", True)):
        failures.append("The 20-day and 50-day trend slopes are not both rising")

    distribution_days = safe_int(row.get("distribution_days"))
    if distribution_days > 4:
        failures.append(f"{distribution_days} recent distribution days show excessive selling pressure")

    leadership = row.get("leadership_percentile")
    if leadership is not None and not pd.isna(leadership) and float(leadership) < 70:
        failures.append(f"Market leadership rank {float(leadership):.0f}% is below the 70% BUY gate")

    intraday_signal = str(row.get("intraday_signal", "")).upper()
    if intraday_signal != "BUY":
        failures.append("Intraday signal has not changed to BUY")
    if intraday_score < 85:
        failures.append(f"Intraday Score {intraday_score:.1f} is below the 85 confirmation threshold")
    if signal == "AVOID" and not failures:
        failures.append("The setup does not meet enough high-probability swing requirements")
    if signal == "WATCH" and not failures:
        failures.append("The stock is close, but at least one production confirmation has not passed.")

    return " | ".join(failures) if failures else "Waiting for additional confirmation."


def render_trade_card(row, rank_num=None):
    signal = row.get("signal", "N/A")
    symbol = row.get("symbol", "N/A")
    prefix = f"#{rank_num} " if rank_num is not None else ""

    with st.container(border=True):
        st.markdown(f"### {prefix}{signal_icon(signal)} {symbol} â {signal}")
        st.write(f"**Swing Score:** {score_display(row.get('swing_score'))}/100")
        st.write(f"**Setup:** {row.get('setup', '')}")
        st.write(f"**Current Price:** {money(row.get('price'))}")
        st.write(f"**Entry Quality:** {score_display(row.get('entry_quality'))}/15")

        if bool(row.get("risk_flag", False)):
            st.error("**Risk Event:** " + str(row.get("risk_reason", "Hard downside catalyst-risk gate active")))
        else:
            leadership = row.get("leadership_percentile")
            leadership_text = "N/A" if leadership is None or pd.isna(leadership) else f"{float(leadership):.0f}th percentile"
            distribution = safe_int(row.get("distribution_days"))
            st.caption(f"Risk gate: PASS | Market leadership: {leadership_text} | Distribution days: {distribution}")

        st.markdown("#### Entry Plan")
        st.write(f"**Preferred Entry Zone:** {money(row.get('entry_low'))} â {money(row.get('entry_high'))}")
        st.write(f"**Stop:** {money(row.get('stop'))}")
        st.write(f"**Target 1:** {money(row.get('target1'))}")
        st.write(f"**Target 2:** {money(row.get('target2'))}")
        st.write(f"**Reward / Risk:** {rr_display(row.get('reward_risk'))}")
        st.markdown("#### Action")

        if signal in {"A+ SWING BUY", "BUY"}:
            st.success(action_text(signal))
            st.markdown("#### Why BUY?")
            st.success("Swing score, entry quality, reward/risk, entry zone, market conditions and intraday confirmation passed.")
        elif signal == "WATCH":
            st.warning(action_text(signal))
            st.markdown("#### Why Not BUY Yet?")
            st.info(why_not_buy(row))
        elif signal == "TOO EXTENDED":
            st.error(action_text(signal))
            st.markdown("#### Why Not BUY Yet?")
            st.info(why_not_buy(row))
        else:
            st.info(action_text(signal))
            st.markdown("#### Why Not BUY Yet?")
            st.info(why_not_buy(row))

        st.caption(f"Intraday confirmation: {row.get('intraday_signal', 'N/A')} ({row.get('intraday_score', 'N/A')}/100)")


def render_validation_summary(validation):
    st.subheader("Walk-Forward Validation")
    if not isinstance(validation, dict):
        st.info("No validation results are available.")
        return

    sample_trades = safe_int(validation.get("sample_trades", validation.get("trades", 0)))
    oos_trades = safe_int(validation.get("out_of_sample_trades", 0))
    oos_expectancy = safe_float(validation.get("out_of_sample_expectancy_r", validation.get("aggregate_oos_expectancy_r", 0)))
    worst_fold_expectancy = safe_float(validation.get("worst_fold_expectancy_r", validation.get("worst_fold_expectancy", 0)))
    grade = validation.get("confidence_grade", validation.get("grade", "INSUFFICIENT"))
    passed = bool(validation.get("validation_pass", False))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated trades", sample_trades)
    c2.metric("OOS trades", oos_trades)
    c3.metric("Aggregate OOS expectancy", f"{oos_expectancy:.3f} R")
    c4.metric("Worst fold expectancy", f"{worst_fold_expectancy:.3f} R")
    st.write(f"**Confidence grade:** {grade}")

    if passed:
        st.success("Configured walk-forward validation checks passed.")
    else:
        st.warning("The historical evidence is not yet strong enough to declare this configuration validated.")

    notes = validation.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if notes:
        with st.expander("Validation notes"):
            for note in notes:
                st.write(f"â¢ {note}")


with st.sidebar:
    st.header("Text alerts")
    sms_enabled = st.toggle("Send SMS alerts", value=False)
    sms_buy_enabled = st.checkbox("Text BUY signals", value=True)
    sms_sell_enabled = st.checkbox("Text SELL signals", value=True)
    st.caption("Phone numbers and Twilio credentials stay in Streamlit Secrets.")

    if sms_enabled:
        if sms_configured():
            st.success("SMS configured")
        else:
            st.warning("SMS secrets are not configured yet.")

    st.divider()
    st.subheader("Tracked positions")
    tracked_positions = st.text_input(
        "Symbols you currently hold",
        "",
        help="Example: NVDA,MU,OWL. SELL-risk alerts are evaluated only for symbols entered here.",
    )


with tab1:
    st.subheader("Full U.S. Market Swing-Trade Scanner")
    st.write("The scanner removes unsuitable securities and illiquid stocks, then analyzes daily swing structure and live intraday momentum separately.")
    st.info("A strong stock is not automatically a BUY. Production rules require daily structure, entry quality, reward/risk, trend health, market leadership, market regime and intraday confirmation to align.")

    c1, c2 = st.columns(2)
    feed = c1.selectbox("Data feed", ["iex", "sip"], index=0, help="Use SIP when your Alpaca plan supports consolidated market data.")
    finalists_n = c2.slider("Finalists for detailed scan", 50, 250, 150, 25)

    if st.button("RUN FULL-MARKET SWING SCAN", type="primary", width="stretch"):
        os.environ["ALPACA_FEED"] = feed
        status = st.status("Scanning the U.S. market...", expanded=True)
        progress = st.progress(0)

        try:
            status.write("1/5 Filtering eligible U.S. securities...")
            elig = eligible_us_equity_universe()
            universe = elig.symbol.tolist()
            status.write(f"Eligible after type filtering: {len(universe):,}")

            now = datetime.now(ET)
            daily_start = now - timedelta(days=45)

            status.write("2/5 Applying price, liquidity, trend and momentum filters...")

            def prog(done, total):
                progress.progress(min(int(done / max(total, 1) * 40), 40))

            daily = get_bars_batched(
                universe,
                daily_start,
                now,
                "1Day",
                feed,
                batch_size=200,
                pause_seconds=0.12,
                progress_callback=prog,
            )

            if daily.empty:
                status.update(label="No daily market data returned.", state="error")
                st.stop()

            daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
            finalists = prefilter_daily(
                daily,
                elig,
                min_price=5,
                min_avg_dollar_volume=10_000_000,
                min_avg_volume=500_000,
                limit=finalists_n,
            )

            if finalists.empty:
                status.update(label="No qualifying finalists found.", state="complete")
                st.stop()

            status.write(f"Finalists after eligibility filters: {len(finalists)}")
            progress.progress(45)

            status.write("3/5 Pulling longer daily history for swing-trade analysis...")
            finalist_symbols = finalists.symbol.tolist()
            swing_start = now - timedelta(days=420)

            swing_daily = get_bars_batched(
                finalist_symbols,
                swing_start,
                now,
                "1Day",
                feed,
                batch_size=100,
                pause_seconds=0.10,
            )

            market_daily = get_bars(["SPY", "QQQ"], swing_start, now, "1Day", feed)

            if not swing_daily.empty:
                swing_daily["timestamp"] = pd.to_datetime(swing_daily["timestamp"], utc=True)
            if not market_daily.empty:
                market_daily["timestamp"] = pd.to_datetime(market_daily["timestamp"], utc=True)

            spy_daily = market_daily[market_daily["symbol"] == "SPY"].copy() if not market_daily.empty else pd.DataFrame()
            qqq_daily = market_daily[market_daily["symbol"] == "QQQ"].copy() if not market_daily.empty else pd.DataFrame()
            leadership_map = relative_strength_percentiles(swing_daily)
            progress.progress(60)

            status.write("4/5 Pulling intraday bars for finalists and SPY...")
            market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)

            intra = get_bars_batched(
                finalist_symbols,
                market_start - timedelta(days=1),
                now,
                "1Min",
                feed,
                batch_size=100,
                pause_seconds=0.10,
            )
            spy = get_bars(["SPY"], market_start - timedelta(days=1), now, "1Min", feed)
            progress.progress(75)

            if intra.empty or spy.empty:
                status.update(label="No intraday bars returned.", state="error")
                st.stop()

            intra["timestamp"] = pd.to_datetime(intra["timestamp"], utc=True)
            spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)

            latest = intra.timestamp.dt.tz_convert(ET).dt.date.max()
            today = intra[intra.timestamp.dt.tz_convert(ET).dt.date == latest]
            spy_today = spy[spy.timestamp.dt.tz_convert(ET).dt.date == latest]

            fmap = finalists.set_index("symbol").to_dict("index")
            advmap = dict(zip(finalists.symbol, finalists.avg_dollar_volume))

            status.write("5/5 Applying swing, risk and intraday confirmation rules...")
            rows = []

            for sym, d in today.groupby("symbol"):
                if len(d) < 20:
                    continue

                ref = fmap.get(sym, {})
                px_value = max(float(ref.get("price", 1)), 0.01)
                adv_dollars = float(advmap.get(sym, 0) or 0)
                adv_shares = adv_dollars / px_value if adv_dollars else None

                prepared = prepare_intraday(d, spy_today, adv_shares)
                if prepared.empty:
                    continue

                r = prepared.iloc[-1]
                intraday_score, intraday_signal, reasons = classify(r, advmap.get(sym, 0))

                stock_swing_daily = swing_daily[swing_daily["symbol"] == sym].copy() if not swing_daily.empty else pd.DataFrame()
                swing = None

                if not stock_swing_daily.empty:
                    swing = score_swing_daily(
                        stock_swing_daily,
                        spy_daily,
                        qqq_daily,
                        leadership_map.get(sym),
                    )

                if not swing:
                    continue

                final_signal, confluence_reason = combine_daily_intraday_signal(
                    swing.get("signal", "N/A"),
                    intraday_signal,
                    intraday_score,
                    risk_flag=bool(swing.get("risk_flag", False)),
                )

                intraday_confirmed = bool(intraday_signal == "BUY" and intraday_score >= 85)

                row = {
                    "symbol": sym,
                    "name": ref.get("name", ""),
                    "signal": final_signal,
                    "swing_score": swing.get("swing_score", 0),
                    "setup": swing.get("setup", ""),
                    "entry_quality": swing.get("entry_quality", 0),
                    "price": round(float(r.close), 2),
                    "entry_low": swing.get("entry_low"),
                    "entry_high": swing.get("entry_high"),
                    "stop": swing.get("stop"),
                    "target1": swing.get("target1"),
                    "target2": swing.get("target2"),
                    "reward_risk": swing.get("reward_risk"),
                    "intraday_signal": intraday_signal,
                    "intraday_score": intraday_score,
                    "change_today_%": round(float(r.get("stock_ret", 0)) * 100, 2),
                    "rel_volume": round(float(r.get("rel_volume", 0)), 2),
                    "vwap": round(float(r.vwap), 2),
                    "intraday_rsi": round(float(r.rsi) if pd.notna(r.rsi) else 50, 1),
                    "swing_rsi": swing.get("rsi14"),
                    "swing_rvol": swing.get("rvol"),
                    "risk_flag": bool(swing.get("risk_flag", False)),
                    "risk_reason": swing.get("risk_reason", ""),
                    "gap_down_pct": swing.get("gap_down_pct", 0),
                    "event_days_ago": swing.get("event_days_ago"),
                    "trend_health": bool(swing.get("trend_health", False)),
                    "distribution_days": swing.get("distribution_days", 0),
                    "leadership_percentile": swing.get("leadership_percentile"),
                    "market_score": swing.get("market_score"),
                    "intraday_confirmed": intraday_confirmed,
                    "vs_SPY_%": round(float(r.get("rs", 0)) * 100, 2),
                    "security_type": ref.get("security_type", "Common-stock candidate"),
                    "decision": confluence_reason,
                    "intraday_reasons": "; ".join(reasons),
                }
                rows.append(row)

            out = pd.DataFrame(rows)
            progress.progress(100)
            status.update(label="Swing scan complete.", state="complete", expanded=False)

            if out.empty:
                st.warning("No finalists had enough data to score.")
            else:
                rank = {"A+ SWING BUY": 0, "BUY": 1, "WATCH": 2, "TOO EXTENDED": 3, "AVOID": 4, "NO BUY": 5, "N/A": 6}
                out["_rank"] = out["signal"].map(rank).fillna(6)
                out = out.sort_values(["_rank", "swing_score", "entry_quality", "intraday_score"], ascending=[True, False, False, False]).drop(columns="_rank")

                buys = out[out["signal"].isin(["A+ SWING BUY", "BUY"])]
                watches = out[out["signal"] == "WATCH"]
                extended = out[out["signal"] == "TOO EXTENDED"]

                if sms_enabled and sms_configured():
                    sent = []
                    if sms_buy_enabled and not buys.empty:
                        for _, alert_row in buys.head(5).iterrows():
                            try:
                                send_sms(build_buy_message(alert_row))
                                sent.append(f"BUY {alert_row['symbol']}")
                            except Exception as sms_error:
                                st.warning(f"Could not text BUY alert for {alert_row['symbol']}: {sms_error}")

                    held = {x.strip().upper() for x in tracked_positions.split(",") if x.strip()}
                    if sms_sell_enabled and held:
                        held_rows = out[out["symbol"].isin(held)]
                        for _, alert_row in held_rows.iterrows():
                            sell_reasons = []
                            current_price = safe_float(alert_row.get("price"))
                            current_vwap = safe_float(alert_row.get("vwap"))
                            current_intraday_score = safe_int(alert_row.get("intraday_score"))
                            current_intraday_signal = alert_row.get("intraday_signal", "")

                            if current_price < current_vwap:
                                sell_reasons.append("price below VWAP")
                            if current_intraday_score < 60:
                                sell_reasons.append(f"intraday score fell to {current_intraday_score}")

                            sell_risk = current_intraday_signal in {"AVOID", "NO BUY"} and bool(sell_reasons)
                            if sell_risk:
                                try:
                                    send_sms(build_sell_message(alert_row["symbol"], alert_row["price"], ", ".join(sell_reasons)))
                                    sent.append(f"SELL-RISK {alert_row['symbol']}")
                                except Exception as sms_error:
                                    st.warning(f"Could not text SELL alert for {alert_row['symbol']}: {sms_error}")

                    if sent:
                        st.info("Text alerts sent: " + ", ".join(sent))

                st.divider()
                st.subheader("Current Market Decision")
                if not buys.empty:
                    st.success(f"ð¢ {len(buys)} CONFIRMED SWING BUY {'SIGNAL' if len(buys) == 1 else 'SIGNALS'}")
                    st.write(", ".join(buys["symbol"].head(8)))
                else:
                    st.error("ð´ NO CONFIRMED SWING BUY RIGHT NOW")
                    st.caption("Do not buy simply because a stock has a high Swing Score. Wait for BUY or A+ SWING BUY.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Finalists", len(out))
                m2.metric("Confirmed BUYs", len(buys))
                m3.metric("WATCH", len(watches))
                m4.metric("TOO EXTENDED", len(extended))

                st.divider()
                st.header("Top 5 Swing Opportunities")
                st.caption("WATCH means wait. It does not mean buy now.")
                for rank_num, (_, row) in enumerate(out.head(5).iterrows(), start=1):
                    render_trade_card(row, rank_num)

                st.divider()
                st.header("ð¢ Confirmed BUY Signals")
                if buys.empty:
                    st.info("No confirmed BUY signals right now.")
                else:
                    for _, row in buys.head(10).iterrows():
                        render_trade_card(row)

                st.divider()
                st.header("ð´ Strong But Too Extended")
                if extended.empty:
                    st.write("None.")
                else:
                    for _, row in extended.head(5).iterrows():
                        render_trade_card(row)

                st.divider()
                st.header("ð¡ WATCH List")
                if watches.empty:
                    st.write("None.")
                else:
                    watch_columns = [c for c in ["symbol", "signal", "swing_score", "setup", "price", "entry_low", "entry_high"] if c in watches.columns]
                    st.dataframe(watches[watch_columns].head(30), width="stretch", hide_index=True)

                st.divider()
                st.header("Stock Detail")
                selected_symbol = st.selectbox("Select a stock", out["symbol"].tolist())
                selected = out[out["symbol"] == selected_symbol].iloc[0]
                render_trade_card(selected)
                st.write(f"**Today's change:** {safe_float(selected.get('change_today_%')):.2f}%")
                st.write(f"**Relative volume:** {safe_float(selected.get('rel_volume')):.2f}x")
                st.write(f"**Relative strength vs SPY:** {safe_float(selected.get('vs_SPY_%')):.2f}%")
                st.write(f"**Decision reason:** {selected.get('decision', '')}")

                with st.expander("Show full research table"):
                    display_columns = [c for c in ["symbol", "name", "signal", "swing_score", "setup", "entry_quality", "price", "entry_low", "entry_high", "stop", "target1", "target2", "reward_risk", "risk_flag", "risk_reason", "gap_down_pct", "event_days_ago", "trend_health", "distribution_days", "leadership_percentile", "market_score", "intraday_confirmed", "intraday_signal", "intraday_score", "change_today_%", "rel_volume", "vs_SPY_%", "decision"] if c in out.columns]
                    st.dataframe(out[display_columns].head(50), width="stretch", hide_index=True)

                st.download_button("Download latest swing scan", data=out.to_csv(index=False).encode("utf-8"), file_name="v3_7_swing_scan_latest.csv", mime="text/csv", width="stretch")

        except Exception as exc:
            status.update(label="Scan stopped because of an error.", state="error")
            st.error(str(exc))
            st.info("If SIP entitlement is mentioned, choose IEX.")


with tab2:
    st.subheader("$2,000 Production-Equivalent Swing Backtester")
    st.info("v3.7 reconstructs the daily and intraday decision chain using historical data available at the configured scan time. Signals are evaluated before simulated entries.")
    st.warning("Backtest results cover only the symbols entered below and do not reconstruct the entire historical U.S. market. Historical performance does not guarantee future profitability.")

    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start", datetime.now(ET).date() - timedelta(days=180), key="swing_bt_start")
    end_date = c2.date_input("End", datetime.now(ET).date() - timedelta(days=1), key="swing_bt_end")
    risk_pct = c3.slider("Risk per trade", 0.25, 2.0, 0.50, 0.25, key="swing_bt_risk") / 100

    c4, c5, c6 = st.columns(3)
    max_positions = c4.slider("Maximum open positions", 1, 5, 3, 1, key="swing_bt_positions")
    max_holding_days = c5.slider("Maximum holding sessions", 5, 30, 20, 5, key="swing_bt_hold")
    scan_time = c6.selectbox("Historical scan time (ET)", ["11:30", "14:00", "15:30"], index=0, key="swing_bt_time")

    c7, c8 = st.columns(2)
    slippage_bps = c7.slider("Estimated slippage (basis points per order)", 0, 25, 5, 1, key="swing_bt_slippage")
    commission_bps = c8.slider("Estimated fees (basis points per order)", 0, 10, 0, 1, key="swing_bt_fees")

    symbols = st.text_input(
        "Backtest symbols",
        "NVDA,MU,AMD,MRVL,FSLR,RIOT,MSFT,AMZN,META,PLTR,AVGO,ANET",
        key="swing_bt_symbols",
        help="Use at least 10 liquid stocks. Relative-strength percentiles are calculated inside this comparison group.",
    )
    btfeed = st.selectbox("Backtest data feed", ["iex", "sip"], index=0, key="swing_bt_feed")
    st.caption("IEX contains only one exchange. Use SIP when your Alpaca plan permits consolidated market data.")
    st.divider()
    st.info("The production backtest runs first. Calibration, threshold research and v3.7 forward-return research use the cached historical audit.")

    if st.button("RUN $2,000 BACKTEST", type="primary", width="stretch"):
        syms = [x.strip().upper() for x in symbols.split(",") if x.strip()]

        if start_date >= end_date:
            st.error("Choose a start date before the end date.")
            st.stop()
        if (end_date - start_date).days > 365:
            st.warning("A range longer than one year can be slow. Start with 6â12 months and then test additional periods.")
        if len(syms) < 5:
            st.error("Enter at least 5 symbols.")
            st.stop()
        if len(syms) < 10:
            st.warning("Ten or more symbols are recommended for more meaningful relative-strength ranking.")

        request_end = end_date + timedelta(days=1)
        warmup_start = start_date - timedelta(days=450)

        with st.spinner("Downloading daily and minute history..."):
            bars = get_bars_batched(syms, start_date, request_end, "1Min", btfeed, batch_size=20, pause_seconds=0.1)
            market_minutes = get_bars(["SPY", "QQQ"], start_date, request_end, "1Min", btfeed)
            daily_history = get_bars_batched(syms, warmup_start, request_end, "1Day", btfeed, batch_size=100, pause_seconds=0.1)
            market_daily = get_bars(["SPY", "QQQ"], warmup_start, request_end, "1Day", btfeed)

            if market_minutes.empty:
                spy = pd.DataFrame()
                qqq = pd.DataFrame()
            else:
                spy = market_minutes[market_minutes["symbol"] == "SPY"].copy()
                qqq = market_minutes[market_minutes["symbol"] == "QQQ"].copy()

        if bars.empty or spy.empty or qqq.empty or daily_history.empty or market_daily.empty or not {"SPY", "QQQ"}.issubset(set(market_daily["symbol"])):
            st.error("The complete daily and minute history was not returned. Try a shorter date range or choose IEX.")
        else:
            complete_symbols = sorted(set(bars["symbol"]) & set(daily_history["symbol"]))
            missing_symbols = sorted(set(syms) - set(complete_symbols))

            if missing_symbols:
                st.warning("Excluded symbols with incomplete data: " + ", ".join(missing_symbols))
            if len(complete_symbols) < 5:
                st.error("Fewer than 5 symbols returned complete data.")
                st.stop()

            bars = bars[bars["symbol"].isin(complete_symbols)].copy()
            daily_history = daily_history[daily_history["symbol"].isin(complete_symbols)].copy()

            with st.spinner("Running production-equivalent v3.7 backtest..."):
                res = swing_backtest(
                    bars,
                    spy,
                    qqq_bars=qqq,
                    daily_bars=daily_history,
                    market_daily_bars=market_daily,
                    starting_capital=2000,
                    risk_pct=risk_pct,
                    max_positions=max_positions,
                    max_holding_days=max_holding_days,
                    scan_time=scan_time,
                    slippage_bps=slippage_bps,
                    commission_bps=commission_bps,
                )

            st.session_state.latest_backtest_result = res
            st.session_state.latest_backtest_daily_bars = daily_history.copy()
            st.session_state.latest_backtest_market_daily = market_daily.copy()
            st.session_state.v37_forward_research_result = None
            st.session_state.v37_enriched_signal_log = None
            st.session_state.latest_backtest_settings = {
                "symbols": ",".join(complete_symbols),
                "start": str(start_date),
                "end": str(end_date),
                "risk_pct": risk_pct,
                "max_positions": max_positions,
                "max_holding_days": max_holding_days,
                "scan_time": scan_time,
                "slippage_bps": slippage_bps,
                "commission_bps": commission_bps,
                "feed": btfeed,
                "version": "v3.7",
            }

            stats = res.get("stats", {})
            st.success("Production backtest completed and v3.7 research data saved.")

            cols = st.columns(4)
            cols[0].metric("Ending $", stats.get("ending_capital", "â"))
            cols[1].metric("Return %", stats.get("total_return_pct", "â"))
            cols[2].metric("Win rate %", stats.get("win_rate_pct", "â"))
            cols[3].metric("Profit factor", stats.get("profit_factor", "â"))

            cols2 = st.columns(4)
            cols2[0].metric("Max DD %", stats.get("max_drawdown_pct", "â"))
            cols2[1].metric("Trades", stats.get("trades", "â"))
            cols2[2].metric("Average expectancy", f"{safe_float(stats.get('expectancy_r')):.3f} R")
            cols2[3].metric("Average trade $", stats.get("avg_trade_dollars", "â"))

            for warning in res.get("warnings", []):
                st.warning(warning)

            st.divider()
            render_validation_summary(res.get("validation", {}))

            diagnostics = res.get("diagnostics", {})
            funnel = diagnostics.get("funnel", pd.DataFrame())
            gate_failures = diagnostics.get("gate_failures", pd.DataFrame())
            near_misses = diagnostics.get("near_misses", pd.DataFrame())
            score_distribution = diagnostics.get("score_distribution", pd.DataFrame())

            st.divider()
            st.subheader("BUY confirmation funnel")
            st.caption("Shows where historical candidates stop progressing under the production rules.")

            if not isinstance(funnel, pd.DataFrame) or funnel.empty:
                st.info("No candidates were available for gate diagnostics.")
            else:
                st.dataframe(funnel, width="stretch", hide_index=True)

            if isinstance(gate_failures, pd.DataFrame) and not gate_failures.empty:
                primary = gate_failures.iloc[0]
                failed_count = safe_int(primary.get("failed", 0))
                failure_pct = safe_float(primary.get("failure_percent", primary.get("failure_rate_pct", 0)), 0)
                primary_gate = primary.get("gate", "Unknown gate")
                st.info(f"Most frequently failed gate: {primary_gate} failed for {failed_count} candidates ({failure_pct:.1f}%).")
                st.markdown("#### Most common failed BUY gates")
                st.dataframe(gate_failures.head(12), width="stretch", hide_index=True)

            if isinstance(score_distribution, pd.DataFrame) and not score_distribution.empty:
                with st.expander("Swing-score distribution"):
                    st.dataframe(score_distribution, width="stretch", hide_index=True)

            st.markdown("#### Closest near misses")
            st.caption("One best non-BUY observation per symbol, ranked by the fewest failed gates. A near miss is not a recommendation.")

            if not isinstance(near_misses, pd.DataFrame) or near_misses.empty:
                st.info("No non-BUY candidates were available to rank.")
            else:
                for _, near_miss in near_misses.head(5).iterrows():
                    with st.container(border=True):
                        symbol = near_miss.get("symbol", "N/A")
                        signal = near_miss.get("signal", "N/A")
                        st.markdown(f"**{symbol} â {signal}**")
                        st.write(f"Session: {near_miss.get('session', 'N/A')} | Gates passed: {near_miss.get('gates_passed', 'N/A')}")
                        st.write(
                            f"Swing Score: {safe_float(near_miss.get('swing_score')):.1f} | "
                            f"Intraday Score: {safe_float(near_miss.get('intraday_score')):.1f} | "
                            f"Entry Quality: {safe_float(near_miss.get('entry_quality')):.1f}/15"
                        )
                        st.warning("Failed BUY gates: " + str(near_miss.get("failed_buy_gates", "N/A")))

                with st.expander("Show full near-miss table"):
                    st.dataframe(near_misses, width="stretch", hide_index=True)

            equity = res.get("equity", pd.DataFrame())
            if isinstance(equity, pd.DataFrame) and not equity.empty:
                st.plotly_chart(px.line(equity, x="date", y="equity", title="$2,000 Equity Curve"), width="stretch")

            st.subheader("Simulated production trades")
            trades = res.get("trades", pd.DataFrame())
            if isinstance(trades, pd.DataFrame):
                st.dataframe(trades, width="stretch", hide_index=True)
                if not trades.empty:
                    st.download_button("Download simulated trades", data=trades.to_csv(index=False).encode("utf-8"), file_name="v3_7_swing_backtest_trades.csv", mime="text/csv", width="stretch")

            with st.expander("Show historical signal audit"):
                signal_log = res.get("signal_log", pd.DataFrame())
                st.caption("Records each reconstructed daily and intraday decision, including observations that never became trades.")
                if isinstance(signal_log, pd.DataFrame):
                    st.dataframe(signal_log, width="stretch", hide_index=True)
                    if not signal_log.empty:
                        st.download_button("Download signal audit", data=signal_log.to_csv(index=False).encode("utf-8"), file_name="v3_7_signal_audit.csv", mime="text/csv", width="stretch")


with tab3:
    st.subheader("Calibration & Walk-Forward Validation")
    st.info("This research lab uses the completed production backtest candidate log for fast threshold calibration. It does not need to download all market data again for every profile.")
    result = st.session_state.latest_backtest_result
    settings = st.session_state.latest_backtest_settings

    if result is None:
        st.warning("Run a backtest first in the '$2,000 Swing Backtester' tab. The completed result will automatically appear here.")
    else:
        if settings:
            with st.expander("Backtest used for this calibration"):
                st.json(settings)
        render_calibration_lab(result)


with tab4:
    st.subheader("v3.6 Empirical Threshold Discovery")
    st.caption("Research only. Live production BUY thresholds are not changed from this screen.")
    st.info("v3.6 studies historical candidate observations and determines which threshold combinations are reachable before making any profitability claim.")

    result = st.session_state.latest_backtest_result
    daily_bars = st.session_state.latest_backtest_daily_bars
    market_daily_bars = st.session_state.latest_backtest_market_daily
    settings = st.session_state.latest_backtest_settings

    if not V36_RESEARCH_AVAILABLE:
        st.warning("The main scanner is running normally, but the optional v3.6 Threshold Discovery module is not available.")
        st.write("Required GitHub file:")
        st.code("threshold_discovery_ui.py")
        if V36_IMPORT_ERROR:
            with st.expander("Show v3.6 module import error"):
                st.code(V36_IMPORT_ERROR)
    elif result is None:
        st.warning("Run the $2,000 backtest first.")
    elif daily_bars is None or not isinstance(daily_bars, pd.DataFrame) or daily_bars.empty:
        st.warning("The saved backtest does not contain the daily-history dataset needed for research.")
        st.info("Run the $2,000 backtest one more time.")
    else:
        signal_log = result.get("signal_log", pd.DataFrame())
        if not isinstance(signal_log, pd.DataFrame) or signal_log.empty:
            st.warning("The completed backtest does not contain a usable historical signal audit.")
        else:
            st.success(f"v3.6 dataset ready: {len(signal_log):,} historical observations.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Candidate observations", f"{len(signal_log):,}")
            c2.metric("Daily stock bars", f"{len(daily_bars):,}")
            c3.metric("Market daily bars", f"{len(market_daily_bars):,}" if isinstance(market_daily_bars, pd.DataFrame) else "0")

            if settings:
                with st.expander("Historical sample used for v3.6 research"):
                    st.json(settings)

            try:
                render_threshold_discovery_lab(result, daily_bars)
            except TypeError:
                try:
                    render_threshold_discovery_lab(result)
                except Exception as exc:
                    st.error("The v3.6 research module loaded, but its UI function does not match the expected interface.")
                    st.exception(exc)
            except Exception as exc:
                st.error("v3.6 Threshold Discovery encountered an error. The production scanner and backtester remain unchanged.")
                st.exception(exc)


with tab5:
    st.subheader("v3.7 Gate Bottleneck + Forward Return Research")
    st.caption("Research only. Production BUY rules remain unchanged.")
    st.info("v3.7 asks a different question from calibration: instead of merely asking which thresholds produce candidates, it studies what happened after each historical observation and whether passing individual BUY gates was actually associated with better future performance.")

    result = st.session_state.latest_backtest_result
    daily_bars = st.session_state.latest_backtest_daily_bars
    settings = st.session_state.latest_backtest_settings

    if result is None:
        st.warning("Run the $2,000 backtest first. The historical signal audit will then be available for v3.7 research.")
    elif daily_bars is None or not isinstance(daily_bars, pd.DataFrame) or daily_bars.empty:
        st.warning("The current saved backtest does not contain the daily-history dataset needed to calculate forward returns.")
        st.info("Return to the '$2,000 Swing Backtester' tab and run the backtest again.")
    else:
        signal_log = result.get("signal_log", pd.DataFrame())
        if not isinstance(signal_log, pd.DataFrame) or signal_log.empty:
            st.warning("The completed backtest contains no usable historical signal observations.")
        else:
            st.success(f"v3.7 research dataset ready: {len(signal_log):,} historical scanner observations.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Signal observations", f"{len(signal_log):,}")
            c2.metric("Historical daily bars", f"{len(daily_bars):,}")
            c3.metric("Symbols", signal_log["symbol"].nunique() if "symbol" in signal_log.columns else 0)

            if settings:
                with st.expander("Historical sample used for v3.7 research"):
                    st.json(settings)

            render_forward_research_lab(result, daily_bars)


st.divider()
st.warning(
    "Research only. Scanner signals, backtests, validation results, calibration results, "
    "threshold studies and forward-return research do not guarantee future returns. "
    "Do not change production BUY thresholds solely because one historical sample looks better. "
    "Require repeated evidence across non-overlapping periods, portfolio simulation and paper trading "
    "before risking real capital."
