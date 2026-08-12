import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from market_data import full_us_equity_universe, get_bars, get_bars_batched
from strategy import prefilter_daily, prepare_intraday, classify
from backtest import backtest

load_dotenv()
ET = ZoneInfo("America/New_York")

st.set_page_config(page_title="Institutional Momentum Scanner v2.1", layout="wide")
st.title("Institutional Momentum Scanner v2.1")
st.caption("Full-market momentum funnel • institutional score • $2,000 simulator • no live orders")

tab1, tab2 = st.tabs(["Live Scanner", "$2,000 Backtester"])

with tab1:
    st.subheader("Full U.S. Market Momentum Scan")
    st.write(
        "The app now screens the entire active U.S. equity universe with daily data first, "
        "then runs the detailed 1-minute scan only on the strongest liquid momentum names."
    )

    c1, c2 = st.columns(2)
    feed = c1.selectbox("Data feed", ["iex", "sip"], index=0)
    candidate_count = c2.slider("Finalists for detailed scan", 50, 250, 150, 25)

    if st.button("RUN FULL-MARKET SCAN", type="primary", use_container_width=True):
        os.environ["ALPACA_FEED"] = feed
        status = st.status("Scanning the U.S. market...", expanded=True)
        progress = st.progress(0)

        try:
            status.write("1/4 Discovering active U.S. equities...")
            universe = full_us_equity_universe()
            status.write(f"Found {len(universe):,} active tradable U.S. equities.")

            now = datetime.now(ET)
            daily_start = now - timedelta(days=45)

            status.write("2/4 Ranking the whole market by liquidity, momentum and volume...")
            def daily_progress(done, total):
                progress.progress(min(int(done / max(total, 1) * 45), 45))

            daily = get_bars_batched(
                universe, daily_start, now, "1Day", feed,
                batch_size=200, pause_seconds=.12,
                progress_callback=daily_progress
            )
            finalists = prefilter_daily(daily, keep=candidate_count)

            if finalists.empty:
                status.update(label="No liquid momentum finalists were found.", state="complete")
                st.stop()

            status.write(f"Daily prefilter kept {len(finalists)} strongest names.")
            progress.progress(55)

            status.write("3/4 Pulling 1-minute bars for finalists and SPY...")
            candidate_symbols = finalists["symbol"].tolist()
            intraday_start = now.replace(hour=9, minute=30, second=0, microsecond=0)

            intraday = get_bars_batched(
                candidate_symbols,
                intraday_start - timedelta(days=1),
                now,
                "1Min",
                feed,
                batch_size=100,
                pause_seconds=.10
            )
            spy = get_bars(
                ["SPY"],
                intraday_start - timedelta(days=1),
                now,
                "1Min",
                feed
            )
            progress.progress(75)

            if intraday.empty:
                status.update(label="No intraday bars returned.", state="error")
                st.stop()

            intraday["timestamp"] = pd.to_datetime(intraday["timestamp"], utc=True)
            spy["timestamp"] = pd.to_datetime(spy["timestamp"], utc=True)

            latest_date = intraday["timestamp"].dt.tz_convert(ET).dt.date.max()
            today_intraday = intraday[
                intraday["timestamp"].dt.tz_convert(ET).dt.date == latest_date
            ]
            spy_today = spy[
                spy["timestamp"].dt.tz_convert(ET).dt.date == latest_date
            ]

            avg_dollar_map = dict(zip(finalists["symbol"], finalists["avg_dollar_volume"]))
            finalist_map = finalists.set_index("symbol").to_dict("index")

            status.write("4/4 Applying VWAP, opening range, volume, trend, RSI and SPY-relative-strength rules...")
            rows = []

            for sym, d in today_intraday.groupby("symbol"):
                if len(d) < 20:
                    continue

                ref = finalist_map.get(sym, {})
                px = max(float(ref.get("price", 1)), .01)
                avg_daily_volume = float(avg_dollar_map.get(sym, 0)) / px if avg_dollar_map.get(sym, 0) else None

                prepared = prepare_intraday(d, spy_today, avg_daily_volume=avg_daily_volume)
                if prepared.empty:
                    continue

                r = prepared.iloc[-1]
                score, sig, reasons = classify(r, avg_dollar_map.get(sym, 0))

                rows.append({
                    "symbol": sym,
                    "score": score,
                    "signal": sig,
                    "price": round(float(r["close"]), 2),
                    "change_today_%": round(float(r.get("stock_ret", 0)) * 100, 2),
                    "rel_volume": round(float(r.get("rel_volume", 0)), 2),
                    "vwap": round(float(r["vwap"]), 2),
                    "rsi": round(float(r["rsi"]) if pd.notna(r["rsi"]) else 50, 1),
                    "vs_SPY_%": round(float(r.get("rs", 0)) * 100, 2),
                    "daily_vol_ratio": round(float(ref.get("daily_volume_ratio", 0)), 2),
                    "reasons": "; ".join(reasons)
                })

            out = pd.DataFrame(rows)
            progress.progress(100)
            status.update(label="Full-market scan complete.", state="complete", expanded=False)

            if out.empty:
                st.warning("The scan finished, but no finalists had enough intraday bars to score.")
            else:
                rank = {"BUY": 0, "WATCH": 1, "NO BUY": 2}
                out["_rank"] = out["signal"].map(rank)
                out = out.sort_values(
                    ["_rank", "score", "rel_volume"],
                    ascending=[True, False, False]
                ).drop(columns="_rank")

                buys = out[out["signal"] == "BUY"]
                watches = out[out["signal"] == "WATCH"]

                a,b,c,d = st.columns(4)
                a.metric("U.S. stocks found", f"{len(universe):,}")
                b.metric("Finalists scored", len(out))
                c.metric("BUY signals", len(buys))
                d.metric("WATCH signals", len(watches))

                if not buys.empty:
                    st.success("BUY candidates: " + ", ".join(buys["symbol"].head(8)))
                else:
                    st.info("NO CONFIRMED BUY right now. That is a valid scanner result.")

                st.subheader("Best opportunities now")
                st.dataframe(out.head(50), use_container_width=True, hide_index=True)

                st.subheader("BUY signals only")
                if buys.empty:
                    st.write("None.")
                else:
                    st.dataframe(buys, use_container_width=True, hide_index=True)

                st.subheader("WATCH list")
                if watches.empty:
                    st.write("None.")
                else:
                    st.dataframe(watches.head(30), use_container_width=True, hide_index=True)

        except Exception as e:
            status.update(label="Scan stopped because of an error.", state="error")
            st.error(str(e))
            st.info("If the message mentions SIP entitlement, choose IEX and run again.")

with tab2:
    st.subheader("$2,000 Historical Performance Simulator")

    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start", datetime(2025,1,1))
    end_date = c2.date_input("End", datetime.now().date() - timedelta(days=1))
    risk_pct = c3.slider("Risk per trade", .25, 2.0, 1.0, .25) / 100

    symbols = st.text_input(
        "Backtest symbols",
        "NVDA,MU,AMD,MRVL,FSLR,RIOT,MSFT,AMZN,META,PLTR,AVGO,ANET"
    )
    backtest_feed = st.selectbox("Backtest data feed", ["iex", "sip"], index=0, key="bt_feed")

    if st.button("RUN $2,000 BACKTEST", type="primary"):
        syms = [x.strip().upper() for x in symbols.split(",") if x.strip()]

        with st.spinner("Downloading historical data and simulating trades..."):
            bars = get_bars_batched(
                syms, start_date, end_date, "1Min", backtest_feed,
                batch_size=30, pause_seconds=.1
            )
            spy = get_bars(["SPY"], start_date, end_date, "1Min", backtest_feed)

        if bars.empty or spy.empty:
            st.error("No historical data returned. Try a shorter period or choose IEX.")
        else:
            result = backtest(
                bars, spy,
                starting_capital=2000,
                risk_pct=risk_pct
            )
            stats = result["stats"]

            cols = st.columns(6)
            metrics = [
                ("ending_capital", "Ending $"),
                ("total_return_pct", "Return %"),
                ("win_rate_pct", "Win rate %"),
                ("profit_factor", "Profit factor"),
                ("max_drawdown_pct", "Max DD %"),
                ("trades", "Trades")
            ]
            for col, (key, label) in zip(cols, metrics):
                col.metric(label, stats.get(key, "—"))

            eq = result["equity"]
            if not eq.empty:
                st.plotly_chart(
                    px.line(eq, x="date", y="equity", title="$2,000 Equity Curve"),
                    use_container_width=True
                )

            st.subheader("Simulated trades")
            st.dataframe(result["trades"], use_container_width=True, hide_index=True)

st.divider()
st.warning(
    "Research only. Simulated results do not guarantee future performance. "
    "Before live trading, add slippage/latency, point-in-time universe controls, "
    "corporate-action handling, survivorship-bias controls and out-of-sample validation."
)
