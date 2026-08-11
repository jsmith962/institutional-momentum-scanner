import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from market_data import full_us_equity_universe, get_bars
from strategy import prepare_day, signal
from backtest import backtest

load_dotenv()
st.set_page_config(page_title="Institutional Momentum Scanner v2", layout="wide")
st.title("Institutional Momentum Scanner v2")
st.caption("Full-market research scanner • historical backtester • $2,000 simulator • no live orders")

tab1, tab2 = st.tabs(["Live Scanner", "$2,000 Backtester"])

with tab1:
    st.subheader("Full U.S. equity scan")
    c1,c2 = st.columns(2)
    max_symbols = c1.number_input("Max symbols for this scan", 100, 6000, 1500, 100)
    feed = c2.selectbox("Data feed", ["sip","iex"], index=0)
    if st.button("Discover full universe and scan", type="primary"):
        os.environ["ALPACA_FEED"] = feed
        with st.spinner("Discovering active U.S. equities..."):
            universe = full_us_equity_universe()
        universe = universe[:max_symbols]
        st.info(f"Universe loaded: {len(universe)} symbols. Historical 1-minute data can be large; scanning is batched.")
        # Use today's bars. This is a practical v2 scanner; production streaming is the next optimization.
        now = datetime.utcnow()
        start = now - timedelta(days=2)
        with st.spinner("Downloading intraday data and scoring candidates..."):
            bars = get_bars(universe, start, now, "1Min", feed)
        if bars.empty:
            st.warning("No bars returned. Check feed entitlement and market hours.")
        else:
            today = pd.to_datetime(bars["timestamp"], utc=True).dt.date.max()
            today_bars = bars[pd.to_datetime(bars["timestamp"], utc=True).dt.date == today]
            rows=[]
            for sym, d in today_bars.groupby("symbol"):
                d = d.sort_values("timestamp")
                if len(d) < 30: continue
                r = prepare_day(d).iloc[-1]
                score, sig, reasons = signal(r)
                rows.append({"symbol":sym,"score":score,"signal":sig,
                              "price":round(r.close,2),"vwap":round(r.vwap,2),
                              "rsi":round(r.rsi,1),"rel_volume":round(r.rel_volume,2),
                              "rs_vs_spy":round(r.rs*100,2),
                              "reasons":"; ".join(reasons)})
            out=pd.DataFrame(rows).sort_values(["score","rel_volume"],ascending=False)
            st.metric("Symbols scored", len(out))
            st.dataframe(out.head(100), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("$2,000 historical performance simulator")
    c1,c2,c3 = st.columns(3)
    start_date = c1.date_input("Start", datetime(2025,1,1))
    end_date = c2.date_input("End", datetime(2026,8,10))
    risk_pct = c3.slider("Risk per trade", .25, 2.0, 1.0, .25)/100
    st.caption("For a practical first backtest, enter a curated universe. Full-market 1-minute history is data-intensive.")
    symbols = st.text_input("Symbols", "NVDA,MU,AMD,MRVL,FSLR,RIOT,MSFT,AMZN,META,PLTR,AVGO,ANET")
    if st.button("Run $2,000 backtest", type="primary"):
        syms=[x.strip().upper() for x in symbols.split(",") if x.strip()]
        with st.spinner("Downloading historical bars..."):
            bars=get_bars(syms, start_date, end_date, "1Min")
            spy=get_bars(["SPY"], start_date, end_date, "1Min")
        if bars.empty or spy.empty:
            st.error("No historical data returned.")
        else:
            result=backtest(bars, spy, starting_capital=2000, risk_pct=risk_pct)
            stats=result["stats"]
            cols=st.columns(6)
            for col,(k,label) in zip(cols,[
                ("ending_capital","Ending $"),("total_return_pct","Return %"),
                ("win_rate_pct","Win rate %"),("profit_factor","Profit factor"),
                ("max_drawdown_pct","Max DD %"),("trades","Trades")]):
                col.metric(label, stats.get(k,"—"))
            eq=result["equity"]
            if not eq.empty:
                st.plotly_chart(px.line(eq,x="date",y="equity",title="$2,000 Equity Curve"),use_container_width=True)
            st.subheader("Trades")
            st.dataframe(result["trades"],use_container_width=True,hide_index=True)

st.divider()
st.warning("Research warning: these results are simulations. They do not guarantee future performance. Before live use, add survivorship-bias controls, point-in-time universe membership, corporate-action handling, slippage/latency modeling, and out-of-sample validation.")
