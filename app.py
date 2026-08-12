import os
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import pandas as pd,streamlit as st,plotly.express as px
from dotenv import load_dotenv
from market_data import eligible_us_equity_universe,get_bars,get_bars_batched
from strategy import prefilter_daily,prepare_intraday,classify
from backtest import backtest
from alerts import sms_configured, send_sms, build_buy_message, build_sell_message

load_dotenv(); ET=ZoneInfo("America/New_York")
st.set_page_config(page_title="Institutional Momentum Scanner v2.3",layout="wide")
st.title("Institutional Momentum Scanner v2.3")
st.caption("Clean-stock universe • momentum score • SMS alerts • $2,000 simulator • no live orders")
tab1,tab2=st.tabs(["Live Scanner","$2,000 Backtester"])

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
        help="Example: NVDA,MU,OWL. SELL alerts are evaluated only for symbols entered here."
    )

with tab1:
    st.subheader("Full U.S. Common-Stock Momentum Scan")
    st.write("Filters many ETFs/ETNs, warrants, preferreds, units, rights, leveraged/inverse products and illiquid names before scoring.")
    c1,c2=st.columns(2)
    feed=c1.selectbox("Data feed",["iex","sip"],index=0)
    finalists_n=c2.slider("Finalists for detailed scan",50,250,150,25)
    if st.button("RUN CLEAN FULL-MARKET SCAN",type="primary",use_container_width=True):
        os.environ["ALPACA_FEED"]=feed
        status=st.status("Scanning the U.S. market...",expanded=True); progress=st.progress(0)
        try:
            status.write("1/4 Filtering eligible U.S. securities...")
            elig=eligible_us_equity_universe(); universe=elig.symbol.tolist()
            status.write(f"Eligible after type filtering: {len(universe):,}")
            now=datetime.now(ET); daily_start=now-timedelta(days=45)
            status.write("2/4 Applying price, liquidity, momentum and volume filters...")
            def prog(done,total): progress.progress(min(int(done/max(total,1)*45),45))
            daily=get_bars_batched(universe,daily_start,now,"1Day",feed,batch_size=200,pause_seconds=.12,progress_callback=prog)
            finalists=prefilter_daily(daily,elig,min_price=5,min_avg_dollar_volume=10_000_000,min_avg_volume=500_000,keep=finalists_n)
            if finalists.empty:
                status.update(label="No qualifying finalists found.",state="complete"); st.stop()
            status.write(f"Finalists after eligibility filters: {len(finalists)}"); progress.progress(55)
            status.write("3/4 Pulling 1-minute bars for finalists and SPY...")
            start=now.replace(hour=9,minute=30,second=0,microsecond=0)
            intra=get_bars_batched(finalists.symbol.tolist(),start-timedelta(days=1),now,"1Min",feed,batch_size=100,pause_seconds=.1)
            spy=get_bars(["SPY"],start-timedelta(days=1),now,"1Min",feed); progress.progress(75)
            if intra.empty:
                status.update(label="No intraday bars returned.",state="error"); st.stop()
            intra["timestamp"]=pd.to_datetime(intra.timestamp,utc=True); spy["timestamp"]=pd.to_datetime(spy.timestamp,utc=True)
            latest=intra.timestamp.dt.tz_convert(ET).dt.date.max()
            today=intra[intra.timestamp.dt.tz_convert(ET).dt.date==latest]
            spy_today=spy[spy.timestamp.dt.tz_convert(ET).dt.date==latest]
            fmap=finalists.set_index("symbol").to_dict("index"); advmap=dict(zip(finalists.symbol,finalists.avg_dollar_volume))
            status.write("4/4 Applying VWAP, opening range, relative volume, EMA, RSI and SPY-relative-strength rules...")
            rows=[]
            for sym,d in today.groupby("symbol"):
                if len(d)<20: continue
                ref=fmap.get(sym,{}); px=max(float(ref.get("price",1)),.01)
                adv=float(advmap.get(sym,0))/px if advmap.get(sym,0) else None
                p=prepare_intraday(d,spy_today,adv)
                if p.empty: continue
                r=p.iloc[-1]; score,sig,reasons=classify(r,advmap.get(sym,0))
                rows.append({"symbol":sym,"name":ref.get("name",""),"score":score,"signal":sig,
                    "price":round(float(r.close),2),"change_today_%":round(float(r.get("stock_ret",0))*100,2),
                    "rel_volume":round(float(r.get("rel_volume",0)),2),"vwap":round(float(r.vwap),2),
                    "rsi":round(float(r.rsi) if pd.notna(r.rsi) else 50,1),"vs_SPY_%":round(float(r.get("rs",0))*100,2),
                    "security_type":ref.get("security_type","Common-stock candidate"),"type_check":"PASS",
                    "price_check":"PASS","liquidity_check":"PASS","reasons":"; ".join(reasons)})
            out=pd.DataFrame(rows); progress.progress(100)
            status.update(label="Clean full-market scan complete.",state="complete",expanded=False)
            if out.empty: st.warning("No finalists had enough intraday bars to score.")
            else:
                rank={"BUY":0,"WATCH":1,"NO BUY":2}; out["_rank"]=out.signal.map(rank)
                out=out.sort_values(["_rank","score","rel_volume"],ascending=[True,False,False]).drop(columns="_rank")
                buys=out[out.signal=="BUY"]; watches=out[out.signal=="WATCH"]

                # SMS alerts are sent when this scan is run.
                if sms_enabled and sms_configured():
                    sent = []
                    if sms_buy_enabled and not buys.empty:
                        for _, alert_row in buys.head(5).iterrows():
                            try:
                                send_sms(build_buy_message(alert_row))
                                sent.append(f"BUY {alert_row['symbol']}")
                            except Exception as sms_error:
                                st.warning(f"Could not text BUY alert for {alert_row['symbol']}: {sms_error}")

                    # SELL alerts apply only to symbols the user says they currently hold.
                    held = {x.strip().upper() for x in tracked_positions.split(",") if x.strip()}
                    if sms_sell_enabled and held:
                        for _, alert_row in out[out["symbol"].isin(held)].iterrows():
                            # Simple v2.3 sell-risk rule:
                            # text when a held stock loses VWAP and is no longer WATCH/BUY,
                            # or the score falls below 60.
                            sell_reasons = []
                            if float(alert_row["price"]) < float(alert_row["vwap"]):
                                sell_reasons.append("price below VWAP")
                            if int(alert_row["score"]) < 60:
                                sell_reasons.append(f"score fell to {int(alert_row['score'])}")
                            if alert_row["signal"] == "NO BUY" and sell_reasons:
                                try:
                                    send_sms(build_sell_message(
                                        alert_row["symbol"],
                                        alert_row["price"],
                                        ", ".join(sell_reasons)
                                    ))
                                    sent.append(f"SELL {alert_row['symbol']}")
                                except Exception as sms_error:
                                    st.warning(f"Could not text SELL alert for {alert_row['symbol']}: {sms_error}")

                    if sent:
                        st.info("Text alerts sent: " + ", ".join(sent))
                a,b,c,d=st.columns(4); a.metric("Eligible U.S. stocks",f"{len(universe):,}"); b.metric("Finalists scored",len(out)); c.metric("BUY signals",len(buys)); d.metric("WATCH signals",len(watches))
                st.success("BUY candidates: "+", ".join(buys.symbol.head(8))) if not buys.empty else st.info("NO CONFIRMED BUY right now.")
                st.subheader("Best opportunities now"); st.dataframe(out.head(50),use_container_width=True,hide_index=True)
                st.subheader("BUY signals only"); st.dataframe(buys,use_container_width=True,hide_index=True) if not buys.empty else st.write("None.")
                st.subheader("WATCH list"); st.dataframe(watches.head(30),use_container_width=True,hide_index=True) if not watches.empty else st.write("None.")
        except Exception as e:
            status.update(label="Scan stopped because of an error.",state="error"); st.error(str(e)); st.info("If SIP entitlement is mentioned, choose IEX.")

with tab2:
    st.subheader("$2,000 Historical Performance Simulator")
    c1,c2,c3=st.columns(3)
    start_date=c1.date_input("Start",datetime(2025,1,1)); end_date=c2.date_input("End",datetime.now().date()-timedelta(days=1))
    risk_pct=c3.slider("Risk per trade",.25,2.0,1.0,.25)/100
    symbols=st.text_input("Backtest symbols","NVDA,MU,AMD,MRVL,FSLR,RIOT,MSFT,AMZN,META,PLTR,AVGO,ANET")
    btfeed=st.selectbox("Backtest data feed",["iex","sip"],index=0,key="bt")
    if st.button("RUN $2,000 BACKTEST",type="primary"):
        syms=[x.strip().upper() for x in symbols.split(",") if x.strip()]
        with st.spinner("Downloading historical data and simulating trades..."):
            bars=get_bars_batched(syms,start_date,end_date,"1Min",btfeed,batch_size=30,pause_seconds=.1)
            spy=get_bars(["SPY"],start_date,end_date,"1Min",btfeed)
        if bars.empty or spy.empty: st.error("No historical data returned.")
        else:
            res=backtest(bars,spy,starting_capital=2000,risk_pct=risk_pct); stats=res["stats"]; cols=st.columns(6)
            for col,(k,label) in zip(cols,[("ending_capital","Ending $"),("total_return_pct","Return %"),("win_rate_pct","Win rate %"),("profit_factor","Profit factor"),("max_drawdown_pct","Max DD %"),("trades","Trades")]): col.metric(label,stats.get(k,"—"))
            if not res["equity"].empty: st.plotly_chart(px.line(res["equity"],x="date",y="equity",title="$2,000 Equity Curve"),use_container_width=True)
            st.subheader("Simulated trades"); st.dataframe(res["trades"],use_container_width=True,hide_index=True)

st.divider()
st.warning("Research only. Simulated results do not guarantee future performance.")
