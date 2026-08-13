import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from market_data import (
    eligible_us_equity_universe,
    get_bars,
    get_bars_batched,
)

from strategy import (
    prefilter_daily,
    prepare_intraday,
    classify,
    score_swing_daily,
)

from backtest import backtest

from alerts import (
    sms_configured,
    send_sms,
    build_buy_message,
    build_sell_message,
)


# ============================================================
# APP CONFIGURATION
# ============================================================

load_dotenv()

ET = ZoneInfo("America/New_York")

st.set_page_config(
    page_title="Institutional Swing Scanner v3",
    layout="wide",
)

st.title("Institutional Swing Scanner v3")

st.caption(
    "Full U.S. market • swing-trade probability framework • "
    "entry-quality protection • SMS alerts • $2,000 simulator • "
    "no live orders"
)

tab1, tab2 = st.tabs(
    [
        "Live Swing Scanner",
        "$2,000 Backtester",
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Text alerts")

    sms_enabled = st.toggle(
        "Send SMS alerts",
        value=False,
    )

    sms_buy_enabled = st.checkbox(
        "Text BUY signals",
        value=True,
    )

    sms_sell_enabled = st.checkbox(
        "Text SELL signals",
        value=True,
    )

    st.caption(
        "Phone numbers and Twilio credentials stay in "
        "Streamlit Secrets."
    )

    if sms_enabled:

        if sms_configured():

            st.success("SMS configured")

        else:

            st.warning(
                "SMS secrets are not configured yet."
            )

    st.divider()

    st.subheader("Tracked positions")

    tracked_positions = st.text_input(
        "Symbols you currently hold",
        "",
        help=(
            "Example: NVDA,MU,OWL. "
            "SELL-risk alerts are evaluated only for "
            "symbols entered here."
        ),
    )


# ============================================================
# LIVE SCANNER TAB
# ============================================================

with tab1:

    st.subheader(
        "Full U.S. Market Swing-Trade Scanner"
    )

    st.write(
        "The scanner first removes unsuitable securities and "
        "illiquid stocks, then analyzes intraday momentum and "
        "daily swing-trade structure separately."
    )

    st.info(
        "A strong stock is not automatically a BUY. "
        "The swing engine can classify a stock as "
        "TOO EXTENDED when the setup is strong but the "
        "current entry is poor."
    )

    c1, c2 = st.columns(2)

    feed = c1.selectbox(
        "Data feed",
        ["iex", "sip"],
        index=0,
    )

    finalists_n = c2.slider(
        "Finalists for detailed scan",
        50,
        250,
        150,
        25,
    )

    if st.button(
        "RUN FULL-MARKET SWING SCAN",
        type="primary",
        use_container_width=True,
    ):

        os.environ["ALPACA_FEED"] = feed

        status = st.status(
            "Scanning the U.S. market...",
            expanded=True,
        )

        progress = st.progress(0)

        try:

            # =================================================
            # STEP 1
            # ELIGIBLE U.S. STOCK UNIVERSE
            # =================================================

            status.write(
                "1/5 Filtering eligible U.S. securities..."
            )

            elig = eligible_us_equity_universe()

            universe = elig.symbol.tolist()

            status.write(
                f"Eligible after type filtering: "
                f"{len(universe):,}"
            )

            now = datetime.now(ET)

            daily_start = now - timedelta(days=45)


            # =================================================
            # STEP 2
            # FAST FULL-MARKET PREFILTER
            # =================================================

            status.write(
                "2/5 Applying price, liquidity, trend and "
                "momentum filters..."
            )

            def prog(done, total):

                progress.progress(
                    min(
                        int(
                            done
                            / max(total, 1)
                            * 40
                        ),
                        40,
                    )
                )

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

                status.update(
                    label="No daily market data returned.",
                    state="error",
                )

                st.stop()

            daily["timestamp"] = pd.to_datetime(
                daily["timestamp"],
                utc=True,
            )

            finalists = prefilter_daily(
                daily,
                elig,
                min_price=5,
                min_avg_dollar_volume=10_000_000,
                min_avg_volume=500_000,
                limit=finalists_n,
            )

            if finalists.empty:

                status.update(
                    label="No qualifying finalists found.",
                    state="complete",
                )

                st.stop()

            status.write(
                f"Finalists after eligibility filters: "
                f"{len(finalists)}"
            )

            progress.progress(45)


            # =================================================
            # STEP 3
            # GET LONGER DAILY HISTORY FOR SWING ANALYSIS
            # =================================================

            status.write(
                "3/5 Pulling longer daily history for "
                "swing-trade analysis..."
            )

            finalist_symbols = (
                finalists
                .symbol
                .tolist()
            )

            swing_start = (
                now
                - timedelta(days=420)
            )

            swing_daily = get_bars_batched(
                finalist_symbols,
                swing_start,
                now,
                "1Day",
                feed,
                batch_size=100,
                pause_seconds=0.10,
            )

            spy_daily = get_bars(
                ["SPY"],
                swing_start,
                now,
                "1Day",
                feed,
            )

            if not swing_daily.empty:

                swing_daily["timestamp"] = pd.to_datetime(
                    swing_daily["timestamp"],
                    utc=True,
                )

            if not spy_daily.empty:

                spy_daily["timestamp"] = pd.to_datetime(
                    spy_daily["timestamp"],
                    utc=True,
                )

            progress.progress(60)


            # =================================================
            # STEP 4
            # INTRADAY DATA
            # =================================================

            status.write(
                "4/5 Pulling intraday bars for finalists "
                "and SPY..."
            )

            market_start = now.replace(
                hour=9,
                minute=30,
                second=0,
                microsecond=0,
            )

            intra = get_bars_batched(
                finalist_symbols,
                market_start - timedelta(days=1),
                now,
                "1Min",
                feed,
                batch_size=100,
                pause_seconds=0.10,
            )

            spy = get_bars(
                ["SPY"],
                market_start - timedelta(days=1),
                now,
                "1Min",
                feed,
            )

            progress.progress(75)

            if intra.empty:

                status.update(
                    label="No intraday bars returned.",
                    state="error",
                )

                st.stop()

            intra["timestamp"] = pd.to_datetime(
                intra["timestamp"],
                utc=True,
            )

            spy["timestamp"] = pd.to_datetime(
                spy["timestamp"],
                utc=True,
            )

            latest = (
                intra
                .timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                .max()
            )

            today = intra[
                intra
                .timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                == latest
            ]

            spy_today = spy[
                spy
                .timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                == latest
            ]

            fmap = (
                finalists
                .set_index("symbol")
                .to_dict("index")
            )

            advmap = dict(
                zip(
                    finalists.symbol,
                    finalists.avg_dollar_volume,
                )
            )


            # =================================================
            # STEP 5
            # SCORE EACH FINALIST
            # =================================================

            status.write(
                "5/5 Applying intraday confirmation, "
                "swing setup, entry quality and risk rules..."
            )

            rows = []

            for sym, d in today.groupby("symbol"):

                if len(d) < 20:

                    continue

                ref = fmap.get(
                    sym,
                    {},
                )

                px = max(
                    float(
                        ref.get(
                            "price",
                            1,
                        )
                    ),
                    0.01,
                )

                adv = (
                    float(
                        advmap.get(
                            sym,
                            0,
                        )
                    )
                    / px
                    if advmap.get(sym, 0)
                    else None
                )


                # ---------------------------------------------
                # INTRADAY ANALYSIS
                # ---------------------------------------------

                p = prepare_intraday(
                    d,
                    spy_today,
                    adv,
                )

                if p.empty:

                    continue

                r = p.iloc[-1]

                intraday_score, intraday_signal, reasons = classify(
                    r,
                    advmap.get(
                        sym,
                        0,
                    ),
                )


                # ---------------------------------------------
                # SWING ANALYSIS
                # ---------------------------------------------

                stock_swing_daily = pd.DataFrame()

                if not swing_daily.empty:

                    stock_swing_daily = swing_daily[
                        swing_daily["symbol"]
                        == sym
                    ].copy()

                swing = None

                if not stock_swing_daily.empty:

                    swing = score_swing_daily(
                        stock_swing_daily,
                        spy_daily,
                    )


                # ---------------------------------------------
                # DEFAULT SWING VALUES
                # ---------------------------------------------

                swing_signal = "N/A"

                swing_score = 0

                setup = ""

                entry_quality = 0

                entry_low = None

                entry_high = None

                stop = None

                target1 = None

                target2 = None

                reward_risk = None

                swing_rsi = None

                swing_rvol = None


                # ---------------------------------------------
                # LOAD SWING RESULTS
                # ---------------------------------------------

                if swing:

                    swing_signal = swing.get(
                        "signal",
                        "N/A",
                    )

                    swing_score = swing.get(
                        "swing_score",
                        0,
                    )

                    setup = swing.get(
                        "setup",
                        "",
                    )

                    entry_quality = swing.get(
                        "entry_quality",
                        0,
                    )

                    entry_low = swing.get(
                        "entry_low"
                    )

                    entry_high = swing.get(
                        "entry_high"
                    )

                    stop = swing.get(
                        "stop"
                    )

                    target1 = swing.get(
                        "target1"
                    )

                    target2 = swing.get(
                        "target2"
                    )

                    reward_risk = swing.get(
                        "reward_risk"
                    )

                    swing_rsi = swing.get(
                        "rsi14"
                    )

                    swing_rvol = swing.get(
                        "rvol"
                    )


                # ---------------------------------------------
                # FINAL DECISION
                # ---------------------------------------------

                if swing:

                    final_signal = swing_signal

                else:

                    final_signal = intraday_signal


                # ---------------------------------------------
                # EXPLANATION
                # ---------------------------------------------

                decision_reason = ""

                if final_signal == "A+ SWING BUY":

                    decision_reason = (
                        "Top-tier swing setup and "
                        "entry quality confirmed."
                    )

                elif final_signal == "BUY":

                    decision_reason = (
                        "Swing setup and entry rules confirmed."
                    )

                elif final_signal == "WATCH":

                    decision_reason = (
                        "Promising setup; wait for a better "
                        "entry or additional confirmation."
                    )

                elif final_signal == "TOO EXTENDED":

                    decision_reason = (
                        "Strong stock but poor entry right now. "
                        "Wait for a pullback or retest."
                    )

                elif final_signal == "AVOID":

                    decision_reason = (
                        "Not enough alignment for a "
                        "high-quality swing entry."
                    )

                else:

                    decision_reason = "; ".join(
                        reasons
                    )


                # ---------------------------------------------
                # SAVE RESULT
                # ---------------------------------------------

                rows.append(
                    {
                        "symbol": sym,

                        "name": ref.get(
                            "name",
                            "",
                        ),

                        "signal": final_signal,

                        "swing_score": swing_score,

                        "setup": setup,

                        "entry_quality": entry_quality,

                        "price": round(
                            float(
                                r.close
                            ),
                            2,
                        ),

                        "entry_low": entry_low,

                        "entry_high": entry_high,

                        "stop": stop,

                        "target1": target1,

                        "target2": target2,

                        "reward_risk": reward_risk,

                        "intraday_signal": intraday_signal,

                        "intraday_score": intraday_score,

                        "change_today_%": round(
                            float(
                                r.get(
                                    "stock_ret",
                                    0,
                                )
                            )
                            * 100,
                            2,
                        ),

                        "rel_volume": round(
                            float(
                                r.get(
                                    "rel_volume",
                                    0,
                                )
                            ),
                            2,
                        ),

                        "vwap": round(
                            float(
                                r.vwap
                            ),
                            2,
                        ),

                        "intraday_rsi": round(
                            float(
                                r.rsi
                            )
                            if pd.notna(
                                r.rsi
                            )
                            else 50,
                            1,
                        ),

                        "swing_rsi": swing_rsi,

                        "swing_rvol": swing_rvol,

                        "vs_SPY_%": round(
                            float(
                                r.get(
                                    "rs",
                                    0,
                                )
                            )
                            * 100,
                            2,
                        ),

                        "security_type": ref.get(
                            "security_type",
                            "Common-stock candidate",
                        ),

                        "type_check": "PASS",

                        "price_check": "PASS",

                        "liquidity_check": "PASS",

                        "decision": decision_reason,

                        "intraday_reasons": "; ".join(
                            reasons
                        ),
                    }
                )


            # =================================================
            # BUILD FINAL TABLE
            # =================================================

            out = pd.DataFrame(
                rows
            )

            progress.progress(
                100
            )

            status.update(
                label="Swing scan complete.",
                state="complete",
                expanded=False,
            )

            if out.empty:

                st.warning(
                    "No finalists had enough data to score."
                )

            else:

                rank = {
                    "A+ SWING BUY": 0,
                    "BUY": 1,
                    "WATCH": 2,
                    "TOO EXTENDED": 3,
                    "AVOID": 4,
                    "NO BUY": 5,
                    "N/A": 6,
                }

                out["_rank"] = (
                    out["signal"]
                    .map(rank)
                    .fillna(6)
                )

                out = (
                    out
                    .sort_values(
                        [
                            "_rank",
                            "swing_score",
                            "entry_quality",
                            "intraday_score",
                        ],
                        ascending=[
                            True,
                            False,
                            False,
                            False,
                        ],
                    )
                    .drop(
                        columns="_rank"
                    )
                )


                # =================================================
                # SIGNAL GROUPS
                # =================================================

                buys = out[
                    out["signal"].isin(
                        [
                            "A+ SWING BUY",
                            "BUY",
                        ]
                    )
                ]

                watches = out[
                    out["signal"]
                    == "WATCH"
                ]

                extended = out[
                    out["signal"]
                    == "TOO EXTENDED"
                ]

                avoid = out[
                    out["signal"]
                    .isin(
                        [
                            "AVOID",
                            "NO BUY",
                        ]
                    )
                ]


                # =================================================
                # SMS ALERTS
                # =================================================

                if (
                    sms_enabled
                    and sms_configured()
                ):

                    sent = []


                    # ---------------------------------------------
                    # BUY SMS
                    # ---------------------------------------------

                    if (
                        sms_buy_enabled
                        and not buys.empty
                    ):

                        for _, alert_row in (
                            buys
                            .head(5)
                            .iterrows()
                        ):

                            try:

                                send_sms(
                                    build_buy_message(
                                        alert_row
                                    )
                                )

                                sent.append(
                                    f"BUY "
                                    f"{alert_row['symbol']}"
                                )

                            except Exception as sms_error:

                                st.warning(
                                    f"Could not text BUY alert for "
                                    f"{alert_row['symbol']}: "
                                    f"{sms_error}"
                                )


                    # ---------------------------------------------
                    # SELL-RISK SMS
                    # ---------------------------------------------

                    held = {
                        x.strip().upper()
                        for x
                        in tracked_positions.split(",")
                        if x.strip()
                    }

                    if (
                        sms_sell_enabled
                        and held
                    ):

                        held_rows = out[
                            out["symbol"]
                            .isin(held)
                        ]

                        for _, alert_row in held_rows.iterrows():

                            sell_reasons = []

                            current_price = float(
                                alert_row["price"]
                            )

                            current_vwap = float(
                                alert_row["vwap"]
                            )

                            current_intraday_score = int(
                                alert_row[
                                    "intraday_score"
                                ]
                            )

                            current_intraday_signal = (
                                alert_row[
                                    "intraday_signal"
                                ]
                            )

                            if (
                                current_price
                                < current_vwap
                            ):

                                sell_reasons.append(
                                    "price below VWAP"
                                )

                            if (
                                current_intraday_score
                                < 60
                            ):

                                sell_reasons.append(
                                    f"intraday score fell to "
                                    f"{current_intraday_score}"
                                )

                            sell_risk = (
                                current_intraday_signal
                                in [
                                    "AVOID",
                                    "NO BUY",
                                ]
                                and sell_reasons
                            )

                            if sell_risk:

                                try:

                                    send_sms(
                                        build_sell_message(
                                            alert_row[
                                                "symbol"
                                            ],
                                            alert_row[
                                                "price"
                                            ],
                                            ", ".join(
                                                sell_reasons
                                            ),
                                        )
                                    )

                                    sent.append(
                                        f"SELL-RISK "
                                        f"{alert_row['symbol']}"
                                    )

                                except Exception as sms_error:

                                    st.warning(
                                        f"Could not text SELL alert for "
                                        f"{alert_row['symbol']}: "
                                        f"{sms_error}"
                                    )

                    if sent:

                        st.info(
                            "Text alerts sent: "
                            + ", ".join(sent)
                        )


                # =================================================
                # SUMMARY METRICS
                # =================================================

                a, b, c, d, e = st.columns(
                    5
                )

                a.metric(
                    "Eligible U.S. stocks",
                    f"{len(universe):,}",
                )

                b.metric(
                    "Finalists scored",
                    len(out),
                )

                c.metric(
                    "Confirmed BUYs",
                    len(buys),
                )

                d.metric(
                    "WATCH",
                    len(watches),
                )

                e.metric(
                    "TOO EXTENDED",
                    len(extended),
                )


                # =================================================
                # BUY MESSAGE
                # =================================================

                if not buys.empty:

                    st.success(
                        "CONFIRMED SWING BUY candidates: "
                        + ", ".join(
                            buys.symbol.head(8)
                        )
                    )

                else:

                    st.info(
                        "NO CONFIRMED SWING BUY right now."
                    )


                # =================================================
                # BEST OPPORTUNITIES
                # =================================================

                st.subheader(
                    "Best Swing Opportunities Now"
                )

                display_columns = [
                    "symbol",
                    "name",
                    "signal",
                    "swing_score",
                    "setup",
                    "entry_quality",
                    "price",
                    "entry_low",
                    "entry_high",
                    "stop",
                    "target1",
                    "target2",
                    "reward_risk",
                    "intraday_signal",
                    "intraday_score",
                    "change_today_%",
                    "rel_volume",
                    "vs_SPY_%",
                    "decision",
                ]

                available_columns = [
                    col
                    for col in display_columns
                    if col in out.columns
                ]

                st.dataframe(
                    out[
                        available_columns
                    ].head(50),
                    use_container_width=True,
                    hide_index=True,
                )


                # =================================================
                # BUY SIGNALS
                # =================================================

                st.subheader(
                    "🟢 Confirmed Swing BUY Signals"
                )

                if not buys.empty:

                    st.dataframe(
                        buys[
                            available_columns
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.write(
                        "None."
                    )


                # =================================================
                # WATCH LIST
                # =================================================

                st.subheader(
                    "🟡 Swing WATCH List"
                )

                if not watches.empty:

                    st.dataframe(
                        watches[
                            available_columns
                        ].head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.write(
                        "None."
                    )


                # =================================================
                # TOO EXTENDED
                # =================================================

                st.subheader(
                    "🔴 Strong Stocks — TOO EXTENDED"
                )

                st.caption(
                    "These stocks may have strong momentum but "
                    "do not currently have an acceptable swing "
                    "entry. The scanner is specifically preventing "
                    "you from chasing them."
                )

                if not extended.empty:

                    st.dataframe(
                        extended[
                            available_columns
                        ].head(30),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.write(
                        "None."
                    )


                # =================================================
                # CANDIDATE DETAIL VIEW
                # =================================================

                st.subheader(
                    "Individual Swing Trade Details"
                )

                selected_symbol = st.selectbox(
                    "Select a stock",
                    out["symbol"].tolist(),
                )

                selected = out[
                    out["symbol"]
                    == selected_symbol
                ].iloc[0]

                m1, m2, m3, m4 = st.columns(
                    4
                )

                m1.metric(
                    "Swing Score",
                    f"{selected['swing_score']}/100",
                )

                m2.metric(
                    "Entry Quality",
                    f"{selected['entry_quality']}/15",
                )

                rr_display = (
                    f"{selected['reward_risk']}:1"
                    if pd.notna(
                        selected[
                            "reward_risk"
                        ]
                    )
                    else "N/A"
                )

                m3.metric(
                    "Reward / Risk",
                    rr_display,
                )

                m4.metric(
                    "Signal",
                    selected[
                        "signal"
                    ],
                )

                st.write(
                    f"**Setup:** "
                    f"{selected['setup']}"
                )

                if pd.notna(
                    selected[
                        "entry_low"
                    ]
                ):

                    st.write(
                        f"**Preferred Entry Zone:** "
                        f"${selected['entry_low']:.2f} – "
                        f"${selected['entry_high']:.2f}"
                    )

                if pd.notna(
                    selected[
                        "stop"
                    ]
                ):

                    st.write(
                        f"**Stop:** "
                        f"${selected['stop']:.2f}"
                    )

                if pd.notna(
                    selected[
                        "target1"
                    ]
                ):

                    st.write(
                        f"**Target 1:** "
                        f"${selected['target1']:.2f}"
                    )

                if pd.notna(
                    selected[
                        "target2"
                    ]
                ):

                    st.write(
                        f"**Target 2:** "
                        f"${selected['target2']:.2f}"
                    )

                st.write(
                    f"**Decision:** "
                    f"{selected['decision']}"
                )

                st.write(
                    f"**Intraday confirmation:** "
                    f"{selected['intraday_signal']} "
                    f"({selected['intraday_score']}/100)"
                )


                # =================================================
                # DOWNLOAD
                # =================================================

                st.download_button(
                    "Download latest swing scan",
                    data=out.to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    ),
                    file_name=(
                        "swing_scan_latest.csv"
                    ),
                    mime="text/csv",
                )


        except Exception as e:

            status.update(
                label=(
                    "Scan stopped because of an error."
                ),
                state="error",
            )

            st.error(
                str(e)
            )

            st.info(
                "If SIP entitlement is mentioned, "
                "choose IEX."
            )


# ============================================================
# BACKTEST TAB
# ============================================================

with tab2:

    st.subheader(
        "$2,000 Historical Performance Simulator"
    )

    c1, c2, c3 = st.columns(
        3
    )

    start_date = c1.date_input(
        "Start",
        datetime(
            2025,
            1,
            1,
        ),
    )

    end_date = c2.date_input(
        "End",
        datetime.now().date()
        - timedelta(
            days=1
        ),
    )

    risk_pct = (
        c3.slider(
            "Risk per trade",
            0.25,
            2.0,
            1.0,
            0.25,
        )
        / 100
    )

    symbols = st.text_input(
        "Backtest symbols",
        (
            "NVDA,MU,AMD,MRVL,FSLR,RIOT,"
            "MSFT,AMZN,META,PLTR,AVGO,ANET"
        ),
    )

    btfeed = st.selectbox(
        "Backtest data feed",
        [
            "iex",
            "sip",
        ],
        index=0,
        key="bt",
    )

    if st.button(
        "RUN $2,000 BACKTEST",
        type="primary",
    ):

        syms = [
            x.strip().upper()
            for x
            in symbols.split(",")
            if x.strip()
        ]

        with st.spinner(
            "Downloading historical data "
            "and simulating trades..."
        ):

            bars = get_bars_batched(
                syms,
                start_date,
                end_date,
                "1Min",
                btfeed,
                batch_size=30,
                pause_seconds=0.1,
            )

            spy = get_bars(
                ["SPY"],
                start_date,
                end_date,
                "1Min",
                btfeed,
            )

        if (
            bars.empty
            or spy.empty
        ):

            st.error(
                "No historical data returned."
            )

        else:

            res = backtest(
                bars,
                spy,
                starting_capital=2000,
                risk_pct=risk_pct,
            )

            stats = res[
                "stats"
            ]

            cols = st.columns(
                6
            )

            labels = [
                (
                    "ending_capital",
                    "Ending $",
                ),
                (
                    "total_return_pct",
                    "Return %",
                ),
                (
                    "win_rate_pct",
                    "Win rate %",
                ),
                (
                    "profit_factor",
                    "Profit factor",
                ),
                (
                    "max_drawdown_pct",
                    "Max DD %",
                ),
                (
                    "trades",
                    "Trades",
                ),
            ]

            for col, (
                key,
                label,
            ) in zip(
                cols,
                labels,
            ):

                col.metric(
                    label,
                    stats.get(
                        key,
                        "—",
                    ),
                )

            if not res[
                "equity"
            ].empty:

                st.plotly_chart(
                    px.line(
                        res[
                            "equity"
                        ],
                        x="date",
                        y="equity",
                        title=(
                            "$2,000 Equity Curve"
                        ),
                    ),
                    use_container_width=True,
                )

            st.subheader(
                "Simulated trades"
            )

            st.dataframe(
                res[
                    "trades"
                ],
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Research only. Scanner signals and simulated results "
    "do not guarantee future performance. Validate the "
    "strategy with out-of-sample testing and paper trading "
    "before risking real capital."
)
