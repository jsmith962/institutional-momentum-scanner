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
    relative_strength_percentiles,
    combine_daily_intraday_signal,
)

from backtest import swing_backtest

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
    page_title="Institutional Swing Scanner v3.2.1",
    layout="wide",
)

st.title("Institutional Swing Scanner v3.2.1")

st.caption(
    "Full U.S. market • catalyst-gap protection • daily + intraday confirmation • "
    "SMS alerts • production-equivalent swing backtester • no live orders"
)

tab1, tab2 = st.tabs(
    [
        "Live Swing Scanner",
        "$2,000 Swing Backtester",
    ]
)


# ============================================================
# HELPER FUNCTIONS FOR MOBILE DISPLAY
# ============================================================

def money(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def score_display(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}"
    except Exception:
        return "N/A"


def rr_display(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.2f}:1"
    except Exception:
        return "N/A"


def action_text(signal):
    if signal == "A+ SWING BUY":
        return "BUY — top-tier setup confirmed"
    if signal == "BUY":
        return "BUY — entry rules confirmed"
    if signal == "WATCH":
        return "WAIT FOR BUY TRIGGER"
    if signal == "TOO EXTENDED":
        return "WAIT FOR PULLBACK / RETEST"
    if signal == "AVOID":
        return "PASS"
    return "WAIT"


def signal_icon(signal):
    if signal in ["A+ SWING BUY", "BUY"]:
        return "🟢"
    if signal == "WATCH":
        return "🟡"
    if signal == "TOO EXTENDED":
        return "🔴"
    return "⚪"


def why_not_buy(row):
    signal = row.get("signal", "")

    if bool(row.get("risk_flag", False)):
        return str(
            row.get(
                "risk_reason",
                "A hard downside catalyst-risk gate is active.",
            )
        )

    if signal in ["A+ SWING BUY", "BUY"]:
        return "All required BUY gates passed."

    if signal == "TOO EXTENDED":
        return (
            "The stock is too extended from its preferred entry. "
            "Wait for a pullback or retest."
        )

    if signal == "AVOID":
        return (
            "The setup does not currently meet enough of the "
            "high-probability swing requirements."
        )

    failures = []

    try:
        swing_score = float(row.get("swing_score", 0))
    except Exception:
        swing_score = 0

    try:
        entry_quality = float(row.get("entry_quality", 0))
    except Exception:
        entry_quality = 0

    try:
        reward_risk = float(row.get("reward_risk", 0))
    except Exception:
        reward_risk = 0

    try:
        price = float(row.get("price", 0))
        entry_low = float(row.get("entry_low", 0))
        entry_high = float(row.get("entry_high", 0))

        inside_entry_zone = (
            entry_low <= price <= entry_high
        )

    except Exception:
        inside_entry_zone = False

    if swing_score < 85:
        failures.append(
            f"Swing Score {swing_score:.1f} is below the 85 BUY threshold"
        )

    if entry_quality < 10:
        failures.append(
            f"Entry Quality {entry_quality:.1f}/15 is below the 10/15 BUY requirement"
        )

    if reward_risk < 2:
        failures.append(
            f"Reward/Risk {reward_risk:.2f}:1 is below the required 2.00:1"
        )

    if not inside_entry_zone:
        failures.append(
            "Current price is outside the preferred entry zone"
        )

    if not bool(row.get("trend_health", True)):
        failures.append(
            "The 20-day and 50-day trend slopes are not both rising"
        )

    try:
        distribution_days = int(row.get("distribution_days", 0))
    except Exception:
        distribution_days = 0

    if distribution_days > 4:
        failures.append(
            f"{distribution_days} recent distribution days show excessive selling pressure"
        )

    leadership = row.get("leadership_percentile")
    if leadership is not None and not pd.isna(leadership):
        if float(leadership) < 70:
            failures.append(
                f"Market leadership rank {float(leadership):.0f}% is below the 70% BUY gate"
            )

    if not bool(row.get("intraday_confirmed", True)):
        failures.append(
            "Live intraday BUY confirmation has not passed"
        )

    if not failures and signal == "WATCH":
        failures.append(
            "Visible price, score and risk gates pass, but the broader "
            "market/regime confirmation has not yet passed the BUY requirement"
        )

    if not failures:
        return "Waiting for additional confirmation."

    return " • ".join(failures)


def render_trade_card(row, rank_num=None):
    symbol = row["symbol"]
    signal = row["signal"]

    title_prefix = (
        f"#{rank_num} "
        if rank_num is not None
        else ""
    )

    with st.container(border=True):
        st.markdown(
            f"### {title_prefix}{signal_icon(signal)} {symbol} — {signal}"
        )

        st.write(
            f"**Swing Score:** {score_display(row.get('swing_score'))}/100"
        )

        st.write(
            f"**Setup:** {row.get('setup', '')}"
        )

        st.write(
            f"**Current Price:** {money(row.get('price'))}"
        )

        st.write(
            f"**Entry Quality:** {score_display(row.get('entry_quality'))}/15"
        )

        if bool(row.get("risk_flag", False)):
            st.error(
                "**Risk Event:** "
                + str(row.get("risk_reason", "Hard risk gate active"))
            )
        else:
            leadership = row.get("leadership_percentile")
            leadership_text = (
                "N/A"
                if leadership is None or pd.isna(leadership)
                else f"{float(leadership):.0f}th percentile"
            )
            st.caption(
                f"Risk gate: PASS • Market leadership: {leadership_text} • "
                f"Distribution days: {int(row.get('distribution_days', 0))}"
            )

        st.markdown("#### Entry Plan")

        st.write(
            f"**Preferred Entry Zone:** "
            f"{money(row.get('entry_low'))} – {money(row.get('entry_high'))}"
        )

        st.write(
            f"**Stop:** {money(row.get('stop'))}"
        )

        st.write(
            f"**Target 1:** {money(row.get('target1'))}"
        )

        st.write(
            f"**Target 2:** {money(row.get('target2'))}"
        )

        st.write(
            f"**Reward / Risk:** {rr_display(row.get('reward_risk'))}"
        )

        st.markdown("#### Action")

        if signal in ["A+ SWING BUY", "BUY"]:
            st.success(
                action_text(signal)
            )

        elif signal == "WATCH":
            st.warning(
                action_text(signal)
            )

        elif signal == "TOO EXTENDED":
            st.error(
                action_text(signal)
            )

        else:
            st.info(
                action_text(signal)
            )

        if signal not in ["A+ SWING BUY", "BUY"]:
            st.markdown(
                "#### Why Not BUY Yet?"
            )

            st.info(
                why_not_buy(row)
            )

        else:
            st.markdown(
                "#### Why BUY?"
            )

            st.success(
                "Swing score, entry quality, reward/risk, entry zone "
                "and market confirmation all passed the BUY rules."
            )

        st.caption(
            f"Intraday confirmation: "
            f"{row.get('intraday_signal', 'N/A')} "
            f"({row.get('intraday_score', 'N/A')}/100)"
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
        "Phone numbers and Twilio credentials stay in Streamlit Secrets."
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
            "SELL-risk alerts are evaluated only for symbols entered here."
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
        "The scanner first removes unsuitable securities and illiquid stocks, "
        "then analyzes intraday momentum and daily swing-trade structure separately."
    )

    st.info(
        "A strong stock is not automatically a BUY. "
        "The swing engine blocks abnormal downside catalyst gaps, requires "
        "rising trends and market leadership, and will not issue a BUY until "
        "the live intraday confirmation also passes."
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
        width="stretch",
    ):

        os.environ["ALPACA_FEED"] = feed

        status = st.status(
            "Scanning the U.S. market...",
            expanded=True,
        )

        progress = st.progress(0)

        try:

            status.write(
                "1/5 Filtering eligible U.S. securities..."
            )

            elig = eligible_us_equity_universe()
            universe = elig.symbol.tolist()

            status.write(
                f"Eligible after type filtering: {len(universe):,}"
            )

            now = datetime.now(ET)
            daily_start = now - timedelta(days=45)

            status.write(
                "2/5 Applying price, liquidity, trend and momentum filters..."
            )

            def prog(done, total):
                progress.progress(
                    min(
                        int(done / max(total, 1) * 40),
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
                f"Finalists after eligibility filters: {len(finalists)}"
            )

            progress.progress(45)

            status.write(
                "3/5 Pulling longer daily history for swing-trade analysis..."
            )

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

            market_daily = get_bars(
                ["SPY", "QQQ"],
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

            if not market_daily.empty:
                market_daily["timestamp"] = pd.to_datetime(
                    market_daily["timestamp"],
                    utc=True,
                )

            spy_daily = (
                market_daily[
                    market_daily["symbol"] == "SPY"
                ].copy()
                if not market_daily.empty
                else pd.DataFrame()
            )

            qqq_daily = (
                market_daily[
                    market_daily["symbol"] == "QQQ"
                ].copy()
                if not market_daily.empty
                else pd.DataFrame()
            )

            leadership_map = relative_strength_percentiles(
                swing_daily
            )

            progress.progress(60)

            status.write(
                "4/5 Pulling intraday bars for finalists and SPY..."
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
                intra.timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                .max()
            )

            today = intra[
                intra.timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                == latest
            ]

            spy_today = spy[
                spy.timestamp
                .dt
                .tz_convert(ET)
                .dt
                .date
                == latest
            ]

            fmap = finalists.set_index(
                "symbol"
            ).to_dict(
                "index"
            )

            advmap = dict(
                zip(
                    finalists.symbol,
                    finalists.avg_dollar_volume,
                )
            )

            status.write(
                "5/5 Applying intraday confirmation, swing setup, "
                "entry quality and risk rules..."
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

                stock_swing_daily = pd.DataFrame()

                if not swing_daily.empty:
                    stock_swing_daily = swing_daily[
                        swing_daily["symbol"] == sym
                    ].copy()

                swing = None

                if not stock_swing_daily.empty:
                    swing = score_swing_daily(
                        stock_swing_daily,
                        spy_daily,
                        qqq_daily,
                        leadership_map.get(sym),
                    )

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
                risk_flag = False
                risk_reason = ""
                gap_down_pct = 0
                event_days_ago = None
                trend_health = False
                distribution_days = 0
                leadership_percentile = None
                market_score = None

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

                    risk_flag = bool(
                        swing.get(
                            "risk_flag",
                            False,
                        )
                    )

                    risk_reason = swing.get(
                        "risk_reason",
                        "",
                    )

                    gap_down_pct = swing.get(
                        "gap_down_pct",
                        0,
                    )

                    event_days_ago = swing.get(
                        "event_days_ago"
                    )

                    trend_health = bool(
                        swing.get(
                            "trend_health",
                            False,
                        )
                    )

                    distribution_days = swing.get(
                        "distribution_days",
                        0,
                    )

                    leadership_percentile = swing.get(
                        "leadership_percentile"
                    )

                    market_score = swing.get(
                        "market_score"
                    )

                if swing:
                    final_signal, confluence_reason = (
                        combine_daily_intraday_signal(
                            swing_signal,
                            intraday_signal,
                            intraday_score,
                            risk_flag=risk_flag,
                        )
                    )
                else:
                    final_signal = (
                        "WATCH"
                        if intraday_signal == "BUY"
                        else intraday_signal
                    )
                    confluence_reason = (
                        "Daily swing history is unavailable, so an intraday BUY "
                        "cannot be promoted to a final BUY."
                    )

                intraday_confirmed = bool(
                    intraday_signal == "BUY"
                    and intraday_score >= 85
                )

                if risk_flag:

                    decision_reason = risk_reason

                elif (
                    final_signal == "WATCH"
                    and (
                        not swing
                        or swing_signal in [
                            "A+ SWING BUY",
                            "BUY",
                        ]
                    )
                ):

                    decision_reason = confluence_reason

                elif final_signal == "A+ SWING BUY":

                    decision_reason = (
                        "Top-tier daily setup and live intraday entry confirmation passed."
                    )

                elif final_signal == "BUY":

                    decision_reason = (
                        "Daily swing setup and live intraday entry confirmation passed."
                    )

                elif final_signal == "WATCH":

                    decision_reason = (
                        "Promising setup; wait for a better entry or "
                        "additional confirmation."
                    )

                elif final_signal == "TOO EXTENDED":

                    decision_reason = (
                        "Strong stock but poor entry right now. "
                        "Wait for a pullback or retest."
                    )

                elif final_signal == "AVOID":

                    decision_reason = (
                        "Not enough alignment for a high-quality swing entry."
                    )

                else:
                    decision_reason = "; ".join(
                        reasons
                    )

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
                            float(r.close),
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
                            float(r.rsi)
                            if pd.notna(r.rsi)
                            else 50,
                            1,
                        ),
                        "swing_rsi": swing_rsi,
                        "swing_rvol": swing_rvol,
                        "risk_flag": risk_flag,
                        "risk_reason": risk_reason,
                        "gap_down_pct": gap_down_pct,
                        "event_days_ago": event_days_ago,
                        "trend_health": trend_health,
                        "distribution_days": distribution_days,
                        "leadership_percentile": leadership_percentile,
                        "market_score": market_score,
                        "intraday_confirmed": intraday_confirmed,
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

                buys = out[
                    out["signal"].isin(
                        [
                            "A+ SWING BUY",
                            "BUY",
                        ]
                    )
                ]

                watches = out[
                    out["signal"] == "WATCH"
                ]

                extended = out[
                    out["signal"] == "TOO EXTENDED"
                ]

                avoid = out[
                    out["signal"].isin(
                        [
                            "AVOID",
                            "NO BUY",
                        ]
                    )
                ]

                if (
                    sms_enabled
                    and sms_configured()
                ):

                    sent = []

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
                                    f"BUY {alert_row['symbol']}"
                                )

                            except Exception as sms_error:

                                st.warning(
                                    f"Could not text BUY alert for "
                                    f"{alert_row['symbol']}: "
                                    f"{sms_error}"
                                )

                    held = {
                        x.strip().upper()
                        for x in tracked_positions.split(",")
                        if x.strip()
                    }

                    if (
                        sms_sell_enabled
                        and held
                    ):

                        held_rows = out[
                            out["symbol"].isin(
                                held
                            )
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

                            if current_price < current_vwap:

                                sell_reasons.append(
                                    "price below VWAP"
                                )

                            if current_intraday_score < 60:

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

                st.divider()

                st.subheader(
                    "Current Market Decision"
                )

                if not buys.empty:

                    st.success(
                        f"🟢 {len(buys)} CONFIRMED SWING BUY "
                        f"{'SIGNAL' if len(buys) == 1 else 'SIGNALS'}"
                    )

                    st.write(
                        ", ".join(
                            buys["symbol"].head(8)
                        )
                    )

                else:

                    st.error(
                        "🔴 NO CONFIRMED SWING BUY RIGHT NOW"
                    )

                    st.caption(
                        "Do not buy simply because a stock has a high "
                        "Swing Score. Wait until the scanner changes "
                        "the signal to BUY or A+ SWING BUY."
                    )

                m1, m2 = st.columns(2)

                m1.metric(
                    "Finalists",
                    len(out),
                )

                m2.metric(
                    "Confirmed BUYs",
                    len(buys),
                )

                m3, m4 = st.columns(2)

                m3.metric(
                    "WATCH",
                    len(watches),
                )

                m4.metric(
                    "TOO EXTENDED",
                    len(extended),
                )

                st.divider()

                st.header(
                    "Top 5 Swing Opportunities"
                )

                st.caption(
                    "These are ranked setups. WATCH means wait — "
                    "it does not mean buy now."
                )

                top_opportunities = out.head(
                    5
                )

                for rank_num, (_, row) in enumerate(
                    top_opportunities.iterrows(),
                    start=1,
                ):

                    render_trade_card(
                        row,
                        rank_num,
                    )

                st.divider()

                st.header(
                    "🟢 Confirmed BUY Signals"
                )

                if not buys.empty:

                    for _, row in (
                        buys
                        .head(10)
                        .iterrows()
                    ):

                        render_trade_card(
                            row
                        )

                else:

                    st.info(
                        "No confirmed BUY signals right now."
                    )

                st.divider()

                st.header(
                    "🔴 Strong But Too Extended"
                )

                st.caption(
                    "These may be good stocks but the current entry "
                    "is too stretched. Wait for a pullback or retest."
                )

                if not extended.empty:

                    for _, row in (
                        extended
                        .head(5)
                        .iterrows()
                    ):

                        render_trade_card(
                            row
                        )

                else:

                    st.write(
                        "None."
                    )

                st.divider()

                st.header(
                    "🟡 WATCH List"
                )

                watch_columns = [
                    "symbol",
                    "signal",
                    "swing_score",
                    "setup",
                    "price",
                    "entry_low",
                    "entry_high",
                ]

                if not watches.empty:

                    st.dataframe(
                        watches[
                            watch_columns
                        ].head(30),
                        width="stretch",
                        hide_index=True,
                    )

                else:

                    st.write(
                        "None."
                    )

                st.divider()

                st.header(
                    "Stock Detail"
                )

                selected_symbol = st.selectbox(
                    "Select a stock",
                    out["symbol"].tolist(),
                )

                selected = out[
                    out["symbol"]
                    == selected_symbol
                ].iloc[0]

                render_trade_card(
                    selected
                )

                st.write(
                    f"**Today's change:** "
                    f"{selected['change_today_%']:.2f}%"
                )

                st.write(
                    f"**Relative volume:** "
                    f"{selected['rel_volume']:.2f}x"
                )

                st.write(
                    f"**Relative strength vs SPY:** "
                    f"{selected['vs_SPY_%']:.2f}%"
                )

                st.write(
                    f"**Decision reason:** "
                    f"{selected['decision']}"
                )

                st.divider()

                with st.expander(
                    "Show full research table"
                ):

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
                        "risk_flag",
                        "risk_reason",
                        "gap_down_pct",
                        "event_days_ago",
                        "trend_health",
                        "distribution_days",
                        "leadership_percentile",
                        "market_score",
                        "intraday_confirmed",
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
                        width="stretch",
                        hide_index=True,
                    )

                st.download_button(
                    "Download latest swing scan",
                    data=out.to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    ),
                    file_name="swing_scan_latest.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as e:

            status.update(
                label="Scan stopped because of an error.",
                state="error",
            )

            st.error(
                str(e)
            )

            st.info(
                "If SIP entitlement is mentioned, choose IEX."
            )


# ============================================================
# BACKTEST TAB
# ============================================================

with tab2:

    st.subheader(
        "$2,000 Production-Equivalent Swing Backtester"
    )

    st.info(
        "This version runs the same daily score, market-regime rules, "
        "leadership gate, catalyst protection and intraday confirmation used "
        "by the live scanner. A signal is evaluated at the selected time and "
        "filled no earlier than the following minute."
    )

    st.warning(
        "Important limitation: results cover only the symbols entered below. "
        "They do not yet reconstruct the entire historical U.S. market or "
        "remove survivorship bias. Treat the results as research, not proof "
        "of future profitability."
    )

    c1, c2, c3 = st.columns(
        3
    )

    start_date = c1.date_input(
        "Start",
        datetime.now(ET).date()
        - timedelta(
            days=180,
        ),
        key="swing_bt_start",
    )

    end_date = c2.date_input(
        "End",
        datetime.now(ET).date()
        - timedelta(
            days=1
        ),
        key="swing_bt_end",
    )

    risk_pct = (
        c3.slider(
            "Risk per trade",
            0.25,
            2.0,
            0.50,
            0.25,
            key="swing_bt_risk",
        )
        / 100
    )

    c4, c5, c6 = st.columns(
        3
    )

    max_positions = c4.slider(
        "Maximum open positions",
        1,
        5,
        3,
        1,
        key="swing_bt_positions",
    )

    max_holding_days = c5.slider(
        "Maximum holding sessions",
        5,
        30,
        20,
        5,
        key="swing_bt_hold",
    )

    scan_time = c6.selectbox(
        "Historical scan time (ET)",
        [
            "11:30",
            "14:00",
            "15:30",
        ],
        index=0,
        key="swing_bt_time",
    )

    c7, c8 = st.columns(
        2
    )

    slippage_bps = c7.slider(
        "Estimated slippage (basis points per order)",
        0,
        25,
        5,
        1,
        key="swing_bt_slippage",
    )

    commission_bps = c8.slider(
        "Estimated fees (basis points per order)",
        0,
        10,
        0,
        1,
        key="swing_bt_fees",
    )

    symbols = st.text_input(
        "Backtest symbols",
        (
            "NVDA,MU,AMD,MRVL,FSLR,RIOT,"
            "MSFT,AMZN,META,PLTR,AVGO,ANET"
        ),
        key="swing_bt_symbols",
        help=(
            "Use at least 10 liquid stocks. Relative-strength percentiles "
            "are calculated inside this comparison group."
        ),
    )

    btfeed = st.selectbox(
        "Backtest data feed",
        [
            "iex",
            "sip",
        ],
        index=0,
        key="swing_bt_feed",
    )

    st.caption(
        "IEX is suitable for testing but contains only one exchange. "
        "Use SIP when your Alpaca plan permits it."
    )

    if st.button(
        "RUN $2,000 BACKTEST",
        type="primary",
        width="stretch",
    ):

        syms = [
            x.strip().upper()
            for x in symbols.split(",")
            if x.strip()
        ]

        if start_date >= end_date:
            st.error(
                "Choose a start date before the end date."
            )
            st.stop()

        if (end_date - start_date).days > 365:
            st.warning(
                "A range longer than one year can be slow or exceed the "
                "minute-data limit. Start with 6–12 months, then test "
                "additional non-overlapping periods."
            )

        if len(syms) < 5:
            st.error(
                "Enter at least 5 symbols so the relative-strength ranking "
                "has a meaningful comparison group."
            )
            st.stop()

        if len(syms) < 10:
            st.warning(
                "Fewer than 10 symbols can make leadership percentiles "
                "unstable. Ten or more is recommended."
            )

        request_end = end_date + timedelta(
            days=1,
        )

        warmup_start = start_date - timedelta(
            days=450,
        )

        with st.spinner(
            "Downloading daily and minute history, rebuilding each signal "
            "and simulating the portfolio..."
        ):

            bars = get_bars_batched(
                syms,
                start_date,
                request_end,
                "1Min",
                btfeed,
                batch_size=20,
                pause_seconds=0.1,
            )

            market_minutes = get_bars(
                [
                    "SPY",
                    "QQQ",
                ],
                start_date,
                request_end,
                "1Min",
                btfeed,
            )

            daily_history = get_bars_batched(
                syms,
                warmup_start,
                request_end,
                "1Day",
                btfeed,
                batch_size=100,
                pause_seconds=0.1,
            )

            market_daily = get_bars(
                [
                    "SPY",
                    "QQQ",
                ],
                warmup_start,
                request_end,
                "1Day",
                btfeed,
            )

            if market_minutes.empty:

                spy = pd.DataFrame()
                qqq = pd.DataFrame()

            else:

                spy = market_minutes[
                    market_minutes["symbol"] == "SPY"
                ].copy()

                qqq = market_minutes[
                    market_minutes["symbol"] == "QQQ"
                ].copy()

        if (
            bars.empty
            or spy.empty
            or qqq.empty
            or daily_history.empty
            or market_daily.empty
            or not {
                "SPY",
                "QQQ",
            }.issubset(
                set(
                    market_daily["symbol"]
                )
            )
        ):

            st.error(
                "The complete daily and minute history was not returned. "
                "Try a shorter date range or choose IEX if SIP entitlement "
                "was rejected."
            )

        else:

            complete_symbols = sorted(
                set(
                    bars["symbol"]
                )
                & set(
                    daily_history["symbol"]
                )
            )

            missing_symbols = sorted(
                set(syms)
                - set(complete_symbols)
            )

            if missing_symbols:

                st.warning(
                    "Excluded symbols with incomplete daily or minute data: "
                    + ", ".join(
                        missing_symbols
                    )
                )

            if len(complete_symbols) < 5:

                st.error(
                    "Fewer than 5 symbols returned complete data. "
                    "Choose a shorter date range or a different feed."
                )
                st.stop()

            bars = bars[
                bars["symbol"].isin(
                    complete_symbols
                )
            ].copy()

            daily_history = daily_history[
                daily_history["symbol"].isin(
                    complete_symbols
                )
            ].copy()

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

            stats = res["stats"]

            cols = st.columns(
                4
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

            cols2 = st.columns(
                4
            )

            labels2 = [
                (
                    "max_drawdown_pct",
                    "Max DD %",
                ),
                (
                    "trades",
                    "Trades",
                ),
                (
                    "expectancy_r",
                    "Average expectancy (R)",
                ),
                (
                    "avg_trade_dollars",
                    "Average trade $",
                ),
            ]

            for col, (
                key,
                label,
            ) in zip(
                cols2,
                labels2,
            ):

                col.metric(
                    label,
                    stats.get(
                        key,
                        "—",
                    ),
                )

            for warning in res.get(
                "warnings",
                [],
            ):

                st.warning(
                    warning
                )

            if stats.get(
                "trades",
                0,
            ) == 0:

                st.info(
                    "No fully confirmed BUY signals became trades during "
                    "this period. That is a valid result; try a longer "
                    "period before changing the strategy rules."
                )

            diagnostics = res.get(
                "diagnostics",
                {},
            )

            funnel = diagnostics.get(
                "funnel",
                pd.DataFrame(),
            )

            gate_failures = diagnostics.get(
                "gate_failures",
                pd.DataFrame(),
            )

            near_misses = diagnostics.get(
                "near_misses",
                pd.DataFrame(),
            )

            st.subheader(
                "BUY confirmation funnel"
            )

            st.caption(
                "This shows where historical candidates stopped progressing. "
                "It does not loosen any BUY rule or predict a future win rate."
            )

            if funnel.empty:

                st.info(
                    "No candidates were available for gate diagnostics."
                )

            else:

                st.dataframe(
                    funnel,
                    width="stretch",
                    hide_index=True,
                )

            if not gate_failures.empty:

                primary = gate_failures.iloc[0]

                st.info(
                    f"Most frequently failed gate: {primary['gate']} failed for "
                    f"{int(primary['failed'])} candidates "
                    f"({float(primary['failure_percent']):.1f}%)."
                )

                st.markdown(
                    "#### Most common failed BUY gates"
                )

                st.dataframe(
                    gate_failures.head(8),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown(
                "#### Closest near misses"
            )

            st.caption(
                "One best non-BUY observation per symbol, ranked by the "
                "fewest failed gates. A near miss is not a recommendation."
            )

            if near_misses.empty:

                st.info(
                    "No non-BUY candidates were available to rank."
                )

            else:

                for _, near_miss in near_misses.head(5).iterrows():

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{near_miss['symbol']} — "
                            f"{near_miss['signal']}**"
                        )

                        st.write(
                            f"Session: {near_miss['session']} • "
                            f"Gates passed: {near_miss['gates_passed']}"
                        )

                        st.write(
                            f"Swing Score: {float(near_miss['swing_score']):.1f} • "
                            f"Intraday Score: "
                            f"{float(near_miss['intraday_score']):.1f} • "
                            f"Entry Quality: "
                            f"{float(near_miss['entry_quality']):.1f}/15"
                        )

                        st.warning(
                            "Failed BUY gates: "
                            f"{near_miss['failed_buy_gates']}"
                        )

                with st.expander(
                    "Show full near-miss table"
                ):

                    st.dataframe(
                        near_misses,
                        width="stretch",
                        hide_index=True,
                    )

            if not res["equity"].empty:

                st.plotly_chart(
                    px.line(
                        res["equity"],
                        x="date",
                        y="equity",
                        title="$2,000 Equity Curve",
                    ),
                    width="stretch",
                )

            st.subheader(
                "Simulated trades"
            )

            st.dataframe(
                res["trades"],
                width="stretch",
                hide_index=True,
            )

            if not res["trades"].empty:

                st.download_button(
                    "Download simulated trades",
                    data=res["trades"].to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    ),
                    file_name="v3_2_1_swing_backtest_trades.csv",
                    mime="text/csv",
                    width="stretch",
                )

            with st.expander(
                "Show historical signal audit"
            ):

                st.caption(
                    "This table records each reconstructed daily and "
                    "intraday decision, including signals that never became "
                    "a trade."
                )

                st.dataframe(
                    res["signal_log"],
                    width="stretch",
                    hide_index=True,
                )

                if not res["signal_log"].empty:

                    st.download_button(
                        "Download signal audit",
                        data=res["signal_log"].to_csv(
                            index=False
                        ).encode(
                            "utf-8"
                        ),
                        file_name="v3_2_1_signal_audit.csv",
                        mime="text/csv",
                        width="stretch",
                    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Research only. Scanner signals and simulated results do not "
    "guarantee future performance. Validate the strategy with "
    "out-of-sample testing and paper trading before risking real capital."
)
