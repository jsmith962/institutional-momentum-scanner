import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from alerts import (
    build_buy_message,
    build_sell_message,
    send_sms,
    sms_configured,
)

from backtest import swing_backtest

from calibration_ui import render_calibration_lab

from forward_research_ui import render_forward_research_lab

from market_data import (
    eligible_us_equity_universe,
    get_bars,
    get_bars_batched,
)

from strategy import (
    classify,
    combine_daily_intraday_signal,
    prefilter_daily,
    prepare_intraday,
    relative_strength_percentiles,
    score_swing_daily,
)


# ============================================================
# OPTIONAL v3.6 RESEARCH UI
# ============================================================

try:
    from threshold_discovery_ui import render_threshold_discovery_lab

    V36_RESEARCH_AVAILABLE = True
    V36_IMPORT_ERROR = None

except Exception as exc:
    render_threshold_discovery_lab = None
    V36_RESEARCH_AVAILABLE = False
    V36_IMPORT_ERROR = str(exc)


# ============================================================
# APP CONFIGURATION
# ============================================================

load_dotenv()

ET = ZoneInfo("America/New_York")

st.set_page_config(
    page_title="Institutional Swing Scanner v3.7",
    layout="wide",
)

st.title("Institutional Swing Scanner v3.7")

st.caption(
    "Full U.S. market | catalyst-gap protection | daily + intraday "
    "confirmation | SMS alerts | production-equivalent backtesting | "
    "walk-forward validation | fast calibration | empirical threshold "
    "discovery | gate bottleneck research | forward-return analysis | "
    "no live orders"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Live Swing Scanner",
        "$2,000 Swing Backtester",
        "Calibration & Validation",
        "v3.6 Threshold Research",
        "v3.7 Forward Research",
    ]
)


# ============================================================
# SESSION STORAGE
# ============================================================

if "latest_backtest_result" not in st.session_state:
    st.session_state.latest_backtest_result = None

if "latest_backtest_settings" not in st.session_state:
    st.session_state.latest_backtest_settings = None

if "latest_backtest_daily_bars" not in st.session_state:
    st.session_state.latest_backtest_daily_bars = None

if "latest_backtest_market_daily" not in st.session_state:
    st.session_state.latest_backtest_market_daily = None

if "v37_forward_research_result" not in st.session_state:
    st.session_state.v37_forward_research_result = None

if "v37_enriched_signal_log" not in st.session_state:
    st.session_state.v37_enriched_signal_log = None


# ============================================================
# BASIC HELPERS
# ============================================================

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


def safe_float(
    value,
    default=0.0,
):

    try:

        if value is None or pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def safe_int(
    value,
    default=0,
):

    try:

        if value is None or pd.isna(value):
            return default

        return int(float(value))

    except Exception:

        return default


def signal_icon(signal):

    if signal in {
        "A+ SWING BUY",
        "BUY",
    }:
        return "🟢"

    if signal == "WATCH":
        return "🟡"

    if signal == "TOO EXTENDED":
        return "🔴"

    return "⚪"


def action_text(signal):

    return {
        "A+ SWING BUY": "BUY — top-tier setup confirmed",
        "BUY": "BUY — entry rules confirmed",
        "WATCH": "WAIT FOR BUY TRIGGER",
        "TOO EXTENDED": "WAIT FOR PULLBACK / RETEST",
        "AVOID": "PASS",
        "NO BUY": "WAIT",
    }.get(
        signal,
        "WAIT",
    )


# ============================================================
# WHY-NOT-BUY EXPLANATION
# ============================================================

def why_not_buy(row):

    signal = row.get(
        "signal",
        "",
    )

    if bool(
        row.get(
            "risk_flag",
            False,
        )
    ):

        return str(
            row.get(
                "risk_reason",
                "A hard downside catalyst-risk gate is active.",
            )
        )

    if signal in {
        "A+ SWING BUY",
        "BUY",
    }:

        return "All required BUY gates passed."

    if signal == "TOO EXTENDED":

        return (
            "The stock is too extended from its preferred entry. "
            "Wait for a pullback or retest."
        )

    failures = []

    swing_score = safe_float(
        row.get(
            "swing_score"
        )
    )

    entry_quality = safe_float(
        row.get(
            "entry_quality"
        )
    )

    reward_risk = safe_float(
        row.get(
            "reward_risk"
        )
    )

    market_score = safe_float(
        row.get(
            "market_score"
        )
    )

    intraday_score = safe_float(
        row.get(
            "intraday_score"
        )
    )

    try:

        price = float(
            row.get(
                "price",
                0,
            )
        )

        entry_low = float(
            row.get(
                "entry_low",
                0,
            )
        )

        entry_high = float(
            row.get(
                "entry_high",
                0,
            )
        )

        inside_entry_zone = (
            entry_low
            <= price
            <= entry_high
        )

    except Exception:

        inside_entry_zone = False

    if swing_score < 85:

        failures.append(
            f"Swing Score {swing_score:.1f} is below the 85 BUY threshold"
        )

    if entry_quality < 10:

        failures.append(
            f"Entry Quality {entry_quality:.1f}/15 is below the 10/15 "
            f"BUY requirement"
        )

    if reward_risk < 2:

        failures.append(
            f"Reward/Risk {reward_risk:.2f}:1 is below the required 2.00:1"
        )

    if market_score < 5:

        failures.append(
            f"Market Score {market_score:.1f}/10 is below the minimum 5/10"
        )

    if not inside_entry_zone:

        failures.append(
            "Current price is outside the preferred entry zone"
        )

    if not bool(
        row.get(
            "trend_health",
            True,
        )
    ):

        failures.append(
            "The 20-day and 50-day trend slopes are not both rising"
        )

    distribution_days = safe_int(
        row.get(
            "distribution_days"
        )
    )

    if distribution_days > 4:

        failures.append(
            f"{distribution_days} recent distribution days show excessive "
            f"selling pressure"
        )

    leadership = row.get(
        "leadership_percentile"
    )

    if (
        leadership is not None
        and not pd.isna(
            leadership
        )
        and float(
            leadership
        )
        < 70
    ):

        failures.append(
            f"Market leadership rank {float(leadership):.0f}% is below "
            f"the 70% BUY gate"
        )

    intraday_signal = str(
        row.get(
            "intraday_signal",
            "",
        )
    ).upper()

    if intraday_signal != "BUY":

        failures.append(
            "Intraday signal has not changed to BUY"
        )

    if intraday_score < 85:

        failures.append(
            f"Intraday Score {intraday_score:.1f} is below the "
            f"85 confirmation threshold"
        )

    if signal == "AVOID" and not failures:

        failures.append(
            "The setup does not meet enough high-probability swing requirements"
        )

    if signal == "WATCH" and not failures:

        failures.append(
            "The stock is close, but at least one production confirmation "
            "has not passed."
        )

    if failures:

        return " | ".join(
            failures
        )

    return "Waiting for additional confirmation."


# ============================================================
# TRADE CARD
# ============================================================

def render_trade_card(
    row,
    rank_num=None,
):

    signal = row.get(
        "signal",
        "N/A",
    )

    symbol = row.get(
        "symbol",
        "N/A",
    )

    prefix = (
        f"#{rank_num} "
        if rank_num is not None
        else ""
    )

    with st.container(
        border=True
    ):

        st.markdown(
            f"### {prefix}{signal_icon(signal)} {symbol} — {signal}"
        )

        st.write(
            f"**Swing Score:** "
            f"{score_display(row.get('swing_score'))}/100"
        )

        st.write(
            f"**Setup:** "
            f"{row.get('setup', '')}"
        )

        st.write(
            f"**Current Price:** "
            f"{money(row.get('price'))}"
        )

        st.write(
            f"**Entry Quality:** "
            f"{score_display(row.get('entry_quality'))}/15"
        )

        if bool(
            row.get(
                "risk_flag",
                False,
            )
        ):

            st.error(
                "**Risk Event:** "
                + str(
                    row.get(
                        "risk_reason",
                        "Hard downside catalyst-risk gate active",
                    )
                )
            )

        else:

            leadership = row.get(
                "leadership_percentile"
            )

            leadership_text = (
                "N/A"
                if leadership is None
                or pd.isna(
                    leadership
                )
                else f"{float(leadership):.0f}th percentile"
            )

            distribution = safe_int(
                row.get(
                    "distribution_days"
                )
            )

            st.caption(
                f"Risk gate: PASS | "
                f"Market leadership: {leadership_text} | "
                f"Distribution days: {distribution}"
            )

        st.markdown(
            "#### Entry Plan"
        )

        st.write(
            f"**Preferred Entry Zone:** "
            f"{money(row.get('entry_low'))} – "
            f"{money(row.get('entry_high'))}"
        )

        st.write(
            f"**Stop:** "
            f"{money(row.get('stop'))}"
        )

        st.write(
            f"**Target 1:** "
            f"{money(row.get('target1'))}"
        )

        st.write(
            f"**Target 2:** "
            f"{money(row.get('target2'))}"
        )

        st.write(
            f"**Reward / Risk:** "
            f"{rr_display(row.get('reward_risk'))}"
        )

        st.markdown(
            "#### Action"
        )

        if signal in {
            "A+ SWING BUY",
            "BUY",
        }:

            st.success(
                action_text(
                    signal
                )
            )

            st.markdown(
                "#### Why BUY?"
            )

            st.success(
                "Swing score, entry quality, reward/risk, entry zone, "
                "market conditions and intraday confirmation passed."
            )

        elif signal == "WATCH":

            st.warning(
                action_text(
                    signal
                )
            )

            st.markdown(
                "#### Why Not BUY Yet?"
            )

            st.info(
                why_not_buy(
                    row
                )
            )

        elif signal == "TOO EXTENDED":

            st.error(
                action_text(
                    signal
                )
            )

            st.markdown(
                "#### Why Not BUY Yet?"
            )

            st.info(
                why_not_buy(
                    row
                )
            )

        else:

            st.info(
                action_text(
                    signal
                )
            )

            st.markdown(
                "#### Why Not BUY Yet?"
            )

            st.info(
                why_not_buy(
                    row
                )
            )

        st.caption(
            f"Intraday confirmation: "
            f"{row.get('intraday_signal', 'N/A')} "
            f"({row.get('intraday_score', 'N/A')}/100)"
        )


# ============================================================
# VALIDATION DISPLAY
# ============================================================

def render_validation_summary(
    validation,
):

    st.subheader(
        "Walk-Forward Validation"
    )

    if not isinstance(
        validation,
        dict,
    ):

        st.info(
            "No validation results are available."
        )

        return

    sample_trades = safe_int(
        validation.get(
            "sample_trades",
            validation.get(
                "trades",
                0,
            ),
        )
    )

    oos_trades = safe_int(
        validation.get(
            "out_of_sample_trades",
            0,
        )
    )

    oos_expectancy = safe_float(
        validation.get(
            "out_of_sample_expectancy_r",
            validation.get(
                "aggregate_oos_expectancy_r",
                0,
            ),
        )
    )

    worst_fold_expectancy = safe_float(
        validation.get(
            "worst_fold_expectancy_r",
            validation.get(
                "worst_fold_expectancy",
                0,
            ),
        )
    )

    grade = validation.get(
        "confidence_grade",
        validation.get(
            "grade",
            "INSUFFICIENT",
        ),
    )

    passed = bool(
        validation.get(
            "validation_pass",
            False,
        )
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Validated trades",
        sample_trades,
    )

    c2.metric(
        "OOS trades",
        oos_trades,
    )

    c3.metric(
        "Aggregate OOS expectancy",
        f"{oos_expectancy:.3f} R",
    )

    c4.metric(
        "Worst fold expectancy",
        f"{worst_fold_expectancy:.3f} R",
    )

    st.write(
        f"**Confidence grade:** {grade}"
    )

    if passed:

        st.success(
            "Configured walk-forward validation checks passed."
        )

    else:

        st.warning(
            "The historical evidence is not yet strong enough to "
            "declare this configuration validated."
        )

    notes = validation.get(
        "notes",
        [],
    )

    if isinstance(
        notes,
        str,
    ):

        notes = [
            notes
        ]

    if notes:

        with st.expander(
            "Validation notes"
        ):

            for note in notes:

                st.write(
                    f"• {note}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Text alerts"
    )

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

            st.success(
                "SMS configured"
            )

        else:

            st.warning(
                "SMS secrets are not configured yet."
            )

    st.divider()

    st.subheader(
        "Tracked positions"
    )

    tracked_positions = st.text_input(
        "Symbols you currently hold",
        "",
        help=(
            "Example: NVDA,MU,OWL. SELL-risk alerts are evaluated only "
            "for symbols entered here."
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
        "The scanner removes unsuitable securities and illiquid stocks, "
        "then analyzes daily swing structure and live intraday momentum separately."
    )

    st.info(
        "A strong stock is not automatically a BUY. Production rules require "
        "daily structure, entry quality, reward/risk, trend health, market "
        "leadership, market regime and intraday confirmation to align."
    )

    c1, c2 = st.columns(
        2
    )

    feed = c1.selectbox(
        "Data feed",
        [
            "iex",
            "sip",
        ],
        index=0,
        help=(
            "Use SIP when your Alpaca plan supports consolidated market data."
        ),
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

        os.environ[
            "ALPACA_FEED"
        ] = feed

        status = st.status(
            "Scanning the U.S. market...",
            expanded=True,
        )

        progress = st.progress(
            0
        )

        try:

            # =================================================
            # 1. UNIVERSE
            # =================================================

            status.write(
                "1/5 Filtering eligible U.S. securities..."
            )

            elig = (
                eligible_us_equity_universe()
            )

            universe = (
                elig.symbol.tolist()
            )

            status.write(
                f"Eligible after type filtering: "
                f"{len(universe):,}"
            )

            now = datetime.now(
                ET
            )

            daily_start = (
                now
                - timedelta(
                    days=45
                )
            )

            # =================================================
            # 2. PREFILTER
            # =================================================

            status.write(
                "2/5 Applying price, liquidity, trend and momentum filters..."
            )

            def prog(
                done,
                total,
            ):

                progress.progress(
                    min(
                        int(
                            done
                            / max(
                                total,
                                1,
                            )
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

            daily[
                "timestamp"
            ] = pd.to_datetime(
                daily[
                    "timestamp"
                ],
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

            progress.progress(
                45
            )

            # =================================================
            # 3. LONG DAILY HISTORY
            # =================================================

            status.write(
                "3/5 Pulling longer daily history for swing-trade analysis..."
            )

            finalist_symbols = (
                finalists.symbol.tolist()
            )

            swing_start = (
                now
                - timedelta(
                    days=420
                )
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

            market_daily = get_bars(
                [
                    "SPY",
                    "QQQ",
                ],
                swing_start,
                now,
                "1Day",
                feed,
            )

            if not swing_daily.empty:

                swing_daily[
                    "timestamp"
                ] = pd.to_datetime(
                    swing_daily[
                        "timestamp"
                    ],
                    utc=True,
                )

            if not market_daily.empty:

                market_daily[
                    "timestamp"
                ] = pd.to_datetime(
                    market_daily[
                        "timestamp"
                    ],
                    utc=True,
                )

            spy_daily = (
                market_daily[
                    market_daily[
                        "symbol"
                    ]
                    == "SPY"
                ].copy()
                if not market_daily.empty
                else pd.DataFrame()
            )

            qqq_daily = (
                market_daily[
                    market_daily[
                        "symbol"
                    ]
                    == "QQQ"
                ].copy()
                if not market_daily.empty
                else pd.DataFrame()
            )

            leadership_map = (
                relative_strength_percentiles(
                    swing_daily
                )
            )

            progress.progress(
                60
            )

            # =================================================
            # 4. INTRADAY
            # =================================================

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
                market_start
                - timedelta(
                    days=1
                ),
                now,
                "1Min",
                feed,
                batch_size=100,
                pause_seconds=0.10,
            )

            spy = get_bars(
                [
                    "SPY"
                ],
                market_start
                - timedelta(
                    days=1
                ),
                now,
                "1Min",
                feed,
            )

            progress.progress(
                75
            )

            if (
                intra.empty
                or spy.empty
            ):

                status.update(
                    label="No intraday bars returned.",
                    state="error",
                )

                st.stop()

            intra[
                "timestamp"
            ] = pd.to_datetime(
                intra[
                    "timestamp"
                ],
                utc=True,
            )

            spy[
                "timestamp"
            ] = pd.to_datetime(
                spy[
                    "timestamp"
                ],
                utc=True,
            )

            latest = (
                intra.timestamp
                .dt
                .tz_convert(
                    ET
                )
                .dt
                .date
                .max()
            )

            today = intra[
                intra.timestamp
                .dt
                .tz_convert(
                    ET
                )
                .dt
                .date
                == latest
            ]

            spy_today = spy[
                spy.timestamp
                .dt
                .tz_convert(
                    ET
                )
                .dt
                .date
                == latest
            ]

            fmap = (
                finalists
                .set_index(
                    "symbol"
                )
                .to_dict(
                    "index"
                )
            )

            advmap = dict(
                zip(
                    finalists.symbol,
                    finalists.avg_dollar_volume,
                )
            )

            # =================================================
            # 5. SCORE
            # =================================================

            status.write(
                "5/5 Applying swing, risk and intraday confirmation rules..."
            )

            rows = []

            for (
                sym,
                d,
            ) in today.groupby(
                "symbol"
            ):

                if len(
                    d
                ) < 20:

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

                adv_dollars = float(
                    advmap.get(
                        sym,
                        0,
                    )
                    or 0
                )

                adv_shares = (
                    adv_dollars
                    / px
                    if adv_dollars
                    else None
                )

                p = prepare_intraday(
                    d,
                    spy_today,
                    adv_shares,
                )

                if p.empty:

                    continue

                r = p.iloc[
                    -1
                ]

                (
                    intraday_score,
                    intraday_signal,
                    reasons,
                ) = classify(
                    r,
                    advmap.get(
                        sym,
                        0,
                    ),
                )

                stock_swing_daily = (
                    swing_daily[
                        swing_daily[
                            "symbol"
                        ]
                        == sym
                    ].copy()
                    if not swing_daily.empty
                    else pd.DataFrame()
                )

                swing = None

                if not stock_swing_daily.empty:

                    swing = score_swing_daily(
                        stock_swing_daily,
                        spy_daily,
                        qqq_daily,
                        leadership_map.get(
                            sym
                        ),
                    )

                if not swing:

                    continue

                final_signal, confluence_reason = (
                    combine_daily_intraday_signal(
                        swing.get(
                            "signal",
                            "N/A",
                        ),
                        intraday_signal,
                        intraday_score,
                        risk_flag=bool(
                            swing.get(
                                "risk_flag",
                                False,
                            )
                        ),
                    )
                )

                intraday_confirmed = bool(
                    intraday_signal
                    == "BUY"
                    and intraday_score
                    >= 85
                )

                row = {
                    "symbol": sym,
                    "name": ref.get(
                        "name",
                        "",
                    ),
                    "signal": final_signal,
                    "swing_score": swing.get(
                        "swing_score",
                        0,
                    ),
                    "setup": swing.get(
                        "setup",
                        "",
                    ),
                    "entry_quality": swing.get(
                        "entry_quality",
                        0,
                    ),
                    "price": round(
                        float(
                            r.close
                        ),
                        2,
                    ),
                    "entry_low": swing.get(
                        "entry_low"
                    ),
                    "entry_high": swing.get(
                        "entry_high"
                    ),
                    "stop": swing.get(
                        "stop"
                    ),
                    "target1": swing.get(
                        "target1"
                    ),
                    "target2": swing.get(
                        "target2"
                    ),
                    "reward_risk": swing.get(
                        "reward_risk"
                    ),
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
                    "swing_rsi": swing.get(
                        "rsi14"
                    ),
                    "swing_rvol": swing.get(
                        "rvol"
                    ),
                    "risk_flag":
