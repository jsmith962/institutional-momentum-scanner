import os
import importlib
import inspect
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
# VERSION
# ============================================================

APP_VERSION = "v3.8"


# ============================================================
# OPTIONAL v3.8 PRODUCTION-vs-CHALLENGER UI
# ============================================================

V38_VALIDATION_AVAILABLE = False
V38_IMPORT_ERROR = None
render_v38_validation_lab = None
V38_MODULE_NAME = None


def _load_v38_ui():
    """
    Tries several compatible module/function names so a naming difference
    in the previously committed v3.8 files does not crash the whole app.
    """

    global V38_VALIDATION_AVAILABLE
    global V38_IMPORT_ERROR
    global render_v38_validation_lab
    global V38_MODULE_NAME

    module_candidates = [
        "portfolio_validation_ui",
        "production_vs_challenger_ui",
        "challenger_validation_ui",
        "production_challenger_ui",
        "v38_portfolio_validation_ui",
        "v38_validation_ui",
        "portfolio_challenger_ui",
    ]

    function_candidates = [
        "render_portfolio_validation_lab",
        "render_production_vs_challenger_lab",
        "render_challenger_validation_lab",
        "render_production_challenger_lab",
        "render_v38_portfolio_validation_lab",
        "render_v38_validation_lab",
    ]

    errors = []

    for module_name in module_candidates:

        try:
            module = importlib.import_module(
                module_name
            )

        except Exception as exc:
            errors.append(
                f"{module_name}: {exc}"
            )
            continue

        for function_name in function_candidates:

            function = getattr(
                module,
                function_name,
                None,
            )

            if callable(
                function
            ):

                render_v38_validation_lab = function
                V38_VALIDATION_AVAILABLE = True
                V38_MODULE_NAME = module_name
                V38_IMPORT_ERROR = None
                return

        errors.append(
            f"{module_name}: module loaded but no recognized "
            f"v3.8 render function was found."
        )

    V38_VALIDATION_AVAILABLE = False

    V38_IMPORT_ERROR = "\n".join(
        errors
    )


_load_v38_ui()


# ============================================================
# APP CONFIGURATION
# ============================================================

load_dotenv()

ET = ZoneInfo(
    "America/New_York"
)

st.set_page_config(
    page_title=f"Institutional Swing Scanner {APP_VERSION}",
    layout="wide",
)

st.title(
    f"Institutional Swing Scanner {APP_VERSION}"
)

st.caption(
    "Full U.S. market | catalyst-gap protection | daily + intraday "
    "confirmation | SMS alerts | production-equivalent backtesting | "
    "adaptive calibration | Production-vs-Challenger portfolio validation | "
    "no live orders"
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Live Swing Scanner",
        "$2,000 Swing Backtester",
        "Calibration & Validation",
        "v3.8 Production vs Challenger",
    ]
)


# ============================================================
# SESSION STORAGE
# ============================================================

SESSION_DEFAULTS = {
    "latest_backtest_result": None,
    "latest_backtest_settings": None,
    "latest_backtest_daily_bars": None,
    "latest_backtest_market_daily": None,
    "latest_backtest_minute_bars": None,
    "latest_backtest_spy_minutes": None,
    "latest_backtest_qqq_minutes": None,
}


for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[
            key
        ] = value


# ============================================================
# HELPERS
# ============================================================

def money(
    value
):
    try:

        if value is None or pd.isna(
            value
        ):
            return "N/A"

        return f"${float(value):,.2f}"

    except Exception:
        return "N/A"


def score_display(
    value
):
    try:

        if value is None or pd.isna(
            value
        ):
            return "N/A"

        return f"{float(value):.1f}"

    except Exception:
        return "N/A"


def rr_display(
    value
):
    try:

        if value is None or pd.isna(
            value
        ):
            return "N/A"

        return f"{float(value):.2f}:1"

    except Exception:
        return "N/A"


def safe_float(
    value,
    default=0.0,
):
    try:

        if value is None or pd.isna(
            value
        ):
            return default

        return float(
            value
        )

    except Exception:
        return default


def safe_int(
    value,
    default=0,
):
    try:

        if value is None or pd.isna(
            value
        ):
            return default

        return int(
            float(
                value
            )
        )

    except Exception:
        return default


def signal_icon(
    signal
):

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


def action_text(
    signal
):

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
# WHY NOT BUY
# ============================================================

def why_not_buy(
    row
):

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

        return (
            "All required BUY gates passed."
        )

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
            f"Entry Quality {entry_quality:.1f}/15 is below the "
            f"10/15 BUY requirement"
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
            f"{distribution_days} recent distribution days show "
            f"excessive selling pressure"
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
            f"Market leadership rank {float(leadership):.0f}% "
            f"is below the 70% BUY gate"
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
            "At least one required production confirmation has not passed."
        )

    return (
        " | ".join(
            failures
        )
        if failures
        else "Waiting for additional confirmation."
    )


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
# v3.8 UI COMPATIBILITY CALLER
# ============================================================

def call_v38_validation_ui(
    result,
    daily_bars,
    market_daily,
    settings,
):

    if not callable(
        render_v38_validation_lab
    ):
        return

    available = {
        "result": result,
        "backtest_result": result,
        "production_result": result,

        "daily_bars": daily_bars,
        "daily_history": daily_bars,
        "stock_daily_bars": daily_bars,

        "market_daily": market_daily,
        "market_daily_bars": market_daily,

        "settings": settings,
        "backtest_settings": settings,
    }

    try:

        signature = inspect.signature(
            render_v38_validation_lab
        )

        kwargs = {}

        for name, parameter in signature.parameters.items():

            if name in available:

                kwargs[
                    name
                ] = available[
                    name
                ]

        if kwargs:

            return render_v38_validation_lab(
                **kwargs
            )

    except Exception:
        pass

    attempts = [
        (
            result,
            daily_bars,
            market_daily,
            settings,
        ),
        (
            result,
            daily_bars,
            market_daily,
        ),
        (
            result,
            daily_bars,
        ),
        (
            result,
        ),
    ]

    last_error = None

    for args in attempts:

        try:

            return render_v38_validation_lab(
                *args
            )

        except TypeError as exc:

            last_error = exc
            continue

    if last_error:
        raise last_error


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
            "Example: NVDA,MU,OWL. SELL-risk alerts are evaluated "
            "only for symbols entered here."
        ),
    )


# ============================================================
# LIVE SCANNER
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
        "v3.8 does NOT automatically replace the production BUY rules. "
        "The live scanner continues to require its existing high-conviction "
        "daily and intraday confirmations while challenger rules are researched "
        "separately."
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
                "3/5 Pulling longer daily history for swing analysis..."
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

            if intra.empty or spy.empty:

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

            for sym, d in today.groupby(
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
                    safe_float(
                        ref.get(
                            "price",
                            1,
                        ),
                        1,
                    ),
                    0.01,
                )

                adv_dollars = safe_float(
                    advmap.get(
                        sym,
                        0,
                    )
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

                if stock_swing_daily.empty:
                    continue

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

                rows.append(
                    {
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
                            safe_float(
                                r.get(
                                    "stock_ret",
                                    0,
                                )
                            )
                            * 100,
                            2,
                        ),
                        "rel_volume": round(
                            safe_float(
                                r.get(
                                    "rel_volume",
                                    0,
                                )
                            ),
                            2,
                        ),
                        "vwap": round(
                            safe_float(
                                r.get(
                                    "vwap",
                                    r.close,
                                )
                            ),
                            2,
                        ),
                        "intraday_rsi": round(
                            safe_float(
                                r.get(
                                    "rsi",
                                    50,
                                ),
                                50,
                            ),
                            1,
                        ),
                        "swing_rsi": swing.get(
                            "rsi14"
                        ),
                        "swing_rvol": swing.get(
                            "rvol"
                        ),
                        "risk_flag": bool(
                            swing.get(
                                "risk_flag",
                                False,
                            )
                        ),
                        "risk_reason": swing.get(
                            "risk_reason",
                            "",
                        ),
                        "gap_down_pct": swing.get(
                            "gap_down_pct",
                            0,
                        ),
                        "event_days_ago": swing.get(
                            "event_days_ago"
                        ),
                        "trend_health": bool(
                            swing.get(
                                "trend_health",
                                False,
                            )
                        ),
                        "distribution_days": swing.get(
                            "distribution_days",
                            0,
                        ),
                        "leadership_percentile": swing.get(
                            "leadership_percentile"
                        ),
                        "market_score": swing.get(
                            "market_score"
                        ),
                        "intraday_confirmed": intraday_confirmed,
                        "vs_SPY_%": round(
                            safe_float(
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
                        "decision": confluence_reason,
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

                out[
                    "_rank"
                ] = (
                    out[
                        "signal"
                    ]
                    .map(
                        rank
                    )
                    .fillna(
                        6
                    )
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
                        columns=[
                            "_rank"
                        ]
                    )
                )

                buys = out[
                    out[
                        "signal"
                    ].isin(
                        [
                            "A+ SWING BUY",
                            "BUY",
                        ]
                    )
                ]

                watches = out[
                    out[
                        "signal"
                    ]
                    == "WATCH"
                ]

                extended = out[
                    out[
                        "signal"
                    ]
                    == "TOO EXTENDED"
                ]

                # =================================================
                # SMS
                # =================================================

                if (
                    sms_enabled
                    and sms_configured()
                ):

                    sent = []

                    if (
                        sms_buy_enabled
                        and not buys.empty
                    ):

                        for _, alert_row in buys.head(
                            5
                        ).iterrows():

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
                                    f"{alert_row['symbol']}: {sms_error}"
                                )

                    held = {
                        x.strip().upper()
                        for x
                        in tracked_positions.split(
                            ","
                        )
                        if x.strip()
                    }

                    if (
                        sms_sell_enabled
                        and held
                    ):

                        held_rows = out[
                            out[
                                "symbol"
                            ].isin(
                                held
                            )
                        ]

                        for _, alert_row in held_rows.iterrows():

                            sell_reasons = []

                            current_price = safe_float(
                                alert_row.get(
                                    "price"
                                )
                            )

                            current_vwap = safe_float(
                                alert_row.get(
                                    "vwap"
                                )
                            )

                            current_intraday_score = safe_int(
                                alert_row.get(
                                    "intraday_score"
                                )
                            )

                            current_intraday_signal = str(
                                alert_row.get(
                                    "intraday_signal",
                                    "",
                                )
                            ).upper()

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
                                in {
                                    "AVOID",
                                    "NO BUY",
                                }
                                and bool(
                                    sell_reasons
                                )
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
                                        f"{alert_row['symbol']}: {sms_error}"
                                    )

                    if sent:

                        st.info(
                            "Text alerts sent: "
                            + ", ".join(
                                sent
                            )
                        )

                # =================================================
                # RESULTS
                # =================================================

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
                            buys[
                                "symbol"
                            ].head(
                                8
                            )
                        )
                    )

                else:

                    st.error(
                        "🔴 NO CONFIRMED SWING BUY RIGHT NOW"
                    )

                    st.caption(
                        "Do not buy simply because a stock has a high "
                        "Swing Score. Wait for BUY or A+ SWING BUY."
                    )

                m1, m2, m3, m4 = st.columns(
                    4
                )

                m1.metric(
                    "Finalists",
                    len(
                        out
                    ),
                )

                m2.metric(
                    "Confirmed BUYs",
                    len(
                        buys
                    ),
                )

                m3.metric(
                    "WATCH",
                    len(
                        watches
                    ),
                )

                m4.metric(
                    "TOO EXTENDED",
                    len(
                        extended
                    ),
                )

                st.divider()

                st.header(
                    "Top 5 Swing Opportunities"
                )

                st.caption(
                    "WATCH means wait. It does not mean buy now."
                )

                for rank_num, (_, row) in enumerate(
                    out.head(
                        5
                    ).iterrows(),
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

                if buys.empty:

                    st.info(
                        "No confirmed BUY signals right now."
                    )

                else:

                    for _, row in buys.head(
                        10
                    ).iterrows():

                        render_trade_card(
                            row
                        )

                st.divider()

                st.header(
                    "🔴 Strong But Too Extended"
                )

                if extended.empty:

                    st.write(
                        "None."
                    )

                else:

                    for _, row in extended.head(
                        5
                    ).iterrows():

                        render_trade_card(
                            row
                        )

                st.divider()

                st.header(
                    "🟡 WATCH List"
                )

                if watches.empty:

                    st.write(
                        "None."
                    )

                else:

                    watch_columns = [
                        c
                        for c in [
                            "symbol",
                            "signal",
                            "swing_score",
                            "setup",
                            "price",
                            "entry_low",
                            "entry_high",
                        ]
                        if c in watches.columns
                    ]

                    st.dataframe(
                        watches[
                            watch_columns
                        ].head(
                            30
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                st.divider()

                st.header(
                    "Stock Detail"
                )

                selected_symbol = st.selectbox(
                    "Select a stock",
                    out[
                        "symbol"
                    ].tolist(),
                )

                selected = out[
                    out[
                        "symbol"
                    ]
                    == selected_symbol
                ].iloc[
                    0
                ]

                render_trade_card(
                    selected
                )

                st.write(
                    f"**Today's change:** "
                    f"{safe_float(selected.get('change_today_%')):.2f}%"
                )

                st.write(
                    f"**Relative volume:** "
                    f"{safe_float(selected.get('rel_volume')):.2f}x"
                )

                st.write(
                    f"**Relative strength vs SPY:** "
                    f"{safe_float(selected.get('vs_SPY_%')):.2f}%"
                )

                st.write(
                    f"**Decision reason:** "
                    f"{selected.get('decision', '')}"
                )

                with st.expander(
                    "Show full research table"
                ):

                    display_columns = [
                        c
                        for c in [
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
                        if c in out.columns
                    ]

                    st.dataframe(
                        out[
                            display_columns
                        ].head(
                            50
                        ),
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
                    file_name="v3_8_swing_scan_latest.csv",
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as exc:

            status.update(
                label="Scan stopped because of an error.",
                state="error",
            )

            st.error(
                str(
                    exc
                )
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
        "The production backtest reconstructs the historical daily and "
        "intraday decision chain. v3.8 additionally stores the historical "
        "datasets so Production and Challenger portfolios can be compared "
        "using the same sample."
    )

    st.warning(
        "Results cover only the symbols entered below. This is not a "
        "complete historical reconstruction of the entire U.S. stock market."
    )

    c1, c2, c3 = st.columns(
        3
    )

    start_date = c1.date_input(
        "Start",
        datetime.now(
            ET
        ).date()
        - timedelta(
            days=180
        ),
        key="swing_bt_start",
    )

    end_date = c2.date_input(
        "End",
        datetime.now(
            ET
        ).date()
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
            "Use at least 10 liquid stocks for a more meaningful "
            "relative-strength comparison group."
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
        "Use SIP when your Alpaca plan permits consolidated data. "
        "Otherwise use IEX."
    )

    if st.button(
        "RUN $2,000 BACKTEST",
        type="primary",
        width="stretch",
    ):

        syms = [
            x.strip().upper()
            for x
            in symbols.split(
                ","
            )
            if x.strip()
        ]

        if start_date >= end_date:

            st.error(
                "Choose a start date before the end date."
            )

            st.stop()

        if len(
            syms
        ) < 5:

            st.error(
                "Enter at least 5 symbols."
            )

            st.stop()

        if len(
            syms
        ) < 10:

            st.warning(
                "Ten or more symbols are recommended."
            )

        if (
            end_date
            - start_date
        ).days > 365:

            st.warning(
                "A range longer than one year can be slow. "
                "Six to twelve months is a good first test."
            )

        request_end = (
            end_date
            + timedelta(
                days=1
            )
        )

        warmup_start = (
            start_date
            - timedelta(
                days=450
            )
        )

        with st.spinner(
            "Downloading historical daily and minute data..."
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
                market_minutes[
                    "symbol"
                ]
                == "SPY"
            ].copy()

            qqq = market_minutes[
                market_minutes[
                    "symbol"
                ]
                == "QQQ"
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
                    market_daily[
                        "symbol"
                    ]
                )
            )
        ):

            st.error(
                "The complete daily and minute history was not returned. "
                "Try a shorter range or choose IEX."
            )

        else:

            complete_symbols = sorted(
                set(
                    bars[
                        "symbol"
                    ]
                )
                & set(
                    daily_history[
                        "symbol"
                    ]
                )
            )

            missing_symbols = sorted(
                set(
                    syms
                )
                - set(
                    complete_symbols
                )
            )

            if missing_symbols:

                st.warning(
                    "Excluded symbols with incomplete data: "
                    + ", ".join(
                        missing_symbols
                    )
                )

            if len(
                complete_symbols
            ) < 5:

                st.error(
                    "Fewer than 5 symbols returned complete data."
                )

                st.stop()

            bars = bars[
                bars[
                    "symbol"
                ].isin(
                    complete_symbols
                )
            ].copy()

            daily_history = daily_history[
                daily_history[
                    "symbol"
                ].isin(
                    complete_symbols
                )
            ].copy()

            with st.spinner(
                "Running production-equivalent portfolio simulation..."
            ):

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

            # =================================================
            # SAVE EVERYTHING NEEDED FOR v3.8
            # =================================================

            st.session_state.latest_backtest_result = res

            st.session_state.latest_backtest_daily_bars = (
                daily_history.copy()
            )

            st.session_state.latest_backtest_market_daily = (
                market_daily.copy()
            )

            st.session_state.latest_backtest_minute_bars = (
                bars.copy()
            )

            st.session_state.latest_backtest_spy_minutes = (
                spy.copy()
            )

            st.session_state.latest_backtest_qqq_minutes = (
                qqq.copy()
            )

            st.session_state.latest_backtest_settings = {
                "symbols": ",".join(
                    complete_symbols
                ),
                "start": str(
                    start_date
                ),
                "end": str(
                    end_date
                ),
                "risk_pct": risk_pct,
                "max_positions": max_positions,
                "max_holding_days": max_holding_days,
                "scan_time": scan_time,
                "slippage_bps": slippage_bps,
                "commission_bps": commission_bps,
                "feed": btfeed,
                "version": APP_VERSION,
            }

            stats = res.get(
                "stats",
                {},
            )

            st.success(
                "Production backtest completed and v3.8 research data saved."
            )

            cols = st.columns(
                4
            )

            cols[
                0
            ].metric(
                "Ending $",
                stats.get(
                    "ending_capital",
                    "—",
                ),
            )

            cols[
                1
            ].metric(
                "Return %",
                stats.get(
                    "total_return_pct",
                    "—",
                ),
            )

            cols[
                2
            ].metric(
                "Win rate %",
                stats.get(
                    "win_rate_pct",
                    "—",
                ),
            )

            cols[
                3
            ].metric(
                "Profit factor",
                stats.get(
                    "profit_factor",
                    "—",
                ),
            )

            cols2 = st.columns(
                4
            )

            cols2[
                0
            ].metric(
                "Max DD %",
                stats.get(
                    "max_drawdown_pct",
                    "—",
                ),
            )

            cols2[
                1
            ].metric(
                "Trades",
                stats.get(
                    "trades",
                    "—",
                ),
            )

            cols2[
                2
            ].metric(
                "Average expectancy",
                f"{safe_float(stats.get('expectancy_r')):.3f} R",
            )

            cols2[
                3
            ].metric(
                "Average trade $",
                stats.get(
                    "avg_trade_dollars",
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

            # =================================================
            # DIAGNOSTICS
            # =================================================

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

            st.divider()

            st.subheader(
                "BUY Confirmation Funnel"
            )

            if (
                isinstance(
                    funnel,
                    pd.DataFrame,
                )
                and not funnel.empty
            ):

                st.dataframe(
                    funnel,
                    width="stretch",
                    hide_index=True,
                )

            else:

                st.info(
                    "No funnel data available."
                )

            if (
                isinstance(
                    gate_failures,
                    pd.DataFrame,
                )
                and not gate_failures.empty
            ):

                primary = gate_failures.iloc[
                    0
                ]

                failed_count = safe_int(
                    primary.get(
                        "failed"
                    )
                )

                failure_pct = safe_float(
                    primary.get(
                        "failure_percent",
                        primary.get(
                            "failure_rate_pct",
                            0,
                        ),
                    )
                )

                st.info(
                    f"Most frequently failed gate: "
                    f"{primary.get('gate', 'Unknown')} failed for "
                    f"{failed_count:,} observations "
                    f"({failure_pct:.1f}%)."
                )

                st.markdown(
                    "#### Most common failed BUY gates"
                )

                st.dataframe(
                    gate_failures.head(
                        12
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.markdown(
                "#### Closest Near Misses"
            )

            if (
                isinstance(
                    near_misses,
                    pd.DataFrame,
                )
                and not near_misses.empty
            ):

                for _, near_miss in near_misses.head(
                    5
                ).iterrows():

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"**{near_miss.get('symbol', 'N/A')} — "
                            f"{near_miss.get('signal', 'N/A')}**"
                        )

                        st.write(
                            f"Session: {near_miss.get('session', 'N/A')} | "
                            f"Gates passed: "
                            f"{near_miss.get('gates_passed', 'N/A')}"
                        )

                        st.write(
                            f"Swing Score: "
                            f"{safe_float(near_miss.get('swing_score')):.1f} | "
                            f"Intraday Score: "
                            f"{safe_float(near_miss.get('intraday_score')):.1f} | "
                            f"Entry Quality: "
                            f"{safe_float(near_miss.get('entry_quality')):.1f}/15"
                        )

                        st.warning(
                            "Failed BUY gates: "
                            f"{near_miss.get('failed_buy_gates', 'N/A')}"
                        )

                with st.expander(
                    "Show full near-miss table"
                ):

                    st.dataframe(
                        near_misses,
                        width="stretch",
                        hide_index=True,
                    )

            else:

                st.info(
                    "No near-miss observations available."
                )

            # =================================================
            # EQUITY CURVE
            # =================================================

            equity = res.get(
                "equity",
                pd.DataFrame(),
            )

            if (
                isinstance(
                    equity,
                    pd.DataFrame,
                )
                and not equity.empty
            ):

                st.plotly_chart(
                    px.line(
                        equity,
                        x="date",
                        y="equity",
                        title="$2,000 Production Equity Curve",
                    ),
                    width="stretch",
                )

            # =================================================
            # TRADES
            # =================================================

            st.subheader(
                "Simulated Production Trades"
            )

            trades = res.get(
                "trades",
                pd.DataFrame(),
            )

            if isinstance(
                trades,
                pd.DataFrame,
            ):

                st.dataframe(
                    trades,
                    width="stretch",
                    hide_index=True,
                )

                if not trades.empty:

                    st.download_button(
                        "Download simulated trades",
                        data=trades.to_csv(
                            index=False
                        ).encode(
                            "utf-8"
                        ),
                        file_name="v3_8_production_trades.csv",
                        mime="text/csv",
                        width="stretch",
                    )

            # =================================================
            # SIGNAL AUDIT
            # =================================================

            with st.expander(
                "Show historical signal audit"
            ):

                signal_log = res.get(
                    "signal_log",
                    pd.DataFrame(),
                )

                st.caption(
                    "Records reconstructed historical scanner observations, "
                    "including observations that never became trades."
                )

                if isinstance(
                    signal_log,
                    pd.DataFrame,
                ):

                    st.dataframe(
                        signal_log,
                        width="stretch",
                        hide_index=True,
                    )

                    if not signal_log.empty:

                        st.download_button(
                            "Download signal audit",
                            data=signal_log.to_csv(
                                index=False
                            ).encode(
                                "utf-8"
                            ),
                            file_name="v3_8_signal_audit.csv",
                            mime="text/csv",
                            width="stretch",
                        )


# ============================================================
# CALIBRATION TAB
# ============================================================

with tab3:

    st.subheader(
        "Calibration & Walk-Forward Validation"
    )

    st.info(
        "Calibration remains research-only. A promising threshold should "
        "not replace the production strategy until it survives realistic "
        "portfolio comparison and out-of-sample validation."
    )

    result = (
        st.session_state.latest_backtest_result
    )

    settings = (
        st.session_state.latest_backtest_settings
    )

    if result is None:

        st.warning(
            "Run a backtest first in the '$2,000 Swing Backtester' tab."
        )

    else:

        if settings:

            with st.expander(
                "Backtest used for calibration"
            ):

                st.json(
                    settings
                )

        try:

            render_calibration_lab(
                result
            )

        except Exception as exc:

            st.error(
                "Calibration encountered an error. "
                "The production scanner is unaffected."
            )

            st.exception(
                exc
            )


# ============================================================
# v3.8 PRODUCTION-vs-CHALLENGER TAB
# ============================================================

with tab4:

    st.subheader(
        "v3.8 Production-vs-Challenger Portfolio Validation"
    )

    st.caption(
        "Research only. Challenger results do not automatically alter "
        "the live production BUY rules."
    )

    st.info(
        "The purpose of v3.8 is to stop judging a strategy only by "
        "individual forward returns. Production and Challenger must be "
        "tested as actual competing portfolios under the same capital, "
        "position limits, holding rules, slippage, fees and historical sample."
    )

    result = (
        st.session_state.latest_backtest_result
    )

    daily_bars = (
        st.session_state.latest_backtest_daily_bars
    )

    market_daily = (
        st.session_state.latest_backtest_market_daily
    )

    settings = (
        st.session_state.latest_backtest_settings
    )

    if not V38_VALIDATION_AVAILABLE:

        st.warning(
            "The main scanner is running normally, but the v3.8 "
            "Production-vs-Challenger UI module was not detected."
        )

        st.caption(
            "This protection prevents an optional research-file problem "
            "from crashing the Live Scanner or Backtester."
        )

        if V38_IMPORT_ERROR:

            with st.expander(
                "Show v3.8 module detection details"
            ):

                st.code(
                    V38_IMPORT_ERROR
                )

    elif result is None:

        st.warning(
            "Run the $2,000 backtest first. v3.8 needs the completed "
            "historical signal audit and portfolio settings."
        )

    elif (
        daily_bars is None
        or not isinstance(
            daily_bars,
            pd.DataFrame,
        )
        or daily_bars.empty
    ):

        st.warning(
            "This saved backtest does not contain the historical daily "
            "bars needed by v3.8."
        )

        st.info(
            "Return to the $2,000 Swing Backtester and run it one more time."
        )

    else:

        signal_log = result.get(
            "signal_log",
            pd.DataFrame(),
        )

        if (
            not isinstance(
                signal_log,
                pd.DataFrame,
            )
            or signal_log.empty
        ):

            st.warning(
                "The completed backtest does not contain a usable "
                "historical signal audit."
            )

        else:

            st.success(
                f"v3.8 dataset ready: "
                f"{len(signal_log):,} historical scanner observations."
            )

            c1, c2, c3 = st.columns(
                3
            )

            c1.metric(
                "Signal observations",
                f"{len(signal_log):,}",
            )

            c2.metric(
                "Historical daily bars",
                f"{len(daily_bars):,}",
            )

            c3.metric(
                "Symbols",
                (
                    signal_log[
                        "symbol"
                    ].nunique()
                    if "symbol"
                    in signal_log.columns
                    else 0
                ),
            )

            if settings:

                with st.expander(
                    "Backtest sample used for v3.8"
                ):

                    st.json(
                        settings
                    )

            if V38_MODULE_NAME:

                st.caption(
                    f"v3.8 UI loaded from: {V38_MODULE_NAME}.py"
                )

            try:

                call_v38_validation_ui(
                    result,
                    daily_bars,
                    market_daily,
                    settings,
                )

            except Exception as exc:

                st.error(
                    "The v3.8 module loaded, but the portfolio-validation "
                    "screen encountered an error. The production scanner "
                    "and backtester remain unchanged."
                )

                st.exception(
                    exc
                )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Research only. Scanner signals, backtests, calibration results and "
    "Production-vs-Challenger comparisons do not guarantee future returns. "
    "Do not promote a Challenger into production because it wins one historical "
    "sample. Require repeated performance across non-overlapping periods, "
    "adequate trade counts, realistic costs, acceptable drawdowns and paper "
    "trading before risking real capital."
)
