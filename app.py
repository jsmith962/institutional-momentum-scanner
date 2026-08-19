import inspect
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
# OPTIONAL v3.7 FORWARD RESEARCH
# ============================================================

try:
    from forward_research import (
        attach_forward_returns,
        run_forward_gate_research,
    )

    from forward_research_ui import (
        render_forward_research_lab,
    )

    V37_AVAILABLE = True
    V37_IMPORT_ERROR = None

except Exception as exc:
    attach_forward_returns = None
    run_forward_gate_research = None
    render_forward_research_lab = None

    V37_AVAILABLE = False
    V37_IMPORT_ERROR = str(exc)


# ============================================================
# OPTIONAL v3.8 PRODUCTION VS CHALLENGER
# ============================================================

try:
    from production_vs_challenger_ui import (
        render_production_vs_challenger_lab,
    )

    V38_AVAILABLE = True
    V38_IMPORT_ERROR = None

except Exception as exc:
    render_production_vs_challenger_lab = None

    V38_AVAILABLE = False
    V38_IMPORT_ERROR = str(exc)


# ============================================================
# APP CONFIGURATION
# ============================================================

load_dotenv()

ET = ZoneInfo(
    "America/New_York"
)

st.set_page_config(
    page_title="Institutional Swing Scanner v3.8",
    layout="wide",
)

st.title(
    "Institutional Swing Scanner v3.8"
)

st.caption(
    "Full U.S. market | catalyst-gap protection | daily + intraday "
    "confirmation | SMS alerts | production-equivalent backtesting | "
    "adaptive calibration | forward-return research | "
    "Production-vs-Challenger portfolio validation | no live orders"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Live Swing Scanner",
        "$2,000 Swing Backtester",
        "Calibration & Validation",
        "v3.7 Forward Research",
        "v3.8 Production vs Challenger",
    ]
)


# ============================================================
# SESSION STORAGE
# ============================================================

DEFAULT_SESSION_VALUES = {
    "latest_backtest_result": None,
    "latest_backtest_settings": None,
    "latest_backtest_daily_bars": None,
    "latest_backtest_market_daily": None,
    "v37_enriched_signal_log": None,
    "v37_forward_research_result": None,
}

for key, value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:
        st.session_state[
            key
        ] = value


# ============================================================
# BASIC HELPERS
# ============================================================

def money(value):

    try:

        if value is None or pd.isna(
            value
        ):
            return "N/A"

        return f"${float(value):,.2f}"

    except Exception:
        return "N/A"


def score_display(value):

    try:

        if value is None or pd.isna(
            value
        ):
            return "N/A"

        return f"{float(value):.1f}"

    except Exception:
        return "N/A"


def rr_display(value):

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
        "A+ SWING BUY": (
            "BUY — top-tier setup confirmed"
        ),
        "BUY": (
            "BUY — entry rules confirmed"
        ),
        "WATCH": (
            "WAIT FOR BUY TRIGGER"
        ),
        "TOO EXTENDED": (
            "WAIT FOR PULLBACK / RETEST"
        ),
        "AVOID": (
            "PASS"
        ),
        "NO BUY": (
            "WAIT"
        ),
    }.get(
        signal,
        "WAIT",
    )


# ============================================================
# WHY NOT BUY
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
                (
                    "A hard downside catalyst-risk "
                    "gate is active."
                ),
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
            "The stock is too extended from its "
            "preferred entry. Wait for a pullback "
            "or retest."
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
            f"Swing Score {swing_score:.1f} "
            f"is below 85"
        )

    if entry_quality < 10:

        failures.append(
            f"Entry Quality {entry_quality:.1f}/15 "
            f"is below 10/15"
        )

    if reward_risk < 2:

        failures.append(
            f"Reward/Risk {reward_risk:.2f}:1 "
            f"is below 2.00:1"
        )

    if market_score < 5:

        failures.append(
            f"Market Score {market_score:.1f}/10 "
            f"is below 5/10"
        )

    if not inside_entry_zone:

        failures.append(
            "Price is outside the preferred entry zone"
        )

    if not bool(
        row.get(
            "trend_health",
            True,
        )
    ):

        failures.append(
            "20-day and 50-day trend slopes "
            "are not both rising"
        )

    distribution_days = safe_int(
        row.get(
            "distribution_days"
        )
    )

    if distribution_days > 4:

        failures.append(
            f"{distribution_days} distribution days "
            f"exceed the production limit"
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
            f"Leadership percentile "
            f"{float(leadership):.0f}% "
            f"is below 70%"
        )

    intraday_signal = str(
        row.get(
            "intraday_signal",
            "",
        )
    ).upper()

    if intraday_signal != "BUY":

        failures.append(
            "Intraday signal is not BUY"
        )

    if intraday_score < 85:

        failures.append(
            f"Intraday Score "
            f"{intraday_score:.1f} "
            f"is below 85"
        )

    if failures:

        return " | ".join(
            failures
        )

    return (
        "Waiting for additional confirmation."
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
            f"### {prefix}"
            f"{signal_icon(signal)} "
            f"{symbol} — {signal}"
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
                        (
                            "Hard downside catalyst-risk "
                            "gate active"
                        ),
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
                else (
                    f"{float(leadership):.0f}th "
                    f"percentile"
                )
            )

            distribution = safe_int(
                row.get(
                    "distribution_days"
                )
            )

            st.caption(
                f"Risk gate: PASS | "
                f"Leadership: {leadership_text} | "
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

            st.success(
                "All required production BUY "
                "confirmations passed."
            )

        elif signal == "WATCH":

            st.warning(
                action_text(
                    signal
                )
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
# AUTOMATIC v3.7 DATASET BUILDER
# ============================================================

def build_v37_dataset_if_possible():

    if not V37_AVAILABLE:
        return None

    existing = st.session_state.get(
        "v37_enriched_signal_log"
    )

    if (
        isinstance(
            existing,
            pd.DataFrame,
        )
        and not existing.empty
    ):

        return existing

    result = st.session_state.get(
        "latest_backtest_result"
    )

    daily_bars = st.session_state.get(
        "latest_backtest_daily_bars"
    )

    if not isinstance(
        result,
        dict,
    ):
        return None

    if (
        not isinstance(
            daily_bars,
            pd.DataFrame,
        )
        or daily_bars.empty
    ):
        return None

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
        return None

    try:

        enriched = attach_forward_returns(
            signal_log,
            daily_bars,
        )

        if (
            isinstance(
                enriched,
                pd.DataFrame,
            )
            and not enriched.empty
        ):

            st.session_state[
                "v37_enriched_signal_log"
            ] = enriched

            research = (
                run_forward_gate_research(
                    enriched
                )
            )

            st.session_state[
                "v37_forward_research_result"
            ] = research

            return enriched

    except Exception:
        return None

    return None


# ============================================================
# FLEXIBLE v3.8 UI CALLER
# ============================================================

def call_v38_ui(
    result,
    enriched,
    daily_bars,
    settings,
):

    if not V38_AVAILABLE:

        st.error(
            "production_vs_challenger_ui.py "
            "could not be loaded."
        )

        if V38_IMPORT_ERROR:

            st.code(
                V38_IMPORT_ERROR
            )

        return

    try:

        signature = inspect.signature(
            render_production_vs_challenger_lab
        )

        kwargs = {}

        for parameter_name in signature.parameters:

            name = parameter_name.lower()

            if name in {
                "backtest_result",
                "result",
                "res",
            }:

                kwargs[
                    parameter_name
                ] = result

            elif name in {
                "enriched_signal_log",
                "signal_log",
                "historical_signal_log",
                "forward_signal_log",
                "historical_signal_audit",
            }:

                kwargs[
                    parameter_name
                ] = enriched

            elif name in {
                "daily_bars",
                "daily_history",
                "historical_daily_bars",
            }:

                kwargs[
                    parameter_name
                ] = daily_bars

            elif name in {
                "settings",
                "backtest_settings",
                "config",
                "configuration",
            }:

                kwargs[
                    parameter_name
                ] = settings

        required_unknown = []

        for name, parameter in (
            signature.parameters.items()
        ):

            if (
                parameter.default
                is inspect.Parameter.empty
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
                and name not in kwargs
            ):

                required_unknown.append(
                    name
                )

        if not required_unknown:

            render_production_vs_challenger_lab(
                **kwargs
            )

            return

    except Exception as exc:

        first_error = exc

    else:
        first_error = None

    attempts = [
        (
            result,
            enriched,
            daily_bars,
            settings,
        ),
        (
            result,
            enriched,
            daily_bars,
        ),
        (
            result,
            enriched,
        ),
        (
            enriched,
            daily_bars,
            settings,
        ),
        (
            enriched,
            settings,
        ),
        (
            enriched,
        ),
        (
            result,
        ),
    ]

    last_error = first_error

    for args in attempts:

        try:

            render_production_vs_challenger_lab(
                *args
            )

            return

        except TypeError as exc:
            last_error = exc

        except Exception as exc:

            st.error(
                "v3.8 loaded successfully, but "
                "the validation engine encountered "
                "an internal error."
            )

            st.exception(
                exc
            )

            return

    st.error(
        "The v3.8 UI function signature does "
        "not match the current app.py."
    )

    if last_error:
        st.exception(
            last_error
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

    if sms_enabled:

        if sms_configured():

            st.success(
                "SMS configured"
            )

        else:

            st.warning(
                "SMS secrets are not configured."
            )

    st.divider()

    tracked_positions = st.text_input(
        "Symbols you currently hold",
        "",
        help="Example: NVDA,MU,OWL",
    )


# ============================================================
# TAB 1 — LIVE SCANNER
# ============================================================

with tab1:

    st.subheader(
        "Full U.S. Market Swing-Trade Scanner"
    )

    st.info(
        "Production BUY rules remain unchanged. "
        "v3.7 and v3.8 are research layers only."
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

            status.write(
                "1/5 Filtering eligible U.S. securities..."
            )

            elig = (
                eligible_us_equity_universe()
            )

            universe = (
                elig.symbol.tolist()
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

            status.write(
                "2/5 Applying liquidity and momentum filters..."
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
                    label=(
                        "No daily market data returned."
                    ),
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
                    label=(
                        "No qualifying finalists found."
                    ),
                    state="complete",
                )

                st.stop()

            progress.progress(
                45
            )

            status.write(
                "3/5 Pulling long daily history..."
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
            )

            qqq_daily = (
                market_daily[
                    market_daily[
                        "symbol"
                    ]
                    == "QQQ"
                ].copy()
            )

            leadership_map = (
                relative_strength_percentiles(
                    swing_daily
                )
            )

            progress.progress(
                60
            )

            status.write(
                "4/5 Pulling intraday confirmation..."
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

            if (
                intra.empty
                or spy.empty
            ):

                status.update(
                    label=(
                        "No intraday data returned."
                    ),
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

            progress.progress(
                75
            )

            status.write(
                "5/5 Applying production BUY gates..."
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

                prepared = prepare_intraday(
                    d,
                    spy_today,
                    adv_shares,
                )

                if prepared.empty:
                    continue

                intraday_row = prepared.iloc[
                    -1
                ]

                (
                    intraday_score,
                    intraday_signal,
                    reasons,
                ) = classify(
                    intraday_row,
                    advmap.get(
                        sym,
                        0,
                    ),
                )

                stock_daily = swing_daily[
                    swing_daily[
                        "symbol"
                    ]
                    == sym
                ].copy()

                if stock_daily.empty:
                    continue

                swing = score_swing_daily(
                    stock_daily,
                    spy_daily,
                    qqq_daily,
                    leadership_map.get(
                        sym
                    ),
                )

                if not swing:
                    continue

                final_signal, decision = (
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
                                intraday_row.close
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
                        "intraday_signal": intraday_signal,
                        "intraday_score": intraday_score,
                        "intraday_confirmed": bool(
                            intraday_signal
                            == "BUY"
                            and intraday_score
                            >= 85
                        ),
                        "vwap": round(
                            safe_float(
                                intraday_row.get(
                                    "vwap"
                                )
                            ),
                            2,
                        ),
                        "rel_volume": round(
                            safe_float(
                                intraday_row.get(
                                    "rel_volume"
                                )
                            ),
                            2,
                        ),
                        "vs_SPY_%": round(
                            safe_float(
                                intraday_row.get(
                                    "rs"
                                )
                            )
                            * 100,
                            2,
                        ),
                        "decision": decision,
                        "intraday_reasons": (
                            "; ".join(
                                reasons
                            )
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
                label=(
                    "Swing scan complete."
                ),
                state="complete",
                expanded=False,
            )

            if out.empty:

                st.warning(
                    "No finalists had enough data to score."
                )

            else:

                signal_rank = {
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
                        signal_rank
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

                if (
                    sms_enabled
                    and sms_configured()
                ):

                    if (
                        sms_buy_enabled
                        and not buys.empty
                    ):

                        for _, alert_row in (
                            buys.head(
                                5
                            ).iterrows()
                        ):

                            try:

                                send_sms(
                                    build_buy_message(
                                        alert_row
                                    )
                                )

                            except Exception as exc:

                                st.warning(
                                    f"BUY SMS failed for "
                                    f"{alert_row['symbol']}: "
                                    f"{exc}"
                                )

                    held = {
                        symbol.strip().upper()
                        for symbol
                        in tracked_positions.split(
                            ","
                        )
                        if symbol.strip()
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

                        for _, alert_row in (
                            held_rows.iterrows()
                        ):

                            reasons = []

                            if safe_float(
                                alert_row.get(
                                    "price"
                                )
                            ) < safe_float(
                                alert_row.get(
                                    "vwap"
                                )
                            ):

                                reasons.append(
                                    "price below VWAP"
                                )

                            if safe_int(
                                alert_row.get(
                                    "intraday_score"
                                )
                            ) < 60:

                                reasons.append(
                                    "intraday momentum weakened"
                                )

                            if reasons:

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
                                                reasons
                                            ),
                                        )
                                    )

                                except Exception:
                                    pass

                st.divider()

                st.subheader(
                    "Current Market Decision"
                )

                if buys.empty:

                    st.error(
                        "🔴 NO CONFIRMED SWING BUY RIGHT NOW"
                    )

                else:

                    st.success(
                        f"🟢 {len(buys)} CONFIRMED "
                        f"SWING BUY SIGNAL"
                        f"{'S' if len(buys) != 1 else ''}"
                    )

                    st.write(
                        ", ".join(
                            buys[
                                "symbol"
                            ].head(
                                10
                            )
                        )
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
                    "BUYs",
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
                    "Extended",
                    len(
                        extended
                    ),
                )

                st.divider()

                st.header(
                    "Top 5 Swing Opportunities"
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
                        "No confirmed BUY signals."
                    )

                else:

                    for _, row in (
                        buys.head(
                            10
                        ).iterrows()
                    ):

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

                    st.dataframe(
                        watches.head(
                            30
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
                    file_name=(
                        "v3_8_swing_scan_latest.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                )

        except Exception as exc:

            status.update(
                label=(
                    "Scan stopped because of an error."
                ),
                state="error",
            )

            st.exception(
                exc
            )


# ============================================================
# TAB 2 — BACKTEST
# ============================================================

with tab2:

    st.subheader(
        "$2,000 Production-Equivalent Swing Backtester"
    )

    st.info(
        "Run this first. The completed backtest is automatically "
        "saved for Calibration, v3.7 Forward Research and v3.8 "
        "Production-vs-Challenger validation."
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
        key="bt_start",
    )

    end_date = c2.date_input(
        "End",
        datetime.now(
            ET
        ).date()
        - timedelta(
            days=1
        ),
        key="bt_end",
    )

    risk_pct = (
        c3.slider(
            "Risk per trade",
            0.25,
            2.0,
            0.50,
            0.25,
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
    )

    max_holding_days = c5.slider(
        "Maximum holding sessions",
        5,
        30,
        20,
        5,
    )

    scan_time = c6.selectbox(
        "Historical scan time",
        [
            "11:30",
            "14:00",
            "15:30",
        ],
        index=0,
    )

    c7, c8 = st.columns(
        2
    )

    slippage_bps = c7.slider(
        "Slippage bps",
        0,
        25,
        5,
    )

    commission_bps = c8.slider(
        "Commission bps",
        0,
        10,
        0,
    )

    symbols_text = st.text_input(
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
    )

    if st.button(
        "RUN $2,000 BACKTEST",
        type="primary",
        width="stretch",
    ):

        symbols = [
            x.strip().upper()
            for x
            in symbols_text.split(
                ","
            )
            if x.strip()
        ]

        if start_date >= end_date:

            st.error(
                "Start date must be before end date."
            )

            st.stop()

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
                symbols,
                start_date,
                request_end,
                "1Min",
                btfeed,
                batch_size=20,
                pause_seconds=0.10,
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
                symbols,
                warmup_start,
                request_end,
                "1Day",
                btfeed,
                batch_size=100,
                pause_seconds=0.10,
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

        if (
            bars.empty
            or daily_history.empty
            or market_minutes.empty
            or market_daily.empty
        ):

            st.error(
                "Complete historical data was not returned."
            )

            st.stop()

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

        if len(
            complete_symbols
        ) < 5:

            st.error(
                "Fewer than five symbols returned complete data."
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
            "Running production-equivalent backtest..."
        ):

            result = swing_backtest(
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

        st.session_state[
            "latest_backtest_result"
        ] = result

        st.session_state[
            "latest_backtest_daily_bars"
        ] = daily_history.copy()

        st.session_state[
            "latest_backtest_market_daily"
        ] = market_daily.copy()

        st.session_state[
            "latest_backtest_settings"
        ] = {
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
            "version": "v3.8",
        }

        # Reset old research whenever a new backtest is run.
        st.session_state[
            "v37_enriched_signal_log"
        ] = None

        st.session_state[
            "v37_forward_research_result"
        ] = None

        # ----------------------------------------------------
        # CRITICAL v3.8 FIX:
        # Automatically construct the enriched forward dataset.
        # ----------------------------------------------------

        enriched = (
            build_v37_dataset_if_possible()
        )

        stats = result.get(
            "stats",
            {},
        )

        st.success(
            "Production backtest completed."
        )

        if (
            isinstance(
                enriched,
                pd.DataFrame,
            )
            and not enriched.empty
        ):

            st.success(
                f"v3.8 research dataset automatically prepared: "
                f"{len(enriched):,} historical scanner observations."
            )

        else:

            st.warning(
                "The production backtest completed, but the "
                "v3.7 forward-return dataset could not yet be "
                "built automatically. Open the v3.7 tab."
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
            "Expectancy",
            f"{safe_float(stats.get('expectancy_r')):.3f} R",
        )

        cols2[
            3
        ].metric(
            "Avg Trade $",
            stats.get(
                "avg_trade_dollars",
                "—",
            ),
        )

        diagnostics = result.get(
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

        if isinstance(
            funnel,
            pd.DataFrame,
        ) and not funnel.empty:

            st.dataframe(
                funnel,
                width="stretch",
                hide_index=True,
            )

        if isinstance(
            gate_failures,
            pd.DataFrame,
        ) and not gate_failures.empty:

            st.markdown(
                "#### Most Common Failed BUY Gates"
            )

            st.dataframe(
                gate_failures.head(
                    12
                ),
                width="stretch",
                hide_index=True,
            )

        if isinstance(
            near_misses,
            pd.DataFrame,
        ) and not near_misses.empty:

            st.markdown(
                "#### Closest Near Misses"
            )

            st.dataframe(
                near_misses.head(
                    20
                ),
                width="stretch",
                hide_index=True,
            )

        equity = result.get(
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
                    title="$2,000 Equity Curve",
                ),
                width="stretch",
            )

        trades = result.get(
            "trades",
            pd.DataFrame(),
        )

        st.subheader(
            "Simulated Production Trades"
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

        with st.expander(
            "Show Historical Signal Audit"
        ):

            signal_log = result.get(
                "signal_log",
                pd.DataFrame(),
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


# ============================================================
# TAB 3 — CALIBRATION
# ============================================================

with tab3:

    st.subheader(
        "Calibration & Walk-Forward Validation"
    )

    result = st.session_state.get(
        "latest_backtest_result"
    )

    settings = st.session_state.get(
        "latest_backtest_settings"
    )

    if result is None:

        st.warning(
            "Run the $2,000 backtest first."
        )

    else:

        if settings:

            with st.expander(
                "Backtest used for calibration"
            ):

                st.json(
                    settings
                )

        render_calibration_lab(
            result
        )


# ============================================================
# TAB 4 — v3.7 FORWARD RESEARCH
# ============================================================

with tab4:

    st.subheader(
        "v3.7 Gate Bottleneck + Forward Return Research"
    )

    if not V37_AVAILABLE:

        st.error(
            "The v3.7 research files could not be loaded."
        )

        if V37_IMPORT_ERROR:

            st.code(
                V37_IMPORT_ERROR
            )

    else:

        result = st.session_state.get(
            "latest_backtest_result"
        )

        daily_bars = st.session_state.get(
            "latest_backtest_daily_bars"
        )

        if result is None:

            st.warning(
                "Run the $2,000 backtest first."
            )

        elif (
            not isinstance(
                daily_bars,
                pd.DataFrame,
            )
            or daily_bars.empty
        ):

            st.warning(
                "Historical daily bars are missing. "
                "Run the backtest again."
            )

        else:

            # Auto-build before rendering.
            enriched = (
                build_v37_dataset_if_possible()
            )

            if (
                isinstance(
                    enriched,
                    pd.DataFrame,
                )
                and not enriched.empty
            ):

                st.success(
                    f"Forward-return dataset ready: "
                    f"{len(enriched):,} observations."
                )

            try:

                render_forward_research_lab(
                    result,
                    daily_bars,
                )

            except Exception as exc:

                st.error(
                    "v3.7 Forward Research encountered an error."
                )

                st.exception(
                    exc
                )


# ============================================================
# TAB 5 — v3.8 PRODUCTION VS CHALLENGER
# ============================================================

with tab5:

    st.subheader(
        "v3.8 Production-vs-Challenger Portfolio Validation"
    )

    st.caption(
        "Research only. Challenger results do not automatically "
        "alter the production BUY rules."
    )

    st.info(
        "v3.8 compares the current production control with bounded "
        "challenger rules using the same historical sample and "
        "portfolio constraints."
    )

    result = st.session_state.get(
        "latest_backtest_result"
    )

    daily_bars = st.session_state.get(
        "latest_backtest_daily_bars"
    )

    settings = st.session_state.get(
        "latest_backtest_settings"
    )

    if result is None:

        st.warning(
            "Run the $2,000 backtest first."
        )

    elif (
        not isinstance(
            daily_bars,
            pd.DataFrame,
        )
        or daily_bars.empty
    ):

        st.warning(
            "Historical daily bars are missing. "
            "Run the backtest again."
        )

    elif not V38_AVAILABLE:

        st.error(
            "production_vs_challenger_ui.py "
            "could not be imported."
        )

        if V38_IMPORT_ERROR:

            with st.expander(
                "Show v3.8 import error"
            ):

                st.code(
                    V38_IMPORT_ERROR
                )

    else:

        # ----------------------------------------------------
        # THIS IS THE FIX FOR THE SCREEN YOU SHOWED ME.
        # v3.8 no longer depends on you manually finding and
        # pressing a hidden v3.7 button first.
        # ----------------------------------------------------

        enriched = (
            build_v37_dataset_if_possible()
        )

        if (
            not isinstance(
                enriched,
                pd.DataFrame,
            )
            or enriched.empty
        ):

            st.warning(
                "The forward-return audit has not been created yet."
            )

            if st.button(
                "PREPARE v3.8 RESEARCH DATASET",
                type="primary",
                width="stretch",
                key="prepare_v38_dataset",
            ):

                with st.spinner(
                    "Building forward-return audit for v3.8..."
                ):

                    enriched = (
                        build_v37_dataset_if_possible()
                    )

                if (
                    isinstance(
                        enriched,
                        pd.DataFrame,
                    )
                    and not enriched.empty
                ):

                    st.success(
                        f"v3.8 dataset prepared: "
                        f"{len(enriched):,} observations."
                    )

                    st.rerun()

                else:

                    st.error(
                        "The enriched dataset could not be built. "
                        "Open the v3.7 tab to view the underlying error."
                    )

        else:

            st.success(
                f"v3.8 dataset ready: "
                f"{len(enriched):,} historical scanner observations."
            )

            c1, c2, c3 = st.columns(
                3
            )

            c1.metric(
                "Signal observations",
                f"{len(enriched):,}",
            )

            c2.metric(
                "Historical daily bars",
                f"{len(daily_bars):,}",
            )

            symbol_count = (
                enriched[
                    "symbol"
                ].nunique()
                if "symbol"
                in enriched.columns
                else 0
            )

            c3.metric(
                "Symbols",
                f"{symbol_count:,}",
            )

            if settings:

                with st.expander(
                    "Backtest sample used for v3.8"
                ):

                    st.json(
                        settings
                    )

            st.divider()

            call_v38_ui(
                result=result,
                enriched=enriched,
                daily_bars=daily_bars,
                settings=settings,
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Research only. Scanner signals, backtests, calibration, "
    "forward-return studies and Production-vs-Challenger comparisons "
    "do not guarantee future returns. Do not promote a Challenger into "
    "production because it wins one historical sample. Require repeated "
    "out-of-sample performance, adequate trade counts, realistic costs, "
    "acceptable drawdowns and paper trading before risking real capital."
)
