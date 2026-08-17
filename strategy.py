import numpy as np
import pandas as pd


# ============================================================
# INSTITUTIONAL SWING STRATEGY v3.4.4
# ============================================================
#
# Goals:
# 1. Keep hard risk gates independent of score.
# 2. Preserve production BUY threshold of 85.
# 3. Make 85 realistically reachable by exceptional setups.
# 4. Avoid rewarding weak setups simply because data is missing.
# 5. Keep compatibility with app.py, backtest.py and calibration.
# ============================================================


# ============================================================
# PRODUCTION THRESHOLDS
# ============================================================

NEGATIVE_GAP_THRESHOLD = -0.05
NEGATIVE_CLOSE_SHOCK_THRESHOLD = -0.07

EVENT_LOOKBACK_SESSIONS = 5
EVENT_MIN_COOLDOWN_SESSIONS = 3

MAX_DISTRIBUTION_DAYS_BUY = 4

MIN_RS_PERCENTILE_BUY = 0.70
MIN_RS_PERCENTILE_A_PLUS = 0.85

MIN_SWING_SCORE_BUY = 85
MIN_ENTRY_QUALITY_BUY = 10
MIN_REWARD_RISK_BUY = 2.0
MIN_MARKET_SCORE_BUY = 5.0

MIN_INTRADAY_CONFIRMATION_SCORE = 85


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def ema(series, periods):
    return series.ewm(
        span=periods,
        adjust=False,
    ).mean()


def sma(series, periods):
    return series.rolling(
        periods
    ).mean()


def rsi_series(series, periods=14):
    delta = series.diff()

    gain = (
        delta
        .clip(lower=0)
        .ewm(
            alpha=1 / periods,
            adjust=False,
        )
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .ewm(
            alpha=1 / periods,
            adjust=False,
        )
        .mean()
    )

    rs = gain / loss.replace(
        0,
        np.nan,
    )

    return (
        100
        - (
            100
            / (
                1 + rs
            )
        )
    )


def atr_series(df, periods=14):
    previous_close = (
        df["close"]
        .shift(1)
    )

    true_range = pd.concat(
        [
            (
                df["high"]
                - df["low"]
            ),
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return (
        true_range
        .ewm(
            alpha=1 / periods,
            adjust=False,
        )
        .mean()
    )


# ============================================================
# CATALYST / DOWNSIDE EVENT RISK
# ============================================================

def downside_event_risk(
    df,
    e20=None,
):
    """
    Detect a recent abnormal downside gap or selloff.

    A recent severe downside event remains blocked until:

    1. At least EVENT_MIN_COOLDOWN_SESSIONS have passed.
    2. Price closes above the event-day high.
    3. Price closes above the current 20EMA.
    """

    default = {
        "risk_flag": False,
        "risk_reason": (
            "No abnormal downside catalyst gap detected"
        ),
        "gap_down_pct": 0.0,
        "event_day_change_pct": 0.0,
        "event_days_ago": None,
        "event_repaired": True,
    }

    if (
        df is None
        or len(df) < 2
    ):
        return default

    d = (
        df.copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    open_px = pd.to_numeric(
        d["open"],
        errors="coerce",
    )

    high_px = pd.to_numeric(
        d["high"],
        errors="coerce",
    )

    close_px = pd.to_numeric(
        d["close"],
        errors="coerce",
    )

    prior_close = (
        close_px
        .shift(1)
    )

    open_gap = (
        open_px
        / prior_close
        - 1
    )

    close_change = (
        close_px
        / prior_close
        - 1
    )

    start = max(
        1,
        len(d)
        - EVENT_LOOKBACK_SESSIONS,
    )

    event_mask = (
        (
            open_gap
            <= NEGATIVE_GAP_THRESHOLD
        )
        |
        (
            close_change
            <= NEGATIVE_CLOSE_SHOCK_THRESHOLD
        )
    )

    recent_mask = (
        event_mask
        .iloc[start:]
        .fillna(False)
        .to_numpy()
    )

    event_positions = (
        np.flatnonzero(
            recent_mask
        )
    )

    if len(
        event_positions
    ) == 0:
        return default

    event_pos = (
        start
        + int(
            event_positions[-1]
        )
    )

    days_ago = (
        len(d)
        - 1
        - event_pos
    )

    gap_pct = safe_float(
        open_gap.iloc[
            event_pos
        ]
    )

    event_change = safe_float(
        close_change.iloc[
            event_pos
        ]
    )

    event_high = safe_float(
        high_px.iloc[
            event_pos
        ]
    )

    current_close = safe_float(
        close_px.iloc[-1]
    )

    if e20 is None:
        e20 = safe_float(
            ema(
                close_px,
                20,
            ).iloc[-1],
            current_close,
        )

    repaired = bool(
        days_ago
        >= EVENT_MIN_COOLDOWN_SESSIONS
        and current_close
        > event_high
        and current_close
        > safe_float(
            e20,
            current_close,
        )
    )

    worst_move = min(
        gap_pct,
        event_change,
    )

    if repaired:
        reason = (
            "Prior downside event completed "
            "its cooldown and repair test"
        )
    else:
        reason = (
            f"Abnormal downside catalyst move "
            f"{worst_move * 100:.1f}% "
            f"{days_ago} session"
            f"{'s' if days_ago != 1 else ''} ago; "
            f"wait at least "
            f"{EVENT_MIN_COOLDOWN_SESSIONS} sessions "
            f"and require a close above the "
            f"event-day high and 20EMA"
        )

    return {
        "risk_flag": not repaired,
        "risk_reason": reason,
        "gap_down_pct": round(
            gap_pct * 100,
            2,
        ),
        "event_day_change_pct": round(
            event_change * 100,
            2,
        ),
        "event_days_ago": int(
            days_ago
        ),
        "event_repaired": repaired,
    }


# ============================================================
# DISTRIBUTION / SELLING PRESSURE
# ============================================================

def distribution_day_count(
    df,
    lookback=20,
):
    """
    Count higher-volume down sessions.
    """

    if (
        df is None
        or len(df) < 2
    ):
        return 0

    d = (
        df.copy()
        .sort_values(
            "timestamp"
        )
        .tail(
            lookback
        )
    )

    close_px = pd.to_numeric(
        d["close"],
        errors="coerce",
    )

    volume = pd.to_numeric(
        d["volume"],
        errors="coerce",
    )

    down_day = (
        close_px
        .pct_change()
        < 0
    )

    higher_volume = (
        volume
        > volume.shift(1)
    )

    return int(
        (
            down_day
            & higher_volume
        )
        .fillna(False)
        .sum()
    )


# ============================================================
# CROSS-SECTIONAL RELATIVE STRENGTH
# ============================================================

def relative_strength_percentiles(
    daily_bars,
):
    """
    Calculate 20/60-day cross-sectional leadership percentile.

    Higher values mean stronger leadership.
    Returns 0.00-1.00.
    """

    if (
        daily_bars is None
        or daily_bars.empty
    ):
        return {}

    rows = []

    for (
        symbol,
        group,
    ) in daily_bars.groupby(
        "symbol"
    ):

        g = group.sort_values(
            "timestamp"
        )

        close_px = (
            pd.to_numeric(
                g["close"],
                errors="coerce",
            )
            .dropna()
        )

        if len(
            close_px
        ) < 61:
            continue

        r20 = safe_float(
            (
                close_px.iloc[-1]
                / close_px.iloc[-21]
            )
            - 1
        )

        r60 = safe_float(
            (
                close_px.iloc[-1]
                / close_px.iloc[-61]
            )
            - 1
        )

        # Weight the intermediate trend slightly more heavily.
        leadership = (
            0.40 * r20
            + 0.60 * r60
        )

        rows.append(
            {
                "symbol": symbol,
                "leadership": leadership,
            }
        )

    if not rows:
        return {}

    ranked = pd.DataFrame(
        rows
    )

    ranked[
        "rs_percentile"
    ] = (
        ranked[
            "leadership"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    return dict(
        zip(
            ranked[
                "symbol"
            ],
            ranked[
                "rs_percentile"
            ],
        )
    )


# ============================================================
# DAILY + INTRADAY CONFLUENCE
# ============================================================

def combine_daily_intraday_signal(
    daily_signal,
    intraday_signal,
    intraday_score,
    risk_flag=False,
):
    """
    A daily BUY cannot become a final BUY without live
    intraday confirmation.
    """

    if risk_flag:
        return (
            "AVOID",
            "A hard risk-event gate is active.",
        )

    if daily_signal in {
        "A+ SWING BUY",
        "BUY",
    }:

        confirmed = bool(
            intraday_signal
            == "BUY"
            and safe_float(
                intraday_score
            )
            >= MIN_INTRADAY_CONFIRMATION_SCORE
        )

        if not confirmed:
            return (
                "WATCH",
                "Daily setup passed, but live intraday "
                "confirmation has not passed yet.",
            )

        return (
            daily_signal,
            "Daily BUY gates and live intraday confirmation passed.",
        )

    if daily_signal == "TOO EXTENDED":
        return (
            "TOO EXTENDED",
            "The daily setup is too extended; "
            "wait for a pullback or retest.",
        )

    if daily_signal == "WATCH":
        return (
            "WATCH",
            "The daily setup is WATCH-only and has "
            "not passed every BUY gate.",
        )

    if daily_signal == "AVOID":
        return (
            "AVOID",
            "The daily setup did not pass every BUY gate.",
        )

    return (
        daily_signal,
        f"The daily signal remained {daily_signal}.",
    )


# ============================================================
# DAILY PREFILTER
# ============================================================

def prefilter_daily(
    bars,
    eligibility,
    min_price=5,
    min_avg_dollar_volume=20_000_000,
    limit=150,
    **kwargs,
):
    """
    Cheap first-stage ranking before running expensive
    swing and intraday scoring.
    """

    if (
        bars is None
        or bars.empty
    ):
        return pd.DataFrame()

    top_n = int(
        kwargs.get(
            "top_n",
            kwargs.get(
                "finalists_n",
                limit,
            ),
        )
    )

    min_avg_volume = safe_float(
        kwargs.get(
            "min_avg_volume",
            0,
        )
    )

    d = bars.copy()

    d[
        "timestamp"
    ] = pd.to_datetime(
        d[
            "timestamp"
        ],
        utc=True,
        errors="coerce",
    )

    meta = (
        eligibility
        .set_index(
            "symbol"
        )
        .to_dict(
            "index"
        )
    )

    rows = []

    for (
        symbol,
        group,
    ) in d.groupby(
        "symbol"
    ):

        if symbol not in meta:
            continue

        g = (
            group
            .sort_values(
                "timestamp"
            )
            .tail(
                260
            )
        )

        if len(
            g
        ) < 20:
            continue

        latest = g.iloc[-1]
        prior = g.iloc[:-1]

        price = safe_float(
            latest[
                "close"
            ]
        )

        if price < min_price:
            continue

        avg_volume = safe_float(
            prior[
                "volume"
            ]
            .tail(20)
            .mean()
        )

        if (
            avg_volume
            < min_avg_volume
        ):
            continue

        avg_dollar_volume = (
            price
            * avg_volume
        )

        if (
            avg_dollar_volume
            < min_avg_dollar_volume
        ):
            continue

        prior_close = safe_float(
            prior[
                "close"
            ].iloc[-1],
            price,
        )

        daily_change_pct = (
            (
                price
                / prior_close
                - 1
            )
            * 100
            if prior_close
            else 0
        )

        latest_open = safe_float(
            latest.get(
                "open"
            ),
            price,
        )

        open_gap_pct = (
            (
                latest_open
                / prior_close
                - 1
            )
            * 100
            if prior_close
            else 0
        )

        daily_volume_ratio = (
            safe_float(
                latest[
                    "volume"
                ]
            )
            / avg_volume
            if avg_volume
            else 0
        )

        close_px = (
            g[
                "close"
            ]
            .astype(
                float
            )
        )

        e20 = safe_float(
            ema(
                close_px,
                20,
            ).iloc[-1],
            price,
        )

        s50 = safe_float(
            sma(
                close_px,
                50,
            ).iloc[-1],
            e20,
        )

        if len(
            close_px
        ) >= 200:
            s200 = safe_float(
                sma(
                    close_px,
                    200,
                ).iloc[-1],
                s50,
            )
        else:
            s200 = s50

        high52 = safe_float(
            g[
                "high"
            ]
            .tail(
                min(
                    252,
                    len(g),
                )
            )
            .max(),
            price,
        )

        near_high = (
            price
            / high52
            if high52
            else 0
        )

        # ====================================================
        # FAST PREFILTER SCORE
        # ====================================================

        score = 0

        # Strong trend
        if (
            price
            > e20
            > s50
        ):
            score += 25

        elif (
            price
            > s50
        ):
            score += 15

        if (
            s50
            > s200
        ):
            score += 10

        # Volume
        if (
            1.5
            <= daily_volume_ratio
            <= 3.5
        ):
            score += 15

        elif (
            1.2
            <= daily_volume_ratio
            < 1.5
        ):
            score += 10

        elif (
            daily_volume_ratio
            > 3.5
        ):
            score += 8

        # Near highs
        if (
            near_high
            >= 0.95
        ):
            score += 15

        elif (
            near_high
            >= 0.90
        ):
            score += 12

        elif (
            near_high
            >= 0.80
        ):
            score += 7

        # Healthy daily momentum
        if (
            0
            <= daily_change_pct
            <= 6
        ):
            score += 8

        elif (
            6
            < daily_change_pct
            <= 10
        ):
            score += 4

        # ====================================================
        # ANTI-CHASE
        # ====================================================

        dist20 = (
            price
            / e20
            - 1
            if e20
            else 0
        )

        if (
            dist20
            > 0.12
        ):
            score -= 25

        elif (
            dist20
            > 0.08
        ):
            score -= 12

        if (
            daily_change_pct
            > 15
        ):
            score -= 25

        elif (
            daily_change_pct
            > 10
        ):
            score -= 12

        risk_event_candidate = bool(
            open_gap_pct
            <= NEGATIVE_GAP_THRESHOLD
            * 100
            or daily_change_pct
            <= NEGATIVE_CLOSE_SHOCK_THRESHOLD
            * 100
        )

        if risk_event_candidate:
            score -= 40

        metadata = meta[
            symbol
        ]

        rows.append(
            {
                "symbol": symbol,
                "name": metadata.get(
                    "name",
                    "",
                ),
                "exchange": metadata.get(
                    "exchange",
                    "",
                ),
                "security_type": metadata.get(
                    "security_type",
                    "",
                ),
                "price": round(
                    price,
                    2,
                ),
                "daily_change_pct": round(
                    daily_change_pct,
                    2,
                ),
                "open_gap_pct": round(
                    open_gap_pct,
                    2,
                ),
                "risk_event_candidate": (
                    risk_event_candidate
                ),
                "daily_volume_ratio": round(
                    daily_volume_ratio,
                    2,
                ),
                "avg_dollar_volume": round(
                    avg_dollar_volume
                ),
                "prefilter_score": round(
                    score,
                    1,
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(
        rows
    )

    return (
        out
        .sort_values(
            [
                "prefilter_score",
                "daily_volume_ratio",
                "avg_dollar_volume",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .head(
            top_n
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# INTRADAY FEATURES
# ============================================================

def add_intraday_features(
    df,
):
    if (
        df is None
        or df.empty
    ):
        return pd.DataFrame()

    d = (
        df.copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    typical_price = (
        d[
            "high"
        ]
        + d[
            "low"
        ]
        + d[
            "close"
        ]
    ) / 3

    cumulative_volume = (
        d[
            "volume"
        ]
        .cumsum()
        .replace(
            0,
            np.nan,
        )
    )

    d[
        "vwap"
    ] = (
        (
            typical_price
            * d[
                "volume"
            ]
        )
        .cumsum()
        / cumulative_volume
    )

    d[
        "ema9"
    ] = ema(
        d[
            "close"
        ],
        9,
    )

    d[
        "ema20"
    ] = ema(
        d[
            "close"
        ],
        20,
    )

    d[
        "atr"
    ] = atr_series(
        d,
        14,
    )

    d[
        "rsi"
    ] = rsi_series(
        d[
            "close"
        ],
        14,
    )

    first = safe_float(
        d[
            "close"
        ].iloc[0]
    )

    if first:
        d[
            "stock_ret"
        ] = (
            d[
                "close"
            ]
            / first
            - 1
        )
    else:
        d[
            "stock_ret"
        ] = 0

    return d


# ============================================================
# PREPARE INTRADAY
# ============================================================

def prepare_intraday(
    day,
    spy_day=None,
    avg_daily_volume=None,
):
    d = add_intraday_features(
        day
    )

    if d.empty:
        return d

    if len(
        d
    ) < 20:
        return d

    opening = (
        d.iloc[
            :min(
                15,
                len(d),
            )
        ]
    )

    d[
        "orb_high"
    ] = safe_float(
        opening[
            "high"
        ].max()
    )

    d[
        "orb_low"
    ] = safe_float(
        opening[
            "low"
        ].min()
    )

    elapsed = np.arange(
        1,
        len(d) + 1,
    )

    avg_daily_volume = safe_float(
        avg_daily_volume
    )

    if (
        avg_daily_volume
        > 0
    ):

        expected_volume = (
            avg_daily_volume
            * np.minimum(
                elapsed
                / 390,
                1,
            )
        )

        expected_volume = np.maximum(
            expected_volume,
            1,
        )

        d[
            "rel_volume"
        ] = (
            d[
                "volume"
            ]
            .cumsum()
            .values
            / expected_volume
        )

    else:

        avg_bar_volume = safe_float(
            d[
                "volume"
            ]
            .head(20)
            .mean(),
            1,
        )

        expected_volume = np.maximum(
            avg_bar_volume
            * elapsed,
            1,
        )

        d[
            "rel_volume"
        ] = (
            d[
                "volume"
            ]
            .cumsum()
            .values
            / expected_volume
        )

    # ========================================================
    # RELATIVE STRENGTH VS SPY
    # ========================================================

    if (
        spy_day is not None
        and not spy_day.empty
    ):

        spy = (
            spy_day.copy()
            .sort_values(
                "timestamp"
            )
        )

        first_spy = safe_float(
            spy[
                "close"
            ].iloc[0]
        )

        if first_spy:
            spy[
                "spy_ret"
            ] = (
                spy[
                    "close"
                ]
                / first_spy
                - 1
            )
        else:
            spy[
                "spy_ret"
            ] = 0

        d[
            "timestamp"
        ] = pd.to_datetime(
            d[
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

        d = pd.merge_asof(
            d.sort_values(
                "timestamp"
            ),
            spy[
                [
                    "timestamp",
                    "spy_ret",
                ]
            ].sort_values(
                "timestamp"
            ),
            on="timestamp",
            direction="backward",
        )

        d[
            "spy_ret"
        ] = (
            d[
                "spy_ret"
            ]
            .fillna(0)
        )

        d[
            "rs"
        ] = (
            d[
                "stock_ret"
            ]
            - d[
                "spy_ret"
            ]
        )

    else:
        d[
            "spy_ret"
        ] = 0

        d[
            "rs"
        ] = d[
            "stock_ret"
        ]

    return d


# ============================================================
# INTRADAY CLASSIFIER
# ============================================================

def classify(
    r,
    avg_dollar_volume=0,
):
    """
    Intraday confirmation engine.

    Maximum score = 100.

    Designed so exceptional but realistic intraday setups can
    reach 85 without requiring every input to be perfect.
    """

    score = 0
    reasons = []

    price = safe_float(
        r.get(
            "close"
        )
    )

    vwap = safe_float(
        r.get(
            "vwap"
        ),
        price,
    )

    ema9_value = safe_float(
        r.get(
            "ema9"
        ),
        price,
    )

    ema20_value = safe_float(
        r.get(
            "ema20"
        ),
        price,
    )

    orb_high = safe_float(
        r.get(
            "orb_high"
        ),
        price,
    )

    relative_volume = safe_float(
        r.get(
            "rel_volume"
        )
    )

    relative_strength = safe_float(
        r.get(
            "rs"
        )
    )

    rsi_value = safe_float(
        r.get(
            "rsi"
        ),
        50,
    )

    atr = safe_float(
        r.get(
            "atr"
        )
    )

    momentum_pct = (
        safe_float(
            r.get(
                "stock_ret"
            )
        )
        * 100
    )

    # ========================================================
    # 1. MOMENTUM QUALITY — 15
    # ========================================================

    if (
        1.0
        <= momentum_pct
        <= 5.0
    ):
        score += 15
        reasons.append(
            "healthy intraday momentum"
        )

    elif (
        0.5
        <= momentum_pct
        < 1.0
    ):
        score += 10

    elif (
        5.0
        < momentum_pct
        <= 7.0
    ):
        score += 10

    elif (
        7.0
        < momentum_pct
        <= 9.0
    ):
        score += 5

    # ========================================================
    # 2. VWAP LOCATION — 15
    # ========================================================

    vwap_extension = (
        price
        / vwap
        - 1
        if vwap
        else 0
    )

    if (
        0.001
        <= vwap_extension
        <= 0.020
    ):
        score += 15
        reasons.append(
            "excellent VWAP position"
        )

    elif (
        0
        <= vwap_extension
        < 0.001
    ):
        score += 10

    elif (
        0.020
        < vwap_extension
        <= 0.035
    ):
        score += 9

    elif (
        0.035
        < vwap_extension
        <= 0.045
    ):
        score += 4

    # ========================================================
    # 3. OPENING RANGE — 10
    # ========================================================

    orb_extension = (
        price
        / orb_high
        - 1
        if orb_high
        else 0
    )

    if (
        0.001
        <= orb_extension
        <= 0.025
    ):
        score += 10
        reasons.append(
            "opening-range breakout confirmed"
        )

    elif (
        -0.003
        <= orb_extension
        < 0.001
    ):
        score += 6

    elif (
        0.025
        < orb_extension
        <= 0.04
    ):
        score += 5

    # ========================================================
    # 4. RELATIVE VOLUME — 15
    # ========================================================

    if (
        1.5
        <= relative_volume
        <= 3.5
    ):
        score += 15
        reasons.append(
            "institutional-quality relative volume"
        )

    elif (
        1.2
        <= relative_volume
        < 1.5
    ):
        score += 11

    elif (
        3.5
        < relative_volume
        <= 5.0
    ):
        score += 9

    elif (
        relative_volume
        > 5.0
    ):
        score += 5

    # ========================================================
    # 5. EMA TREND — 15
    # ========================================================

    if (
        price
        > ema9_value
        > ema20_value
    ):
        score += 15
        reasons.append(
            "intraday EMA trend aligned"
        )

    elif (
        price
        > ema20_value
        and ema9_value
        >= ema20_value
    ):
        score += 11

    elif (
        price
        > ema20_value
    ):
        score += 6

    # ========================================================
    # 6. RELATIVE STRENGTH VS SPY — 10
    # ========================================================

    if (
        0.0075
        <= relative_strength
        <= 0.035
    ):
        score += 10
        reasons.append(
            "outperforming SPY"
        )

    elif (
        0.003
        <= relative_strength
        < 0.0075
    ):
        score += 8

    elif (
        0
        <= relative_strength
        < 0.003
    ):
        score += 5

    elif (
        relative_strength
        > 0.035
    ):
        score += 7

    # ========================================================
    # 7. RSI — 10
    # ========================================================

    if (
        52
        <= rsi_value
        <= 68
    ):
        score += 10

    elif (
        48
        <= rsi_value
        < 52
    ):
        score += 7

    elif (
        68
        < rsi_value
        <= 72
    ):
        score += 8

    elif (
        72
        < rsi_value
        <= 76
    ):
        score += 4

    # ========================================================
    # 8. LIQUIDITY — 5
    # ========================================================

    avg_dollar_volume = safe_float(
        avg_dollar_volume
    )

    if (
        avg_dollar_volume
        >= 100_000_000
    ):
        score += 5

    elif (
        avg_dollar_volume
        >= 50_000_000
    ):
        score += 4

    elif (
        avg_dollar_volume
        >= 25_000_000
    ):
        score += 3

    # ========================================================
    # ANTI-CHASE
    # ========================================================

    ema20_extension = (
        price
        / ema20_value
        - 1
        if ema20_value
        else 0
    )

    atr_extension = (
        (
            price
            - vwap
        )
        / atr
        if atr
        > 0
        else 0
    )

    too_extended = bool(
        vwap_extension
        > 0.055
        or ema20_extension
        > 0.07
        or rsi_value
        > 78
        or momentum_pct
        > 10
        or atr_extension
        > 2.5
    )

    if too_extended:
        score -= 25
        reasons.append(
            "TOO EXTENDED - wait for pullback"
        )

    score = int(
        clamp(
            score,
            0,
            100,
        )
    )

    # ========================================================
    # ENTRY QUALITY — 15
    # ========================================================

    entry_quality = 0

    if (
        0
        <= vwap_extension
        <= 0.025
    ):
        entry_quality += 4

    if (
        price
        > ema20_value
        and ema20_extension
        <= 0.04
    ):
        entry_quality += 3

    if (
        52
        <= rsi_value
        <= 72
    ):
        entry_quality += 2

    if (
        1.2
        <= relative_volume
        <= 3.5
    ):
        entry_quality += 2

    if (
        0.003
        <= relative_strength
        <= 0.03
    ):
        entry_quality += 2

    if (
        momentum_pct
        <= 6
    ):
        entry_quality += 2

    entry_quality = int(
        clamp(
            entry_quality,
            0,
            15,
        )
    )

    # ========================================================
    # FINAL INTRADAY BUY GATE
    # ========================================================

    buy = bool(
        score
        >= MIN_INTRADAY_CONFIRMATION_SCORE
        and entry_quality
        >= 10
        and price
        > vwap
        and price
        > ema20_value
        and ema9_value
        >= ema20_value
        and relative_volume
        >= 1.2
        and relative_strength
        >= 0
        and 48
        <= rsi_value
        <= 76
        and not too_extended
    )

    watch = bool(
        score
        >= 65
        or (
            score
            >= 58
            and price
            > ema20_value
            and relative_strength
            >= 0
        )
    )

    if buy:
        signal = "BUY"

        reasons.append(
            f"entry quality "
            f"{entry_quality}/15"
        )

    elif watch:
        signal = "WATCH"

    else:
        signal = "AVOID"

    return (
        score,
        signal,
        reasons,
    )


# ============================================================
# LEGACY DAILY SCORER
# ============================================================

def _score_swing_daily_legacy(
    stock_daily,
    spy_daily=None,
):
    """
    Legacy compatibility function.

    The production scanner uses score_swing_daily() below.
    """

    return score_swing_daily(
        stock_daily,
        spy_daily=spy_daily,
    )


# ============================================================
# PRODUCTION DAILY SWING SCORER
# ============================================================

def score_swing_daily(
    stock_daily,
    spy_daily=None,
    qqq_daily=None,
    rs_percentile=None,
):
    """
    Institutional daily swing-trade scoring engine.

    Maximum score = 100.

    Production BUY still requires:
        Swing Score >= 85
        Entry Quality >= 10
        Reward/Risk >= 2.0
        Market Score >= 5
        Price in preferred entry zone
        Healthy 20EMA + 50SMA trend
        Distribution days <= 4
        Leadership >= 70th percentile
        No catalyst-risk block
        Not too extended

    A+ BUY uses even stronger thresholds.
    """

    if (
        stock_daily is None
        or stock_daily.empty
    ):
        return None

    d = (
        stock_daily.copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        d
    ) < 60:
        return None

    close = pd.to_numeric(
        d[
            "close"
        ],
        errors="coerce",
    )

    volume = pd.to_numeric(
        d[
            "volume"
        ],
        errors="coerce",
    )

    price = safe_float(
        close.iloc[-1]
    )

    if price <= 0:
        return None

    # ========================================================
    # CORE INDICATORS
    # ========================================================

    e10_series = ema(
        close,
        10,
    )

    e20_series = ema(
        close,
        20,
    )

    s50_series = sma(
        close,
        50,
    )

    e10 = safe_float(
        e10_series.iloc[-1],
        price,
    )

    e20 = safe_float(
        e20_series.iloc[-1],
        price,
    )

    s50 = safe_float(
        s50_series.iloc[-1],
        e20,
    )

    rsi14 = safe_float(
        rsi_series(
            close,
            14,
        ).iloc[-1],
        50,
    )

    atr14 = safe_float(
        atr_series(
            d,
            14,
        ).iloc[-1],
        0,
    )

    if atr14 <= 0:
        atr14 = (
            price
            * 0.02
        )

    average_20_volume = safe_float(
        volume
        .iloc[:-1]
        .tail(20)
        .mean()
    )

    rvol = (
        safe_float(
            volume.iloc[-1]
        )
        / average_20_volume
        if average_20_volume
        else 0
    )

    prior20high = safe_float(
        d[
            "high"
        ]
        .shift(1)
        .tail(20)
        .max(),
        price,
    )

    high252 = safe_float(
        d[
            "high"
        ]
        .tail(
            min(
                252,
                len(d),
            )
        )
        .max(),
        price,
    )

    prior_close = safe_float(
        close.iloc[-2],
        price,
    )

    day_change = (
        price
        / prior_close
        - 1
        if prior_close
        else 0
    )

    latest_open = safe_float(
        d[
            "open"
        ].iloc[-1],
        price,
    )

    open_gap = (
        latest_open
        / prior_close
        - 1
        if prior_close
        else 0
    )

    # ========================================================
    # HARD RISK DATA
    # ========================================================

    risk_event = downside_event_risk(
        d,
        e20=e20,
    )

    distribution_days = (
        distribution_day_count(
            d,
            lookback=20,
        )
    )

    # ========================================================
    # TREND HEALTH
    # ========================================================

    e20_rising = bool(
        len(
            e20_series
        ) >= 6
        and e20
        > safe_float(
            e20_series.iloc[-6],
            e20,
        )
    )

    s50_rising = bool(
        len(
            s50_series
        ) >= 11
        and s50
        > safe_float(
            s50_series.iloc[-11],
            s50,
        )
    )

    trend_health = bool(
        e20_rising
        and s50_rising
    )

    # ========================================================
    # LEADERSHIP VALUE
    # ========================================================

    if (
        rs_percentile is None
        or pd.isna(
            rs_percentile
        )
    ):
        rs_percentile_value = None
    else:
        rs_percentile_value = clamp(
            safe_float(
                rs_percentile
            ),
            0,
            1,
        )

    # ========================================================
    # SETUP TYPE
    # ========================================================

    if risk_event[
        "risk_flag"
    ]:
        setup = (
            "NEGATIVE CATALYST GAP"
        )

    elif (
        price
        >= prior20high
        and rvol
        >= 1.4
        and day_change
        < 0.10
    ):
        setup = (
            "BREAKOUT"
        )

    elif (
        day_change
        >= 0.12
    ):
        setup = (
            "GAP MOMENTUM"
        )

    elif (
        price
        > e20
        > s50
        and abs(
            price
            - e20
        )
        <= max(
            0.8
            * atr14,
            0.03
            * price,
        )
    ):
        setup = (
            "20EMA PULLBACK"
        )

    elif (
        price
        > e10
        > e20
        > s50
        and price
        <= e10
        + 0.7
        * atr14
    ):
        setup = (
            "10EMA CONTINUATION"
        )

    elif (
        price
        > e20
        > s50
        and price
        >= 0.97
        * prior20high
    ):
        setup = (
            "BASE / NEAR BREAKOUT"
        )

    else:
        setup = (
            "TREND MOMENTUM"
        )

    # ========================================================
    # 1. TREND + RELATIVE STRENGTH — 20
    # ========================================================

    trend_score = 0

    if (
        price
        > e10
        > e20
        > s50
    ):
        trend_score += 10

    elif (
        price
        > e20
        > s50
    ):
        trend_score += 8

    elif (
        price
        > s50
    ):
        trend_score += 4

    if e20_rising:
        trend_score += 2

    if s50_rising:
        trend_score += 2

    stock20 = 0
    stock60 = 0

    if len(
        close
    ) >= 21:
        old = safe_float(
            close.iloc[-21],
            price,
        )

        if old:
            stock20 = (
                price
                / old
                - 1
            )

    if len(
        close
    ) >= 61:
        old60 = safe_float(
            close.iloc[-61],
            price,
        )

        if old60:
            stock60 = (
                price
                / old60
                - 1
            )

    spy20 = 0
    spy60 = 0

    if (
        spy_daily is not None
        and not spy_daily.empty
        and len(
            spy_daily
        ) >= 21
    ):

        spy = (
            spy_daily.copy()
            .sort_values(
                "timestamp"
            )
        )

        spy_close = pd.to_numeric(
            spy[
                "close"
            ],
            errors="coerce",
        )

        spy_last = safe_float(
            spy_close.iloc[-1]
        )

        old_spy = safe_float(
            spy_close.iloc[-21],
            spy_last,
        )

        if old_spy:
            spy20 = (
                spy_last
                / old_spy
                - 1
            )

        if len(
            spy_close
        ) >= 61:

            old_spy60 = safe_float(
                spy_close.iloc[-61],
                spy_last,
            )

            if old_spy60:
                spy60 = (
                    spy_last
                    / old_spy60
                    - 1
                )

    if (
        stock20
        > spy20
    ):
        trend_score += 3

    if (
        stock60
        > spy60
    ):
        trend_score += 3

    trend_score = clamp(
        trend_score,
        0,
        20,
    )

    # ========================================================
    # 2. ACCUMULATION / VOLUME — 15
    # ========================================================

    accumulation_score = 0

    if (
        1.5
        <= rvol
        <= 3.5
    ):
        accumulation_score += 7

    elif (
        1.2
        <= rvol
        < 1.5
    ):
        accumulation_score += 5

    elif (
        rvol
        > 3.5
    ):
        accumulation_score += 4

    recent = (
        d.tail(20)
    )

    up_days = (
        recent[
            "close"
        ]
        > recent[
            "open"
        ]
    )

    up_volume = (
        safe_float(
            recent.loc[
                up_days,
                "volume",
            ].mean()
        )
        if up_days.any()
        else 0
    )

    down_volume = (
        safe_float(
            recent.loc[
                ~up_days,
                "volume",
            ].mean()
        )
        if (
            ~up_days
        ).any()
        else 0
    )

    if (
        up_volume
        > down_volume
        * 1.20
    ):
        accumulation_score += 8

    elif (
        up_volume
        > down_volume
        * 1.05
    ):
        accumulation_score += 6

    elif (
        up_volume
        > down_volume
    ):
        accumulation_score += 4

    if (
        distribution_days
        >= 6
    ):
        accumulation_score -= 5

    elif (
        distribution_days
        == 5
    ):
        accumulation_score -= 3

    accumulation_score = clamp(
        accumulation_score,
        0,
        15,
    )

    # ========================================================
    # 3. ENTRY QUALITY — 15
    # ========================================================

    entry_quality = 0

    dist20 = (
        price
        / e20
        - 1
        if e20
        else 0
    )

    if (
        0
        <= dist20
        <= 0.035
    ):
        entry_quality += 6

    elif (
        0.035
        < dist20
        <= 0.06
    ):
        entry_quality += 5

    elif (
        0.06
        < dist20
        <= 0.08
    ):
        entry_quality += 3

    elif (
        -0.02
        <= dist20
        < 0
    ):
        entry_quality += 3

    if (
        52
        <= rsi14
        <= 68
    ):
        entry_quality += 4

    elif (
        48
        <= rsi14
        < 52
    ):
        entry_quality += 3

    elif (
        68
        < rsi14
        <= 72
    ):
        entry_quality += 3

    elif (
        72
        < rsi14
        <= 75
    ):
        entry_quality += 1

    if setup in {
        "20EMA PULLBACK",
        "10EMA CONTINUATION",
    }:
        entry_quality += 3

    elif (
        setup
        == "BREAKOUT"
        and day_change
        <= 0.08
    ):
        entry_quality += 3

    elif (
        setup
        == "BASE / NEAR BREAKOUT"
    ):
        entry_quality += 2

    if (
        abs(
            day_change
        )
        <= 0.05
    ):
        entry_quality += 2

    elif (
        abs(
            day_change
        )
        <= 0.08
    ):
        entry_quality += 1

    entry_quality = int(
        clamp(
            entry_quality,
            0,
            15,
        )
    )

    # ========================================================
    # 4. MARKET REGIME — 10
    # ========================================================

    market_score = 5

    if (
        spy_daily is not None
        and not spy_daily.empty
        and len(
            spy_daily
        ) >= 50
    ):

        spy = (
            spy_daily.copy()
            .sort_values(
                "timestamp"
            )
        )

        spy_close = pd.to_numeric(
            spy[
                "close"
            ],
            errors="coerce",
        )

        spy_last = safe_float(
            spy_close.iloc[-1]
        )

        spy_e20 = safe_float(
            ema(
                spy_close,
                20,
            ).iloc[-1],
            spy_last,
        )

        spy_s50 = safe_float(
            sma(
                spy_close,
                50,
            ).iloc[-1],
            spy_e20,
        )

        market_score = 0

        if (
            spy_last
            > spy_e20
        ):
            market_score += 3

        if (
            spy_last
            > spy_s50
        ):
            market_score += 2

        if (
            spy_e20
            > spy_s50
        ):
            market_score += 2

        if (
            qqq_daily is not None
            and not qqq_daily.empty
            and len(
                qqq_daily
            ) >= 50
        ):

            qqq = (
                qqq_daily.copy()
                .sort_values(
                    "timestamp"
                )
            )

            qqq_close = pd.to_numeric(
                qqq[
                    "close"
                ],
                errors="coerce",
            )

            qqq_last = safe_float(
                qqq_close.iloc[-1]
            )

            qqq_e20 = safe_float(
                ema(
                    qqq_close,
                    20,
                ).iloc[-1],
                qqq_last,
            )

            qqq_s50 = safe_float(
                sma(
                    qqq_close,
                    50,
                ).iloc[-1],
                qqq_e20,
            )

            if (
                qqq_last
                > qqq_e20
            ):
                market_score += 2

            if (
                qqq_e20
                > qqq_s50
            ):
                market_score += 1

    market_score = clamp(
        market_score,
        0,
        10,
    )

    # ========================================================
    # 5. SETUP QUALITY — 10
    # ========================================================

    setup_scores = {
        "20EMA PULLBACK": 10,
        "BREAKOUT": 10,
        "10EMA CONTINUATION": 9,
        "BASE / NEAR BREAKOUT": 8,
        "TREND MOMENTUM": 6,
        "GAP MOMENTUM": 3,
        "NEGATIVE CATALYST GAP": 0,
    }

    setup_score = (
        setup_scores.get(
            setup,
            5,
        )
    )

    # ========================================================
    # PREFERRED ENTRY ZONE
    # ========================================================

    if (
        setup
        == "20EMA PULLBACK"
    ):

        entry_low = max(
            e20
            - 0.30
            * atr14,
            0.01,
        )

        entry_high = (
            e20
            + 0.65
            * atr14
        )

    elif (
        setup
        == "10EMA CONTINUATION"
    ):

        entry_low = max(
            e10
            - 0.35
            * atr14,
            0.01,
        )

        entry_high = (
            e10
            + 0.60
            * atr14
        )

    elif setup in {
        "BREAKOUT",
        "BASE / NEAR BREAKOUT",
    }:

        entry_low = max(
            prior20high
            - 0.25
            * atr14,
            0.01,
        )

        entry_high = (
            prior20high
            + 0.65
            * atr14
        )

    else:

        entry_low = max(
            min(
                e20,
                price
                - 0.50
                * atr14,
            ),
            0.01,
        )

        entry_high = (
            price
            + 0.25
            * atr14
        )

    inside_entry_zone = bool(
        entry_low
        <= price
        <= entry_high
    )

    # ========================================================
    # STOP / TARGETS
    # ========================================================

    structural_stop = (
        e20
        - 0.50
        * atr14
    )

    atr_stop = (
        price
        - 2.0
        * atr14
    )

    stop = max(
        min(
            structural_stop,
            s50,
        ),
        atr_stop,
    )

    if (
        stop
        >= price
    ):
        stop = (
            price
            - max(
                atr14,
                price
                * 0.025,
            )
        )

    stop = max(
        stop,
        0.01,
    )

    risk = max(
        price
        - stop,
        0.01,
    )

    target1 = (
        price
        + 2.0
        * risk
    )

    target2 = (
        price
        + 3.0
        * risk
    )

    # Do not automatically punish a breakout simply because
    # it is near/above its old 52-week high.
    realistic_target = target1

    if (
        high252
        > price
        and high252
        > price
        + 0.75
        * risk
        and high252
        < target1
    ):
        realistic_target = high252

    reward_risk = (
        max(
            realistic_target
            - price,
            0,
        )
        / risk
    )

    # ========================================================
    # 6. REWARD / RISK — 10
    # ========================================================

    if (
        reward_risk
        >= 3.0
    ):
        rr_score = 10

    elif (
        reward_risk
        >= 2.5
    ):
        rr_score = 9

    elif (
        reward_risk
        >= 2.0
    ):
        rr_score = 8

    elif (
        reward_risk
        >= 1.7
    ):
        rr_score = 5

    elif (
        reward_risk
        >= 1.4
    ):
        rr_score = 3

    else:
        rr_score = 0

    # ========================================================
    # 7. LIQUIDITY / VOLATILITY — 5
    # ========================================================

    atr_pct = (
        atr14
        / price
        if price
        else 0
    )

    average_dollar_volume = safe_float(
        (
            d[
                "close"
            ]
            * d[
                "volume"
            ]
        )
        .tail(20)
        .mean()
    )

    vol_liq_score = 0

    if (
        0.015
        <= atr_pct
        <= 0.055
    ):
        vol_liq_score += 3

    elif (
        0.010
        <= atr_pct
        <= 0.075
    ):
        vol_liq_score += 2

    if (
        average_dollar_volume
        >= 100_000_000
    ):
        vol_liq_score += 2

    elif (
        average_dollar_volume
        >= 20_000_000
    ):
        vol_liq_score += 1

    vol_liq_score = clamp(
        vol_liq_score,
        0,
        5,
    )

    # ========================================================
    # 8. LEADERSHIP — 10
    # ========================================================

    # Missing cross-sectional data receives a neutral 5,
    # but cannot satisfy the BUY leadership hard gate unless
    # the caller intentionally supplied no ranking universe.
    if (
        rs_percentile_value
        is None
    ):
        leadership_score = 5

    elif (
        rs_percentile_value
        >= 0.95
    ):
        leadership_score = 10

    elif (
        rs_percentile_value
        >= 0.90
    ):
        leadership_score = 9

    elif (
        rs_percentile_value
        >= 0.85
    ):
        leadership_score = 8

    elif (
        rs_percentile_value
        >= 0.80
    ):
        leadership_score = 7

    elif (
        rs_percentile_value
        >= 0.70
    ):
        leadership_score = 6

    elif (
        rs_percentile_value
        >= 0.50
    ):
        leadership_score = 3

    else:
        leadership_score = 1

    # ========================================================
    # 9. CATALYST / EVENT QUALITY — 5
    # ========================================================

    if (
        risk_event[
            "risk_flag"
        ]
    ):
        catalyst_score = 0

    elif (
        risk_event[
            "event_days_ago"
        ]
        is not None
        and risk_event[
            "event_repaired"
        ]
    ):
        catalyst_score = 3

    else:
        catalyst_score = 5

    # ========================================================
    # TOTAL SCORE
    #
    # Trend             20
    # Accumulation      15
    # Entry quality     15
    # Market            10
    # Setup             10
    # Reward/Risk       10
    # Leadership        10
    # Catalyst           5
    # Liquidity          5
    # --------------------
    # Total            100
    # ========================================================

    swing_score = (
        trend_score
        + accumulation_score
        + entry_quality
        + market_score
        + setup_score
        + rr_score
        + leadership_score
        + catalyst_score
        + vol_liq_score
    )

    # Hard-event penalty remains independent of scoring.
    if (
        risk_event[
            "risk_flag"
        ]
    ):
        swing_score -= 25

    swing_score = round(
        clamp(
            swing_score,
            0,
            100,
        ),
        1,
    )

    # ========================================================
    # ANTI-CHASE
    # ========================================================

    positive_gap_extension = bool(
        open_gap
        >= 0.08
        or day_change
        >= 0.10
    )

    too_extended = bool(
        price
        > e20
        * 1.10
        or rsi14
        > 76
        or positive_gap_extension
    )

    # ========================================================
    # HARD BUY GATES
    # ========================================================

    leadership_buy_gate = bool(
        rs_percentile_value
        is None
        or rs_percentile_value
        >= MIN_RS_PERCENTILE_BUY
    )

    leadership_a_plus_gate = bool(
        rs_percentile_value
        is None
        or rs_percentile_value
        >= MIN_RS_PERCENTILE_A_PLUS
    )

    distribution_gate = bool(
        distribution_days
        <= MAX_DISTRIBUTION_DAYS_BUY
    )

    # ========================================================
    # FINAL DAILY SIGNAL
    # ========================================================

    if (
        risk_event[
            "risk_flag"
        ]
    ):
        signal = "AVOID"

    elif too_extended:
        signal = "TOO EXTENDED"

    elif (
        swing_score
        >= 92
        and entry_quality
        >= 12
        and reward_risk
        >= 2.5
        and market_score
        >= 7
        and inside_entry_zone
        and trend_health
        and distribution_gate
        and leadership_a_plus_gate
    ):
        signal = (
            "A+ SWING BUY"
        )

    elif (
        swing_score
        >= MIN_SWING_SCORE_BUY
        and entry_quality
        >= MIN_ENTRY_QUALITY_BUY
        and reward_risk
        >= MIN_REWARD_RISK_BUY
        and market_score
        >= MIN_MARKET_SCORE_BUY
        and inside_entry_zone
        and trend_health
        and distribution_gate
        and leadership_buy_gate
    ):
        signal = "BUY"

    elif (
        swing_score
        >= 72
    ):
        signal = "WATCH"

    else:
        signal = "AVOID"

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "signal": signal,

        "swing_score": swing_score,

        "setup": setup,

        "price": round(
            price,
            2,
        ),

        "entry_quality": round(
            entry_quality,
            1,
        ),

        "entry_low": round(
            entry_low,
            2,
        ),

        "entry_high": round(
            entry_high,
            2,
        ),

        "stop": round(
            stop,
            2,
        ),

        "target1": round(
            target1,
            2,
        ),

        "target2": round(
            target2,
            2,
        ),

        "reward_risk": round(
            reward_risk,
            2,
        ),

        "rsi14": round(
            rsi14,
            1,
        ),

        "rvol": round(
            rvol,
            2,
        ),

        "too_extended": (
            too_extended
        ),

        "inside_entry_zone": (
            inside_entry_zone
        ),

        "risk_flag": bool(
            risk_event[
                "risk_flag"
            ]
        ),

        "risk_reason": (
            risk_event[
                "risk_reason"
            ]
        ),

        "gap_down_pct": (
            risk_event[
                "gap_down_pct"
            ]
        ),

        "event_day_change_pct": (
            risk_event[
                "event_day_change_pct"
            ]
        ),

        "event_days_ago": (
            risk_event[
                "event_days_ago"
            ]
        ),

        "event_repaired": bool(
            risk_event[
                "event_repaired"
            ]
        ),

        "open_gap_pct": round(
            open_gap
            * 100,
            2,
        ),

        "day_change_pct": round(
            day_change
            * 100,
            2,
        ),

        "trend_health": (
            trend_health
        ),

        "e20_rising": (
            e20_rising
        ),

        "s50_rising": (
            s50_rising
        ),

        "distribution_days": int(
            distribution_days
        ),

        "market_score": round(
            market_score,
            1,
        ),

        "leadership_percentile": (
            None
            if rs_percentile_value
            is None
            else round(
                rs_percentile_value
                * 100,
                1,
            )
        ),

        "leadership_score": round(
            leadership_score,
            1,
        ),

        "trend_score": round(
            trend_score,
            1,
        ),

        "accumulation_score": round(
            accumulation_score,
            1,
        ),

        "setup_score": round(
            setup_score,
            1,
        ),

        "rr_score": round(
            rr_score,
            1,
        ),

        "catalyst_score": round(
            catalyst_score,
            1,
        ),

        "vol_liq_score": round(
            vol_liq_score,
            1,
        ),
    }
