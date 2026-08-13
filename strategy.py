import numpy as np
import pandas as pd


# ============================================================
# INSTITUTIONAL MOMENTUM + SWING STRATEGY v3
# ============================================================


# Conservative hard gates. These are deliberately separate from the
# 0-100 score so a high score cannot override an abnormal risk event.
NEGATIVE_GAP_THRESHOLD = -0.05
NEGATIVE_CLOSE_SHOCK_THRESHOLD = -0.07
EVENT_LOOKBACK_SESSIONS = 5
EVENT_MIN_COOLDOWN_SESSIONS = 3
MAX_DISTRIBUTION_DAYS_BUY = 4
MIN_RS_PERCENTILE_BUY = 0.70
MIN_RS_PERCENTILE_A_PLUS = 0.85
MIN_INTRADAY_CONFIRMATION_SCORE = 85


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def rsi_series(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_series(df, n=14):
    prev = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / n, adjust=False).mean()


def downside_event_risk(df, e20=None):
    """Detect a recent abnormal downside gap/selloff and its repair status.

    The rule intentionally does not depend on an earnings-calendar vendor.
    Earnings misses, guidance cuts and other material negative catalysts all
    receive the same protection. A recent event remains blocked for at least
    three completed sessions and until price closes above the event-day high
    and the 20-day EMA.
    """

    default = {
        "risk_flag": False,
        "risk_reason": "No abnormal downside catalyst gap detected",
        "gap_down_pct": 0.0,
        "event_day_change_pct": 0.0,
        "event_days_ago": None,
        "event_repaired": True,
    }

    if df is None or len(df) < 2:
        return default

    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    open_px = pd.to_numeric(d["open"], errors="coerce")
    high_px = pd.to_numeric(d["high"], errors="coerce")
    close_px = pd.to_numeric(d["close"], errors="coerce")
    prior_close = close_px.shift(1)

    open_gap = open_px / prior_close - 1
    close_change = close_px / prior_close - 1

    start = max(1, len(d) - EVENT_LOOKBACK_SESSIONS)
    event_mask = (
        (open_gap <= NEGATIVE_GAP_THRESHOLD)
        | (close_change <= NEGATIVE_CLOSE_SHOCK_THRESHOLD)
    )
    event_positions = np.flatnonzero(event_mask.iloc[start:].fillna(False).to_numpy())

    if len(event_positions) == 0:
        return default

    event_pos = start + int(event_positions[-1])
    days_ago = len(d) - 1 - event_pos
    gap_pct = safe_float(open_gap.iloc[event_pos])
    event_change = safe_float(close_change.iloc[event_pos])
    event_high = safe_float(high_px.iloc[event_pos])
    current_close = safe_float(close_px.iloc[-1])

    if e20 is None:
        e20 = safe_float(ema(close_px, 20).iloc[-1], current_close)

    repaired = (
        days_ago >= EVENT_MIN_COOLDOWN_SESSIONS
        and current_close > event_high
        and current_close > safe_float(e20, current_close)
    )

    hard_block = not repaired
    worst_move = min(gap_pct, event_change)
    reason = (
        f"Abnormal downside catalyst move {worst_move * 100:.1f}% "
        f"{days_ago} session{'s' if days_ago != 1 else ''} ago; "
        "wait for at least 3 sessions and a close above the event-day high "
        "and 20-day EMA"
    )

    if repaired:
        reason = "Prior downside event has completed its cooldown and repair test"

    return {
        "risk_flag": bool(hard_block),
        "risk_reason": reason,
        "gap_down_pct": round(gap_pct * 100, 2),
        "event_day_change_pct": round(event_change * 100, 2),
        "event_days_ago": int(days_ago),
        "event_repaired": bool(repaired),
    }


def distribution_day_count(df, lookback=20):
    """Count high-volume down days, a simple institutional-selling proxy."""

    if df is None or len(df) < 2:
        return 0

    d = df.copy().sort_values("timestamp").tail(lookback)
    close_px = pd.to_numeric(d["close"], errors="coerce")
    volume = pd.to_numeric(d["volume"], errors="coerce")
    down = close_px.pct_change() < 0
    higher_volume = volume > volume.shift(1)
    return int((down & higher_volume).fillna(False).sum())


def relative_strength_percentiles(daily_bars):
    """Return cross-sectional 20/60-day leadership percentiles by symbol."""

    if daily_bars is None or daily_bars.empty:
        return {}

    rows = []
    for symbol, group in daily_bars.groupby("symbol"):
        g = group.sort_values("timestamp")
        close_px = pd.to_numeric(g["close"], errors="coerce").dropna()
        if len(close_px) < 61:
            continue

        r20 = safe_float(close_px.iloc[-1] / close_px.iloc[-21] - 1)
        r60 = safe_float(close_px.iloc[-1] / close_px.iloc[-61] - 1)
        rows.append({"symbol": symbol, "leadership": 0.40 * r20 + 0.60 * r60})

    if not rows:
        return {}

    ranked = pd.DataFrame(rows)
    ranked["rs_percentile"] = ranked["leadership"].rank(pct=True, method="average")
    return dict(zip(ranked["symbol"], ranked["rs_percentile"]))


def combine_daily_intraday_signal(
    daily_signal,
    intraday_signal,
    intraday_score,
    risk_flag=False,
):
    """Require live confirmation before promoting a daily setup to BUY."""

    if risk_flag:
        return "AVOID", "A hard risk-event gate is active."

    if daily_signal in {"A+ SWING BUY", "BUY"}:
        confirmed = (
            intraday_signal == "BUY"
            and safe_float(intraday_score) >= MIN_INTRADAY_CONFIRMATION_SCORE
        )
        if not confirmed:
            return (
                "WATCH",
                "Daily setup passed, but live intraday confirmation has not passed yet.",
            )

    return daily_signal, "Daily and intraday signal rules are aligned."


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

    if bars is None or bars.empty:
        return pd.DataFrame()

    top_n = int(
        kwargs.get(
            "top_n",
            kwargs.get("finalists_n", limit)
        )
    )

    min_avg_volume = safe_float(
        kwargs.get("min_avg_volume", 0)
    )

    d = bars.copy()

    d["timestamp"] = pd.to_datetime(
        d["timestamp"],
        utc=True,
        errors="coerce"
    )

    meta = eligibility.set_index("symbol").to_dict("index")

    rows = []

    for sym, g in d.groupby("symbol"):

        if sym not in meta:
            continue

        g = g.sort_values("timestamp").tail(260)

        if len(g) < 20:
            continue

        latest = g.iloc[-1]
        prior = g.iloc[:-1]

        price = safe_float(latest["close"])

        if price < min_price:
            continue

        avg_vol = safe_float(
            prior["volume"].tail(20).mean()
        )

        if avg_vol < min_avg_volume:
            continue

        avg_dollar_volume = price * avg_vol

        if avg_dollar_volume < min_avg_dollar_volume:
            continue

        prior_close = safe_float(
            prior["close"].iloc[-1],
            price
        )

        daily_change_pct = (
            (price / prior_close - 1) * 100
            if prior_close
            else 0
        )

        latest_open = safe_float(
            latest.get("open"),
            price,
        )

        open_gap_pct = (
            (latest_open / prior_close - 1) * 100
            if prior_close
            else 0
        )

        daily_volume_ratio = (
            safe_float(latest["volume"]) / avg_vol
            if avg_vol
            else 0
        )

        close = g["close"].astype(float)

        e20 = safe_float(
            ema(close, 20).iloc[-1],
            price
        )

        s50 = safe_float(
            sma(close, 50).iloc[-1],
            e20
        )

        if len(close) >= 200:
            s200 = safe_float(
                sma(close, 200).iloc[-1],
                s50
            )
        else:
            s200 = s50

        high52 = safe_float(
            g["high"].tail(
                min(252, len(g))
            ).max(),
            price
        )

        near_high = (
            price / high52
            if high52
            else 0
        )

        # --------------------------------------------
        # FAST PREFILTER SCORE
        # --------------------------------------------

        score = 0

        if price > e20 > s50:
            score += 25
        elif price > s50:
            score += 15

        if s50 > s200:
            score += 10

        if daily_volume_ratio >= 2:
            score += 15
        elif daily_volume_ratio >= 1.5:
            score += 12
        elif daily_volume_ratio >= 1.1:
            score += 7

        if near_high >= .90:
            score += 15
        elif near_high >= .80:
            score += 8

        # --------------------------------------------
        # ANTI-CHASE PENALTY
        # --------------------------------------------

        dist20 = (
            price / e20 - 1
            if e20
            else 0
        )

        if dist20 > .12:
            score -= 20
        elif dist20 > .08:
            score -= 10

        if daily_change_pct > 15:
            score -= 20
        elif daily_change_pct > 10:
            score -= 10

        # A severe downside catalyst should not rank like a healthy pullback.
        # Keep the row eligible for an explicit AVOID explanation, but push it
        # well below clean momentum candidates in the expensive second stage.
        risk_event_candidate = (
            open_gap_pct <= NEGATIVE_GAP_THRESHOLD * 100
            or daily_change_pct <= NEGATIVE_CLOSE_SHOCK_THRESHOLD * 100
        )

        if risk_event_candidate:
            score -= 40

        m = meta[sym]

        rows.append(
            {
                "symbol": sym,
                "name": m.get("name", ""),
                "exchange": m.get("exchange", ""),
                "security_type": m.get(
                    "security_type",
                    ""
                ),
                "price": round(price, 2),
                "daily_change_pct":
                    round(daily_change_pct, 2),
                "open_gap_pct":
                    round(open_gap_pct, 2),
                "risk_event_candidate":
                    bool(risk_event_candidate),
                "daily_volume_ratio":
                    round(daily_volume_ratio, 2),
                "avg_dollar_volume":
                    round(avg_dollar_volume),
                "prefilter_score":
                    round(score, 1),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    return (
        out.sort_values(
            [
                "prefilter_score",
                "daily_volume_ratio"
            ],
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# INTRADAY FEATURES
# ============================================================

def add_intraday_features(df):

    if df is None or df.empty:
        return pd.DataFrame()

    d = (
        df.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    typical = (
        d.high +
        d.low +
        d.close
    ) / 3

    cumulative_volume = (
        d.volume
        .cumsum()
        .replace(0, np.nan)
    )

    d["vwap"] = (
        (typical * d.volume).cumsum()
        / cumulative_volume
    )

    d["ema9"] = ema(
        d.close,
        9
    )

    d["ema20"] = ema(
        d.close,
        20
    )

    d["atr"] = atr_series(
        d,
        14
    )

    d["rsi"] = rsi_series(
        d.close,
        14
    )

    first = safe_float(
        d.close.iloc[0]
    )

    d["stock_ret"] = (
        d.close / first - 1
        if first
        else 0
    )

    return d


# ============================================================
# PREPARE INTRADAY
# ============================================================

def prepare_intraday(
    day,
    spy_day=None,
    avg_daily_volume=None
):

    d = add_intraday_features(day)

    if d.empty:
        return d

    if len(d) < 20:
        return d

    opening = d.iloc[
        :min(15, len(d))
    ]

    d["orb_high"] = opening.high.max()
    d["orb_low"] = opening.low.min()

    elapsed = np.arange(
        1,
        len(d) + 1
    )

    avg_daily_volume = safe_float(
        avg_daily_volume
    )

    if avg_daily_volume > 0:

        expected_volume = (
            avg_daily_volume
            * np.minimum(
                elapsed / 390,
                1
            )
        )

        expected_volume = np.maximum(
            expected_volume,
            1
        )

        d["rel_volume"] = (
            d.volume.cumsum().values
            / expected_volume
        )

    else:

        avg_bar_volume = safe_float(
            d.volume.head(20).mean(),
            1
        )

        expected_volume = np.maximum(
            avg_bar_volume * elapsed,
            1
        )

        d["rel_volume"] = (
            d.volume.cumsum().values
            / expected_volume
        )

    # --------------------------------------------
    # RELATIVE STRENGTH VS SPY
    # --------------------------------------------

    if (
        spy_day is not None
        and not spy_day.empty
    ):

        s = (
            spy_day.copy()
            .sort_values("timestamp")
        )

        first_spy = safe_float(
            s.close.iloc[0]
        )

        s["spy_ret"] = (
            s.close / first_spy - 1
            if first_spy
            else 0
        )

        d["timestamp"] = pd.to_datetime(
            d["timestamp"],
            utc=True
        )

        s["timestamp"] = pd.to_datetime(
            s["timestamp"],
            utc=True
        )

        d = pd.merge_asof(
            d.sort_values("timestamp"),
            s[
                [
                    "timestamp",
                    "spy_ret"
                ]
            ].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        d["spy_ret"] = (
            d["spy_ret"]
            .fillna(0)
        )

        d["rs"] = (
            d["stock_ret"]
            - d["spy_ret"]
        )

    else:

        d["spy_ret"] = 0

        d["rs"] = d["stock_ret"]

    return d


# ============================================================
# LIVE BUY / WATCH CLASSIFIER
# ============================================================

def classify(
    r,
    avg_dollar_volume=0
):

    score = 0
    reasons = []

    price = safe_float(
        r.get("close")
    )

    vwap = safe_float(
        r.get("vwap"),
        price
    )

    ema9 = safe_float(
        r.get("ema9"),
        price
    )

    ema20 = safe_float(
        r.get("ema20"),
        price
    )

    orb_high = safe_float(
        r.get("orb_high"),
        price
    )

    rv = safe_float(
        r.get("rel_volume")
    )

    rs = safe_float(
        r.get("rs")
    )

    rsi = safe_float(
        r.get("rsi"),
        50
    )

    atr = safe_float(
        r.get("atr")
    )

    mom = (
        safe_float(
            r.get("stock_ret")
        ) * 100
    )

    # ========================================================
    # 1. MOMENTUM QUALITY
    # ========================================================

    if 1 <= mom <= 5:

        score += 15

        reasons.append(
            "healthy momentum"
        )

    elif .5 <= mom < 1:

        score += 9

    elif 5 < mom <= 8:

        score += 7

    elif mom > 8:

        score += 2

        reasons.append(
            "large intraday move"
        )

    # ========================================================
    # 2. ENTRY LOCATION / VWAP
    # ========================================================

    vwap_extension = (
        price / vwap - 1
        if vwap
        else 0
    )

    if .001 <= vwap_extension <= .025:

        score += 15

        reasons.append(
            "good VWAP entry"
        )

    elif 0 <= vwap_extension < .001:

        score += 8

    elif .025 < vwap_extension <= .04:

        score += 6

    # ========================================================
    # 3. OPENING RANGE
    # ========================================================

    if orb_high:

        orb_extension = (
            price / orb_high - 1
        )

        if .001 <= orb_extension <= .025:

            score += 10

            reasons.append(
                "ORB confirmed"
            )

        elif -.003 <= orb_extension < .001:

            score += 5

    # ========================================================
    # 4. RELATIVE VOLUME
    # ========================================================

    if 1.5 <= rv <= 3.5:

        score += 15

        reasons.append(
            "strong volume"
        )

    elif 1.2 <= rv < 1.5:

        score += 10

    elif 3.5 < rv <= 5:

        score += 7

    elif rv > 5:

        score += 3

        reasons.append(
            "extreme volume spike"
        )

    # ========================================================
    # 5. TREND
    # ========================================================

    if price > ema9 > ema20:

        score += 15

        reasons.append(
            "EMA trend aligned"
        )

    elif price > ema20:

        score += 8

    # ========================================================
    # 6. RELATIVE STRENGTH VS SPY
    # ========================================================

    if .0075 <= rs <= .035:

        score += 10

        reasons.append(
            "outperforming SPY"
        )

    elif 0 <= rs < .0075:

        score += 5

    elif rs > .035:

        score += 5

    # ========================================================
    # 7. RSI
    # ========================================================

    if 52 <= rsi <= 68:

        score += 10

    elif 45 <= rsi < 52:

        score += 5

    elif 68 < rsi <= 72:

        score += 7

    elif 72 < rsi <= 76:

        score += 3

    # ========================================================
    # 8. LIQUIDITY
    # ========================================================

    avg_dollar_volume = safe_float(
        avg_dollar_volume
    )

    if avg_dollar_volume >= 100_000_000:

        score += 5

    elif avg_dollar_volume >= 25_000_000:

        score += 3

    # ========================================================
    # 9. ANTI-CHASE FILTER
    # ========================================================

    ema20_extension = (
        price / ema20 - 1
        if ema20
        else 0
    )

    atr_extension = (
        (price - vwap) / atr
        if atr > 0
        else 0
    )

    too_extended = (

        vwap_extension > .055

        or ema20_extension > .07

        or rsi > 78

        or mom > 10

        or atr_extension > 2.5
    )

    if too_extended:

        score -= 25

        reasons.append(
            "TOO EXTENDED - wait for pullback"
        )

    # ========================================================
    # 10. ENTRY QUALITY SCORE
    # ========================================================

    entry_quality = 0

    if 0 <= vwap_extension <= .025:
        entry_quality += 4

    if (
        price > ema20
        and ema20_extension <= .04
    ):
        entry_quality += 3

    if 52 <= rsi <= 72:
        entry_quality += 2

    if 1.2 <= rv <= 3.5:
        entry_quality += 2

    if .003 <= rs <= .03:
        entry_quality += 2

    if mom <= 6:
        entry_quality += 2

    score = max(
        0,
        min(
            100,
            int(score)
        )
    )

    # ========================================================
    # FINAL BUY GATE
    # ========================================================

    buy = (

        score >= 85

        and entry_quality >= 10

        and price > vwap

        and price > ema20

        and ema9 >= ema20

        and rv >= 1.2

        and rs >= 0

        and 48 <= rsi <= 76

        and not too_extended
    )

    watch = (

        score >= 65

        or (

            score >= 58

            and price > ema20

            and rs >= 0
        )
    )

    if buy:

        signal = "BUY"

        reasons.append(
            f"entry quality {entry_quality}/15"
        )

    elif watch:

        signal = "WATCH"

        if too_extended:

            reasons.append(
                "strong stock but bad entry now"
            )

    else:

        signal = "AVOID"

    return score, signal, reasons


# ============================================================
# LEGACY DAILY SWING TRADE SCORER (kept for reference; not exported)
# ============================================================

def _score_swing_daily_legacy(
    stock_daily,
    spy_daily=None
):

    if (
        stock_daily is None
        or stock_daily.empty
    ):

        return None

    d = (
        stock_daily.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(d) < 60:
        return None

    close = d.close.astype(float)
    volume = d.volume.astype(float)

    price = safe_float(
        close.iloc[-1]
    )

    e10 = safe_float(
        ema(close, 10).iloc[-1],
        price
    )

    e20 = safe_float(
        ema(close, 20).iloc[-1],
        price
    )

    s50 = safe_float(
        sma(close, 50).iloc[-1],
        e20
    )

    rsi14 = safe_float(
        rsi_series(close).iloc[-1],
        50
    )

    atr14 = safe_float(
        atr_series(d).iloc[-1]
    )

    avg20 = safe_float(
        volume.iloc[:-1]
        .tail(20)
        .mean()
    )

    rvol = (
        volume.iloc[-1] / avg20
        if avg20
        else 0
    )

    prior20high = safe_float(
        d.high.shift(1)
        .tail(20)
        .max(),
        price
    )

    day_change = (
        price / close.iloc[-2] - 1
    )

    # ========================================================
    # SETUP TYPE
    # ========================================================

    if (
        price >= prior20high
        and rvol >= 1.4
        and day_change < .12
    ):

        setup = "BREAKOUT"

    elif day_change >= .12:

        setup = "GAP MOMENTUM"

    elif (
        price > e20 > s50
        and abs(price - e20)
        <= max(
            .8 * atr14,
            .03 * price
        )
    ):

        setup = "20EMA PULLBACK"

    elif (
        price > e10 > e20 > s50
        and price
        <= e10 + .7 * atr14
    ):

        setup = "10EMA CONTINUATION"

    elif (
        price > e20 > s50
        and price >= .97 * prior20high
    ):

        setup = "BASE / NEAR BREAKOUT"

    else:

        setup = "TREND MOMENTUM"

    # ========================================================
    # ENTRY QUALITY 0-15
    # ========================================================

    dist20 = (
        price / e20 - 1
        if e20
        else 0
    )

    entry_quality = 0

    if 0 <= dist20 <= .04:

        entry_quality += 6

    elif .04 < dist20 <= .07:

        entry_quality += 4

    if 52 <= rsi14 <= 68:

        entry_quality += 4

    elif 68 < rsi14 <= 72:

        entry_quality += 2

    if setup in [
        "20EMA PULLBACK",
        "10EMA CONTINUATION"
    ]:

        entry_quality += 3

    elif setup == "BREAKOUT":

        entry_quality += 3

    if abs(day_change) <= .06:

        entry_quality += 2

    entry_quality = min(
        15,
        entry_quality
    )

    # ========================================================
    # STOP + TARGETS
    # ========================================================

    stop = max(
        s50,
        price - 2 * atr14
    )

    if stop >= price:

        stop = (
            price
            - max(
                atr14,
                price * .025
            )
        )

    risk = max(
        price - stop,
        .01
    )

    target1 = (
        price + 2 * risk
    )

    target2 = (
        price + 3 * risk
    )

    reward_risk = 2.0

    # ========================================================
    # SWING SCORE
    # ========================================================

    score = 0

    # Trend / relative strength - 20
    if price > e20 > s50:
        score += 15
    elif price > s50:
        score += 8

    if spy_daily is not None and not spy_daily.empty:

        spy = (
            spy_daily.copy()
            .sort_values("timestamp")
        )

        if len(spy) >= 21:

            stock20 = (
                close.iloc[-1]
                / close.iloc[-21]
                - 1
            )

            spy20 = (
                spy.close.iloc[-1]
                / spy.close.iloc[-21]
                - 1
            )

            if stock20 > spy20:

                score += 5

    # Accumulation - 15
    if rvol >= 1.5:

        score += 8

    elif rvol >= 1.2:

        score += 5

    recent = d.tail(20)

    up = (
        recent.close
        > recent.open
    )

    up_vol = safe_float(
        recent.loc[
            up,
            "volume"
        ].mean()
    )

    down_vol = safe_float(
        recent.loc[
            ~up,
            "volume"
        ].mean()
    )

    if up_vol > down_vol:

        score += 7

    # Entry quality - 15
    score += entry_quality

    # Setup quality - 10
    if setup == "20EMA PULLBACK":

        score += 10

    elif setup in [
        "BREAKOUT",
        "10EMA CONTINUATION"
    ]:

        score += 9

    elif setup == "BASE / NEAR BREAKOUT":

        score += 8

    else:

        score += 5

    # Market regime placeholder - 10
    market_score = 5

    if (
        spy_daily is not None
        and not spy_daily.empty
        and len(spy_daily) >= 50
    ):

        spy_close = (
            spy_daily
            .sort_values("timestamp")
            .close
        )

        spy20 = ema(
            spy_close,
            20
        ).iloc[-1]

        spy50 = sma(
            spy_close,
            50
        ).iloc[-1]

        market_score = 0

        if spy_close.iloc[-1] > spy20:
            market_score += 4

        if spy_close.iloc[-1] > spy50:
            market_score += 3

        if spy20 > spy50:
            market_score += 3

    score += market_score

    # Reward / risk - 10
    if reward_risk >= 2:
        score += 8

    # Sector - neutral 5/10 until app.py feeds sector data
    score += 5

    # Catalyst / earnings - neutral 2.5/5 until app.py adds data
    score += 2.5

    # Liquidity / volatility - 5
    atr_pct = (
        atr14 / price
        if price
        else 0
    )

    if .015 <= atr_pct <= .055:

        score += 3

    adv = safe_float(
        (
            d.close
            * d.volume
        )
        .tail(20)
        .mean()
    )

    if adv >= 100_000_000:

        score += 2

    elif adv >= 20_000_000:

        score += 1

    score = round(
        min(
            100,
            score
        ),
        1
    )

    too_extended = (

        price > e20 * 1.10

        or rsi14 > 76

        or day_change > .12
    )

    # Preferred entry zone
    if setup == "20EMA PULLBACK":

        entry_low = (
            e20 - .25 * atr14
        )

        entry_high = (
            e20 + .50 * atr14
        )

    elif setup == "10EMA CONTINUATION":

        entry_low = (
            e10 - .30 * atr14
        )

        entry_high = (
            e10 + .45 * atr14
        )

    else:

        entry_low = (
            prior20high
            - .20 * atr14
        )

        entry_high = (
            prior20high
            + .50 * atr14
        )

    inside_entry = (
        entry_low
        <= price
        <= entry_high
    )

    if too_extended:

        signal = "TOO EXTENDED"

    elif (
        score >= 90
        and entry_quality >= 12
        and inside_entry
    ):

        signal = "A+ SWING BUY"

    elif (
        score >= 85
        and entry_quality >= 10
        and inside_entry
    ):

        signal = "BUY"

    elif score >= 75:

        signal = "WATCH"

    else:

        signal = "AVOID"

    return {

        "signal": signal,

        "swing_score": score,

        "setup": setup,

        "price":
            round(price, 2),

        "entry_quality":
            entry_quality,

        "entry_low":
            round(entry_low, 2),

        "entry_high":
            round(entry_high, 2),

        "stop":
            round(stop, 2),

        "target1":
            round(target1, 2),

        "target2":
            round(target2, 2),

        "reward_risk":
            round(reward_risk, 2),

        "rsi14":
            round(rsi14, 1),

        "rvol":
            round(rvol, 2),

        "too_extended":
            too_extended,
    }

def score_swing_daily(
    stock_daily,
    spy_daily=None,
    qqq_daily=None,
    rs_percentile=None,
):
    """
    Daily swing-trade scorer.
    Returns setup, swing score, entry quality, preferred entry,
    stop, targets and BUY/WATCH/TOO EXTENDED/AVOID decision.

    BUY signals must pass independent risk-event, trend-health,
    distribution and cross-sectional leadership gates.
    """

    if stock_daily is None or stock_daily.empty:
        return None

    d = (
        stock_daily.copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(d) < 60:
        return None

    close = d["close"].astype(float)
    volume = d["volume"].astype(float)

    price = safe_float(close.iloc[-1])

    e10 = safe_float(
        ema(close, 10).iloc[-1],
        price
    )

    e20_series = ema(close, 20)
    s50_series = sma(close, 50)

    e20 = safe_float(
        e20_series.iloc[-1],
        price
    )

    s50 = safe_float(
        s50_series.iloc[-1],
        e20
    )

    rsi14 = safe_float(
        rsi_series(close, 14).iloc[-1],
        50
    )

    atr14 = safe_float(
        atr_series(d, 14).iloc[-1],
        0
    )

    avg20vol = safe_float(
        volume.iloc[:-1].tail(20).mean(),
        0
    )

    rvol = (
        float(volume.iloc[-1]) / avg20vol
        if avg20vol > 0
        else 0
    )

    prior20high = safe_float(
        d["high"].shift(1).tail(20).max(),
        price
    )

    high252 = safe_float(
        d["high"].tail(min(252, len(d))).max(),
        price
    )

    prior_close = safe_float(
        close.iloc[-2],
        price
    )

    day_change = (
        price / prior_close - 1
        if prior_close
        else 0
    )

    latest_open = safe_float(
        d["open"].iloc[-1],
        price,
    )

    open_gap = (
        latest_open / prior_close - 1
        if prior_close
        else 0
    )

    risk_event = downside_event_risk(
        d,
        e20=e20,
    )

    distribution_days = distribution_day_count(
        d,
        lookback=20,
    )

    e20_rising = (
        len(e20_series) >= 6
        and e20 > safe_float(e20_series.iloc[-6], e20)
    )

    s50_rising = (
        len(s50_series) >= 11
        and s50 > safe_float(s50_series.iloc[-11], s50)
    )

    trend_health = bool(
        e20_rising
        and s50_rising
    )

    rs_percentile_value = (
        None
        if rs_percentile is None or pd.isna(rs_percentile)
        else max(0.0, min(1.0, safe_float(rs_percentile)))
    )

    # ========================================================
    # SETUP TYPE
    # ========================================================

    if risk_event["risk_flag"]:
        setup = "NEGATIVE CATALYST GAP"

    elif (
        price >= prior20high
        and rvol >= 1.4
        and day_change < 0.12
    ):
        setup = "BREAKOUT"

    elif day_change >= 0.12:
        setup = "GAP MOMENTUM"

    elif (
        price > e20 > s50
        and abs(price - e20)
        <= max(0.8 * atr14, 0.03 * price)
    ):
        setup = "20EMA PULLBACK"

    elif (
        price > e10 > e20 > s50
        and price <= e10 + 0.7 * atr14
    ):
        setup = "10EMA CONTINUATION"

    elif (
        price > e20 > s50
        and price >= 0.97 * prior20high
    ):
        setup = "BASE / NEAR BREAKOUT"

    else:
        setup = "TREND MOMENTUM"

    # ========================================================
    # TREND + RELATIVE STRENGTH — 20 POINTS
    # ========================================================

    trend_score = 0

    if price > e20 > s50:
        trend_score += 8
    elif price > s50:
        trend_score += 4

    if e20_rising:
        trend_score += 2

    if s50_rising:
        trend_score += 2

    stock20 = 0
    stock60 = 0

    if len(close) >= 21:
        old = safe_float(close.iloc[-21], price)
        if old:
            stock20 = price / old - 1

    if len(close) >= 61:
        old60 = safe_float(close.iloc[-61], price)
        if old60:
            stock60 = price / old60 - 1

    spy20 = 0
    spy60 = 0

    if (
        spy_daily is not None
        and not spy_daily.empty
        and len(spy_daily) >= 21
    ):
        spy = (
            spy_daily.copy()
            .sort_values("timestamp")
        )

        spy_close = spy["close"].astype(float)

        old_spy = safe_float(
            spy_close.iloc[-21],
            spy_close.iloc[-1]
        )

        if old_spy:
            spy20 = (
                float(spy_close.iloc[-1])
                / old_spy
                - 1
            )

        if len(spy_close) >= 61:
            old_spy60 = safe_float(
                spy_close.iloc[-61],
                spy_close.iloc[-1],
            )
            if old_spy60:
                spy60 = float(spy_close.iloc[-1]) / old_spy60 - 1

    if stock20 > spy20:
        trend_score += 4

    if stock60 > spy60:
        trend_score += 4

    trend_score = min(20, trend_score)

    # ========================================================
    # ACCUMULATION / VOLUME — 15 POINTS
    # ========================================================

    accumulation_score = 0

    if rvol >= 1.5:
        accumulation_score += 7
    elif rvol >= 1.2:
        accumulation_score += 5

    recent = d.tail(20)

    up_days = recent["close"] > recent["open"]

    up_volume = (
        safe_float(
            recent.loc[
                up_days,
                "volume"
            ].mean(),
            0
        )
        if up_days.any()
        else 0
    )

    down_volume = (
        safe_float(
            recent.loc[
                ~up_days,
                "volume"
            ].mean(),
            0
        )
        if (~up_days).any()
        else 0
    )

    if up_volume > down_volume * 1.15:
        accumulation_score += 8
    elif up_volume > down_volume:
        accumulation_score += 5

    if distribution_days >= 6:
        accumulation_score -= 5
    elif distribution_days == 5:
        accumulation_score -= 3

    accumulation_score = max(
        0,
        min(
            15,
            accumulation_score
        )
    )

    # ========================================================
    # ENTRY QUALITY — 15 POINTS
    # ========================================================

    entry_quality = 0

    dist20 = (
        price / e20 - 1
        if e20
        else 0
    )

    if 0 <= dist20 <= 0.04:
        entry_quality += 6
    elif 0.04 < dist20 <= 0.07:
        entry_quality += 4
    elif -0.02 <= dist20 < 0:
        entry_quality += 3

    if 52 <= rsi14 <= 68:
        entry_quality += 4
    elif 45 <= rsi14 < 52:
        entry_quality += 2
    elif 68 < rsi14 <= 72:
        entry_quality += 2

    if setup in [
        "20EMA PULLBACK",
        "10EMA CONTINUATION"
    ]:
        entry_quality += 3

    elif setup == "BREAKOUT":
        if day_change <= 0.08:
            entry_quality += 3

    elif setup == "BASE / NEAR BREAKOUT":
        entry_quality += 2

    if abs(day_change) <= 0.06:
        entry_quality += 2
    elif abs(day_change) <= 0.10:
        entry_quality += 1

    entry_quality = min(
        15,
        entry_quality
    )

    # ========================================================
    # MARKET REGIME — 10 POINTS
    # ========================================================

    market_score = 5

    if (
        spy_daily is not None
        and not spy_daily.empty
        and len(spy_daily) >= 50
    ):
        spy = (
            spy_daily.copy()
            .sort_values("timestamp")
        )

        spy_close = spy["close"].astype(float)

        spy_e20 = safe_float(
            ema(spy_close, 20).iloc[-1],
            spy_close.iloc[-1]
        )

        spy_s50 = safe_float(
            sma(spy_close, 50).iloc[-1],
            spy_e20
        )

        market_score = 0

        if spy_close.iloc[-1] > spy_e20:
            market_score += 3

        if spy_close.iloc[-1] > spy_s50:
            market_score += 2

        if spy_e20 > spy_s50:
            market_score += 2

        if (
            qqq_daily is not None
            and not qqq_daily.empty
            and len(qqq_daily) >= 50
        ):
            qqq = qqq_daily.copy().sort_values("timestamp")
            qqq_close = qqq["close"].astype(float)
            qqq_e20 = safe_float(
                ema(qqq_close, 20).iloc[-1],
                qqq_close.iloc[-1],
            )
            qqq_s50 = safe_float(
                sma(qqq_close, 50).iloc[-1],
                qqq_e20,
            )

            if qqq_close.iloc[-1] > qqq_e20:
                market_score += 2

            if qqq_e20 > qqq_s50:
                market_score += 1

    # ========================================================
    # SETUP QUALITY — 10 POINTS
    # ========================================================

    setup_scores = {
        "20EMA PULLBACK": 10,
        "BREAKOUT": 9,
        "10EMA CONTINUATION": 9,
        "BASE / NEAR BREAKOUT": 8,
        "GAP MOMENTUM": 4,
        "NEGATIVE CATALYST GAP": 0,
        "TREND MOMENTUM": 5,
    }

    setup_score = setup_scores.get(
        setup,
        5
    )

    # ========================================================
    # ENTRY ZONE
    # ========================================================

    if setup == "20EMA PULLBACK":

        entry_low = max(
            e20 - 0.25 * atr14,
            0.01
        )

        entry_high = (
            e20 + 0.50 * atr14
        )

    elif setup == "10EMA CONTINUATION":

        entry_low = max(
            e10 - 0.30 * atr14,
            0.01
        )

        entry_high = (
            e10 + 0.45 * atr14
        )

    elif setup in [
        "BREAKOUT",
        "BASE / NEAR BREAKOUT"
    ]:

        entry_low = max(
            prior20high - 0.20 * atr14,
            0.01
        )

        entry_high = (
            prior20high + 0.50 * atr14
        )

    else:

        entry_low = max(
            e20,
            price - 0.50 * atr14
        )

        entry_high = (
            price + 0.25 * atr14
        )

    # ========================================================
    # STOP / TARGETS
    # ========================================================

    structural_stop = (
        e20 - 0.50 * atr14
    )

    atr_stop = (
        price - 2.0 * atr14
    )

    stop = max(
        min(structural_stop, s50),
        atr_stop
    )

    if stop >= price:

        stop = (
            price
            - max(
                atr14,
                price * 0.025
            )
        )

    stop = max(
        stop,
        0.01
    )

    risk = max(
        price - stop,
        0.01
    )

    target1 = (
        price + 2.0 * risk
    )

    target2 = (
        price + 3.0 * risk
    )

    realistic_target = target1

    if (
        high252 > price
        and high252 < target1
    ):
        realistic_target = high252

    reward_risk = (
        max(
            realistic_target - price,
            0
        )
        / risk
    )

    # ========================================================
    # REWARD / RISK SCORE — 10 POINTS
    # ========================================================

    if reward_risk >= 3:
        rr_score = 10

    elif reward_risk >= 2.5:
        rr_score = 9

    elif reward_risk >= 2:
        rr_score = 8

    elif reward_risk >= 1.7:
        rr_score = 5

    elif reward_risk >= 1.4:
        rr_score = 3

    else:
        rr_score = 0

    # ========================================================
    # LIQUIDITY / VOLATILITY — 5 POINTS
    # ========================================================

    atr_pct = (
        atr14 / price
        if price
        else 0
    )

    adv = safe_float(
        (
            d["close"]
            * d["volume"]
        )
        .tail(20)
        .mean(),
        0
    )

    vol_liq_score = 0

    if 0.015 <= atr_pct <= 0.055:
        vol_liq_score += 3
    elif 0.01 <= atr_pct <= 0.075:
        vol_liq_score += 2

    if adv >= 100_000_000:
        vol_liq_score += 2
    elif adv >= 20_000_000:
        vol_liq_score += 1

    # ========================================================
    # CROSS-SECTIONAL LEADERSHIP / CATALYST RISK
    # ========================================================

    if rs_percentile_value is None:
        leadership_score = 5
    elif rs_percentile_value >= 0.90:
        leadership_score = 10
    elif rs_percentile_value >= 0.80:
        leadership_score = 8
    elif rs_percentile_value >= 0.70:
        leadership_score = 6
    elif rs_percentile_value >= 0.50:
        leadership_score = 4
    else:
        leadership_score = 1

    catalyst_score = (
        0
        if risk_event["risk_flag"]
        else 2.5
    )

    # ========================================================
    # TOTAL SWING SCORE
    # ========================================================

    swing_score = (
        trend_score
        + accumulation_score
        + entry_quality
        + market_score
        + leadership_score
        + setup_score
        + rr_score
        + catalyst_score
        + vol_liq_score
    )

    if risk_event["risk_flag"]:
        swing_score -= 25

    swing_score = round(
        max(
            0,
            min(
                100,
                swing_score
            )
        ),
        1
    )

    # ========================================================
    # HARD ANTI-CHASE FILTER
    # ========================================================

    positive_gap_extension = (
        open_gap >= 0.08
        or day_change >= 0.10
    )

    too_extended = (
        price > e20 * 1.10
        or rsi14 > 76
        or positive_gap_extension
    )

    inside_entry_zone = (
        entry_low
        <= price
        <= entry_high
    )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    leadership_buy_gate = (
        rs_percentile_value is None
        or rs_percentile_value >= MIN_RS_PERCENTILE_BUY
    )

    leadership_a_plus_gate = (
        rs_percentile_value is None
        or rs_percentile_value >= MIN_RS_PERCENTILE_A_PLUS
    )

    distribution_gate = (
        distribution_days <= MAX_DISTRIBUTION_DAYS_BUY
    )

    if risk_event["risk_flag"]:

        signal = "AVOID"

    elif too_extended:

        signal = "TOO EXTENDED"

    elif (
        swing_score >= 90
        and entry_quality >= 12
        and reward_risk >= 2
        and market_score >= 7
        and inside_entry_zone
        and trend_health
        and distribution_gate
        and leadership_a_plus_gate
    ):

        signal = "A+ SWING BUY"

    elif (
        swing_score >= 85
        and entry_quality >= 10
        and reward_risk >= 2
        and market_score >= 5
        and inside_entry_zone
        and trend_health
        and distribution_gate
        and leadership_buy_gate
    ):

        signal = "BUY"

    elif swing_score >= 75:

        signal = "WATCH"

    else:

        signal = "AVOID"

    return {
        "signal": signal,
        "swing_score": swing_score,
        "setup": setup,
        "price": round(price, 2),
        "entry_quality": round(
            entry_quality,
            1
        ),
        "entry_low": round(
            entry_low,
            2
        ),
        "entry_high": round(
            entry_high,
            2
        ),
        "stop": round(
            stop,
            2
        ),
        "target1": round(
            target1,
            2
        ),
        "target2": round(
            target2,
            2
        ),
        "reward_risk": round(
            reward_risk,
            2
        ),
        "rsi14": round(
            rsi14,
            1
        ),
        "rvol": round(
            rvol,
            2
        ),
        "too_extended": bool(
            too_extended
        ),
        "inside_entry_zone": bool(
            inside_entry_zone
        ),
        "risk_flag": bool(
            risk_event["risk_flag"]
        ),
        "risk_reason": risk_event[
            "risk_reason"
        ],
        "gap_down_pct": risk_event[
            "gap_down_pct"
        ],
        "event_day_change_pct": risk_event[
            "event_day_change_pct"
        ],
        "event_days_ago": risk_event[
            "event_days_ago"
        ],
        "event_repaired": bool(
            risk_event["event_repaired"]
        ),
        "open_gap_pct": round(
            open_gap * 100,
            2
        ),
        "day_change_pct": round(
            day_change * 100,
            2
        ),
        "trend_health": bool(
            trend_health
        ),
        "e20_rising": bool(
            e20_rising
        ),
        "s50_rising": bool(
            s50_rising
        ),
        "distribution_days": int(
            distribution_days
        ),
        "market_score": round(
            market_score,
            1
        ),
        "leadership_percentile": (
            None
            if rs_percentile_value is None
            else round(rs_percentile_value * 100, 1)
        ),
        "leadership_score": round(
            leadership_score,
            1
        ),
        "trend_score": round(
            trend_score,
            1
        ),
        "accumulation_score": round(
            accumulation_score,
            1
        ),
    }
