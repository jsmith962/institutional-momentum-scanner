import numpy as np
import pandas as pd

def add_features(df):
    d = df.copy()
    d["typical"] = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (d["typical"] * d["volume"]).cumsum() / d["volume"].replace(0, np.nan).cumsum()
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    tr = pd.concat([
        d["high"]-d["low"],
        (d["high"]-d["close"].shift()).abs(),
        (d["low"]-d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100/(1+rs))
    return d

def prepare_day(day, spy_day=None):
    d = add_features(day.sort_values("timestamp").reset_index(drop=True))
    if len(d) < 30:
        return d
    d["minute"] = np.arange(len(d))
    d["orb_high"] = d.loc[:14, "high"].max()
    d["orb_low"] = d.loc[:14, "low"].min()
    base = d["volume"].iloc[:15].mean()
    d["rel_volume"] = d["volume"] / max(base, 1)
    if spy_day is not None and not spy_day.empty:
        s = spy_day.sort_values("timestamp").reset_index(drop=True)
        s["spy_ret"] = s["close"] / s["close"].iloc[0] - 1
        d["stock_ret"] = d["close"] / d["close"].iloc[0] - 1
        d = d.merge(s[["timestamp","spy_ret"]], on="timestamp", how="left")
        d["rs"] = d["stock_ret"] - d["spy_ret"]
    else:
        d["rs"] = 0.0
    return d

def score_row(r):
    score = 0
    reasons = []
    if r["close"] > r["vwap"] * 1.001:
        score += 15; reasons.append("above VWAP")
    if r["close"] > r["orb_high"]:
        score += 15; reasons.append("opening-range breakout")
    if r["rel_volume"] >= 2:
        score += 15; reasons.append("2x+ volume")
    elif r["rel_volume"] >= 1.5:
        score += 10
    if r["close"] > r["ema9"] > r["ema20"]:
        score += 10; reasons.append("bullish EMA structure")
    elif r["close"] > r["ema20"]:
        score += 5
    if r["rs"] >= .01:
        score += 15; reasons.append("strong vs SPY")
    elif r["rs"] >= .005:
        score += 10
    if 50 <= r["rsi"] <= 72:
        score += 10
    elif r["rsi"] > 80:
        score -= 10; reasons.append("overextended")
    elif r["rsi"] > 72:
        score -= 5
    if r["atr"] > 0 and r["close"] > 0:
        score += 5
    return max(0, min(100, int(score))), reasons

def signal(r, min_score=85):
    score, reasons = score_row(r)
    buy = (
        score >= min_score and
        r["close"] > r["vwap"] and
        r["close"] > r["orb_high"] and
        r["rel_volume"] >= 1.5 and
        r["rs"] >= .005 and
        45 <= r["rsi"] <= 78
    )
    return score, ("BUY" if buy else ("WATCH" if score >= 70 else "NO BUY")), reasons
