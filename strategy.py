import numpy as np
import pandas as pd

def prefilter_daily(daily_bars, min_price=3.0, min_avg_dollar_volume=5_000_000,
                    min_avg_volume=300_000, keep=150):
    if daily_bars.empty:
        return pd.DataFrame()
    d = daily_bars.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    rows = []
    for sym, g in d.groupby("symbol"):
        g = g.sort_values("timestamp").tail(25)
        if len(g) < 6:
            continue
        latest = g.iloc[-1]
        prior = g.iloc[:-1]
        price = float(latest["close"])
        avg_vol = float(prior["volume"].tail(20).mean())
        avg_price = float(prior["close"].tail(20).mean())
        avg_dollar = avg_vol * avg_price
        if price < min_price or avg_vol < min_avg_volume or avg_dollar < min_avg_dollar_volume:
            continue
        prev_close = float(prior["close"].iloc[-1])
        change_pct = (price / prev_close - 1) * 100 if prev_close else 0
        vol_ratio = float(latest["volume"]) / max(avg_vol, 1)
        pre_score = min(max(change_pct, -5), 12) * 5 + min(vol_ratio, 6) * 12
        rows.append({
            "symbol": sym, "price": price,
            "daily_change_pct": change_pct,
            "daily_volume_ratio": vol_ratio,
            "avg_dollar_volume": avg_dollar,
            "prefilter_score": pre_score
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out[(out["daily_change_pct"] >= -1) | (out["daily_volume_ratio"] >= 1.4)]
    return out.sort_values(
        ["prefilter_score","daily_volume_ratio","daily_change_pct"],
        ascending=False
    ).head(keep).reset_index(drop=True)

def add_intraday_features(df):
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    typical = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap"] = (typical*d["volume"]).cumsum() / d["volume"].replace(0,np.nan).cumsum()
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
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0,np.nan)
    d["rsi"] = 100 - (100/(1+rs))
    return d

def prepare_intraday(day, spy_day=None, avg_daily_volume=None):
    d = add_intraday_features(day)
    if len(d) < 20:
        return d
    d["orb_high"] = d.iloc[:15]["high"].max()
    d["stock_ret"] = d["close"]/d["open"].iloc[0]-1
    elapsed = np.arange(1, len(d)+1)
    if avg_daily_volume and avg_daily_volume > 0:
        expected = avg_daily_volume*np.minimum(elapsed/390.0,1)
        d["rel_volume"] = d["volume"].cumsum()/np.maximum(expected,1)
    else:
        d["rel_volume"] = 1.0
    if spy_day is not None and not spy_day.empty:
        s = spy_day.sort_values("timestamp").copy()
        s["spy_ret"] = s["close"]/s["open"].iloc[0]-1
        d = d.merge(s[["timestamp","spy_ret"]], on="timestamp", how="left")
        d["spy_ret"] = d["spy_ret"].ffill().fillna(0)
        d["rs"] = d["stock_ret"]-d["spy_ret"]
    else:
        d["rs"] = d["stock_ret"]
    return d

def classify(r, avg_dollar_volume=0):
    score = 0
    reasons = []
    momentum = float(r.get("stock_ret",0))*100
    if momentum >= 3: score += 15; reasons.append("strong momentum")
    elif momentum >= 1.5: score += 11
    elif momentum >= .5: score += 6
    if r["close"] > r["vwap"]*1.002: score += 15; reasons.append("above VWAP")
    elif r["close"] > r["vwap"]: score += 9
    if r["close"] > r["orb_high"]*1.001: score += 15; reasons.append("opening-range breakout")
    elif r["close"] >= r["orb_high"]*.997: score += 7
    rv = float(r.get("rel_volume",0))
    if rv >= 2.5: score += 15; reasons.append("2.5x+ relative volume")
    elif rv >= 1.8: score += 12
    elif rv >= 1.3: score += 7
    if r["close"] > r["ema9"] > r["ema20"]: score += 10; reasons.append("bullish EMA structure")
    elif r["close"] > r["ema20"]: score += 5
    rs = float(r.get("rs",0))
    if rs >= .015: score += 10; reasons.append("strong vs SPY")
    elif rs >= .0075: score += 7
    elif rs >= 0: score += 3
    rsi = float(r["rsi"]) if pd.notna(r["rsi"]) else 50
    if 52 <= rsi <= 72: score += 10
    elif 45 <= rsi < 52: score += 5
    elif 72 < rsi <= 78: score += 4
    elif rsi > 82: score -= 10; reasons.append("overextended RSI")
    elif rsi > 78: score -= 5
    if avg_dollar_volume >= 100_000_000: score += 5
    elif avg_dollar_volume >= 25_000_000: score += 3
    if r["close"] > r["vwap"]*1.06: score -= 10; reasons.append("too extended from VWAP")
    elif r["close"] > r["vwap"]*1.04: score -= 5
    score = max(0,min(100,int(score)))
    hard_buy = (
        score >= 82 and r["close"] > r["vwap"]
        and r["close"] >= r["orb_high"]*.997
        and rv >= 1.3 and rs >= 0 and 45 <= rsi <= 80
    )
    signal = "BUY" if hard_buy else ("WATCH" if score >= 68 else "NO BUY")
    return score, signal, reasons
