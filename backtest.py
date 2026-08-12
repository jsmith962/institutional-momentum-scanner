import pandas as pd
import numpy as np
from strategy import prepare_intraday, classify

def backtest(bars, spy_bars, starting_capital=2000, risk_pct=.01,
             reward_risk=2.0, commission=0.0):
    capital = float(starting_capital)
    equity, trades = [], []

    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["timestamp"], utc=True).dt.date
    spy_bars = spy_bars.copy()
    spy_bars["date"] = pd.to_datetime(spy_bars["timestamp"], utc=True).dt.date

    for date, day_all in bars.groupby("date"):
        spy_day = spy_bars[spy_bars["date"] == date]
        for symbol, day in day_all.groupby("symbol"):
            d = prepare_intraday(day, spy_day)
            if len(d) < 30:
                continue
            in_trade = False
            for i in range(15, len(d)):
                r = d.iloc[i]
                if in_trade:
                    hit_stop = r["low"] <= stop
                    hit_target = r["high"] >= target
                    if hit_stop or hit_target:
                        exit_price = stop if hit_stop else target
                        pnl = (exit_price-entry)*qty - commission
                        capital += pnl
                        trades.append({
                            "date":date,"symbol":symbol,"entry_time":entry_time,
                            "exit_time":r["timestamp"],"entry":entry,"exit":exit_price,
                            "qty":qty,"pnl":pnl,"reason":"stop" if hit_stop else "target"
                        })
                        in_trade = False
                        break
                else:
                    score, sig, _ = classify(r)
                    if sig == "BUY" and pd.notna(r["atr"]) and r["atr"] > 0:
                        entry = float(r["close"])
                        stop = entry - 1.25*float(r["atr"])
                        per_share_risk = entry-stop
                        qty = min(int((capital*risk_pct)/max(per_share_risk,.01)),
                                  int(capital/entry))
                        if qty <= 0:
                            continue
                        target = entry + reward_risk*per_share_risk
                        entry_time = r["timestamp"]
                        in_trade = True
            if in_trade:
                last = d.iloc[-1]
                exit_price = float(last["close"])
                pnl = (exit_price-entry)*qty - commission
                capital += pnl
                trades.append({
                    "date":date,"symbol":symbol,"entry_time":entry_time,
                    "exit_time":last["timestamp"],"entry":entry,"exit":exit_price,
                    "qty":qty,"pnl":pnl,"reason":"EOD"
                })
        equity.append({"date":date,"equity":capital})

    trades_df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity)
    if eq.empty:
        return {"trades":trades_df,"equity":eq,"stats":{}}

    peak = eq["equity"].cummax()
    dd = eq["equity"]/peak - 1
    n = len(trades_df)
    wins = trades_df[trades_df["pnl"]>0] if n else pd.DataFrame()
    losses = trades_df[trades_df["pnl"]<0] if n else pd.DataFrame()
    pf = wins["pnl"].sum()/abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() else np.inf

    stats = {
        "starting_capital":round(starting_capital,2),
        "ending_capital":round(capital,2),
        "total_return_pct":round((capital/starting_capital-1)*100,2),
        "trades":n,
        "win_rate_pct":round(len(wins)/n*100,2) if n else 0,
        "profit_factor":round(pf,2) if np.isfinite(pf) else "inf",
        "avg_trade_dollars":round(trades_df["pnl"].mean(),2) if n else 0,
        "max_drawdown_pct":round(dd.min()*100,2)
    }
    return {"trades":trades_df,"equity":eq,"stats":stats}
