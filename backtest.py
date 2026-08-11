import pandas as pd
import numpy as np
from strategy import prepare_day, signal

def backtest(bars, spy_bars, starting_capital=2000, risk_pct=.01, reward_risk=2.0,
             max_daily_loss=.03, commission=0.0):
    capital = float(starting_capital)
    equity = []
    trades = []
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["timestamp"], utc=True).dt.date
    spy_bars = spy_bars.copy()
    spy_bars["date"] = pd.to_datetime(spy_bars["timestamp"], utc=True).dt.date

    for date, day_all in bars.groupby("date"):
        spy_day = spy_bars[spy_bars["date"] == date]
        daily_start = capital
        day_trades = 0
        for symbol, day in day_all.groupby("symbol"):
            d = prepare_day(day, spy_day)
            if len(d) < 30: continue
            in_trade = False
            entry = stop = target = qty = None
            entry_time = None
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
                            "date": date, "symbol": symbol, "entry_time": entry_time,
                            "exit_time": r["timestamp"], "entry": entry,
                            "exit": exit_price, "qty": qty, "pnl": pnl,
                            "return_pct": pnl/(entry*qty) if entry*qty else 0,
                            "reason": "stop" if hit_stop else "target"
                        })
                        in_trade = False
                        day_trades += 1
                        break
                else:
                    score, sig, reasons = signal(r)
                    if sig == "BUY" and r["atr"] > 0:
                        entry = float(r["close"])
                        stop = entry - 1.25*float(r["atr"])
                        risk_per_share = entry-stop
                        if risk_per_share <= 0: continue
                        risk_dollars = capital*risk_pct
                        qty = int(risk_dollars/risk_per_share)
                        qty = min(qty, int(capital/entry))
                        if qty <= 0: continue
                        target = entry + reward_risk*risk_per_share
                        entry_time = r["timestamp"]
                        in_trade = True
            # close any open position at final bar
            if in_trade:
                r = d.iloc[-1]
                exit_price = float(r["close"])
                pnl = (exit_price-entry)*qty - commission
                capital += pnl
                trades.append({
                    "date": date, "symbol": symbol, "entry_time": entry_time,
                    "exit_time": r["timestamp"], "entry": entry,
                    "exit": exit_price, "qty": qty, "pnl": pnl,
                    "return_pct": pnl/(entry*qty) if entry*qty else 0,
                    "reason": "EOD"
                })
        if daily_start > 0 and (capital/daily_start-1) <= -max_daily_loss:
            pass
        equity.append({"date": date, "equity": capital})

    trades_df = pd.DataFrame(trades)
    eq = pd.DataFrame(equity)
    if eq.empty:
        return {"trades": trades_df, "equity": eq, "stats": {}}

    peak = eq["equity"].cummax()
    drawdown = eq["equity"]/peak - 1
    n = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0] if n else pd.DataFrame()
    losses = trades_df[trades_df["pnl"] < 0] if n else pd.DataFrame()
    profit_factor = wins["pnl"].sum()/abs(losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() else np.inf
    days = max(1, (eq["date"].iloc[-1] - eq["date"].iloc[0]).days)
    cagr = (capital/starting_capital)**(365/days)-1 if capital > 0 else -1
    stats = {
        "starting_capital": starting_capital,
        "ending_capital": round(capital,2),
        "total_return_pct": round((capital/starting_capital-1)*100,2),
        "cagr_pct": round(cagr*100,2),
        "trades": n,
        "win_rate_pct": round(len(wins)/n*100,2) if n else 0,
        "profit_factor": round(profit_factor,2) if np.isfinite(profit_factor) else "inf",
        "avg_trade_dollars": round(trades_df["pnl"].mean(),2) if n else 0,
        "max_drawdown_pct": round(drawdown.min()*100,2),
    }
    return {"trades": trades_df, "equity": eq, "stats": stats}
