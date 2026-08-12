import pandas as pd,numpy as np
from strategy import prepare_intraday,classify

def backtest(bars,spy_bars,starting_capital=2000,risk_pct=.01,reward_risk=2.0,commission=0.0):
    capital=float(starting_capital); equity=[]; trades=[]
    bars=bars.copy(); bars["date"]=pd.to_datetime(bars.timestamp,utc=True).dt.date
    spy_bars=spy_bars.copy(); spy_bars["date"]=pd.to_datetime(spy_bars.timestamp,utc=True).dt.date
    for date,day_all in bars.groupby("date"):
        spy=spy_bars[spy_bars.date==date]
        for sym,day in day_all.groupby("symbol"):
            d=prepare_intraday(day,spy)
            if len(d)<30: continue
            active=False
            for i in range(15,len(d)):
                r=d.iloc[i]
                if active:
                    hs=r.low<=stop; ht=r.high>=target
                    if hs or ht:
                        xp=stop if hs else target; pnl=(xp-entry)*qty-commission; capital+=pnl
                        trades.append({"date":date,"symbol":sym,"entry_time":etime,"exit_time":r.timestamp,"entry":entry,"exit":xp,"qty":qty,"pnl":pnl,"reason":"stop" if hs else "target"})
                        active=False; break
                else:
                    score,sig,_=classify(r)
                    if sig=="BUY" and pd.notna(r.atr) and r.atr>0:
                        entry=float(r.close); stop=entry-1.25*float(r.atr); risk=entry-stop
                        qty=min(int((capital*risk_pct)/max(risk,.01)),int(capital/entry))
                        if qty<=0: continue
                        target=entry+reward_risk*risk; etime=r.timestamp; active=True
            if active:
                r=d.iloc[-1]; xp=float(r.close); pnl=(xp-entry)*qty-commission; capital+=pnl
                trades.append({"date":date,"symbol":sym,"entry_time":etime,"exit_time":r.timestamp,"entry":entry,"exit":xp,"qty":qty,"pnl":pnl,"reason":"EOD"})
        equity.append({"date":date,"equity":capital})
    t=pd.DataFrame(trades); e=pd.DataFrame(equity)
    if e.empty:return {"trades":t,"equity":e,"stats":{}}
    peak=e.equity.cummax(); dd=e.equity/peak-1; n=len(t)
    wins=t[t.pnl>0] if n else pd.DataFrame(); losses=t[t.pnl<0] if n else pd.DataFrame()
    pf=wins.pnl.sum()/abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() else np.inf
    return {"trades":t,"equity":e,"stats":{
        "starting_capital":round(starting_capital,2),"ending_capital":round(capital,2),
        "total_return_pct":round((capital/starting_capital-1)*100,2),"trades":n,
        "win_rate_pct":round(len(wins)/n*100,2) if n else 0,
        "profit_factor":round(pf,2) if np.isfinite(pf) else "inf",
        "avg_trade_dollars":round(t.pnl.mean(),2) if n else 0,
        "max_drawdown_pct":round(dd.min()*100,2)}}
