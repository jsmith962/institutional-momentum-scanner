import os,time,pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame,TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass,AssetStatus

def clients():
    k=os.getenv("ALPACA_API_KEY"); s=os.getenv("ALPACA_SECRET_KEY")
    if not k or not s: raise RuntimeError("Missing Alpaca credentials in Streamlit Secrets.")
    return StockHistoricalDataClient(k,s),TradingClient(k,s,paper=True)

def _bad_name(n):
    n=(n or "").lower()
    bad=["etf","etn","warrant","preferred","depositary share"," unit"," units"," rights"," leveraged"," inverse"," 2x"," 3x"," ultra"]
    return any(x in n for x in bad)

def eligible_us_equity_universe():
    _,t=clients()
    assets=t.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE,asset_class=AssetClass.US_EQUITY))
    rows=[]
    for a in assets:
        sym=getattr(a,"symbol",None); name=getattr(a,"name","") or ""; ex=str(getattr(a,"exchange",""))
        if not sym or not getattr(a,"tradable",False) or not ex or "OTC" in ex.upper(): continue
        if _bad_name(name): continue
        if any(x in sym for x in ["/","^","."]): continue
        rows.append({"symbol":sym,"name":name,"exchange":ex,"security_type":"Common-stock candidate","type_check":"PASS"})
    return pd.DataFrame(rows).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

def _tf(x): return TimeFrame(1,TimeFrameUnit.Day) if x=="1Day" else TimeFrame(1,TimeFrameUnit.Minute)

def get_bars(symbols,start,end,timeframe="1Min",feed=None):
    if not symbols:return pd.DataFrame()
    d,_=clients()
    req=StockBarsRequest(symbol_or_symbols=symbols,timeframe=_tf(timeframe),start=start,end=end,feed=feed or os.getenv("ALPACA_FEED","iex"),adjustment="split")
    x=d.get_stock_bars(req).df
    return x.reset_index() if not x.empty else pd.DataFrame()

def get_bars_batched(symbols,start,end,timeframe="1Day",feed=None,batch_size=200,pause_seconds=.12,progress_callback=None):
    fs=[]; total=len(symbols)
    for i in range(0,total,batch_size):
        try:
            x=get_bars(symbols[i:i+batch_size],start,end,timeframe,feed)
            if not x.empty: fs.append(x)
        except Exception: pass
        if progress_callback: progress_callback(min(i+batch_size,total),total)
        time.sleep(pause_seconds)
    return pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()
