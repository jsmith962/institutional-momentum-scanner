import os
import time
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

def _keys():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials in Streamlit Secrets.")
    return key, secret

def clients():
    key, secret = _keys()
    return StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=True)

def full_us_equity_universe():
    _, trading = clients()
    assets = trading.get_all_assets(GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY
    ))
    symbols = []
    for a in assets:
        sym = getattr(a, "symbol", None)
        exch = str(getattr(a, "exchange", ""))
        if sym and getattr(a, "tradable", False) and exch and "OTC" not in exch.upper():
            symbols.append(sym)
    return sorted(set(symbols))

def _tf(name):
    return TimeFrame(1, TimeFrameUnit.Day) if name == "1Day" else TimeFrame(1, TimeFrameUnit.Minute)

def get_bars(symbols, start, end, timeframe="1Min", feed=None):
    if not symbols:
        return pd.DataFrame()
    data, _ = clients()
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=_tf(timeframe),
        start=start,
        end=end,
        feed=feed or os.getenv("ALPACA_FEED", "iex"),
        adjustment="split"
    )
    df = data.get_stock_bars(req).df
    return df.reset_index() if not df.empty else pd.DataFrame()

def get_bars_batched(symbols, start, end, timeframe="1Day", feed=None,
                     batch_size=200, pause_seconds=.12, progress_callback=None):
    frames = []
    total = len(symbols)
    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df = get_bars(batch, start, end, timeframe, feed)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
        if progress_callback:
            progress_callback(min(i + len(batch), total), total)
        time.sleep(pause_seconds)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
