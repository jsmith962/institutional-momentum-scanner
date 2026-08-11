import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

ET = ZoneInfo("America/New_York")

def clients():
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
    return StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=True)

def full_us_equity_universe():
    _, trading = clients()
    assets = trading.get_all_assets(GetAssetsRequest(
        status=AssetStatus.ACTIVE,
        asset_class=AssetClass.US_EQUITY
    ))
    out = []
    for a in assets:
        if getattr(a, "tradable", False) and getattr(a, "exchange", None) and str(a.exchange) != "OTC":
            out.append(a.symbol)
    return sorted(set(out))

def get_bars(symbols, start, end, timeframe="1Min", feed=None):
    data, _ = clients()
    tf = TimeFrame(1, TimeFrameUnit.Minute) if timeframe == "1Min" else TimeFrame(1, TimeFrameUnit.Day)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end,
                           feed=feed or os.getenv("ALPACA_FEED", "sip"), adjustment="split")
    return data.get_stock_bars(req).df.reset_index()
