# Institutional Momentum Scanner v2

Full-market scanner + historical backtester + $2,000 performance simulator.

## Components
- `app.py` — Streamlit dashboard
- `market_data.py` — Alpaca asset discovery and historical data
- `strategy.py` — feature engineering and 0–100 scoring
- `backtest.py` — event-driven historical simulation
- `requirements.txt`
- `.env.example`

## Safety
This version is **research/paper-trading only**. It does not submit live orders.

## Full-market design
The app dynamically discovers active US equities from Alpaca's assets endpoint and filters to tradable, exchange-listed stocks. It then downloads historical bars in batches.

## Backtest model
The default strategy is intentionally conservative:
- Universe liquidity filter
- 1-minute intraday bars
- Opening-range breakout after the first 15 minutes
- Above VWAP
- Relative-volume confirmation
- EMA trend confirmation
- Relative-strength confirmation against SPY
- RSI extension penalty
- Entry after confirmation
- Fixed fractional risk per trade
- ATR-based stop
- Risk/reward target
- One position at a time
- Daily loss guard
- No overnight holding in the default backtest

## $2,000 simulator
Set starting capital to $2,000. The report calculates:
- Ending equity
- Total return
- CAGR
- Win rate
- Profit factor
- Average trade
- Max drawdown
- Number of trades
- Benchmark comparison vs SPY

## Setup
1. Create Alpaca paper/data API keys.
2. Copy `.env.example` to `.env`.
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

For serious testing, use a SIP/all-exchange data subscription. The basic/free feed may not provide complete consolidated U.S. market coverage.
