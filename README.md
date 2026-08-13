# Institutional Swing Scanner v3.1

A Streamlit research scanner for liquid U.S. stocks. It combines daily swing
structure with live intraday confirmation and does not place orders.

## v3.1 safety and quality gates

- Blocks an opening gap down of 5% or more, or a one-day downside shock of 7%
  or more, instead of misclassifying it as a normal 20EMA pullback.
- Keeps the event blocked for at least three completed sessions and until price
  closes above both the event-day high and the 20-day EMA.
- Requires the 20-day EMA and 50-day SMA slopes to be rising for BUY.
- Rejects BUY when recent high-volume distribution days exceed the limit.
- Ranks 20/60-day relative strength across the finalist group and requires at
  least the 70th percentile for BUY and 85th percentile for A+ BUY.
- Confirms the market with both SPY and QQQ.
- Requires the daily signal and live intraday signal to agree; otherwise a
  daily BUY is downgraded to WATCH.
- Shows the exact failed gates and risk-event explanation in each trade card.
- Fixes BUY SMS messages to use `swing_score` and include entry/stop levels.

These rules are designed to reduce false positives. They do not guarantee a
profit or establish that the strategy has an edge. Validate with walk-forward,
out-of-sample and paper-trading results before using real money.

## Required Streamlit secrets

```toml
ALPACA_API_KEY = "..."
ALPACA_SECRET_KEY = "..."
```

Optional SMS alerts:

```toml
TWILIO_ACCOUNT_SID = "..."
TWILIO_AUTH_TOKEN = "..."
TWILIO_FROM_NUMBER = "+1..."
ALERT_TO_NUMBER = "+1..."
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Important backtest limitation

The current `$2,000 Backtester` simulates the intraday confirmation engine. It
does not yet reproduce the entire daily-plus-intraday v3.1 selection pipeline,
so its results must not be treated as validation of the live scanner.
