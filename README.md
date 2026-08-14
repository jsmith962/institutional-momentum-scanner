# Institutional Swing Scanner v3.2.1

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

## v3.2 production-equivalent swing backtester

- Reconstructs the same daily scorer, SPY/QQQ regime rules, relative-strength
  gate, catalyst protection, intraday classifier and confirmation gate used by
  the live scanner.
- Builds the current daily candle only from minute bars visible at the selected
  historical scan time, preventing the completed day's close from leaking into
  an earlier decision.
- Fills confirmed BUY signals on the next available minute rather than the bar
  that created the signal.
- Simulates a cash-limited portfolio with stop-based sizing, configurable risk,
  maximum open positions, slippage and fees.
- Holds positions across sessions, fills overnight gaps at the opening price,
  sells half at 2R, moves the remaining stop to breakeven, targets 3R and exits
  on the next session after a completed close below the 20-day EMA.
- Produces an equity curve, drawdown, profit factor, average expectancy in R,
  complete trade ledger and historical signal audit.
- Uses conservative stop-first handling when one minute touches both a stop and
  a target and the true intrabar sequence is unknown.

The v3.2 backtest evaluates only the comparison symbols entered by the user.
It does not yet reconstruct the full point-in-time U.S. stock universe and can
still contain selection or survivorship bias. A small trade sample is not
evidence of profitability.

## v3.2.1 signal diagnostics

- Adds a BUY-confirmation funnel from all evaluated candidates through the
  daily score, complete daily BUY gates, intraday confirmation and completed
  simulated trades.
- Counts the most common failed BUY gates and reports their failure rates.
- Ranks the closest non-BUY observation for each symbol by the fewest failed
  gates, while clearly labeling near misses as non-recommendations.
- Adds every daily and intraday gate result, plus readable failed-gate reasons,
  to the historical signal audit.
- Replaces the ambiguous “daily and intraday rules are aligned” audit text with
  the actual daily outcome or confirmation status.
- Preserves every v3.2 BUY threshold and all portfolio simulation behavior.

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

## Backtest interpretation

Use at least ten liquid comparison symbols, realistic slippage and a long enough
period to generate many independent trades. The rule score is not a predicted
win probability, and all displayed results remain hypothetical.
