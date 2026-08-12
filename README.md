# Institutional Momentum Scanner v2.1

This update fixes the main weakness in v2: it no longer scans only the first symbols alphabetically.

New scan flow:
1. Discover all active tradable U.S. equities.
2. Pull recent daily bars in batches.
3. Remove illiquid / low-priced names.
4. Rank by momentum, volume surge and dollar liquidity.
5. Keep the strongest finalists.
6. Pull 1-minute bars only for those finalists.
7. Apply VWAP, opening-range breakout, relative volume, EMA trend, RSI and SPY-relative-strength scoring.
8. Show BUY / WATCH / NO BUY candidates.

Research / paper-testing only. No live orders.
