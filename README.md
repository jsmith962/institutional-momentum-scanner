# Institutional Momentum Scanner v2.3

Version 2.3 adds SMS alerts using Twilio.

## SMS behavior
- BUY texts: sent for qualifying BUY candidates when you run the scanner.
- SELL texts: evaluated only for symbols you enter under **Tracked positions**.
- Current SELL-risk rule: a tracked stock is `NO BUY` and has lost VWAP and/or its score has fallen below 60.
- This version does **not** place orders.

## Streamlit Secrets
Add these to the same Streamlit Secrets area where your Alpaca keys are stored:

TWILIO_ACCOUNT_SID = "..."
TWILIO_AUTH_TOKEN = "..."
TWILIO_FROM_NUMBER = "+1..."
ALERT_TO_NUMBER = "+1..."

Keep your existing Alpaca secrets too.

## Important limitation
Streamlit Community Cloud can hibernate inactive apps. SMS alerts in this version happen when the scanner actually runs. For truly continuous background alerts while the app is closed, deploy the scanner worker to an always-on scheduler/server in a later version.
