import os
from twilio.rest import Client

def sms_configured():
    required = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_FROM_NUMBER",
        "ALERT_TO_NUMBER",
    ]
    return all(os.getenv(k) for k in required)

def send_sms(body):
    if not sms_configured():
        raise RuntimeError(
            "SMS is not configured. Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER, and ALERT_TO_NUMBER to Streamlit Secrets."
        )
    client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN"),
    )
    msg = client.messages.create(
        body=body,
        from_=os.getenv("TWILIO_FROM_NUMBER"),
        to=os.getenv("ALERT_TO_NUMBER"),
    )
    return msg.sid

def build_buy_message(row):
    return (
        f"BUY ALERT: {row['symbol']} | Score {int(row['score'])}/100 | "
        f"Price ${float(row['price']):.2f} | Change {float(row['change_today_%']):.2f}% | "
        f"Rel Vol {float(row['rel_volume']):.2f}x. "
        f"Research signal only; verify entry and risk before trading."
    )

def build_sell_message(symbol, price, reason):
    return (
        f"SELL ALERT: {symbol} | Price ${float(price):.2f} | {reason}. "
        f"Research signal only; verify before trading."
    )
