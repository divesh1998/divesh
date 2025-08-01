import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import os
from PIL import Image

st.set_page_config(page_title="📈 Divesh Market Zone", layout="wide")
st.title("📈 Divesh Market Zone")

# Create save folder
if not os.path.exists("saved_charts"):
    os.makedirs("saved_charts")

# --- Symbols including NSE stocks and NIFTY 50 ---
symbols = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "NIFTY 50": "^NSEI",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}
symbol = st.selectbox("Select Asset", list(symbols.keys()))
symbol_yf = symbols[symbol]
timeframes = {"1H": "1h", "15M": "15m", "5M": "5m"}

# --- Data Fetch ---
def get_data(symbol, interval, period='5d'):
    df = yf.download(symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# --- Trend Detection ---
def detect_trend(df):
    return "Uptrend" if df["Close"].iloc[-1] > df["Close"].iloc[-2] else "Downtrend"

# --- Price Action Pattern Detection ---
def detect_price_action(df):
    patterns = []

    # Bullish Engulfing
    if df['Close'].iloc[-1] > df['Open'].iloc[-1] and df['Open'].iloc[-2] > df['Close'].iloc[-2]:
        if df['Open'].iloc[-1] < df['Close'].iloc[-2] and df['Close'].iloc[-1] > df['Open'].iloc[-2]:
            patterns.append("📈 Bullish Engulfing")

    # Bearish Engulfing
    if df['Close'].iloc[-1] < df['Open'].iloc[-1] and df['Open'].iloc[-2] < df['Close'].iloc[-2]:
        if df['Open'].iloc[-1] > df['Close'].iloc[-2] and df['Close'].iloc[-1] < df['Open'].iloc[-2]:
            patterns.append("📉 Bearish Engulfing")

    # Inside Bar
    if df['High'].iloc[-1] < df['High'].iloc[-2] and df['Low'].iloc[-1] > df['Low'].iloc[-2]:
        patterns.append("🔒 Inside Bar")

    return patterns

# --- Signal Generator (trend filtered) ---
def generate_signal(df):
    df['EMA10'] = df['Close'].ewm(span=10).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['Signal'] = 0
    trend = detect_trend(df)

    if trend == "Uptrend":
        df.loc[df['EMA10'] > df['EMA20'], 'Signal'] = 1
    elif trend == "Downtrend":
        df.loc[df['EMA10'] < df['EMA20'], 'Signal'] = -1

    return df

# --- SL/TP ---
def generate_sl_tp(price, signal, trend):
    atr = 0.015 if trend == "Uptrend" else 0.02
    rr = 2.0  # fixed R:R ratio

    if signal == 1:  # Buy
        sl = price * (1 - atr)
        tp = price + (price - sl) * rr
    elif signal == -1:  # Sell
        sl = price * (1 + atr)
        tp = price - (sl - price) * rr
    else:
        sl = tp = price
    return round(sl, 2), round(tp, 2)

# --- Accuracy ---
def backtest_accuracy(df):
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['StrategyReturn'] = df['Signal'].shift(1) * df['Return']
    total = df[df['Signal'] != 0]
    correct = df[df['StrategyReturn'] > 0]
    return round(len(correct) / len(total) * 100, 2) if len(total) else 0

# --- Elliott Wave Logic ---
def detect_elliott_wave_breakout(df):
    if len(df) < 6:
        return False, ""
    wave1_start = df['Low'].iloc[-6]
    wave1_end = df['High'].iloc[-5]
    wave2 = df['Low'].iloc[-4]
    current_price = df['Close'].iloc[-1]
    trend = detect_trend(df)

    if trend == "Uptrend" and current_price > wave1_end:
        return True, "🌀 Elliott Wave 3 Uptrend Breakout Detected!"
    elif trend == "Downtrend" and current_price < wave2:
        return True, "🌀 Elliott Wave 3 Downtrend Breakout Detected!"
    return False, ""

# --- Upload chart image ---
uploaded_image = st.file_uploader("📸 Upload Chart", type=["png", "jpg", "jpeg"])
trade_reason = st.text_area("📝 Enter Trade Reason")

if st.button("💾 Save Chart & Reason"):
    if uploaded_image is not None:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_image.name}"
        filepath = os.path.join("saved_charts", filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_image.read())
        with open(filepath + ".txt", "w", encoding="utf-8") as f:
            f.write(trade_reason)
        st.success("✅ Chart and Reason Saved!")

# --- Show saved charts ---
st.subheader("📁 Saved Charts")
for file in os.listdir("saved_charts"):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        st.image(os.path.join("saved_charts", file), width=350)
        txt_file = os.path.join("saved_charts", file + ".txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                reason = f.read()
            st.caption(f"📝 Reason: {reason}")

# --- Multi-timeframe Analysis ---
for tf_label, tf_code in timeframes.items():
    st.markdown("---")
    st.subheader(f"🕒 Timeframe: {tf_label}")

    df = get_data(symbol_yf, tf_code)
    trend = detect_trend(df)
    df = generate_signal(df)
    signal = df["Signal"].iloc[-1]
    acc = backtest_accuracy(df)
    price = round(df["Close"].iloc[-1], 2)
    sl, tp = generate_sl_tp(price, signal, trend)

    reward = abs(tp - price)
    risk = abs(price - sl)
    rr_ratio = round(reward / risk, 2) if risk != 0 else "∞"

    signal_text = "Buy" if signal == 1 else "Sell" if signal == -1 else "No Signal"

    st.write(f"**Trend:** `{trend}`")
    st.write(f"**Signal:** `{signal_text}`")
    st.write(f"**Accuracy:** `{acc}%`")
    st.write(f"**Entry Price:** `{price}`")
    st.write(f"**SL:** `{sl}` | **TP:** `{tp}`")
    st.write(f"📊 **Risk/Reward Ratio:** `{rr_ratio}`")

    breakout, message = detect_elliott_wave_breakout(df)
    if breakout:
        st.warning(message)

    # Show price action patterns
    patterns = detect_price_action(df)
    if patterns:
        for p in patterns:
            st.info(f"📌 {p}")

    st.line_chart(df[['Close']])
