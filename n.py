import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import os
from PIL import Image

# ======================
# Streamlit UI Config
# ======================
st.set_page_config(page_title="📈 Divesh Market Zone", layout="wide")
st.title("📈 Divesh Market Zone")

if not os.path.exists("saved_charts"):
    os.makedirs("saved_charts")

# Symbols
symbols = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold (XAU)": "GC=F",
    "Silver (XAG)": "SI=F",
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS"
}

# Timeframes
timeframes = {
    "1 Minute": "1m",
    "5 Minute": "5m",
    "15 Minute": "15m",
    "1 Hour": "1h",
    "1 Day": "1d"
}

# ======================
# Helper Functions
# ======================

def detect_price_action_flags(df):
    flags = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        o1, c1, h1, l1 = df.iloc[i-1][["Open", "Close", "High", "Low"]]
        o2, c2, h2, l2 = df.iloc[i][["Open", "Close", "High", "Low"]]
        pattern_found = False

        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:  # Bullish Engulfing
            pattern_found = True
        elif c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:  # Bearish Engulfing
            pattern_found = True
        elif h2 < h1 and l2 > l1:  # Inside Bar
            pattern_found = True
        body = abs(c2 - o2)
        wick = h2 - l2
        if body < wick * 0.3:  # Pin Bar
            pattern_found = True
        if c1 < o1 and abs(c2 - o2) < 0.2 * (h2 - l2):  # Morning Star
            if i+1 < len(df):
                o3, c3 = df.iloc[i+1][["Open", "Close"]]
                if c3 > o3:
                    pattern_found = True
        if c1 > o1 and abs(c2 - o2) < 0.2 * (h2 - l2):  # Evening Star
            if i+1 < len(df):
                o3, c3 = df.iloc[i+1][["Open", "Close"]]
                if c3 < o3:
                    pattern_found = True

        if pattern_found:
            flags.iloc[i] = True
    return flags

def detect_elliott_wave_flags(df):
    flags = pd.Series(False, index=df.index)
    for i in range(6, len(df)):
        wave1_end = df['High'].iloc[i-5]
        wave2 = df['Low'].iloc[i-4]
        current_price = df['Close'].iloc[i]
        trend = "Uptrend" if df['Close'].iloc[i] > df['Close'].iloc[i-1] else "Downtrend"

        if trend == "Uptrend" and current_price > wave1_end:
            flags.iloc[i] = True
        elif trend == "Downtrend" and current_price < wave2:
            flags.iloc[i] = True
    return flags

def backtest_strategy_accuracy(df, use_elliott=False, use_price_action=False):
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Signal'] = 0
    df.loc[(df['EMA20'] > df['EMA50']) & (df['RSI'] > 50), 'Signal'] = 1
    df.loc[(df['EMA20'] < df['EMA50']) & (df['RSI'] < 50), 'Signal'] = -1

    if use_price_action:
        pa_flags = detect_price_action_flags(df)
        df.loc[~pa_flags, 'Signal'] = 0

    if use_elliott:
        ew_flags = detect_elliott_wave_flags(df)
        df.loc[~ew_flags, 'Signal'] = 0

    df['Return'] = df['Close'].pct_change().shift(-1)
    df['StrategyReturn'] = df['Signal'].shift(1) * df['Return']

    total_signals = df[df['Signal'] != 0]
    correct = df[df['StrategyReturn'] > 0]

    accuracy = round(len(correct) / len(total_signals) * 100, 2) if len(total_signals) else 0
    return accuracy

# ======================
# User Inputs
# ======================
col1, col2 = st.columns(2)
with col1:
    selected_symbol = st.selectbox("Select Symbol", list(symbols.keys()))
with col2:
    selected_tf = st.selectbox("Select Timeframe", list(timeframes.keys()))

ticker = symbols[selected_symbol]
tf = timeframes[selected_tf]

# ======================
# Fetch Data
# ======================
df = yf.download(ticker, period="6mo", interval=tf)
df.dropna(inplace=True)

# ======================
# Accuracy Calculations
# ======================
acc_ema_rsi = backtest_strategy_accuracy(df)
acc_price_action = backtest_strategy_accuracy(df, use_price_action=True)
acc_elliott = backtest_strategy_accuracy(df, use_elliott=True)
acc_combined = backtest_strategy_accuracy(df, use_price_action=True, use_elliott=True)

# ======================
# Display Results
# ======================
st.subheader(f"📊 Accuracy Results for {selected_symbol} ({selected_tf})")
st.write(f"EMA + RSI Accuracy: **{acc_ema_rsi}%**")
st.write(f"Price Action Filter Accuracy: **{acc_price_action}%**")
st.write(f"Elliott Wave Filter Accuracy: **{acc_elliott}%**")
st.write(f"Combined (Elliott + Price Action) Accuracy: **{acc_combined}%**")
