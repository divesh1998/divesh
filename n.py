import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import os
from PIL import Image

st.set_page_config(page_title="📈 Divesh Market Zone", layout="wide")
st.title("📈 Divesh Market Zone - Pro Version")

# Create save folder
if not os.path.exists("saved_charts"):
    os.makedirs("saved_charts")

# --- Symbols ---
symbols = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "NIFTY 50": "^NSEI",

    # Existing
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",

    # Newly added
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "ITC Limited": "ITC.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC Limited": "NTPC.NS"
}

symbol = st.selectbox("Select Asset", list(symbols.keys()))
symbol_yf = symbols[symbol]
timeframes = {
    "1H": "1h",
    "15M": "15m",
    "5M": "5m"
}

# --- Data Fetch ---
def get_data(symbol, interval, period='30d'):
    df = yf.download(symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# --- Trend Detection ---
def detect_trend(df):
    return "Uptrend" if df["Close"].iloc[-1] > df["Close"].iloc[-2] else "Downtrend"

# --- Price Action Detection ---
def detect_price_action(df):
    patterns = []
    for i in range(2, len(df)):
        o1, c1, h1, l1 = df.iloc[i-1][["Open", "Close", "High", "Low"]]
        o2, c2, h2, l2 = df.iloc[i][["Open", "Close", "High", "Low"]]

        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            patterns.append((df.index[i], "Bullish Engulfing"))
        elif c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
            patterns.append((df.index[i], "Bearish Engulfing"))
        elif h2 < h1 and l2 > l1:
            patterns.append((df.index[i], "Inside Bar"))
        body = abs(c2 - o2)
        wick = h2 - l2
        if body < wick * 0.3:
            patterns.append((df.index[i], "Pin Bar"))
        if c1 < o1 and abs(c2 - o2) < 0.2 * (h2 - l2):
            if i+1 < len(df):
                o3, c3 = df.iloc[i+1][["Open", "Close"]]
                if c3 > o3:
                    patterns.append((df.index[i+1], "Morning Star"))
        if c1 > o1 and abs(c2 - o2) < 0.2 * (h2 - l2):
            if i+1 < len(df):
                o3, c3 = df.iloc[i+1][["Open", "Close"]]
                if c3 < o3:
                    patterns.append((df.index[i+1], "Evening Star"))
    return patterns

# --- Elliott Wave ---
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

# --- Signal Generator with RSI filter ---
def generate_signals(df):
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
    trend = detect_trend(df)
    if trend == "Uptrend" and df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] and df['RSI'].iloc[-1] > 50:
        df.iloc[-1, df.columns.get_loc('Signal')] = 1
    elif trend == "Downtrend" and df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] and df['RSI'].iloc[-1] < 50:
        df.iloc[-1, df.columns.get_loc('Signal')] = -1
    return df

# --- SL/TP ---
def generate_sl_tp(price, signal, trend):
    atr = 0.015 if trend == "Uptrend" else 0.02
    rr = 2.0
    if signal == 1:
        sl = price * (1 - atr)
        tp = price + (price - sl) * rr
    elif signal == -1:
        sl = price * (1 + atr)
        tp = price - (sl - price) * rr
    else:
        sl = tp = price
    return round(sl, 2), round(tp, 2)

# --- Fixed Backtest Strategy Accuracy ---
def backtest_strategy_accuracy(df, use_elliott=False, use_price_action=False):
    df = df.copy()

    # Indicators
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Generate signals for full history
    df['Signal'] = 0
    for i in range(len(df)):
        if df['EMA20'].iloc[i] > df['EMA50'].iloc[i] and df['RSI'].iloc[i] > 50:
            df.iloc[i, df.columns.get_loc('Signal')] = 1
        elif df['EMA20'].iloc[i] < df['EMA50'].iloc[i] and df['RSI'].iloc[i] < 50:
            df.iloc[i, df.columns.get_loc('Signal')] = -1

    # Elliott Wave filter
    if use_elliott:
        for i in range(6, len(df)):
            sub_df = df.iloc[:i+1]
            breakout, _ = detect_elliott_wave_breakout(sub_df)
            if not breakout:
                df.iloc[i, df.columns.get_loc('Signal')] = 0

    # Price Action filter
    if use_price_action:
        valid_dates = [p[0] for p in detect_price_action(df)]
        df['Signal'] = df.apply(lambda row: row['Signal'] if row.name in valid_dates else 0, axis=1)

    # Backtest
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['StrategyReturn'] = df['Signal'].shift(1) * df['Return']

    total_signals = df[df['Signal'] != 0]
    correct = df[df['StrategyReturn'] > 0]
    accuracy = round(len(correct) / len(total_signals) * 100, 2) if len(total_signals) else 0

    return accuracy

# --- Main Display ---
for tf_label, tf_code in timeframes.items():
    st.markdown("---")
    st.subheader(f"🕒 Timeframe: {tf_label}")

    df = get_data(symbol_yf, tf_code)
    trend = detect_trend(df)
    df = generate_signals(df)

    signal = int(df['Signal'].iloc[-1])
    price = round(df['Close'].iloc[-1], 2)
    sl, tp = generate_sl_tp(price, signal, trend)
    rr_ratio = round(abs(tp - price) / abs(price - sl), 2) if price != sl else "∞"

    signal_text = "Buy" if signal == 1 else "Sell" if signal == -1 else "No Signal"
    acc_ema_rsi = backtest_strategy_accuracy(df)
    acc_epa_rsi = backtest_strategy_accuracy(df, use_elliott=True, use_price_action=True)

    st.write(f"**Trend:** `{trend}`")
    st.write(f"**Signal:** `{signal_text}`")
    st.metric("📘 EMA+RSI Accuracy", f"{acc_ema_rsi}%")
    st.metric("🔮 Elliott+PA+RSI Accuracy", f"{acc_epa_rsi}%")
    st.write(f"**Entry Price:** `{price}` | **SL:** `{sl}` | **TP:** `{tp}`")
    st.write(f"📊 **Risk/Reward Ratio:** `{rr_ratio}`")

    breakout, message = detect_elliott_wave_breakout(df)
    if breakout:
        st.warning(message)

    st.markdown("### 📈 Profit Probability Estimate")
    st.info(f"📘 **EMA+RSI Strategy Profit Chance:** `{acc_ema_rsi}%` | Loss: `{100 - acc_ema_rsi}%`")
    st.success(f"🔮 **Elliott+PA+RSI Strategy Profit Chance:** `{acc_epa_rsi}%` | Loss: `{100 - acc_epa_rsi}%`")
