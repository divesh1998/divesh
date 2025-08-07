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

# --- Symbols ---
symbols = {
    "Bitcoin (BTC)": "BTC-USD",
    "Gold (XAU)": "GC=F",
    "NIFTY 50": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
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
timeframes = {"1H": "1h", "15M": "15m", "5M": "5m"}

# --- Data Fetch ---
def get_data(symbol, interval, period='7d'):
    df = yf.download(symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# --- Improved Trend Detection ---
def detect_trend(df, period=20):
    if len(df) < period:
        return "Sideways"
    recent = df['Close'].tail(period)
    slope = recent.iloc[-1] - recent.iloc[0]
    threshold = 0.01 * recent.mean()
    if slope > threshold:
        return "Uptrend"
    elif slope < -threshold:
        return "Downtrend"
    else:
        return "Sideways"

# --- EMA Crossover Signals ---
def generate_signals(df):
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['Prev_EMA20'] = df['EMA20'].shift(1)
    df['Prev_EMA50'] = df['EMA50'].shift(1)
    df['Signal'] = 0
    df.loc[(df['Prev_EMA20'] < df['Prev_EMA50']) & (df['EMA20'] > df['EMA50']), 'Signal'] = 1
    df.loc[(df['Prev_EMA20'] > df['Prev_EMA50']) & (df['EMA20'] < df['EMA50']), 'Signal'] = -1
    return df

# --- Price Action Patterns ---
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

# --- Strategy Confidence ---
def strategy_confidence(row):
    score = 0
    reasons = []
    if row.get("Bullish Engulfing"):
        score += 1
        reasons.append("📈 Price Action bullish")
    if row.get("Bearish Engulfing"):
        score -= 1
        reasons.append("📉 Price Action bearish")
    if row.get("Elliott_Wave_Breakout"):
        score += 1
        reasons.append("🔮 Elliott Wave breakout")
    if row.get("EMA_Trend") == "Uptrend":
        score += 1
        reasons.append("✅ EMA Uptrend")
    elif row.get("EMA_Trend") == "Downtrend":
        score -= 1
        reasons.append("❌ EMA Downtrend")
    return score, ", ".join(reasons)

# --- Improved Backtest Logic ---
def backtest_strategy_accuracy(df, use_elliott=False, use_price_action=False, holding_period=3):
    df = generate_signals(df.copy())
    df['Future_Close'] = df['Close'].shift(-holding_period)
    df['Return'] = df['Future_Close'] / df['Close'] - 1
    df['StrategyReturn'] = df['Signal'] * df['Return']
    if use_elliott:
        breakout, _ = detect_elliott_wave_breakout(df)
        if not breakout:
            df['Signal'] = 0
            df['StrategyReturn'] = 0
    if use_price_action:
        patterns = detect_price_action(df)
        if not patterns:
            df['Signal'] = 0
            df['StrategyReturn'] = 0
    total_signals = df[df['Signal'] != 0]
    correct = total_signals[total_signals['StrategyReturn'] > 0]
    accuracy = round(len(correct) / len(total_signals) * 100, 2) if len(total_signals) else 0
    return accuracy

# --- Daily Accuracy ---
def accuracy_over_days(df, holding_period=3):
    df = generate_signals(df.copy())
    df['Date'] = df.index.date
    df['Future_Close'] = df['Close'].shift(-holding_period)
    df['Return'] = df['Future_Close'] / df['Close'] - 1
    df['StrategyReturn'] = df['Signal'] * df['Return']
    accuracy_df = df.groupby('Date').apply(
        lambda x: (x['StrategyReturn'] > 0).sum() / (x['Signal'] != 0).sum() * 100
        if (x['Signal'] != 0).sum() > 0 else 0
    ).reset_index(name="Daily Accuracy")
    return accuracy_df

# --- Chart Upload ---
uploaded_image = st.file_uploader("📸 Upload Chart", type=["png", "jpg", "jpeg"])
trade_reason = st.text_area("📜 Enter Trade Reason")
if st.button("💾 Save Chart & Reason"):
    if uploaded_image is not None:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_image.name}"
        filepath = os.path.join("saved_charts", filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_image.read())
        with open(filepath + ".txt", "w", encoding="utf-8") as f:
            f.write(trade_reason)
        st.success("✅ Chart and Reason Saved!")

# --- Show Saved Charts ---
st.subheader("📁 Saved Charts")
for file in os.listdir("saved_charts"):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        st.image(os.path.join("saved_charts", file), width=350)
        txt_file = os.path.join("saved_charts", file + ".txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                reason = f.read()
            st.caption(f"📜 Reason: {reason}")

# --- Multi-Timeframe Analysis ---
for tf_label, tf_code in timeframes.items():
    st.markdown("---")
    st.subheader(f"🕒 Timeframe: {tf_label}")
    df = get_data(symbol_yf, tf_code)
    trend = detect_trend(df, 20)
    df = generate_signals(df)
    if not df[df["Signal"] != 0].empty:
        signal_index = df[df["Signal"] != 0].index[-1]
        signal = df.loc[signal_index, "Signal"]
        price = round(df.loc[signal_index, "Close"], 2)
    else:
        signal_index = df.index[-1]
        signal = 0
        price = round(df["Close"].iloc[-1], 2)
    sl, tp = generate_sl_tp(price, signal, trend)
    reward = abs(tp - price)
    risk = abs(price - sl)
    rr_ratio = round(reward / risk, 2) if risk != 0 else "∞"
    signal_text = "Buy" if signal == 1 else "Sell" if signal == -1 else "No Signal"
    acc_ema = backtest_strategy_accuracy(df, holding_period=3)
    acc_epa = backtest_strategy_accuracy(df, use_elliott=True, use_price_action=True, holding_period=3)

    st.write(f"**Trend:** `{trend}`")
    st.write(f"**Signal:** `{signal_text}`")
    st.metric("📘 Only EMA Accuracy", f"{acc_ema}%")
    st.metric("🔮 Elliott + Price Action Accuracy", f"{acc_epa}%")
    st.write(f"**Entry Price:** `{price}` | **SL:** `{sl}` | **TP:** `{tp}`")
    st.write(f"📊 **Risk/Reward Ratio:** `{rr_ratio}`")

    breakout, message = detect_elliott_wave_breakout(df)
    if breakout:
        st.warning(message)

    patterns = detect_price_action(df)
    row = {
        "Bullish Engulfing": any("Bullish Engulfing" in p[1] for p in patterns),
        "Bearish Engulfing": any("Bearish Engulfing" in p[1] for p in patterns),
        "Elliott_Wave_Breakout": breakout,
        "EMA_Trend": trend
    }

    conf_score, conf_reason = strategy_confidence(row)
    st.subheader("📊 Pro Strategy Confidence Meter")
    if conf_score >= 3:
        st.success(f"✅ **Strong Buy Signal!**\nConfidence Score: {conf_score}/5")
    elif conf_score <= -3:
        st.error(f"❌ **Strong Sell Signal!**\nConfidence Score: {conf_score}/5")
    else:
        st.warning(f"⚠️ **Sideways / Neutral Market**\nConfidence Score: {conf_score}/5")

    acc_df = accuracy_over_days(df)
    # st.line_chart(acc_df.set_index("Date"))
    st.markdown("### 📈 Profit Probability Estimate")
    st.info(f"📘 **EMA Strategy Profit Chance:** `{acc_ema}%` | Loss: `{100 - acc_ema}%`")
    st.success(f"🔮 **Elliott + PA Strategy Profit Chance:** `{acc_epa}%` | Loss: `{100 - acc_epa}%`")
