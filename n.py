import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from PIL import Image

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Divesh Market Zone", layout="wide")
st.title("📈 Divesh Market Zone")
st.markdown("Live BTC/Gold/NIFTY50/NSE Stock + Signals + S/R + Price Action + Elliott Wave + SL/TP + Save/Export")

# --- Symbol Selection ---
symbols = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Gold (XAUUSD=X)": "XAUUSD=X",
    "NIFTY 50 (^NSEI)": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "SBIN": "SBIN.NS",
}
symbol_label = st.selectbox("Select Symbol", list(symbols.keys()))
symbol = symbols[symbol_label]

# --- Download Data ---
@st.cache_data(ttl=600)
def load_data(symbol):
    df = yf.download(tickers=symbol, interval="60m", period="7d")
    return df

df = load_data(symbol)

# --- Safe Trend Detection ---
def detect_trend(df):
    if df.empty or len(df) < 2:
        return "No Data"
    return "Uptrend" if df["Close"].iloc[-1] > df["Close"].iloc[-2] else "Downtrend"

trend = detect_trend(df)
st.subheader(f"📊 Trend: {trend}")

# --- Plot Chart ---
if not df.empty:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'))
    fig.update_layout(title=f"{symbol} Price Chart", xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for this symbol.")

# --- Upload & Save Chart Section ---
st.header("📤 Upload Chart for Trade Reason")
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chart", use_column_width=True)
    reason = st.text_area("Trade Reason")
    if st.button("💾 Save Chart"):
        now = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        image.save(f"saved_chart_{now}.png")
        with open(f"saved_chart_{now}.txt", "w") as f:
            f.write(reason)
        st.success("✅ Chart and reason saved!")

