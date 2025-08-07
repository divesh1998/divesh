import yfinance as yf import pandas as pd import streamlit as st from datetime import datetime import os from PIL import Image

st.set_page_config(page_title="📈 Divesh Market Zone", layout="wide") st.title("📈 Divesh Market Zone")

Create save folder

if not os.path.exists("saved_charts"): os.makedirs("saved_charts")

--- Symbols ---

symbols = { "Bitcoin (BTC)": "BTC-USD", "Gold (XAU)": "GC=F" } symbol = st.selectbox("Select Asset", list(symbols.keys())) symbol_yf = symbols[symbol] timeframes = {"1H": "1h", "15M": "15m", "5M": "5m"}

--- Data Fetch ---

def get_data(symbol, interval, period='7d'): df = yf.download(symbol, interval=interval, period=period) if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0) df.dropna(inplace=True) return df

--- Elliott Wave Detection ---

def detect_elliott_wave(df): if len(df) < 30: return None

waves = {}

# Wave 1: First swing high from a low
low_idx = df['Low'].idxmin()
wave1_end_idx = df['High'][low_idx:].idxmax()
wave1_start_price = df['Low'].loc[low_idx]
wave1_end_price = df['High'].loc[wave1_end_idx]

if wave1_end_idx <= low_idx:
    return None

# Wave 2: Retracement from Wave 1
wave2_idx = df['Low'][wave1_end_idx:].idxmin()
wave2_price = df['Low'].loc[wave2_idx]

# Wave 3: Breaks Wave 1 high and goes further
wave3_idx = df['High'][wave2_idx:].idxmax()
wave3_price = df['High'].loc[wave3_idx]

# Wave 4: Retracement again
wave4_idx = df['Low'][wave3_idx:].idxmin()
wave4_price = df['Low'].loc[wave4_idx]

# Wave 5: Final high
wave5_idx = df['High'][wave4_idx:].idxmax()
wave5_price = df['High'].loc[wave5_idx]

waves['Wave 1'] = (low_idx, wave1_end_idx)
waves['Wave 2'] = (wave1_end_idx, wave2_idx)
waves['Wave 3'] = (wave2_idx, wave3_idx)
waves['Wave 4'] = (wave3_idx, wave4_idx)
waves['Wave 5'] = (wave4_idx, wave5_idx)

return waves

--- Display Elliott Wave Info ---

def show_elliott_wave(df, waves): if not waves: st.warning("Elliott Wave structure not detected.") return

st.subheader("🔮 Elliott Wave Structure Detected")
for wave, (start_idx, end_idx) in waves.items():
    st.write(f"{wave}: {df.index.get_loc(start_idx)} to {df.index.get_loc(end_idx)}")
    st.line_chart(df['Close'].loc[start_idx:end_idx])

--- Main Execution ---

selected_tf = st.selectbox("Select Timeframe", list(timeframes.keys())) df = get_data(symbol_yf, timeframes[selected_tf]) waves = detect_elliott_wave(df) show_elliott_wave(df, waves)

