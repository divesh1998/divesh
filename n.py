# streamlit_strategy_with_sl_tp_and_advanced_elliott.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import os
from math import isnan

st.set_page_config(page_title="📈 Divesh Market Zone — SL/TP Backtest + Elliott", layout="wide")

# -----------------------
# Helper / Indicator funcs
# -----------------------
def atr(df, period=14):
    """Compute ATR (True Range based)."""
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr_series = df['TR'].rolling(period, min_periods=1).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return atr_series

def ema(df, span):
    return df['Close'].ewm(span=span, adjust=False).mean()

def rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -----------------------
# Price action detection (same improved heuristics)
# -----------------------
def detect_price_action(df):
    patterns = []
    if len(df) < 3:
        return patterns
    for i in range(2, len(df)):
        o1, c1, h1, l1 = df.iloc[i-2][["Open","Close","High","Low"]]
        o2, c2, h2, l2 = df.iloc[i-1][["Open","Close","High","Low"]]
        o3, c3, h3, l3 = df.iloc[i][["Open","Close","High","Low"]]

        # Engulfing
        if (c2 < o2) and (c3 > o3) and (c3 > o2) and (o3 < c2):
            patterns.append((df.index[i], "Bullish Engulfing"))
        if (c2 > o2) and (c3 < o3) and (c3 < o2) and (o3 > c2):
            patterns.append((df.index[i], "Bearish Engulfing"))

        # Inside bar (middle candle inside previous)
        if (h2 < h1) and (l2 > l1):
            patterns.append((df.index[i-1], "Inside Bar"))

        # Pin bar (small body, long wick)
        body = abs(c3 - o3)
        upper_wick = h3 - max(c3, o3)
        lower_wick = min(c3, o3) - l3
        if not np.isnan(body):
            if body < (upper_wick + lower_wick) * 0.5 and max(upper_wick, lower_wick) > body * 1.5:
                patterns.append((df.index[i], "Pin Bar"))

        # Morning/Evening star (3-candle)
        if (c1 < o1) and (abs(c2 - o2) < (h2 - l2) * 0.3) and (c3 > o3):
            patterns.append((df.index[i], "Morning Star"))
        if (c1 > o1) and (abs(c2 - o2) < (h2 - l2) * 0.3) and (c3 < o3):
            patterns.append((df.index[i], "Evening Star"))
    return patterns

# -----------------------
# ZigZag pivot finder for Elliott
# -----------------------
def zigzag_pivots(df, pct=0.03):
    """
    Find pivot highs/lows by percentage move threshold (simple zigzag).
    pct: minimum percent move from pivot to count (0.03 = 3%)
    Returns list of pivots as (index, 'H'/'L', value)
    """
    highs = df['High'].values
    lows = df['Low'].values
    idxs = df.index
    pivots = []
    # Start searching from first candle
    last_pivot_type = None
    last_pivot_val = None
    last_pivot_idx = None

    # initialize pivot as first close
    last_pivot_val = df['Close'].iloc[0]
    last_pivot_idx = idxs[0]
    last_pivot_type = 'L'  # start assuming low, will adapt

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        change = (price - last_pivot_val) / last_pivot_val
        if last_pivot_type == 'L':
            if change >= pct:  # new high pivot
                pivots.append((last_pivot_idx, 'L', last_pivot_val))
                last_pivot_type = 'H'
                last_pivot_val = price
                last_pivot_idx = idxs[i]
            elif price < last_pivot_val:
                last_pivot_val = price
                last_pivot_idx = idxs[i]
        else:
            if change <= -pct:
                pivots.append((last_pivot_idx, 'H', last_pivot_val))
                last_pivot_type = 'L'
                last_pivot_val = price
                last_pivot_idx = idxs[i]
            elif price > last_pivot_val:
                last_pivot_val = price
                last_pivot_idx = idxs[i]

    # append final pivot
    pivots.append((last_pivot_idx, last_pivot_type, last_pivot_val))
    return pivots

# -----------------------
# Advanced Elliott-ish heuristic
# -----------------------
def detect_advanced_elliott(df, zigzag_pct=0.03):
    """
    Heuristic detection:
      - Build pivot list (zigzag)
      - Search for last 5-6 pivot pattern L H L H L H (start with L)
      - Validate basic Fibonacci retracement of wave2 (~38-78% of wave1)
      - Confirm wave3 larger than wave1 (common in Elliott)
    Returns (True/False, message)
    """
    if len(df) < 20:
        return False, ""
    pivots = zigzag_pivots(df, pct=zigzag_pct)
    # Need at least 6 pivots for wave1..3 structure
    if len(pivots) < 6:
        return False, ""
    # focus last 6 pivots
    last = pivots[-6:]
    types = ''.join([p[1] for p in last])
    # ensure starts with L (trough)
    if not types.startswith('L'):
        # try to find window of 6 with L start
        found = False
        for s in range(len(pivots)-5):
            ts = ''.join([p[1] for p in pivots[s:s+6]])
            if ts.startswith('L'):
                last = pivots[s:s+6]
                types = ts
                found = True
                break
        if not found:
            return False, ""
    # extract numeric values
    Ls = [p for p in last if p[1]=='L']
    Hs = [p for p in last if p[1]=='H']
    if len(Hs) < 2 or len(Ls) < 2:
        return False, ""
    # basic wave points
    wave1_high = Hs[0][2]
    wave2_low = Ls[1][2]
    wave3_high = Hs[1][2]
    # check wave2 retracement of wave1 (using prior low - wave1 high)
    wave1_start_low = Ls[0][2]
    if wave1_high == 0:
        return False, ""
    retracement = (wave1_high - wave2_low) / (wave1_high - wave1_start_low) if (wave1_high - wave1_start_low) != 0 else 0
    # wave3 should be larger than wave1 (heuristic)
    wave3_strength = wave3_high / wave1_high if wave1_high != 0 else 0

    # Heuristic checks:
    if 0.38 <= retracement <= 0.786 and wave3_strength > 1.05:
        # consider current price vs wave1_high for breakout
        current_price = df['Close'].iloc[-1]
        if current_price > wave1_high:
            return True, f"Advanced Elliott-like breakout: wave1_high={wave1_high:.2f}, wave3={wave3_high:.2f}"
    return False, ""

# -----------------------
# SL/TP trade simulator (per-signal)
# -----------------------
def simulate_trade_from_index(df, entry_idx, direction, sl_price, tp_price, exit_on_close=True):
    """
    direction: 1 for long, -1 for short
    entry_idx: index label where entry happens (we assume entry at close of that candle)
    sl_price, tp_price: price levels
    Returns:
      dict: { 'result': 'tp'/'sl'/'timeout', 'return': pct_return, 'exit_index': idx }
    Behavior:
      - Scan subsequent candles: if High >= TP and Low <= SL in same candle, resolve by which touched first:
         we'll approximate order by comparing distances from open:
           - compute distance from open to SL and to TP; smaller absolute distance assumed hit first.
      - If only TP or only SL touched, resolve accordingly.
      - If none touched until data end, return 'timeout' with return equal to last_close/entry_close - 1 (direction adjusted).
    """
    pos = df.index.get_loc(entry_idx)
    entry_close = df['Close'].iloc[pos]
    for j in range(pos+1, len(df)):
        low = df['Low'].iloc[j]
        high = df['High'].iloc[j]
        openp = df['Open'].iloc[j]
        closep = df['Close'].iloc[j]
        # check tp/sl touches
        touched_tp = (high >= tp_price) if direction == 1 else (low <= tp_price)
        touched_sl = (low <= sl_price) if direction == 1 else (high >= sl_price)
        if touched_tp and touched_sl:
            # approximate intrabar which touched first using distance from open
            dist_tp = abs((tp_price - openp))
            dist_sl = abs((sl_price - openp))
            if dist_tp < dist_sl:
                ret = (tp_price - entry_close) / entry_close if direction==1 else (entry_close - tp_price) / entry_close
                return {'result':'tp', 'return': ret, 'exit_index': df.index[j]}
            else:
                ret = (sl_price - entry_close) / entry_close if direction==1 else (entry_close - sl_price) / entry_close
                return {'result':'sl', 'return': ret, 'exit_index': df.index[j]}
        elif touched_tp:
            ret = (tp_price - entry_close) / entry_close if direction==1 else (entry_close - tp_price) / entry_close
            return {'result':'tp', 'return': ret, 'exit_index': df.index[j]}
        elif touched_sl:
            ret = (sl_price - entry_close) / entry_close if direction==1 else (entry_close - sl_price) / entry_close
            return {'result':'sl', 'return': ret, 'exit_index': df.index[j]}
    # if reached end without SL/TP hit, use last close as exit (timeout)
    last_close = df['Close'].iloc[-1]
    ret = (last_close - entry_close) / entry_close if direction==1 else (entry_close - last_close) / entry_close
    return {'result':'timeout', 'return': ret, 'exit_index': df.index[-1]}

# -----------------------
# Full SL/TP backtester that uses signals and simulates trades
# -----------------------
def backtest_with_sl_tp(ticker, interval="15m", use_elliott=False, use_price_action=False,
                        sl_method='ATR', sl_value=1.5, tp_multiplier=2.0,
                        min_signals=20, periods=None):
    """
    sl_method: 'ATR' or 'PCT' (percent)
    sl_value: if ATR -> multiplier of ATR; if PCT -> decimal fraction (e.g., 0.015 => 1.5%)
    tp_multiplier: TP = entry + (entry - SL)*tp_multiplier for longs (reverse for shorts)
    """
    if periods is None:
        periods = ["30d", "90d", "180d", "1y"]

    best_report = None

    for period in periods:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            continue
        df = df.dropna().copy()
        if len(df) < 60:
            continue

        # Indicators
        df['EMA20'] = ema(df, 20)
        df['EMA50'] = ema(df, 50)
        df['RSI'] = rsi(df, 14)
        df['ATR14'] = atr(df, 14)

        # generate signals
        df['Signal'] = 0
        for i in range(len(df)):
            if i < 50:
                continue
            e20 = df['EMA20'].iloc[i]
            e50 = df['EMA50'].iloc[i]
            rsi_v = df['RSI'].iloc[i]
            if pd.isna(e20) or pd.isna(e50) or pd.isna(rsi_v):
                continue
            sig = 0
            if (e20 > e50) and (rsi_v > 50):
                sig = 1
            elif (e20 < e50) and (rsi_v < 50):
                sig = -1

            if sig != 0:
                # apply advanced elliott filter
                if use_elliott:
                    sub = df.iloc[:i+1]
                    ok, _ = detect_advanced_elliott(sub, zigzag_pct=0.03)
                    if not ok:
                        sig = 0
                # apply price action filter: need at least one recent pattern within last 5 candles
                if sig != 0 and use_price_action:
                    sub = df.iloc[:i+1]
                    pats = detect_price_action(sub)
                    recent = [p for p in pats if p[0] >= sub.index[-5]]
                    if not recent:
                        sig = 0
            df.iat[i, df.columns.get_loc('Signal')] = sig

        # simulate trades
        trades = []
        for i in range(len(df)):
            if df['Signal'].iloc[i] == 0:
                continue
            entry_idx = df.index[i]
            direction = int(df['Signal'].iloc[i])
            entry_price = df['Close'].iloc[i]
            # compute SL based on method
            if sl_method == 'ATR':
                atr_val = df['ATR14'].iloc[i]
                if pd.isna(atr_val) or atr_val == 0:
                    continue
                atr_amt = atr_val * sl_value
                if direction == 1:
                    sl_price = entry_price - atr_amt
                    tp_price = entry_price + (entry_price - sl_price) * tp_multiplier
                else:
                    sl_price = entry_price + atr_amt
                    tp_price = entry_price - (sl_price - entry_price) * tp_multiplier
            else:  # PCT
                pct = sl_value
                if direction == 1:
                    sl_price = entry_price * (1 - pct)
                    tp_price = entry_price + (entry_price - sl_price) * tp_multiplier
                else:
                    sl_price = entry_price * (1 + pct)
                    tp_price = entry_price - (sl_price - entry_price) * tp_multiplier

            sim = simulate_trade_from_index(df, entry_idx, direction, sl_price, tp_price)
            trades.append({
                'entry_index': entry_idx,
                'entry_price': entry_price,
                'direction': direction,
                'sl': sl_price,
                'tp': tp_price,
                'result': sim['result'],
                'return': sim['return'],
                'exit_index': sim['exit_index']
            })

        # aggregate stats
        trades_df = pd.DataFrame(trades)
        n_total = len(trades_df)
        if n_total == 0:
            best_report = {
                "period": period,
                "n_trades": 0,
                "win_rate": None,
                "avg_return": None,
                "expectancy": None,
                "net_pnl": None,
                "trades_df": trades_df
            }
            # try next period
            continue

        wins = trades_df[trades_df['return'] > 0]
        losses = trades_df[trades_df['return'] <= 0]
        win_rate = round(len(wins) / n_total * 100, 2)
        avg_return = round(trades_df['return'].mean() * 100, 3)  # percent
        avg_win = round(wins['return'].mean() * 100, 3) if len(wins) else 0
        avg_loss = round(losses['return'].mean() * 100, 3) if len(losses) else 0
        net_pnl = round(trades_df['return'].sum() * 100, 3)
        expectancy = round(( (len(wins)/n_total) * (avg_win/100) + (len(losses)/n_total) * (avg_loss/100) ) * 100, 3)

        report = {
            "period": period,
            "n_trades": n_total,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "net_pnl": net_pnl,
            "trades_df": trades_df
        }

        # return if we have enough trades to be meaningful
        if n_total >= min_signals:
            return report

        # keep last computed as best effort
        best_report = report

    # after trying all periods return best effort
    return best_report

# -----------------------
# Streamlit UI
# -----------------------
st.title("📈 Divesh Market Zone — SL/TP Backtest + Advanced Elliott Filter")

# Symbols list (editable by you)
symbols = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Gold (XAU)": "GC=F",
    "Nifty 50": "^NSEI"
}

col1, col2 = st.columns([2,1])
with col1:
    asset = st.selectbox("Select asset", list(symbols.keys()), index=0)
    ticker = symbols[asset]
    tf = st.selectbox("Timeframe", ["15m","1h","1d"], index=0)
    st.write("Note: 15m and 1h data may be rate-limited by yfinance. Increase period if signals are few.")
with col2:
    sl_method = st.selectbox("SL method", ["ATR","PCT"])
    if sl_method == "ATR":
        sl_mult = st.number_input("ATR multiplier for SL", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
        sl_value = sl_mult
    else:
        pct = st.number_input("Fixed SL percent (e.g. 0.015 = 1.5%)", min_value=0.001, max_value=0.1, value=0.015, step=0.001)
        sl_value = pct
    tp_mult = st.number_input("TP multiplier (RR)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    min_signals = st.number_input("Minimum signals (for period auto-expand)", min_value=5, max_value=200, value=20, step=1)

col3, col4 = st.columns(2)
with col3:
    use_elliott = st.checkbox("Use Advanced Elliott filter", value=True)
    use_pa = st.checkbox("Use Price Action filter", value=True)
with col4:
    st.write("Advanced Elliott uses zigzag pivots + Fibonacci heuristics.")
    st.write("SL uses ATR14 by default when ATR method selected.")

run = st.button("Run Backtest")

if run:
    with st.spinner("Running backtest (may take few seconds depending on timeframe & period)..."):
        report = backtest_with_sl_tp(ticker, interval=tf, use_elliott=use_elliott, use_price_action=use_pa,
                                    sl_method=sl_method, sl_value=sl_value, tp_multiplier=tp_mult,
                                    min_signals=min_signals)
    st.markdown("### Results")
    if report is None:
        st.error("No data / no trades found for the selected asset & timeframe.")
    else:
        st.write(f"**Period used:** {report.get('period')}")
        st.write(f"**Number of trades tested:** {report.get('n_trades')}")
        if report.get('n_trades') == 0:
            st.warning("No trades generated. Try increasing the period or turning off filters.")
        else:
            st.metric("Win Rate", f"{report.get('win_rate')}%")
            st.metric("Avg Return (per trade)", f"{report.get('avg_return')}%")
            st.metric("Avg Win / Avg Loss", f"{report.get('avg_win')}% / {report.get('avg_loss')}%")
            st.metric("Net PnL (sum %)", f"{report.get('net_pnl')}%")
            st.write(f"**Expectancy (approx)**: {report.get('expectancy')}%")
            # show sample trades
            st.markdown("#### Sample trades (first 50)")
            st.dataframe(report.get('trades_df').head(50).assign(
                entry_index=lambda d: d['entry_index'].astype(str),
                exit_index=lambda d: d['exit_index'].astype(str),
                return_pct=lambda d: (d['return']*100).round(3)
            ).drop(columns=['return']).rename(columns={'return_pct':'return (%)'}))
            # Option to download trades CSV
            csv = report.get('trades_df').to_csv(index=False)
            st.download_button("Download trades CSV", csv, file_name=f"trades_{ticker}_{tf}.csv", mime="text/csv")

st.markdown("---")
st.caption("Heuristics used are helpful but not perfect. Use these results as guidance, not trading advice.")
