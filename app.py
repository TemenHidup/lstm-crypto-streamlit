# app4_with_save_load.py
# Modified from user's app4.py — adds Save/Load pretrained models and Compare Pretrained Models
# KEY: This file only ADDS features; it does NOT modify existing logic for other pages.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px


# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

# ============================================
# CONFIG & THEME (Clean Dark Mode)
# ============================================
st.set_page_config(page_title="Cryptocurrency Dashboard", layout="wide")

# ensure saved_models folder exists
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42)

# (CSS and matplotlib config unchanged — omitted here for brevity but kept identical to original)
st.markdown("""
<style>
body { 
    background: linear-gradient(135deg, #0b0f14 0%, #131720 100%); 
    color: #e6eef6; 
}
section[data-testid="stSidebar"] {
    background: rgba(10, 12, 15, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.08);
    font-size: 18px;
}
.main-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
.control-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 14px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}
.section-header {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255,255,255,0.22);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 12px 20px;
    border-radius: 12px;
    margin: 20px 0 15px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.32);
}
.metric-value { color: #00E0B8; font-size: 32px; font-weight: 700; }
.metric-label { color: #cdd5df; font-size: 14px; }
.title { 
    color: #00E0B8; 
    font-size: 32px; 
    font-weight: 800; 
    text-shadow: 0 0 8px rgba(0,224,184,0.45); 
    margin: 0;
}
.small-muted { color: #9aa3ad; font-size: 13px; }
section[data-testid="stSidebar"] * {
    font-size: 18px !important;
}
.stButton > button {
    background: rgba(0, 224, 184, 0.18);
    border: 1px solid rgba(0, 224, 184, 0.55);
    color: #00E0B8;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    transition: 0.25s ease;
    backdrop-filter: blur(4px);
}
.stButton > button:hover {
    background: rgba(0, 224, 184, 0.32);
    box-shadow: 0 0 15px rgba(0, 224, 184, 0.9);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.sidebar-menu-item {
    padding: 10px 16px;
    border-radius: 6px;
    cursor: pointer;
    color: #ddd;
    font-size: 17px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sidebar-menu-item:hover {
    background-color: rgba(255,255,255,0.08);
}

.sidebar-menu-item-active {
    background-color: #1e6ff7 !important;
    color: white !important;
    font-weight: 600 !important;
}

.sidebar-icon {
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)


plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams['figure.facecolor'] = '#0E1117'
plt.rcParams['axes.facecolor'] = '#0E1117'
plt.rcParams['text.color'] = '#e6eef6'
plt.rcParams['axes.labelcolor'] = '#e6eef6'
plt.rcParams['xtick.color'] = '#e6eef6'
plt.rcParams['ytick.color'] = '#e6eef6'

# ============================================
# SESSION STATE INIT
# ============================================
for key, val in {
    'X_train': None,
    'y_train': None,
    'scaler': None,
    'model': None,
    'predict_year': None,
    'opt_results': {},
    'forecast_days': 7,  # default
    'model_loaded': False,
    'loaded_model_meta': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# Helper functions
# ============================================
coin_map = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'BNB': 'BNB-USD'
}
coin_short = {
    'Bitcoin': 'BTC',
    'Ethereum': 'ETH',
    'BNB': 'BNB'
}

@st.cache_data
def download_data(ticker):
    df = yf.download(ticker, start='2020-12-01', end='2025-12-01')
    return df


def prepare_lstm_data(df, window=60, horizon=1):
    """
    Prepare data for LSTM.
    horizon: number of future days to predict (1..14)
    Returns X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values.astype(float)

    n = len(data)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    # SCALER FIT ONLY ON TRAIN
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    close_idx = features.index("Close")

    # WINDOWING FUNCTION — supports horizon
    def create_window(dataset):
        X_list, y_list = [], []
        # ensure there's room for horizon
        for i in range(window, len(dataset) - (horizon - 1)):
            X_list.append(dataset[i-window:i, :])         # all OHLCV
            if horizon == 1:
                y_list.append(dataset[i, close_idx])      # target = Close (single)
            else:
                # collect next `horizon` closes (scaled)
                y_list.append(dataset[i:i+horizon, close_idx])
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        if horizon == 1:
            y_arr = y_arr.reshape(-1)  # shape (N,)
        else:
            # shape (N, horizon)
            pass
        return X_arr, y_arr

    # TRAIN
    X_train, y_train = create_window(train_scaled)

    # VAL (needs tail(train))
    val_input = np.concatenate([train_scaled[-window:], val_scaled], axis=0)
    X_val, y_val = create_window(val_input)

    # TEST (needs tail(val))
    test_input = np.concatenate([val_scaled[-window:], test_scaled], axis=0)
    X_test, y_test = create_window(test_input)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def prepare_lstm_data_with_scaler(df, scaler, window=60, horizon=1):
    """
    Prepare windows using an already-fitted scaler (used when loading pretrained models).
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values.astype(float)

    n = len(data)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    train_scaled = scaler.transform(train)
    val_scaled   = scaler.transform(val)
    test_scaled  = scaler.transform(test)

    close_idx = features.index("Close")

    def create_window(dataset):
        X_list, y_list = [], []
        for i in range(window, len(dataset) - (horizon - 1)):
            X_list.append(dataset[i-window:i, :])
            if horizon == 1:
                y_list.append(dataset[i, close_idx])
            else:
                y_list.append(dataset[i:i+horizon, close_idx])
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        if horizon == 1:
            y_arr = y_arr.reshape(-1)
        return X_arr, y_arr

    X_train, y_train = create_window(train_scaled)

    val_input = np.concatenate([train_scaled[-window:], val_scaled], axis=0)
    X_val, y_val = create_window(val_input)

    test_input = np.concatenate([val_scaled[-window:], test_scaled], axis=0)
    X_test, y_test = create_window(test_input)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_shape, horizon=1):
    """
    Build LSTM. If horizon>1, final Dense will output `horizon` values (multi-step direct).
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model


def get_optimizer(name):
    name = name.lower()
    if name == 'adam': return Adam()
    if name == 'rmsprop': return RMSprop()
    if name == 'sgd': return SGD()
    return Adam()


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # avoid division by zero in MAPE
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    accuracy = 100 - mape
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape} # 'Accuracy': accuracy


def inverse_close_only_multi(scaler, scaled_preds, last_known_row, features=['Open','High','Low','Close','Volume']):
    close_idx = features.index("Close")
    arr = np.array(scaled_preds)
    if arr.ndim == 2:
        results = []
        for i in range(arr.shape[0]):
            preds = arr[i]
            last = last_known_row.copy()
            outs = []
            for p in preds:
                row = last.copy()
                row[close_idx] = p
                inv = scaler.inverse_transform(row.reshape(1, -1))[0][close_idx]
                outs.append(inv)
                last = row.copy()
            results.append(outs)
        return np.array(results)
    else:
        preds = arr.flatten()
        last = last_known_row.copy()
        outs = []
        for p in preds:
            row = last.copy()
            row[close_idx] = p
            inv = scaler.inverse_transform(row.reshape(1, -1))[0][close_idx]
            outs.append(inv)
            last = row.copy()
        return np.array(outs)


def generate_rolling_forecast(model, scaler, df, forecast_days=7, window=60):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values.astype(float)
    scaled_all = scaler.transform(data)
    last_window = scaled_all[-window:].copy()
    close_idx = features.index("Close")

    preds_scaled = []
    curr = last_window.copy()

    try:
        out_shape = model.output_shape
        horizon_model = int(out_shape[1])
    except Exception:
        horizon_model = None

    if horizon_model is None:
        for _ in range(forecast_days):
            pred = model.predict(curr.reshape(1, window, -1))[0]
            pred_val = pred.flatten()[0]
            preds_scaled.append(pred_val)
            next_row = curr[-1].copy()
            next_row[close_idx] = pred_val
            curr = np.vstack([curr[1:], next_row])
    else:
        if horizon_model >= forecast_days:
            pred = model.predict(curr.reshape(1, window, -1))[0].flatten()[:forecast_days]
            preds_scaled = list(pred)
        else:
            direct = model.predict(curr.reshape(1, window, -1))[0].flatten()
            for p in direct:
                preds_scaled.append(p)
                next_row = curr[-1].copy()
                next_row[close_idx] = p
                curr = np.vstack([curr[1:], next_row])
            remaining = forecast_days - len(direct)
            for _ in range(remaining):
                pred = model.predict(curr.reshape(1, window, -1))[0].flatten()
                p = pred[0]
                preds_scaled.append(p)
                next_row = curr[-1].copy()
                next_row[close_idx] = p
                curr = np.vstack([curr[1:], next_row])

    preds_scaled = np.array(preds_scaled)
    if preds_scaled.size == 0:
        return np.array([])

    last_window = scaled_all[-window:]
    reference_row = last_window[-1]

    inv = inverse_close_only_multi(scaler, preds_scaled, reference_row)
    return inv


# ---- New helpers to save/load model + scaler + metadata ----

def save_model_files(model, scaler, coin_short_name, optimizer_name, window, horizon):
    base = f"{coin_short_name}_{optimizer_name}"
    model_path = os.path.join("saved_models", base + ".h5")
    scaler_path = os.path.join("saved_models", base + "_scaler.pkl")
    meta_path = os.path.join("saved_models", base + "_meta.json")

    # save model
    model.save(model_path)

    # save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # save metadata
    meta = {'window': int(window), 'horizon': int(horizon)}
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    return model_path, scaler_path, meta_path


# def load_model_files(coin, optimizer):
#     model_path = f"saved_models/{coin}_{optimizer}.keras"
#     scaler_path = f"saved_models/{coin}_scaler.pkl"
#     meta_path   = f"saved_models/{coin}_meta.json"

#     model = load_model(model_path,compile=False)

#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)

#     with open(meta_path, 'r') as f:
#         meta = json.load(f)

#     return model, scaler, meta


def load_model_files(short, optimizer):
    

    
    model_path = f"saved_models/{short}_{optimizer}.keras"
    scaler_path = f"saved_models/{short}_scaler.pkl"
    meta_path   = f"saved_models/{short}_meta.json"

    # ============================
    # FILE EXISTENCE CHECK
    # ============================
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    # ============================
    # LOAD SCALER FIRST
    # ============================
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # ============================
    # LOAD META FILE
    # ============================
    with open(meta_path, "r") as f:
        meta = json.load(f)

    window  = meta.get("window", 60)
    horizon = meta.get("horizon", 1)

    # ============================
    # BUILD DUMMY INPUT FOR SAFE LOAD
    # (Keras 3 sometimes fails unless
    #   model.build(input_shape) called manually)
    # ============================
    dummy_input = tf.zeros((1, window, 5))

    # ============================
    # LOAD MODEL (export format friendly)
    # ============================
    try:
        model = load_model(
            model_path,
            compile=False,       # IMPORTANT to avoid optimizer issues
            safe_mode=False      # allow full deserialization
        )

        # Force-build model to avoid LSTM name_scope error
        model.predict(dummy_input)

    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    return model, scaler, meta


# ============================================
# LAYOUT: Sidebar (Navigation only)
# ============================================
st.sidebar.title('Website Navigation')
page = st.sidebar.radio('', ['Dashboard', 'Prediction', 'Comparison','Information'])

# reset logic
if "last_page" not in st.session_state:
    st.session_state.last_page = page

if st.session_state.last_page != page:
    st.session_state.opt_results = {}
    st.session_state.last_page = page


# ============================================================
# 🔥 MAIN CONTENT START
# ============================================================
st.markdown('</div>', unsafe_allow_html=True)

# Top Dashboard Header
st.markdown("""
<div class="section-header">
    <h2 class='title'>Cryptocurrency Dashboard</h2>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE: HOME
# (unchanged content — omitted for brevity in comments but kept)
# ============================================
if page == "Information":

    st.markdown("""
        <div class="section-header">
            <h3 style="margin:0; color:#e6eef6;">Welcome to the Cryptocurrency Dashboard</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🌍 About This Dashboard
    This platform provides a clean and powerful interface to explore cryptocurrency price movements, view historical charts, and generate LSTM-based price forecasts for Bitcoin (BTC), Ethereum (ETH), and Binance Coin (BNB).

    The system uses **pretrained LSTM models** to deliver fast and accurate predictions without retraining.

    ---

    ## Available Pages

    ### **1️⃣ Dashboard — Historical Data & Charts**
    This page displays:
    - Candlestick charts (Close Price) for BTC, ETH, and BNB  
    - Individual Line charts for each coin  
    - Clean and interactive visualization of price movements from **Dec 2020 – Dec 2025**

    Use this page to understand market trends before forecasting.

    ---

    ## **2️⃣ Prediction — LSTM Forecasting (Pretrained Models)**  
    Below is a quick visual guide:

    """)

    # ---------------- IMAGES FOR PREDICTION PAGE ----------------
    st.markdown("#### 🔹 Choose Coin")
    st.image("images/Choosecoin.png", width=450)

    st.markdown("#### 🔹 Choose Optimizer")
    st.image("images/Chooseopt.png", width=450)

    st.markdown("#### 🔹 Choose Forecast Days")
    st.image("images/Forecastday.png", width=450)

    st.markdown("""
    Here you can:
    - Choose **Coin (BTC/ETH/BNB)**  
    - Choose **Optimizer (Adam, RMSProp, SGD)**  
    - Choose **Forecast Days**  
    - Load the pretrained model instantly  
    - Generate:
    - Training / Validation / Test predictions  
    - Multi-day future forecast  
    - Download predicted forecast values as CSV  

    No more training time — pretrained models load instantly.

    ---

    ### **3️⃣ Optimizer Comparison**
    Visual guide for this page:
    """)

    # --------------- IMAGE FOR COMPARISON ----------------
    st.markdown("#### 🔹 Choose Coin")
    st.image("images/Choosecoin.png", width=450)
    
    st.markdown("#### 🔹 Forecast Days & Comparison Input")
    st.image("images/Forecastcompare.png", width=450)

    st.markdown("""
    Compare optimizer performance in two modes:

    #### **Compare Pretrained Models**
    - Choose **Coin (BTC/ETH/BNB)**  
    - Choose **Forecast Days**  
    - Loads all three pretrained models for the selected coin  
    - Instantly compares their performance and forecast outcomes  

    This is useful for research, analysis, or finding the best optimizer for crypto prediction.

    ---

    ## 📝 Tips for Best Use
    - **Prediction page** is ideal for fast, clean forecasting  
    - **Dashboard page** helps visualize historical behavior  
    - **Comparison page** is excellent for analysis and reporting  
    - Models were trained with a **60-day sliding window**, giving strong pattern learning on volatile crypto data  
    - Forecasts become less certain the further they project into the future  

    ---

    ## 🚀 Ready to Explore?
    Use the sidebar to navigate between features and start analyzing cryptocurrency trends and predictions!
    """)



    # Stop UI from rendering prediction/comparison on home page
    st.stop()




def safe_series(x):    
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    arr = np.array(x)
    if arr.ndim > 1:
        return pd.Series(arr[:, 0])
    return pd.Series(arr)


def historical_page():

    st.markdown("""
        <div class="section-header">
            <h3 style="margin:0; color:#e6eef6;">Historical Price Data</h3>
        </div>
    """, unsafe_allow_html=True)

    # ===============================
    # LOAD RAW DATA
    # ===============================
    # NOTE: Keep your original download_data and coin_map calls
    df_btc = download_data(coin_map["Bitcoin"]).copy()
    df_eth = download_data(coin_map["Ethereum"]).copy()
    df_bnb = download_data(coin_map["BNB"]).copy()

    # FIX ALL OHLC COLUMNS
    for df in [df_btc, df_eth, df_bnb]:
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = safe_series(df[col]).astype(float)
            
    # Map for easy access
    df_map = {"Bitcoin": df_btc, "Ethereum": df_eth, "BNB": df_bnb}


    # ===============================
    # PLOT 1 — 3 CANDLESTICK CHARTS
    # ===============================
    st.markdown("### 🕯 Candlestick Charts")

    coins_to_plot = ["Bitcoin", "Ethereum", "BNB"]
    colors = {"Bitcoin": {"inc": "#00E0B8", "dec": "#EF476F"},
              "Ethereum": {"inc": "#FFD166", "dec": "#EF476F"},
              "BNB": {"inc": "#2A9D8F", "dec": "#EF476F"}}

    for coin_choice in coins_to_plot:
        df = df_map[coin_choice]
        symbol = coin_choice
        
        st.markdown(f"#### {symbol} Candlestick Chart")

        fig_candle = go.Figure(data=[
            go.Candlestick(
                x=df.index,
                open=safe_series(df["Open"]),
                high=safe_series(df["High"]),
                low=safe_series(df["Low"]),
                close=safe_series(df["Close"]),
                increasing_line_color=colors[coin_choice]["inc"],
                decreasing_line_color=colors[coin_choice]["dec"],
                name=symbol
            )
        ])

        fig_candle.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        # Hides range slider and default range buttons to keep the chart clean
        fig_candle.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig_candle, use_container_width=True)

    # --- Horizontal Rule to separate plots ---
    st.markdown("---")

    # ===============================
    # PLOT 2 — INTERACTIVE LINE CHART (1D ONLY)
    # ===============================
    st.markdown("### 📈 Close Price Trend")

    # Dropdown for selecting the coin for the line graph
    coin_choice_line = st.selectbox("Select coin to view Close Price trend", 
                                    ["Bitcoin","Ethereum","BNB"], 
                                    key="line_select")

    df_selected = df_map[coin_choice_line]
    symbol_line = coin_choice_line
    
    # Define a color for the selected line
    line_color_map = {"Bitcoin": "#00E0B8", "Ethereum": "#FFD166", "BNB": "#2A9D8F"}
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_selected.index,
        y=safe_series(df_selected["Close"]),
        name=symbol_line,
        mode="lines",
        line=dict(width=2, color=line_color_map[coin_choice_line])
    ))

    fig_line.update_layout(
        title=f"{symbol_line} Close Price History",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Close Price (USD)"
    )

    st.plotly_chart(fig_line, use_container_width=True)



if page == "Dashboard":
    historical_page()

# ============================================
# Page: Prediction (Simplified Pretrained Mode)
# ============================================
if page == 'Prediction':

    st.markdown("""
    <div class="section-header">
        <h3 style="margin:0; color:#e6eef6;">Prediction Section</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])

    # ============================
    # LEFT PANEL — USER INPUT
    # ============================
    with col1:

        # Choose coin + optimizer (dipakai untuk load model)
        coin = st.selectbox('Choose Coin', list(coin_map.keys()))
        optimizer = st.selectbox('Choose Optimizer', ['Adam', 'RMSProp', 'SGD'])

        # Forecast days
        forecast_days = st.slider('Forecast horizon (days)', 1, 14, 7)
        st.session_state['forecast_days'] = forecast_days

        st.markdown("### Load Model")

        # LOAD MODEL (tanpa dropdown tambahan)
        if st.button('Load Model'):

            # =====================================================
            # RESET SESSION STATE — WAJIB supaya tidak crash
            # =====================================================
            keys_to_clear = [
                "model", "scaler", "meta",
                "X_train", "y_train",
                "X_val", "y_val",
                "X_test", "y_test",
                "window", "horizon",
                "fig1", "fig2"
            ]
            for k in keys_to_clear:
                st.session_state[k] = None
        
            # =====================================================
            # LOAD MODEL + SCALER + META
            # =====================================================
            short = coin_short[coin]
        
            try:
                m, sc, meta = load_model_files(short, optimizer)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()
        
            # =====================================================
            # PREPARE FRESH DATASETS (MENCEGAH SHAPE MISMATCH)
            # =====================================================
            df = download_data(coin_map[coin])
            w = meta.get('window', 60)
            h = meta.get('horizon', 1)
        
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data_with_scaler(
                df, sc, window=w, horizon=h
            )
        
            # =====================================================
            # SAVE INTO SESSION_STATE
            # =====================================================
            st.session_state.model = m
            st.session_state.scaler = sc
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_val = X_val
            st.session_state.y_val = y_val
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.window = w
            st.session_state.horizon = h
        
            st.success(f"Loaded pretrained model: {short}_{optimizer}.keras (window={w}, horizon={h})")
        

        # SHOW PLOT
        if st.button('Show Plot'):
            if st.session_state.model is None:
                st.error("Model not loaded")
            else:
                df = download_data(coin_map[coin])
                model = st.session_state.model
                scaler = st.session_state.scaler
                window = st.session_state.window
                horizon = st.session_state.horizon
                forecast_days = st.session_state['forecast_days']

                # ============================
                # PREDICT TRAIN / VAL / TEST
                # ============================
                scaled_all = scaler.transform(df[['Open','High','Low','Close','Volume']])
                last_window = scaled_all[-window:]
                reference_row = last_window[-1]

                # TRAIN
                train_raw = model.predict(st.session_state.X_train)
                train_pred = inverse_close_only_multi(scaler, train_raw.flatten(), reference_row)
                train_start = window
                train_end = train_start + len(train_pred)
                train_dates = df.index[train_start:train_end]

                # VAL
                val_raw = model.predict(st.session_state.X_val)
                val_pred = inverse_close_only_multi(scaler, val_raw.flatten(), reference_row)
                val_start = train_end
                val_end = val_start + len(val_pred)
                val_dates = df.index[val_start:val_end]

                # TEST
                test_raw = model.predict(st.session_state.X_test)
                test_pred = inverse_close_only_multi(scaler, test_raw.flatten(), reference_row)
                test_start = len(df) - len(test_pred)
                test_end = test_start + len(test_pred)
                test_dates = df.index[test_start:test_end]

                # FORECAST
                forecast = generate_rolling_forecast(
                    model, scaler, df, forecast_days=forecast_days, window=window
                )
                future_dates = pd.date_range(start=test_dates[-1], periods=len(forecast)+1)[1:]

                # ============================
                # DOWNLOAD FORECAST CSV
                # ============================
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Forecast_Close": forecast
                })

                csv_data = forecast_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"{coin}_{optimizer}_forecast_{forecast_days}d.csv",
                    mime="text/csv",
                )


                # ============================
                # PLOT: FULL
                # ============================
                fig1, ax1 = plt.subplots(figsize=(13,5))
                ax1.plot(df.index, df['Close'], label='Actual', linewidth=2)
                ax1.plot(train_dates, train_pred, label='Train Prediction', linewidth=2)
                ax1.plot(val_dates, val_pred, label='Validation Prediction', linewidth=2)
                ax1.plot(test_dates, test_pred, label='Test Prediction', linewidth=2)
                ax1.plot(future_dates, forecast, linestyle='dashed',
                         label=f'Forecast ({forecast_days}d)', linewidth=2)
                ax1.legend()
                # st.pyplot(fig1)

                # ============================
                # PLOT: ZOOM 60 DAYS
                # ============================
                zoom_days = 60
                df_zoom = df.iloc[-zoom_days:]
                fig2, ax2 = plt.subplots(figsize=(13,5))
                ax2.plot(df_zoom.index, df_zoom['Close'], label='Actual (Last 60d)', color='cyan')

                mask_zoom = (test_dates >= df_zoom.index[0])
                ax2.plot(test_dates[mask_zoom], test_pred[mask_zoom], label='Test (Zoom)', color='orange')

                ax2.plot(future_dates, forecast, linestyle='dashed', color='magenta', linewidth=3)
                ax2.legend()
                # st.pyplot(fig2)
                st.session_state.fig1 = fig1
                st.session_state.fig2 = fig2


    # RIGHT PANEL — DATA & EVALUATION
    with col2:
        st.markdown("""
        <div class="section-header" style="margin-bottom:10px;">
            <h3 style="margin:0; color:#e6eef6;">Data & Evaluation</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ===================== SAFETY CHECK FOR PREDICTION PAGE =====================

        # Cek apakah model sudah di-load
        if "model" not in st.session_state or st.session_state.model is None:
            st.warning("⚠ Model not loaded. Please load a pretrained model first.")
            st.stop()
        
        # Cek apakah X_test ada
        if "X_test" not in st.session_state or st.session_state.X_test is None:
            st.warning("⚠ Test data not ready. Please load a pretrained model first.")
            st.stop()
        
        # Cek scaler
        if "scaler" not in st.session_state or st.session_state.scaler is None:
            st.warning("⚠ Scaler missing. Please load a pretrained model first.")
            st.stop()

        
        df = download_data(coin_map[coin])
        st.dataframe(df)

        if st.session_state.model is None:
            st.info("Please train the model first to see the evaluation metrics.")
        else:
            test_pred_raw = st.session_state.model.predict(st.session_state.X_test)
            scaled_all = st.session_state.scaler.transform(df[['Open','High','Low','Close','Volume']].values)
            last_known_row = scaled_all[-1]  # baris terakhir

            # inverse pred (if multi-step, take only first-step for evaluation against y_test)
            if test_pred_raw.ndim == 2 and test_pred_raw.shape[1] > 1:
                test_pred_scaled_for_eval = test_pred_raw[:, 0]
            else:
                test_pred_scaled_for_eval = test_pred_raw.flatten()

            test_pred_rescaled = inverse_close_only_multi(
                st.session_state.scaler,
                test_pred_scaled_for_eval,
                last_known_row
            )

            # inverse true y_test (if y_test is multi-output and we evaluated single-step, adapt accordingly)
            if st.session_state.y_test.ndim == 2:
                ytest_for_eval = st.session_state.y_test[:, 0]
            else:
                ytest_for_eval = st.session_state.y_test.flatten()

            y_test_rescaled = inverse_close_only_multi(
                st.session_state.scaler,
                ytest_for_eval,
                last_known_row
            )

            metrics = compute_metrics(y_test_rescaled, test_pred_rescaled)

            c1, c2 = st.columns(2)
            c1.metric('RMSE', f"{metrics['RMSE']:.2f}")
            c2.metric('MAE', f"{metrics['MAE']:.2f}")
            

            c3, c4 = st.columns(2)
            c3.metric('R2', f"{metrics['R2']:.4f}")
            c4.metric('MAPE', f"{metrics['MAPE']:.2f}%")
            # c5.metric('Accuracy', f"{metrics['Accuracy']:.2f}%")

    # ===============================
    # FULL-WIDTH PLOTTING AREA
    # ===============================
    # ===============================
    # FULL-WIDTH PLOTTING AREA (SAFE)
    # ===============================
    if st.session_state.get("fig1") is not None and st.session_state.get("fig2") is not None:

        st.markdown("""
        <div class="section-header">
            <h3 style="margin:0; color:#e6eef6;">Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)

        st.pyplot(st.session_state.fig1)
        st.pyplot(st.session_state.fig2)




# ============================================
# Page: Optimizer Comparison
# ============================================
elif page == 'Comparison':

    st.markdown("""
    <div class="section-header">
        <h3 style="margin:0; color:#e6eef6;">Optimizer Comparison</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,2])

    with col1:
        coin = st.selectbox('Choose Coin', list(coin_map.keys()), key='coc')
        forecast_days_opt = st.slider('Forecast days for comparison', min_value=1, max_value=14, value=7, step=1)
        run_btn = st.button('Compare Models')

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.write("Tips: ")
        st.write("This will train SGD, Adam, and RMSProp on the same dataset.")
        st.write("Useful for comparing performance differences.")
        st.markdown('</div>', unsafe_allow_html=True)

        if run_btn:

            # =====================================================
            # RESET SEMUA SESSION STATE — WAJIB untuk hindari crash
            # =====================================================
            keys_to_clear = [
                "model", "scaler", "meta",
                "X_train", "y_train",
                "X_val", "y_val",
                "X_test", "y_test",
                "window", "horizon",
                "fig1", "fig2"
            ]
            for k in keys_to_clear:
                st.session_state[k] = None
        
            st.session_state.opt_results = {}
            df = download_data(coin_map[coin])
        
            # ---- Preprocess fresh for optimizer comparison (horizon=1) ----
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_lstm_data(df, window=60, horizon=1)
        
            # store into session_state (optional)
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_val = X_val
            st.session_state.y_val = y_val
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler  = scaler
        
            st.success("Preprocessed for optimizer comparison (horizon=1).")
        
            X_train = st.session_state.X_train
            y_train = st.session_state.y_train
            X_val   = st.session_state.X_val
            y_val   = st.session_state.y_val
            X_test  = st.session_state.X_test
            y_test  = st.session_state.y_test
            scaler  = st.session_state.scaler
        
            # ---- Prepare reference rows ----
            last_row_train = X_train[-1, -1].copy()
            last_row_val   = X_val[-1, -1].copy()
        
            features = ['Open','High','Low','Close','Volume']
            data = df[features].values.astype(float)
            scaled_all = scaler.transform(data)
        
            n = len(scaled_all)
            train_end = int(n * 0.70)
            val_end   = int(n * 0.85)
        
            train_scaled = scaled_all[:train_end]
            val_scaled   = scaled_all[train_end:val_end]
            test_scaled  = scaled_all[val_end:]
        
            test_input = np.concatenate([val_scaled[-60:], test_scaled], axis=0)
            reference_row_test = test_input[60 - 1]
        
            # ---- Load pretrained models ----
            optimizers = ["SGD", "Adam", "RMSProp"]
            loaded = {}
            missing = []
        
            prog = st.progress(0)
            status = st.empty()
        
            for i, opt_name in enumerate(optimizers):
                short = coin_short[coin]
                base = f"{short}_{opt_name}"
        
                model_path = f"saved_models/{base}.keras"
                scaler_path = f"saved_models/{short}_scaler.pkl"
                meta_path   = f"saved_models/{short}_meta.json"
        
                if not (os.path.exists(model_path) and 
                        os.path.exists(scaler_path) and 
                        os.path.exists(meta_path)):
                    missing.append(base)
                    continue
        
                try:
                    # -------------------------------
                    # LOAD MODEL (compile=False)
                    # -------------------------------
                    m = load_model(model_path, compile=False)
        
                    # -------------------------------
                    # FORCE BUILD MODEL (Anti-crash)
                    # -------------------------------
                    with open(meta_path, "r") as f:
                        meta_preview = json.load(f)
        
                    dummy = tf.zeros((1, meta_preview["window"], 5))
                    m.predict(dummy)   # WAJIB
        
                    # -------------------------------
                    # LOAD SCALER + META
                    # -------------------------------
                    with open(scaler_path, "rb") as f:
                        sc_loaded = pickle.load(f)
                    with open(meta_path, "r") as f:
                        meta_loaded = json.load(f)
        
                    loaded[opt_name] = {
                        "model": m,
                        "scaler": sc_loaded,
                        "meta": meta_loaded
                    }
        
                except Exception as e:
                    st.error(f"Failed to load {base}: {e}")
        
                prog.progress((i + 1) / len(optimizers))



                if missing:
                    st.error(f'Missing pretrained files for: {missing}. Please train & save those models first.')
                else:
                    results = {}
                    for opt_name, obj in loaded.items():
                        status.text(f'Preparing predictions for {opt_name}')
                        m = obj['model']
                        sc = obj['scaler']
                        meta = obj['meta']
                        w = meta.get('window', 60)

                        # Recreate windows WITH the loaded scaler so predictions are consistent
                        X_tr, y_tr, X_v, y_v, X_te, y_te = prepare_lstm_data_with_scaler(df, sc, window=w, horizon=1)

                        # reference rows
                        last_row_train = X_tr[-1, -1].copy()
                        last_row_val = X_v[-1, -1].copy()

                        # reconstruct test_input reference row similar to training block
                        data = df[['Open','High','Low','Close','Volume']].values.astype(float)
                        scaled_all_local = sc.transform(data)
                        n = len(scaled_all_local)
                        train_end = int(n * 0.70)
                        val_end   = int(n * 0.85)
                        train_scaled = scaled_all_local[:train_end]
                        val_scaled   = scaled_all_local[train_end:val_end]
                        test_scaled  = scaled_all_local[val_end:]
                        test_input = np.concatenate([val_scaled[-60:], test_scaled], axis=0)
                        reference_row_test_local = test_input[60 - 1]

                        pred_train_raw = m.predict(X_tr)
                        pred_train = inverse_close_only_multi(sc, pred_train_raw.flatten(), last_row_train)

                        pred_val_raw = m.predict(X_v)
                        pred_val = inverse_close_only_multi(sc, pred_val_raw.flatten(), last_row_val)

                        pred_test_raw = m.predict(X_te)
                        pred_test = inverse_close_only_multi(sc, pred_test_raw.flatten(), reference_row_test_local)

                        true_train = inverse_close_only_multi(sc, y_tr.reshape(-1,1).flatten(), last_row_train)
                        true_val   = inverse_close_only_multi(sc, y_v.reshape(-1,1).flatten(), last_row_val)
                        true_test  = inverse_close_only_multi(sc, y_te.reshape(-1,1).flatten(), reference_row_test_local)

                        if len(true_test) != len(pred_test):
                            mlen = min(len(true_test), len(pred_test))
                            true_test = true_test[:mlen]
                            pred_test = pred_test[:mlen]

                        metrics = compute_metrics(true_test, pred_test)

                        forecast = generate_rolling_forecast(m, sc, df, forecast_days=forecast_days_opt, window=w)

                        results[opt_name] = {
                            "pred_train": pred_train,
                            "pred_val": pred_val,
                            "pred_test": pred_test,
                            "true_train": true_train,
                            "true_val": true_val,
                            "true_test": true_test,
                            "metrics": metrics,
                            "forecast": forecast,
                            "loss": [],
                            "val_loss": []
                        }

                    st.session_state.opt_results = results
                    st.success('Loaded and compared pretrained models successfully')

    # SHOW RESULTS (unchanged rendering logic)
    if st.session_state.opt_results:

        res = st.session_state.opt_results

        # metrics table
        mt = []
        for k, v in res.items():
            m = v['metrics']
            mt.append({
                'Optimizer': k,
                'RMSE': m['RMSE'],
                'MAE': m['MAE'],
                'R2': m['R2'],
                'MAPE': m['MAPE'],
                # 'Accuracy': m['Accuracy']
            })
        metrics_df = pd.DataFrame(mt).set_index('Optimizer')
        st.subheader("Metrics Comparison")
        st.table(metrics_df)

        # ============================================================
        #   FIND BEST OPTIMIZER (Based on RMSE)
        # ============================================================

        results = st.session_state.opt_results

        best_opt = None
        best_rmse = float("inf")

        for opt_name, res_dict in results.items():
            rmse = res_dict["metrics"]["RMSE"]
            if rmse < best_rmse:
                best_rmse = rmse
                best_opt = opt_name

        st.markdown(f"""
        ### 🏆 Best Optimizer Result

        **Best Optimizer for {coin}:**  
        👉 <span style='color:#00E0B8; font-size:22px; font-weight:700;'>{best_opt}</span>  
        **Lowest RMSE:** {best_rmse:.4f}

        """, unsafe_allow_html=True)


        # st.subheader("Optimizer Loss Comparison")

        # fig_loss, ax_loss = plt.subplots(figsize=(10,4))

        # for k, v in res.items():
        #     ax_loss.plot(v["loss"], label=f"{k} Train Loss")
        #     ax_loss.plot(v["val_loss"], linestyle='dashed', label=f"{k} Val Loss")

        # ax_loss.set_xlabel("Epoch")
        # ax_loss.set_ylabel("Loss (MSE)")
        # ax_loss.legend()
        # st.pyplot(fig_loss)

        # training comparison
        df_full = download_data(coin_map[coin])
        st.subheader("Prediction Comparison")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_full.index, df_full['Close'], label='Actual', linewidth=2)

        window = 60

        X_train = st.session_state.X_train
        X_val   = st.session_state.X_val
        X_test  = st.session_state.X_test

        train_len = len(X_train)
        val_len   = len(X_val)
        test_len  = len(X_test)

        train_dates = df_full.index[window : window + train_len]

        val_start = window + train_len
        val_end   = val_start + val_len
        val_dates = df_full.index[val_start:val_end]

        test_start = val_end
        test_end   = test_start + test_len
        test_dates = df_full.index[test_start:test_end]

        for k, v in res.items():
            ax.plot(train_dates, v["pred_train"], label=f"{k} Train")
            ax.plot(val_dates,   v["pred_val"],   label=f"{k} Val")
            ax.plot(test_dates,  v["pred_test"],  label=f"{k} Test")

        ax.legend()
        st.pyplot(fig)

        # ZOOM 60 and other plots (logic unchanged)
        st.subheader("Optimizer Comparison — Zoom 60 Days")

        zoom_days = 60
        df_zoom = df_full.iloc[-zoom_days:]

        figz, axz = plt.subplots(figsize=(12,5))
        axz.plot(df_zoom.index, df_zoom['Close'], label="Actual", linewidth=2)

        mask_test_zoom = (test_dates >= df_zoom.index[0])

        colors = {"SGD":"orange", "Adam":"cyan", "RMSProp":"magenta"}

        for opt_name, v in res.items():
            axz.plot(test_dates[mask_test_zoom], v["pred_test"][mask_test_zoom],
                    label=f"{opt_name} Test", linewidth=2, color=colors[opt_name])

        axz.set_title("Optimizer Comparison — Zoomed Test Prediction (Last 60 days)")
        axz.legend()
        st.pyplot(figz)

        st.subheader("Optimizer Comparison — Zoomed Test Prediction + Forecast (Last 60 days)")

        zoom_days = 60
        df_zoom = df_full.iloc[-zoom_days:].copy()

        fig4, ax4 = plt.subplots(figsize=(12,5))

        ax4.plot(df_zoom.index, df_zoom['Close'], label='Actual', linewidth=2, color='lightblue')

        colors = {"SGD":"orange", "Adam":"cyan", "RMSProp":"magenta"}

        for opt_name, v in res.items():
            test_start = window + train_len + val_len
            test_end   = test_start + test_len
            test_dates = df_full.index[test_start:test_end]

            mask_zoom = (test_dates >= df_zoom.index[0])
            zoom_pred = v["pred_test"][mask_zoom]
            zoom_dates = test_dates[mask_zoom]

            ax4.plot(
                zoom_dates,
                zoom_pred,
                label=f"{opt_name} Test",
                linewidth=2,
                color=colors[opt_name]
            )

        last_actual_date = df_full.index[-1]
        future_dates = pd.date_range(
        start=last_actual_date,
        periods=forecast_days_opt + 1
            )[1:]

        for opt_name, v in res.items():
            forecast = v["forecast"]
            m = min(len(future_dates), len(forecast))
            plot_dates = future_dates[:m]
            plot_forecast = forecast[:m]

            ax4.plot(
                plot_dates,
                plot_forecast,
                linestyle='dashed',
                linewidth=2.5,
                color=colors[opt_name],
                label=f"{opt_name} Forecast"
            )

        ax4.set_title("Optimizer Comparison — Zoomed Test Prediction (Last 60 days) + Forecast")
        ax4.legend()
        st.pyplot(fig4)


# ============================================================
# CLOSE MAIN CARD
# ============================================================

st.markdown("</div>", unsafe_allow_html=True)






