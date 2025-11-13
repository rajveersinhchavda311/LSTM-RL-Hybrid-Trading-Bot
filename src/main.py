import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf # Import TensorFlow
import tf2onnx # Import tf2onnx for ONNX conversion
import sys # New import for system path diagnostics
import os # Import os for environment variables
import random # Import random module
import time # Import time for retries
import requests # Import requests for handling exceptions

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure reproducibility for TensorFlow operations
tf.config.experimental.enable_op_determinism()

try:
    from stable_baselines3 import PPO # New import for RL
    print("Successfully imported stable_baselines3.PPO")
except ImportError as e:
    print(f"Error importing stable_baselines3.PPO: {e}")
    print("Python Path (sys.path):")
    for p in sys.path:
        print(f"  {p}")
    print("Please ensure stable-baselines3 is installed in your active virtual environment.")
    sys.exit(1) # Exit if essential module is not found

from indicators import calculate_indicators
from signals import generate_signals
from support_resistance import detect_support_resistance, detect_fractals, kmeans_support_resistance
from fibo import fibonacci_levels
from visualization import plot_signals
from lstm_model import prepare_lstm_data, build_lstm_model, predict_lstm_price
from rl_trading_env import TradingEnv # New import for RL environment

# 1. Data Preparation
# -------------------
ticker = "IOC.NS"  # IOCL (Indian Oil Corporation) on NSE

max_retries = 5
retries = 0
data = pd.DataFrame() # Initialize empty DataFrame

while retries < max_retries:
    try:
        print(f"Attempting to download data for {ticker} (Attempt {retries + 1}/{max_retries})")
        data = yf.download(ticker, period="5y", interval="1d", timeout=15) # Increased timeout
        if not data.empty:
            print(f"Successfully downloaded data for {ticker}.")
            break
        else:
            print(f"Downloaded data for {ticker} is empty. Retrying...")

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception) as e:
        print(f"Error downloading data for {ticker}: {e}")
        print(f"Retrying in 5 seconds...")
        time.sleep(5)
    retries += 1

if data.empty:
    print(f"Failed to download data for {ticker} after {max_retries} attempts. Exiting.")
    sys.exit(1)

data.reset_index(inplace=True)

# Flatten MultiIndex columns if present (e.g., ('Close', 'ITC.NS') -> 'Close')
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if col[0] else col[1] for col in data.columns.values]

# Ensure all price columns are 1D pandas Series, not DataFrames or 2D arrays
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        arr = data[col]
        # If it's a DataFrame, flatten to Series
        if isinstance(arr, pd.DataFrame):
            data[col] = arr.iloc[:, 0]
        # If it's a numpy array with shape (n, 1), flatten it
        elif hasattr(arr, 'values') and len(arr.values.shape) == 2 and arr.values.shape[1] == 1:
            data[col] = pd.Series(arr.values.flatten(), index=data.index)
        # If it's not a Series, convert to Series
        elif not isinstance(arr, pd.Series):
            data[col] = pd.Series(arr, index=data.index)

# 2. Indicator Calculation
# ------------------------
data = calculate_indicators(data)

# 3. ATR-based Dynamic Stop-Loss/Take-Profit
# ------------------------------------------
data['stop_loss'] = data['Close'] - 1.5 * data['atr']
data['take_profit'] = data['Close'] + (data['Close'] - data['stop_loss'])

# 4. Signal Generation
# --------------------
fib_levels = fibonacci_levels(data)
data = generate_signals(data, fib_levels=fib_levels)

# Calculate Support/Resistance Levels (moved here to ensure presence before LSTM prep)
# -----------------------------------
print("Type of data before support/resistance:", type(data))

# Standard S/R
resistance, support = detect_support_resistance(data)
data['resistance'], data['support'] = resistance, support
# Fractal S/R
data['fract_resistance'], data['fract_support'] = detect_fractals(data)
# K-means S/R (as horizontal lines)
kmeans_levels = kmeans_support_resistance(data, n_clusters=5)

print("\n--- DataFrame state BEFORE dropna ---")
print(data.info()) # This will show non-null counts
print(data.head())
print("-------------------------------------")

# Drop NaN values introduced by indicator and signal calculations to clean data for LSTM and RL
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

print("\n--- DataFrame state AFTER dropna ---")
print(data.info()) # Check again after dropping NaNs
print(data.head())
print("------------------------------------")

print("\n--- DataFrame state before LSTM data preparation ---")
print("Columns:", data.columns.tolist())
print("Is DataFrame empty:", data.empty)
if not data.empty:
    print("Head of DataFrame:", data.head())
    print("Tail of DataFrame:", data.tail())
print("--------------------------------------------------")

# Define features for LSTM
lstm_features = [
    'Close', 'Open', 'High', 'Low', 'Volume',
    'bb_upper', 'bb_lower', 
    'macd_diff', # MACD_hist as macd_diff
    'rsi',
    'obv',
    'ichimoku_a', 'ichimoku_b' # Ichimoku_base/conv as ichimoku_a/b
]
n_steps = 60 # Look-back window for LSTM
target_days = 5 # Predict 5 days into the future

# Prepare LSTM data
X, y_price, y_signal_confidence, scaler = prepare_lstm_data(data, lstm_features, n_steps, target_days)

# Split data into training and testing sets for LSTM
X_train, X_test, y_price_train, y_price_test, y_signal_train, y_signal_test = train_test_split(
    X, y_price, y_signal_confidence, test_size=0.2, random_state=42
)

# Build and train LSTM model (placeholder - actual training takes time)
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = build_lstm_model(input_shape)

print("\n--- Starting LSTM Model Training (this may take a while) ---")
# For demonstration, we'll use a small number of epochs
lstm_model.fit(X_train, np.column_stack((y_price_train, y_signal_train)), epochs=5, batch_size=32, verbose=1)
print("--- LSTM Model Training Complete ---")

# --- Real-Time Inference Optimization ---
print("\n--- Starting Model Optimization (Quantization & ONNX Conversion) ---")

# Model Quantization (TensorFlow Lite)
converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Apply the suggested flags to handle dynamic shapes/tensor list ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the TFLite model
with open("optimized_lstm_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved optimized_lstm_model.tflite")

# Save the Keras model as a SavedModel for more robust ONNX conversion
saved_model_path = "lstm_saved_model"
tf.saved_model.save(lstm_model, saved_model_path)
print(f"Saved Keras model as SavedModel at {saved_model_path}")

# ONNX Conversion from SavedModel
try:
    onnx_model, _ = tf2onnx.convert.from_saved_model(saved_model_path, output_path="optimized_lstm_model.onnx")
    print("Saved optimized_lstm_model.onnx")
except Exception as e:
    print(f"ONNX Conversion failed: {e}")
    print("You might need to install tf2onnx: pip install tf2onnx")

print("--- Model Optimization Complete ---")

# Get the index of 'Close' in the features list for inverse scaling
close_feature_index = lstm_features.index('Close')

# 5. Ambiguity Resolver using LSTM Prediction
# -------------------------------------------
# Iterate through data to resolve ambiguous signals
for i in range(len(data) - n_steps + 1):
    if data['final_signal'].iloc[i + n_steps - 1] == 'Ambiguous':
        # Get the sequence for prediction
        current_sequence = data[lstm_features].iloc[i : i + n_steps].values
        scaled_current_sequence = scaler.transform(current_sequence)
        
        # Predict future price and confidence
        predicted_price_5d, predicted_confidence_lstm = predict_lstm_price(lstm_model, scaled_current_sequence, scaler, close_feature_index)
        
        # Resolve ambiguity based on LSTM prediction
        if predicted_price_5d > data['Close'].iloc[i + n_steps -1]: # Compare predicted price to current price
            data.loc[i + n_steps -1, 'final_signal'] = 'Buy (LSTM)'
            data.loc[i + n_steps -1, 'confidence'] = int(predicted_confidence_lstm * 100) # Convert back to 0-100
        else:
            data.loc[i + n_steps -1, 'final_signal'] = 'Sell (LSTM)'
            data.loc[i + n_steps -1, 'confidence'] = int(predicted_confidence_lstm * 100) # Convert back to 0-100

# 6. Backtesting Strategy
# -----------------------
print("\n--- Starting Backtesting ---")

initial_capital = 100000.0  # Starting capital for backtesting
capital = initial_capital
position = 0  # 0 means no position, 1 means long position
trade_entry_price = 0
total_profit_loss = 0
num_trades = 0
num_wins = 0

trade_log = [] # To store details of each trade

for i in range(len(data)):
    signal = data['final_signal'].iloc[i]
    current_price = data['Close'].iloc[i]
    trade_date = data['Date'].iloc[i]

    if 'Buy' in signal and position == 0:
        # Enter a long position
        position = 1
        trade_entry_price = current_price
        print(f"Buy Signal at {trade_date} - Entry Price: {trade_entry_price:.2f}")

    elif 'Sell' in signal and position == 1:
        # Exit the long position
        position = 0
        profit_loss = (current_price - trade_entry_price) / trade_entry_price * 100 # Percentage profit/loss
        
        # Assume a fixed share size or capital deployment for simplicity.
        # For simplicity, let's assume we invest 100% of available capital
        # This is a very simplified model; real backtesting needs position sizing
        trade_value = capital # simplified: trade with current capital
        pnl_cash = trade_value * (current_price - trade_entry_price) / trade_entry_price
        
        capital += pnl_cash
        total_profit_loss += pnl_cash
        num_trades += 1

        if profit_loss > 0:
            num_wins += 1
            outcome = "Win"
        else:
            outcome = "Loss"
        
        trade_log.append({
            'Date': trade_date,
            'Type': 'Sell',
            'Entry Price': trade_entry_price,
            'Exit Price': current_price,
            'P/L (%)': profit_loss,
            'P/L (Cash)': pnl_cash,
            'Capital After': capital,
            'Outcome': outcome
        })
        print(f"Sell Signal at {trade_date} - Exit Price: {current_price:.2f} | P/L: {profit_loss:.2f}% | Capital: {capital:.2f}")

# Final calculation if a position is still open at the end
if position == 1:
    # Close the position at the last available price
    current_price = data['Close'].iloc[-1]
    trade_date = data['Date'].iloc[-1]
    profit_loss = (current_price - trade_entry_price) / trade_entry_price * 100
    pnl_cash = capital * (current_price - trade_entry_price) / trade_entry_price
    capital += pnl_cash
    total_profit_loss += pnl_cash
    num_trades += 1 # Count as a trade
    
    if profit_loss > 0:
        num_wins += 1
        outcome = "Win"
    else:
        outcome = "Loss"
    
    trade_log.append({
        'Date': trade_date,
        'Type': 'Forced Sell (End of Data)',
        'Entry Price': trade_entry_price,
        'Exit Price': current_price,
        'P/L (%)': profit_loss,
        'P/L (Cash)': pnl_cash,
        'Capital After': capital,
        'Outcome': outcome
    })
    print(f"Forced Sell at {trade_date} (End of Data) - Exit Price: {current_price:.2f} | P/L: {profit_loss:.2f}% | Capital: {capital:.2f}")


print("\n--- Backtesting Results ---")
print(f"Initial Capital: {initial_capital:.2f}")
print(f"Final Capital: {capital:.2f}")
print(f"Total P/L: {total_profit_loss:.2f}")
print(f"Number of Trades: {num_trades}")
if num_trades > 0:
    win_rate = (num_wins / num_trades) * 100
    print(f"Win Rate: {win_rate:.2f}%")
else:
    print("No trades were executed.")

print("\n--- Trade Log (Last 5 Trades) ---")
for trade in trade_log[-5:]:
    print(trade)

print("--- Backtesting Complete ---")

# 7. Reinforcement Learning (RL) Integration (Placeholder)
# --------------------------------------------------------
print("\n--- Starting Reinforcement Learning Integration (Placeholder) ---")

# Initialize the custom trading environment
# Ensure data fed to TradingEnv has all necessary features and is cleaned.
rl_env = TradingEnv(df=data.copy(), lookback_window=lstm_features.__len__() ) # Use a copy to avoid modifying original df for RL

# Define and train the PPO model (placeholder for actual training)
# The policy network for PPO will learn to map observations to actions.
model = PPO("MlpPolicy", rl_env, verbose=1) # MlpPolicy for multi-layer perceptron policy

# In a real scenario, you would train the model for many timesteps:
# model.learn(total_timesteps=100000) # This will take a long time to run
print("PPO model initialized. Training would go here.\n(Skipping extensive training for demonstration purposes)")

print("--- Reinforcement Learning Integration Complete (Placeholder) ---")

# 8. Visualization
# ----------------
print("\n--- DataFrame state before plotting ---")
print("Columns:", data.columns.tolist())
print("Is 'support' in columns:", 'support' in data.columns)
print("Is DataFrame empty:", data.empty)
if not data.empty:
    print("Head of DataFrame (before plot):", data[['Date', 'Close', 'support', 'resistance']].head())
    print("Tail of DataFrame (before plot):", data[['Date', 'Close', 'support', 'resistance']].tail())
print("---------------------------------------")

# Pass kmeans_levels to plot_signals if you want to plot them
plot_signals(data, fib_levels=fib_levels)
# Optionally, plot K-means S/R as horizontal lines
for level in kmeans_levels:
    plt.axhline(level, color='grey', linestyle='-.', alpha=0.7, label='KMeans S/R')

# Backtesting Visualization (Simple)
plt.figure(figsize=(16, 8))
plt.plot(data['Date'], data['Close'], label='Close Price', alpha=0.7)

# Plot buy and sell signals
buy_signals = data[data['final_signal'].str.contains('Buy', na=False)]
sell_signals = data[data['final_signal'].str.contains('Sell', na=False)]
ambiguous_signals = data[data['final_signal'].str.contains('Ambiguous', na=False)]

plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal', alpha=1)
plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal', alpha=1)
plt.scatter(ambiguous_signals['Date'], ambiguous_signals['Close'], marker='x', color='orange', s=100, label='Ambiguous Signal', alpha=1)

plt.title(f'Stock Signals for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Print actionable buy/sell recommendations
latest_signal = data.iloc[-1]['final_signal']
latest_date = data.iloc[-1]['Date']
latest_close = data.iloc[-1]['Close']
latest_confidence = data.iloc[-1]['confidence']
print(f"\nLatest signal for {ticker} on {latest_date}: {latest_signal} with confidence {latest_confidence} at price {latest_close}")

# Optionally, print all recent signals (including 'Hold')
print("\nRecent signals (all types) for the last 15 days:")
print(data[['Date', 'Close', 'final_signal', 'confidence']].tail(15))

# --- Future Price Prediction and Visualization ---
print("\n--- Future Price Prediction ---")

# Get the latest data sequence for prediction
# Ensure we have enough data points for the lookback window
if len(data) >= n_steps:
    latest_sequence_df = data[lstm_features].iloc[-n_steps:]
    latest_scaled_sequence = scaler.transform(latest_sequence_df.values)
    
    # Make prediction
    predicted_price_5d, predicted_confidence_lstm = predict_lstm_price(lstm_model, latest_scaled_sequence, scaler, close_feature_index)
    
    print(f"Predicted Close price in {target_days} days: {predicted_price_5d:.2f}")
    print(f"Predicted signal confidence: {int(predicted_confidence_lstm * 100)}%")

    # Plotting the prediction
    plt.figure(figsize=(10, 6))
    
    # Plot recent historical data
    recent_data = data.tail(n_steps + target_days) # Get enough recent data for context
    plt.plot(recent_data['Date'], recent_data['Close'], label='Historical Close Price', color='blue')
    
    # Plot the predicted point
    last_historical_date = data['Date'].iloc[-1]
    # Create a future date for the prediction point (approx. target_days from last historical date)
    future_date = last_historical_date + pd.Timedelta(days=target_days)
    
    plt.scatter(future_date, predicted_price_5d, color='red', s=100, label='Predicted Price', zorder=5)
    plt.plot([last_historical_date, future_date], [data['Close'].iloc[-1], predicted_price_5d], linestyle='--', color='orange', label='Prediction Line')

    plt.title(f'LSTM Future Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"Not enough data ({len(data)} rows) for a {n_steps}-day lookback window prediction.")

# --- Current Support and Resistance Levels ---
print("\n--- Current Support and Resistance Levels ---")

# Get the last non-NaN values for S/R columns
current_support = data['support'].iloc[-1]
current_resistance = data['resistance'].iloc[-1]
current_fract_support = data['fract_support'].iloc[-1]
current_fract_resistance = data['fract_resistance'].iloc[-1]

print(f"Latest Standard Support: {current_support:.2f}")
print(f"Latest Standard Resistance: {current_resistance:.2f}")
print(f"Latest Fractal Support: {current_fract_support:.2f}")
print(f"Latest Fractal Resistance: {current_fract_resistance:.2f}")

# K-means levels are already a list of distinct levels
if kmeans_levels:
    print("K-Means Support/Resistance Levels:")
    for i, level in enumerate(kmeans_levels):
        print(f"  Level {i+1}: {level:.2f}")
else:
    print("No K-Means Support/Resistance Levels detected.")

# --- Adaptive RSI (Future Enhancement) ---
# Implementing GNQTS algorithm for adaptive RSI is complex and would require significant
# additional code. This is noted as a future, more advanced enhancement.

# --- Real-time Inference Optimization (Deployment Concern) ---
# Optimization for real-time inference typically involves model quantization, TF-Lite conversion,
# and deployment strategies, which are beyond the scope of this script but crucial for production.