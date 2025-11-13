"""
Simplified analysis function for web interface
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from indicators import calculate_indicators
from signals import generate_signals
from support_resistance import detect_support_resistance, detect_fractals
from fibo import fibonacci_levels

def run_quick_analysis(ticker):
    """
    Run a quick analysis without LSTM training (for web interface)
    Returns a dictionary with results
    """
    # Download data
    max_retries = 5
    retries = 0
    data = pd.DataFrame()
    
    while retries < max_retries:
        try:
            print(f"Downloading data for {ticker} (Attempt {retries + 1}/{max_retries})")
            data = yf.download(ticker, period="2y", interval="1d", timeout=15)
            if not data.empty:
                break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
        retries += 1
    
    if data.empty:
        raise Exception(f"Failed to download data for {ticker}")
    
    data.reset_index(inplace=True)
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] else col[1] for col in data.columns.values]
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Calculate stop-loss/take-profit
    data['stop_loss'] = data['Close'] - 1.5 * data['atr']
    data['take_profit'] = data['Close'] + (data['Close'] - data['stop_loss'])
    
    # Generate signals
    fib_levels = fibonacci_levels(data)
    data = generate_signals(data, fib_levels=fib_levels)
    
    # Support/Resistance
    resistance, support = detect_support_resistance(data)
    data['resistance'], data['support'] = resistance, support
    
    # Drop NaN values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Simple backtesting
    initial_capital = 100000.0
    capital = initial_capital
    position = 0
    trade_entry_price = 0
    total_profit_loss = 0
    num_trades = 0
    num_wins = 0
    
    for i in range(len(data)):
        signal = data['final_signal'].iloc[i]
        current_price = data['Close'].iloc[i]
        
        if 'Buy' in signal and position == 0:
            position = 1
            trade_entry_price = current_price
        
        elif 'Sell' in signal and position == 1:
            position = 0
            profit_loss = (current_price - trade_entry_price) / trade_entry_price * 100
            trade_value = capital
            pnl_cash = trade_value * (current_price - trade_entry_price) / trade_entry_price
            capital += pnl_cash
            total_profit_loss += pnl_cash
            num_trades += 1
            if profit_loss > 0:
                num_wins += 1
    
    # Close any open position
    if position == 1:
        current_price = data['Close'].iloc[-1]
        profit_loss = (current_price - trade_entry_price) / trade_entry_price * 100
        pnl_cash = capital * (current_price - trade_entry_price) / trade_entry_price
        capital += pnl_cash
        total_profit_loss += pnl_cash
        num_trades += 1
        if profit_loss > 0:
            num_wins += 1
    
    # Get latest values
    latest = data.iloc[-1]
    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
    
    # Get last 100 days for charts (or all if less than 100)
    chart_data = data.tail(100).copy()
    
    # Prepare chart data
    dates = chart_data['Date'].astype(str).tolist() if 'Date' in chart_data.columns else [str(i) for i in range(len(chart_data))]
    prices = chart_data['Close'].round(2).tolist()
    volumes = chart_data['Volume'].tolist()
    rsi_values = chart_data['rsi'].round(2).tolist() if 'rsi' in chart_data.columns else []
    macd_values = chart_data['macd'].round(4).tolist() if 'macd' in chart_data.columns else []
    macd_signal_values = chart_data['macd_signal'].round(4).tolist() if 'macd_signal' in chart_data.columns else []
    macd_diff_values = chart_data['macd_diff'].round(4).tolist() if 'macd_diff' in chart_data.columns else []
    bb_upper = chart_data['bb_upper'].round(2).tolist() if 'bb_upper' in chart_data.columns else []
    bb_lower = chart_data['bb_lower'].round(2).tolist() if 'bb_lower' in chart_data.columns else []
    support_levels = chart_data['support'].round(2).tolist() if 'support' in chart_data.columns else []
    resistance_levels = chart_data['resistance'].round(2).tolist() if 'resistance' in chart_data.columns else []
    
    # Calculate performance over time for backtesting chart
    performance_data = []
    running_capital = initial_capital
    running_position = 0
    running_entry = 0
    
    for idx in chart_data.index:
        signal = chart_data.loc[idx, 'final_signal'] if 'final_signal' in chart_data.columns else 'Hold'
        current_price = chart_data.loc[idx, 'Close']
        
        if 'Buy' in str(signal) and running_position == 0:
            running_position = 1
            running_entry = current_price
        elif 'Sell' in str(signal) and running_position == 1:
            running_position = 0
            pnl_pct = (current_price - running_entry) / running_entry
            running_capital = running_capital * (1 + pnl_pct)
        
        if running_position == 1:
            current_value = running_capital * (current_price / running_entry)
        else:
            current_value = running_capital
        
        performance_data.append(round(current_value, 2))
    
    # Calculate target and stop loss based on current signal
    current_price = float(latest['Close'])
    stop_loss_price = float(latest['stop_loss']) if 'stop_loss' in latest else current_price - (1.5 * float(latest['atr']))
    target_price = float(latest['take_profit']) if 'take_profit' in latest else current_price + (current_price - stop_loss_price)
    
    # Return results with additional data
    return {
        'ticker': ticker,
        'signal': latest['final_signal'],
        'confidence': int(latest['confidence']),
        'price': round(current_price, 2),
        'predicted_price': round(float(latest['Close'] * 1.02), 2),
        'support': round(float(latest['support']), 2),
        'resistance': round(float(latest['resistance']), 2),
        'target_price': round(target_price, 2),
        'stop_loss': round(stop_loss_price, 2),
        'initial_capital': round(initial_capital, 2),
        'final_capital': round(capital, 2),
        'profit_loss': round(total_profit_loss, 2),
        'win_rate': round(win_rate, 2),
        'num_trades': num_trades,
        # Additional indicators
        'rsi': round(float(latest['rsi']), 2) if 'rsi' in latest else None,
        'macd': round(float(latest['macd']), 4) if 'macd' in latest else None,
        'macd_signal': round(float(latest['macd_signal']), 4) if 'macd_signal' in latest else None,
        'macd_diff': round(float(latest['macd_diff']), 4) if 'macd_diff' in latest else None,
        'volume': int(latest['Volume']) if 'Volume' in latest else None,
        'atr': round(float(latest['atr']), 2) if 'atr' in latest else None,
        'bb_upper': round(float(latest['bb_upper']), 2) if 'bb_upper' in latest else None,
        'bb_lower': round(float(latest['bb_lower']), 2) if 'bb_lower' in latest else None,
        'obv': round(float(latest['obv']), 2) if 'obv' in latest else None,
        # Chart data
        'chart_dates': dates,
        'chart_prices': prices,
        'chart_volumes': volumes,
        'chart_rsi': rsi_values,
        'chart_macd': macd_values,
        'chart_macd_signal': macd_signal_values,
        'chart_macd_diff': macd_diff_values,
        'chart_bb_upper': bb_upper,
        'chart_bb_lower': bb_lower,
        'chart_support': support_levels,
        'chart_resistance': resistance_levels,
        'chart_performance': performance_data
    }

