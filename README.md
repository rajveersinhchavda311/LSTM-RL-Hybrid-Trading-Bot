# AI Stock Predictor Trading System

A hybrid AI-powered algorithmic trading system that combines Long Short-Term Memory (LSTM) neural networks for stock price prediction with reinforcement learning (RL) for strategy optimization. The system analyzes historical stock data, generates trading signals, and provides backtesting capabilities for evaluating performance.

## Features

- **LSTM-Based Price Prediction**: Uses deep learning to forecast stock prices and resolve ambiguous trading signals
- **Technical Indicators**: Implements Bollinger Bands, RSI, MACD, Ichimoku Cloud, OBV, and more
- **Signal Generation**: Combines technical analysis with Fibonacci retracements and support/resistance levels
- **Reinforcement Learning Environment**: Custom Gymnasium environment for RL-based strategy optimization
- **Backtesting Pipeline**: Comprehensive evaluation of trading strategies with P/L analysis and win rates
- **Model Optimization**: Automatic conversion to TensorFlow Lite and ONNX formats for deployment
- **Web Interface**: Flask-based web application for real-time analysis (under development)
- **Visualization**: Charts and plots for signal analysis and performance metrics

## Architecture

```
Data Acquisition → Feature Engineering → LSTM Prediction → Signal Generation → RL Optimization → Backtesting → Visualization
```

## Technologies Used

- **Machine Learning**: TensorFlow/Keras, Stable Baselines3, Scikit-learn
- **Data Processing**: Pandas, NumPy, yfinance
- **Technical Analysis**: TA-Lib, Custom indicators
- **Reinforcement Learning**: Gymnasium, PPO algorithm
- **Visualization**: Matplotlib, Plotly
- **Web Framework**: Flask
- **Model Formats**: TensorFlow Lite, ONNX

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for data download

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajveersinhchavda311/ALGO_TRADING.git
   cd ALGO_TRADING
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Activate (Windows)
   venv\Scripts\activate
   # Activate (Linux/Mac)
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Complete Analysis Pipeline

Run the main analysis script for full pipeline execution:

```bash
python src/main.py
```

This will:
- Download 5 years of historical data for IOC.NS (configurable)
- Calculate technical indicators
- Train LSTM model for price prediction
- Generate trading signals
- Run backtesting simulation
- Display performance metrics and visualizations

### Web Application

Start the Flask web interface:

```bash
python app.py
```

Access at `http://127.0.0.1:5000` (interface under development)

### Custom Stock Analysis

Edit `src/main.py` to change the ticker symbol:

```python
ticker = "YOUR_STOCK_SYMBOL.NS"  # e.g., "RELIANCE.NS"
```

## Project Structure

```
├── src/
│   ├── main.py              # Main analysis pipeline
│   ├── lstm_model.py        # LSTM model implementation
│   ├── rl_trading_env.py    # RL environment
│   ├── indicators.py        # Technical indicators
│   ├── signals.py           # Signal generation logic
│   ├── support_resistance.py # S/R level detection
│   ├── fibo.py              # Fibonacci calculations
│   ├── visualization.py     # Plotting functions
│   └── web_analysis.py      # Web scraping utilities
├── scripts/
│   └── generate_report.py   # Report generation script
├── models/                  # Saved models (ignored)
├── data/                    # Downloaded data (ignored)
├── outputs/                 # Generated outputs (ignored)
├── notebooks/               # Jupyter notebooks
├── templates/               # Flask templates
├── app.py                   # Flask application
├── requirements.txt         # Python dependencies
└── README.md
```

## Configuration

### Model Parameters
- **Lookback Window**: 60 days for LSTM input
- **Prediction Horizon**: 5 days ahead
- **Training Epochs**: 5 (increase for better performance)
- **Batch Size**: 32

### Trading Parameters
- **Initial Capital**: ₹100,000
- **Stop Loss**: 1.5x ATR
- **Take Profit**: Symmetric to stop loss

## Results

The system provides:
- Trading signals with confidence scores
- Backtesting results (P/L, win rate, trade log)
- Performance visualizations
- Optimized models in multiple formats

## Future Enhancements

- Complete RL training implementation
- Real-time trading integration
- Multiple asset portfolio optimization
- Advanced risk management
- Web interface completion
- API endpoints for external integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Use at your own risk.

## Disclaimer

This is not financial advice. Algorithmic trading involves significant risk of loss. Always backtest thoroughly and never risk more than you can afford to lose.
