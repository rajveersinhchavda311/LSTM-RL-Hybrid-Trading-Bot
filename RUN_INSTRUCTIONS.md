# How to Run Stock Suggester AI

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading stock data)

## Setup Instructions

### Option 1: Using Existing Virtual Environment

If you already have a virtual environment set up:

1. **Activate the virtual environment:**
   - **PowerShell** (in project root):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Command Prompt** (in project root):
     ```cmd
     venv\Scripts\activate.bat
     ```

2. **Install/Update dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Create New Virtual Environment

1. **Create a new virtual environment:**
   - **PowerShell** (in project root):
     ```powershell
     python -m venv venv
     ```
   - **Command Prompt** (in project root):
     ```cmd
     python -m venv venv
     ```

2. **Activate the virtual environment:**
   - **PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Command Prompt:**
     ```cmd
     venv\Scripts\activate.bat
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Method 1: Run Main Analysis Script (Recommended)

This runs the complete stock analysis pipeline including:
- Data download
- Technical indicator calculation
- LSTM model training
- Signal generation
- Backtesting
- Visualization

**PowerShell or Command Prompt** (with venv activated, in project root):
```bash
python src/main.py
```

**What to expect:**
- The script will download 5 years of data for IOC.NS (Indian Oil Corporation)
- It will train an LSTM model (takes a few minutes)
- Generate buy/sell signals
- Run backtesting
- Display charts and results
- Save optimized models (TFLite and ONNX formats)

**Note:** The first run may take 5-10 minutes depending on your system. Subsequent runs will be faster if models are cached.

### Method 2: Run Flask Web Application

The Flask app is currently a skeleton. To run it:

**PowerShell or Command Prompt** (with venv activated, in project root):
```bash
python app.py
```

**What to expect:**
- Flask server starts on `http://127.0.0.1:5000`
- Currently, it will error because `templates/index.html` doesn't exist
- You'll need to create the frontend templates first

## Changing the Stock Ticker

To analyze a different stock, edit `src/main.py`:

1. Open `src/main.py`
2. Find line 45: `ticker = "IOC.NS"`
3. Change it to your desired ticker (e.g., `"AAPL"`, `"RELIANCE.NS"`, `"TCS.NS"`)
4. Save and run again

**Format:**
- US stocks: `"AAPL"`, `"GOOGL"`, `"MSFT"`
- Indian stocks (NSE): `"RELIANCE.NS"`, `"TCS.NS"`, `"INFY.NS"`
- Indian stocks (BSE): `"RELIANCE.BO"`, `"TCS.BO"`

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Make sure your virtual environment is activated and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: TensorFlow/GPU errors
**Solution:** If you have GPU issues, TensorFlow will fall back to CPU automatically. For CPU-only installation:
```bash
pip install tensorflow-cpu
```

### Issue: Data download fails
**Solution:** 
- Check your internet connection
- The script has retry logic (5 attempts)
- Some tickers may not be available on yfinance
- Try a different ticker symbol

### Issue: Out of memory during LSTM training
**Solution:**
- Reduce the number of epochs in `src/main.py` (line 165)
- Reduce `n_steps` (lookback window) in `src/main.py` (line 148)
- Close other applications to free up RAM

### Issue: PowerShell execution policy error
**Solution:** Run this in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Project Structure

```
stock-suggester-ai/
├── src/                    # Main source code
│   ├── main.py            # Main analysis pipeline
│   ├── indicators.py      # Technical indicators
│   ├── signals.py         # Signal generation
│   ├── lstm_model.py      # LSTM model
│   ├── rl_trading_env.py  # RL environment
│   └── ...
├── app.py                 # Flask web app
├── requirements.txt       # Python dependencies
├── venv/                  # Virtual environment
├── data/                  # Data storage (created automatically)
├── models/                # Model storage (created automatically)
└── outputs/               # Output files (created automatically)
```

## Output Files

After running, you'll find:
- `optimized_lstm_model.tflite` - Optimized TensorFlow Lite model
- `optimized_lstm_model.onnx` - ONNX format model
- `lstm_saved_model/` - Saved TensorFlow model directory
- Charts displayed in matplotlib windows
- Console output with backtesting results

## Next Steps

1. **Customize parameters:** Edit signal thresholds, LSTM architecture, etc.
2. **Add more features:** Implement additional technical indicators
3. **Complete Flask app:** Create frontend templates and connect to backend
4. **Train RL model:** Uncomment and run the RL training (line 340 in main.py)
5. **Deploy:** Use the optimized models for real-time predictions

