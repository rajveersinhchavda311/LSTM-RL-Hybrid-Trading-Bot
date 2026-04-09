**Stock Suggester AI — Repository Report (Expanded)**

Generated: 2025-12-02

**Executive Summary**
- **Purpose:** `stock-suggester-ai` is a research/prototype project that combines technical indicators, an LSTM forecasting model, and optional RL components to generate buy/sell signals and evaluate trading strategies via a simple backtester. It includes a lightweight Flask endpoint for quick analyses.
- **Status:** Source code and model artifacts are present. Local `data/`, `notebooks/`, `outputs/`, and `models/` are empty in this workspace snapshot; the pipeline downloads data at runtime. A TensorFlow SavedModel and an optimized TFLite model are present (`lstm_saved_model/` and `optimized_lstm_model.tflite`).

**Repository Root & Important Paths**
- Project root: `c:\Users\rajveersinh chavda\stock-suggester-ai`
- Key files:
  - `src/main.py` — Full pipeline orchestration (data, indicators, LSTM, model optimization, backtest, visualization)
  - `src/lstm_model.py` — Data prep and LSTM model builder/predictor
  - `src/indicators.py`, `src/signals.py` — Feature engineering and rule-based signals
  - `src/web_analysis.py`, `app.py` — Quick web-facing analysis and Flask server
  - `optimized_lstm_model.tflite` — Optimized inference model
  - `lstm_saved_model/` — Full SavedModel (retrainable)
  - `scripts/generate_report.py` — manifest/inspection script (added)

**Environment & Dependencies**
- Primary runtime: Python (3.8+ recommended)
- Core packages (from `requirements.txt`): `yfinance`, `pandas`, `numpy`, `matplotlib`, `ta`, `scikit-learn`, `tensorflow`, `tf2onnx`, `stable-baselines3`, `gymnasium`, `flask`, `requests`
- Notes:
  - If you only need inference, replace `tensorflow` with `tensorflow-cpu` to avoid GPU dependencies on non-GPU machines.
  - `stable-baselines3` and `gymnasium` are only necessary if using the RL components.

**File Inventory (recommended items to include in the final report)**
- Run the manifest generator to produce exact sizes and a CSV listing: `python scripts\generate_report.py`. That script creates `report/manifest.json` and `report/files.csv`.
- Suggested inventory columns for the formal report:
  - Path, File type (code/data/model), Size (MB), SHA256 (optional), Last modified

**Data: sources, schema, and preprocessing**
- Source: `yfinance` (downloaded at runtime). Default in `src/main.py` is `IOC.NS` for 5 years of daily data.
- Raw columns expected: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- Preprocessing performed in pipeline:
  - Flatten yfinance MultiIndex columns if present
  - Calculate technical indicators: Bollinger Bands, RSI, MACD (and macd_diff), OBV, Ichimoku, ATR
  - Add derived columns: `stop_loss`, `take_profit`, support/resistance, Fibonacci levels
  - Drop NaNs before model training and backtesting

**Feature Engineering**
- Primary features used by LSTM (explicit list):
  - `Close, Open, High, Low, Volume, bb_upper, bb_lower, macd_diff, rsi, obv, ichimoku_a, ichimoku_b`
- Suggestions to improve features:
  - Add log returns, multiple SMA/EMA windows, volatility (rolling std), normalized volume, sentiment indicators, holiday flags
  - Use lagged returns and engineered features like momentum and mean-reversion signals

**Model Architecture & Hyperparameters**
- LSTM (from `src/lstm_model.py`):
  - Layers: 3 × LSTM(256) with `return_sequences=True` for first two, `Dropout(0.3)` after each LSTM, final `Dense(2)`
  - Output interpretation: first value → predicted normalized Close; second value → predicted signal confidence
  - Loss: MSE; Optimizer: Adam
- Defaults used in demo pipeline (`src/main.py`):
  - `n_steps` (lookback window): 60
  - `target_days`: 5
  - `epochs`: 5 (demo), `batch_size`: 32
  - Determinism: seeds set (`42`) and TF op determinism enabled where possible

**How to compute parameter count and model sizes**
- Parameter count: load the Keras model and run `model.summary()` (e.g., in a helper script or small notebook).
- SavedModel size: sum of bytes for files under `lstm_saved_model/`.
- TFLite size: `os.path.getsize('optimized_lstm_model.tflite') / (1024*1024)` MB.

**Training procedure & reproducibility**
- Reproduction steps:
  1. Create & activate a venv and install `requirements.txt`.
  2. Optionally set `ticker`, `epochs`, `n_steps` in `src/main.py`.
  3. Run `python src/main.py` to execute the full pipeline (downloads data, trains model, converts and saves artifacts, runs backtest).
- For strict reproducibility, capture commit hash, pin package versions (use `pip freeze`), and run on CPU with TF deterministic ops enabled.

**Evaluation: metrics, calculation and reporting format**
- Regression metrics to compute on a hold-out test set:
  - **MSE** = mean((y_true - y_pred)^2)
  - **RMSE** = sqrt(MSE)
  - **MAE** = mean(|y_true - y_pred|)
  - **R2** (optional)
- Trading/backtest metrics to report:
  - **Initial capital** (e.g., 100,000)
  - **Final capital** and **Total return**
  - **Annualized return** (compounded)
  - **Sharpe ratio** (annualized, assume 252 trading days)
  - **Max drawdown** (peak-to-trough metric)
  - **Win rate**, **#trades**, average P/L per trade
- Suggested output formats:
  - `metrics.json` with numeric summaries
  - `equity_curve.csv` with daily equity values
  - `predictions_test.csv` with columns: `date, actual_close, predicted_close, residual`

**Visualizations to produce for the report**
- Training loss and validation loss vs epochs (PNG)
- Predicted vs Actual close price on test set (PNG)
- Residual histogram and QQ-plot (PNG)
- Equity curve with buy/sell markers (PNG)
- Signal timeline (series of signals over the period) (PNG)
- Feature correlation heatmap (PNG)

**Backtesting methodology (detailed)**
- Current implementation (in `src/main.py`):
  - Entry: when `final_signal` contains 'Buy' and no current position
  - Exit: when `final_signal` contains 'Sell' and position is open
  - Ambiguous signals resolved by LSTM predicted price vs current price
  - Position sizing: 100% of capital (not realistic)
  - Fees/slippage: not modeled in simple backtest (RL env does have `trade_fee_pct`)
- Recommended production changes:
  - Implement position sizing (fixed fraction, volatility-adjusted, or risk-per-trade)
  - Model transaction costs and slippage explicitly
  - Use event-driven or vectorized backtester for realistic fills (e.g., `backtrader`, `zipline`, `vectorbt`)

**Reinforcement Learning integration details**
- The custom environment (`TradingEnv`) provides a flattened observation vector: history of features + portfolio state, action space `Discrete(3)` (Buy, Sell, Hold). Reward approximates percent change in net worth and penalizes losses.
- The project initializes PPO in `src/main.py` as a placeholder. Effective RL deployment requires long training runs and stable environment wrappers.

**Deployment & inference guidance**
- Inference with TFLite (recommended for low-latency deployment):
  - Use the TensorFlow Lite `Interpreter` or `tflite_runtime` for embedded/edge deployments
  - Ensure input sequences are scaled with the same `MinMaxScaler` used during training (save/load scaler)
- Serving options:
  - Lightweight: Flask endpoint (`app.py`) that calls `src/web_analysis.run_quick_analysis` for fast analysis (no LSTM training)
  - Production: FastAPI or a small microservice that loads the TFLite model and exposes a prediction endpoint with authentication and rate limiting

**Risks & limitations**
- Model generalization risk: LSTMs can overfit; results depend heavily on lookback windows, feature set, and data quality
- Data integrity: `yfinance` may return adjusted prices and occasional gaps; validate timestamps and outliers
- Backtest realism: all-in trades and missing fees produce optimistic results; do not use for live trading without improvements

**Reproducibility checklist**
- [ ] Commit hash for the repo used in experiments
- [ ] `requirements.txt` with pinned versions (or `environment.yml` / `pip freeze` output)
- [ ] Random seeds and TF determinism documented (present in `src/main.py`)
- [ ] `report/manifest.json` and `report/files.csv` included (generated by `scripts/generate_report.py`)
- [ ] `outputs/` containing `metrics.json`, `equity_curve.csv`, `predictions_test.csv`, and PNG figures

**Recommended next steps & timeline estimates**
1. Generate manifest and file inventory: `python scripts\generate_report.py` (5 minutes)
2. Run the full pipeline with `epochs=20` to generate training metrics and outputs (30–120 minutes depending on hardware)
3. Replace naive backtester with `vectorbt` or `backtrader` to get accurate equity and transaction cost modeling (2–4 hours)
4. Produce a Jupyter notebook that reproduces the pipeline and saves all figures and metrics in `outputs/` (2–4 hours)
5. Add unit tests and CI to ensure reproducibility and prevent regressions (2–6 hours)

**Appendix: useful commands & snippets**
- Create venv and install dependencies (PowerShell):
```powershell
cd "c:\Users\rajveersinh chavda\stock-suggester-ai"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Generate manifest and file listing:
```powershell
python scripts\generate_report.py
type report\manifest.json
type report\files.csv
```
- Inspect TFLite model (example):
```python
from tensorflow.lite import Interpreter
interp = Interpreter(model_path='optimized_lstm_model.tflite')
interp.allocate_tensors()
print('Inputs:', interp.get_input_details())
print('Outputs:', interp.get_output_details())
```

**Suggested report tables**
- Dataset summary (Ticker, Start, End, #Rows, Frequency)
- Feature summary (Feature, Type, Transformation)
- Model summary (Model name, Layers, Params, Epochs, SavedModel size, TFLite size)
- Backtest summary (Initial capital, Final capital, Total return, Annualized return, Sharpe, Max drawdown, #trades, Win rate)

---

If you want, I can now run `python scripts\generate_report.py` and paste the resulting `report/manifest.json` here, or run `python src\main.py` to produce the training and backtest outputs. Tell me which to run next.
**Stock Suggester AI — Repository Report**

Generated: 2025-12-02

**Executive Summary**
- Purpose: A stock suggestion / prediction repository that uses technical indicators + an LSTM model (and placeholder RL) to generate buy/sell signals and run simple backtests. Includes a small Flask web endpoint for fast analysis.
- Status: Code and model artifacts present. Data, notebooks, and outputs are empty in this copy. A TFLite and SavedModel artifact exist (optimized_lstm_model.tflite, `lstm_saved_model/`).

**Repository Location**
- Root: `c:\Users\rajveersinh chavda\stock-suggester-ai`

**Top-level inventory (important files / folders)**
- `.git/`, `.gitignore`, `.env`
- `app.py` — Flask web API (routes: `/`, `/analyze` which calls `src/web_analysis.run_quick_analysis`).
- `RUN_INSTRUCTIONS.md` — detailed run instructions (useful, non-empty).
- `requirements.txt` — dependency list.
- `optimized_lstm_model.tflite` — TFLite model artifact (present)
- `lstm_saved_model/` — TensorFlow SavedModel directory (contains `saved_model.pb`, `variables/`, `assets/`)
- `models/`, `data/`, `notebooks/`, `outputs/` — present but empty in this workspace copy
- `src/` — core code (see next section)
- `templates/`, `frontend/` — frontend skeleton (templates contains `index.html`) 
- `scripts/generate_report.py` — added to automate manifest and basic model inspection

Note: The `venv/` folder is present in the workspace snapshot — do not commit virtual environment contents to git normally.

**Requirements (from `requirements.txt`)**
- yfinance
- pandas
- matplotlib
- ta
- numpy
- scikit-learn
- tensorflow
- tf2onnx
- stable-baselines3
- gymnasium
- flask
- requests

These are the packages needed to run the main pipeline, quick web analysis, and RL components. If you don't need RL or ONNX conversion you can omit `stable-baselines3`, `gymnasium`, and `tf2onnx`.

**Source code summary (`src/` key modules)**
- `lstm_model.py`
  - prepare_lstm_data(df, features, n_steps, target_days): scales features with `MinMaxScaler`, creates sequences, returns X, y_price, y_signal_confidence, scaler.
  - build_lstm_model(input_shape): 3-layer LSTM network:
    - LSTM(256, return_sequences=True) -> Dropout(0.3)
    - LSTM(256, return_sequences=True) -> Dropout(0.3)
    - LSTM(256) -> Dropout(0.3)
    - Dense(2) (outputs: [price, confidence])
    - compiled with `optimizer='adam'`, `loss='mse'`.
  - predict_lstm_price(model, data_sequence, scaler, close_feature_index): prediction helper which inverse-transforms the Close price using the same scaler.

- `main.py` — full pipeline (data download, indicators, signals, LSTM training, model optimization, backtesting, RL placeholder, visualization)
  - Default ticker: `IOC.NS` (5 years, daily interval)
  - LSTM feature list used:
    - `Close, Open, High, Low, Volume, bb_upper, bb_lower, macd_diff, rsi, obv, ichimoku_a, ichimoku_b`
  - LSTM configuration (in demo run): `n_steps = 60`, `target_days = 5`, training `epochs=5`, `batch_size=32` (these are conservative/demo values and are configurable in code)
  - Optimization: converts trained Keras model to TFLite and ONNX (via `tf2onnx`) and saves `optimized_lstm_model.tflite` and `optimized_lstm_model.onnx` (ONNX conversion guarded with try/except).
  - Backtesting: simple entry/exit using `final_signal` from `signals.py`, starting capital 100000, computes P/L, win rate, and prints a trade log and visualizations.

- `indicators.py` — computes Bollinger Bands, RSI (with placeholder GNQTS optimizer), MACD, OBV, Ichimoku, ATR using `ta` package.

- `signals.py` — rule-based signal generation combining BB, MACD diff, OBV, RSI, Ichimoku
  - Outputs `signal`, `watch_signal`, `confidence` (0-100), and `final_signal` (uses signal unless Hold then uses watch signals)

- `rl_trading_env.py` — custom `TradingEnv` (gymnasium) with Discrete(3) actions (Buy, Sell, Hold); observation is a flattened historical feature window + portfolio state; reward approximates percent change in net worth with small penalty on losses.

- `web_analysis.py` — `run_quick_analysis(ticker)` performs a 2-year quick analysis without LSTM training and returns JSON-ready results for the Flask app; includes chart data arrays for drawing in a frontend.

- `app.py` — Flask server exposing `/analyze` POST for quick web analysis.

**Model artifacts present**
- `optimized_lstm_model.tflite` — present at project root (TFLite). You can inspect input/output details with the TensorFlow Lite `Interpreter` or `tflite_runtime`.
- `lstm_saved_model/` — present and contains `saved_model.pb`, `variables/*`, and `assets/`.

Files/directories that are empty or missing data in this copy
- `data/` — empty (no local OHLCV CSV). The code downloads data via `yfinance` at runtime.
- `outputs/` — empty (no generated plots or CSV results present)
- `models/` — empty (no checkpoints besides SavedModel)
- `notebooks/` — empty (no notebooks included here)

Because datasets and outputs are missing, numeric results (training curves, exact backtest numbers) are not available in this snapshot and must be generated by running the pipeline.

**Repro / Run instructions (PowerShell commands)**
1) Create & activate a virtual environment (recommended)
```powershell
cd "c:\Users\rajveersinh chavda\stock-suggester-ai"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2) Install dependencies
```powershell
pip install -r requirements.txt
```

3) Quick web analysis (no model training)
```powershell
python app.py
# then POST JSON {"ticker":"AAPL"} to http://127.0.0.1:5000/analyze
```

4) Run the full analysis pipeline (downloads data, trains LSTM for demo epochs, converts to TFLite/ONNX, runs backtest)
```powershell
python src\main.py
```

5) Generate the static inventory/manifest (script added to repo):
```powershell
python scripts\generate_report.py
# Output: report/manifest.json and report/files.csv
```

Notes:
- `src/main.py` will download 5 years of data for `IOC.NS` by default. Change `ticker = "IOC.NS"` in `src/main.py` or call the code differently to analyze other tickers.
- Training times depend on CPU/GPU, available RAM, and number of epochs. The demo uses `epochs=5`.

**Data & Evaluation Items (to include in a formal report)**
- Data description: source (`yfinance`), ticker(s) used, date ranges, sample counts
- Preprocessing: missing-value handling, indicator windows, lookback window (`n_steps`)
- Feature list and feature engineering (list above)
- Model details: architecture, parameter count, training hyperparameters (optimizer, batch size, epochs, loss), model size (SavedModel + TFLite bytes)
- Metrics (to compute): MSE, MAE, RMSE for price predictions; backtest metrics: total return, annualized return, Sharpe ratio, max drawdown, win rate, number of trades
- Visuals: training loss curve, prediction vs actual price plots, residual histogram, equity curve, buy/sell signal markers on price chart, correlation heatmap

**Backtest assumptions (from `src/main.py` / `web_analysis.py`)**
- Starting capital: 100000
- Position sizing: simplified (full capital deployed on each trade) — note: this is not realistic; use per-trade allocation rules in production
- Transaction costs: not explicitly modeled in the simple backtest (except RL env has `trade_fee_pct`); production backtest should include fees and slippage

**Autogenerated manifest & inspection script**
- `scripts/generate_report.py` (added) will create `report/manifest.json` and `report/files.csv`. The script attempts to inspect `optimized_lstm_model.tflite` using TF Lite Interpreter and will list files in `lstm_saved_model/`. Run it after activating your virtual environment.

**Key code excerpts & configuration (for quick reference)**
- LSTM features used:
  - `['Close', 'Open', 'High', 'Low', 'Volume', 'bb_upper', 'bb_lower', 'macd_diff', 'rsi', 'obv', 'ichimoku_a', 'ichimoku_b']`
- LSTM lookback / prediction horizon:
  - `n_steps = 60` (lookback)
  - `target_days = 5` (predict 5 days ahead)
- LSTM architecture: 3 × LSTM(256) with Dropout(0.3) and final Dense(2)

**Limitations & Risks**
- No train/validation/test metrics are present in the repository snapshot — you must run `src/main.py` or a notebook to generate numeric results.
- Backtesting logic is simplistic (all-in allocation, no fees/slippage). Treat live/production claims cautiously.
- RL components are placeholders — PPO is initialized but not fully trained by default.

**Recommended next steps (for a complete report)**
1. Run `python scripts\generate_report.py` to create the manifest and gather model metadata.
2. Run `python src\main.py` in an environment with internet access to produce training metrics, `outputs/` artifacts, and saved models. Increase `epochs` for a meaningful train.
3. Collect the following and add to the report: training loss curves, evaluation MSE/RMSE/MAE on a held-out set, backtest metrics (annualized return, Sharpe ratio, max drawdown), and example trades CSV.
4. Replace the simple backtest with a proper backtesting engine (e.g., `vectorbt`, `backtrader`, or a custom engine that supports fees and position sizing).
5. Harden deployment: add unit tests, CI, and remove `venv/` from the repository.

**Appendix — Useful file paths & commands**
- Project root: `c:\Users\rajveersinh chavda\stock-suggester-ai`
- Main pipeline: `python src\main.py`
- Quick web analysis: `python app.py` (POST to `/analyze`)
- Manifest generator: `python scripts\generate_report.py` (writes `report/manifest.json`)
- Inspect TFLite manually (Python snippet):
```python
from tensorflow.lite import Interpreter
interp = Interpreter(model_path='optimized_lstm_model.tflite')
interp.allocate_tensors()
print(interp.get_input_details())
print(interp.get_output_details())
```

---

If you'd like, I can now: run `scripts/generate_report.py` and paste the resulting `report/manifest.json`, or run `src/main.py` to generate model training outputs (this will download data and train models). Which should I run next?
