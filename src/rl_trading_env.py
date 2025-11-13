import gymnasium as gym # Using gymnasium as recommended by Stable Baselines3
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    The agent learns to make trading decisions (Buy, Sell, Hold) based on market data.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, lookback_window=60, initial_balance=10000, trade_fee_pct=0.001):
        super().__init__()
        self.df = df.dropna().reset_index(drop=True) # Ensure no NaNs and reset index
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.trade_fee_pct = trade_fee_pct

        # Define action space: Buy (0), Sell (1), Hold (2)
        self.action_space = spaces.Discrete(3)

        # Define observation space:
        # Features: OHLC, BB_upper, BB_lower, MACD_diff, RSI, OBV, Ichimoku_a, Ichimoku_b
        # Plus portfolio value, shares held, current balance
        # We need to determine the exact shape based on your data's features.
        # For simplicity, let's assume the features from lstm_features in main.py.
        # The lower and upper bounds might need fine-tuning based on actual data ranges.
        num_features = 12 # Based on lstm_features count in main.py: Close,Open,High,Low,Volume,bb_upper,bb_lower,macd_diff,rsi,obv,ichimoku_a,ichimoku_b
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback_window, num_features + 3), dtype=np.float32)
        
        self.current_step = self.lookback_window # Start from a point where lookback data is available
        self.balance = initial_balance
        self.shares = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.episode_history = []

    def _get_obs(self):
        # Get historical data for the lookback window
        end = self.current_step
        start = end - self.lookback_window
        obs_df = self.df.iloc[start:end]

        # Features relevant for observation (matching LSTM features)
        # Note: 'Volume' is often very large, scaling might be needed here or in a wrapper
        features_for_obs = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'bb_upper', 'bb_lower', 'macd_diff', 'rsi', 'obv',
            'ichimoku_a', 'ichimoku_b'
        ]
        
        # Get numerical features as a NumPy array
        historical_features = obs_df[features_for_obs].values
        
        # Add portfolio state to observation
        portfolio_state = np.array([
            self.balance, # Current cash balance
            self.shares,  # Number of shares held
            self.net_worth # Current total net worth
        ])
        
        # Repeat portfolio state for each step in the lookback window if needed for consistent shape
        # Or, append to the last step of historical_features
        # For simplicity now, let's append it to the last row (current day's features)
        # This might need refinement based on how the RL algorithm expects state.
        
        # A simpler approach: Concatenate historical features with current portfolio state
        # Ensure portfolio state is correctly broadcast or appended
        # This requires historical_features to be 2D (lookback_window, num_features)
        # And portfolio_state to be 1D (3,)
        
        # Let's reshape portfolio_state to (1, 3) and concatenate for a (lookback_window, num_features + 3) result
        # This would require repeating the portfolio state for each lookback step, or a more complex observation wrapper

        # Simplest approach for initial setup: just concatenate flattened historical data and current portfolio state
        # This changes the shape from (lookback_window, num_features + 3) to (1D array)
        # For a Box space with 2D shape, we need to carefully construct it.

        # Let's assume the observation space expects the lookback_window * num_features, plus the 3 portfolio states at the end.
        # This would make the observation space 1D: (lookback_window * num_features + 3,)
        # Re-evaluating observation_space definition to match this simpler flattening strategy.
        # The current spaces.Box(shape=(lookback_window, num_features + 3)) implies a 2D observation.
        # If we stick to 2D observation (like LSTM input), portfolio state should be appended to each timestep.
        
        # Let's assume `historical_features` should be augmented with `portfolio_state` for *each* timestep.
        # This means `portfolio_state` needs to be broadcast or concatenated row-wise.
        # For now, let's add it to the last row of the historical features as a simple start.
        
        current_day_features = historical_features[-1].copy() # Get current day's features
        current_day_features = np.concatenate((current_day_features, portfolio_state))
        
        # For the observation space to be (lookback_window, num_features + 3), we need to ensure all rows have portfolio data.
        # A common way is to make portfolio state part of the overall observation vector.
        # Let's simplify the observation_space to be 1D: flattened historical features + portfolio state.
        
        # Revised observation structure: Flattened historical features + current portfolio state
        # Total elements = (lookback_window * num_features) + 3
        
        flattened_historical = historical_features.flatten()
        observation = np.concatenate((flattened_historical, portfolio_state))
        
        # Update observation_space shape to match the flattened observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=((self.lookback_window * num_features) + 3,), 
                                            dtype=np.float32)
        
        return observation

    def _get_info(self):
        return {"net_worth": self.net_worth, "balance": self.balance, "shares": self.shares}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.episode_history = []

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.df) - 1: # End of data
            done = True
        else:
            done = False

        current_price = self.df['Close'].iloc[self.current_step]
        previous_net_worth = self.net_worth

        # Execute action
        if action == 0:  # Buy
            if self.balance > current_price * (1 + self.trade_fee_pct): # Ensure enough balance for purchase + fee
                buy_shares = self.balance // (current_price * (1 + self.trade_fee_pct))
                self.shares += buy_shares
                self.balance -= buy_shares * current_price * (1 + self.trade_fee_pct)
        elif action == 1: # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price * (1 - self.trade_fee_pct)
                self.shares = 0

        # Update net worth
        self.net_worth = self.balance + self.shares * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Reward function (simple: change in net worth, penalize for volatility/fees)
        reward = (self.net_worth - previous_net_worth) / previous_net_worth # Percentage change
        
        # Incorporate risk-adjusted returns (e.g., Sharpe Ratio like component or penalty for drawdowns)
        # For simplicity, let's add a small penalty for trade fees directly into the reward.
        # This can be made more sophisticated with portfolio volatility as in the paper.
        
        # If net_worth drops, negative reward
        if self.net_worth < previous_net_worth:
            reward -= 0.01 # Small penalty for losing money, can be tuned

        # Observation and Info for the next step
        observation = self._get_obs()
        info = self._get_info()

        # Store episode history for analysis (optional)
        self.episode_history.append({
            'date': self.df['Date'].iloc[self.current_step],
            'close': current_price,
            'action': action,
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares': self.shares,
            'reward': reward
        })

        return observation, reward, done, False, info # last False is `truncated`

    def render(self):
        # For plotting or visualization, not critical for core RL logic.
        pass

    def close(self):
        pass 