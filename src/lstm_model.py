import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_lstm_data(df, features, n_steps, target_days):
    """
    Prepares data for LSTM: scales features and creates sequences.
    
    Args:
        df (pd.DataFrame): Input DataFrame with all features.
        features (list): List of feature column names.
        n_steps (int): Number of time steps (look-back window) for LSTM input.
        target_days (int): Number of days to predict into the future (e.g., 5 for next_5d_price).
        
    Returns:
        tuple: X (features), y_price (target price), y_signal (target signal confidence), scaler
    """
    data = df[features].values
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y_price, y_signal_confidence = [], [], []
    for i in range(len(scaled_data) - n_steps - target_days + 1):
        # Features (OHLC + BB + MACD + RSI + OBV + Ichimoku)
        X.append(scaled_data[i:(i + n_steps)])
        
        # Target: Price of 'Close' after target_days
        y_price.append(scaled_data[i + n_steps + target_days -1, features.index('Close')])

        # Target: Signal Confidence (placeholder for now, will refine)
        # This part assumes you have a 'confidence' column, and it's already scaled/handled
        # For initial LSTM, we might simplify this to just price prediction
        # For actual signal confidence output, you'd need to define how it's derived from future data
        y_signal_confidence.append(df['confidence'].iloc[i + n_steps + target_days - 1] / 100.0) # Assuming confidence is 0-100
        
    return np.array(X), np.array(y_price), np.array(y_signal_confidence), scaler

def build_lstm_model(input_shape):
    """
    Builds a 3-layer LSTM model.
    
    Args:
        input_shape (tuple): (n_steps, n_features)
        
    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        # Output: Price + Signal Confidence
        Dense(2) # First output for price, second for signal confidence (e.g., probability)
    ])
    
    model.compile(optimizer='adam', loss='mse') # Using MSE for regression
    return model

def predict_lstm_price(model, data_sequence, scaler, close_feature_index):
    """
    Makes a prediction with the trained LSTM model.
    
    Args:
        model (tf.keras.Model): Trained LSTM model.
        data_sequence (np.array): A single sequence of scaled data for prediction (n_steps, n_features).
        scaler (MinMaxScaler): The scaler used for feature scaling.
        close_feature_index (int): Index of the 'Close' price in the features list.

    Returns:
        float: Predicted next price.
    """
    # Expand dims for single prediction (batch_size, n_steps, n_features)
    prediction = model.predict(np.expand_dims(data_sequence, axis=0))
    
    # Inverse transform only the 'Close' price prediction part
    # Create a dummy array with the same shape as original scaled_data for inverse_transform
    dummy_array = np.zeros((1, scaler.n_features_in_))
    dummy_array[0, close_feature_index] = prediction[0, 0] # Assign predicted price to Close column
    
    # Inverse transform the dummy array to get actual price
    predicted_price = scaler.inverse_transform(dummy_array)[0, close_feature_index]
    
    # The second output from Dense(2) is for signal confidence
    predicted_confidence = prediction[0, 1] # Assuming confidence is the second output
    
    return predicted_price, predicted_confidence
