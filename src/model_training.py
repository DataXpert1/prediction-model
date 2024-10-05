import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import pickle

def train_xgboost(X_train, y_train, n_estimators=500, learning_rate=0.05):
    """Train an XGBoost model."""
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_lstm(X_train, y_train, input_shape):
    """Train an LSTM model for time-series predictions."""
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(4))  # Output layer with 4 targets: stock price, market share, revenue, expense

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    return model

def save_model(model, filepath):
    """Save the trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def reshape_for_lstm(X):
    """Reshape data for LSTM model input (3D)."""
    return np.array(X).reshape((X.shape[0], 1, X.shape[1]))
