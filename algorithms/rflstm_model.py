import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import tensorflow as tf
from datetime import datetime, timedelta
import random as python_random

from algorithms.backtest_strategy import backtest_strategy

def run_rf_lstm(stock_symbol, start_date, backtest_start_date, backtest_end_date):

    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(stock_symbol, start=start_date, end=end_date)

    stock = yf.Ticker(stock_symbol)

    info = stock.info
    data['RSI'] = compute_RSI(data['Close'], 14)
    rsi = float(data['RSI'].tail(1).values[0])  
    market_cap = info.get('marketCap', 'N/A')
    beta = info.get('beta', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    eps = info.get('trailingEps', 'N/A')

    data['Prev_Close'] = data['Close'].shift(1)
    data['SMA_20'] = data['Prev_Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Prev_Close'].ewm(span=20, adjust=False).mean()
    data['Upper_Band'] = data['Prev_Close'].rolling(window=20).mean() + 2 * data['Prev_Close'].rolling(window=20).std()
    data['Lower_Band'] = data['Prev_Close'].rolling(window=20).mean() - 2 * data['Prev_Close'].rolling(window=20).std()
    data['ATR'] = data['High'] - data['Low']
    data['RSI'] = 100 - (100 / (1 + data['Prev_Close'].diff().rolling(window=14).apply(lambda x: np.sum(np.where(x > 0, x, 0)) / np.sum(np.where(x < 0, -x, 0)))))
    data['MACD'] = data['Prev_Close'].ewm(span=12, adjust=False).mean() - data['Prev_Close'].ewm(span=26, adjust=False).mean()
    data.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'SMA_20', 'EMA_20', 'Upper_Band', 'Lower_Band', 'ATR', 'RSI', 'MACD']
    target = ['Close']
    X = data[features]
    y = data[target].values.ravel()

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    N = 5  
    top_features = [features[i] for i in indices[:N]]

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(data[top_features])
    scaled_target = scaler_target.fit_transform(data[target])

    time_step = 10
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(scaled_features[i:(i + time_step)])
        y.append(scaled_target[i + time_step])
    X = np.array(X)
    y = np.array(y)

    total_samples = len(X)
    train_size = int(0.7 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size - val_size  

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    best_params = {
        'lstm_units': 100,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 120,
        'batch_size': 64
    }

    model = Sequential([
        LSTM(best_params['lstm_units'], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(best_params['dropout']),
        LSTM(int(best_params['lstm_units'] / 2), activation='relu'),  
        Dropout(best_params['dropout']),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, model_checkpoint]
    )

    model.load_weights('best_model.h5')

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    y_pred_train_inv = scaler_target.inverse_transform(y_pred_train)
    y_pred_val_inv = scaler_target.inverse_transform(y_pred_val)
    y_pred_test_inv = scaler_target.inverse_transform(y_pred_test)

    y_train_inv = scaler_target.inverse_transform(y_train)
    y_val_inv = scaler_target.inverse_transform(y_val)
    y_test_inv = scaler_target.inverse_transform(y_test)

    mse_train, rmse_train, mae_train, mape_train, r2_train = calculate_performance(y_train_inv.flatten(), y_pred_train_inv.flatten())
    mse_val, rmse_val, mae_val, mape_val, r2_val = calculate_performance(y_val_inv.flatten(), y_pred_val_inv.flatten())
    mse, rmse, mae, mape, r2 = calculate_performance(y_test_inv.flatten(), y_pred_test_inv.flatten())

    print(f"Training Set - MSE: {mse_train}, RMSE: {rmse_train}, MAE: {mae_train}, MAPE: {mape_train}, R2: {r2_train}")
    print(f"Validation Set - MSE: {mse_val}, RMSE: {rmse_val}, MAE: {mae_val} , MAPE: {mape_val}, R2: {r2_val}")
    print(f"Test Set - MSE: {mse}, RMSE: {rmse}, MAE: {mae} , MAPE: {mape}, R2: {r2}")

    test_dates = data.index[train_size+val_size+time_step:]

    comparison_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Closing Price': y_test_inv.flatten(),
        'Predicted Closing Price': y_pred_test_inv.flatten()
    })

    print(comparison_df.tail(10))

    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])

    plot_data = {
        "dates": comparison_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "actual": comparison_df['Actual Closing Price'].tolist(),
        "predicted": comparison_df['Predicted Closing Price'].tolist(),
    }

    json_plot_data = json.dumps(plot_data)

    returns = np.diff(y_test_inv.flatten()) / y_test_inv.flatten()[:-1]
    volatility = np.std(returns)

    print(f'Volatility: {volatility}')

    future_data = data[top_features].iloc[-time_step:].values
    future_predictions = []

    for _ in range(7):
        scaled_future_data = scaler_features.transform(pd.DataFrame(future_data, columns=top_features))  
        future_input = scaled_future_data[-time_step:]
        future_input = np.expand_dims(future_input, axis=0)
        future_pred = model.predict(future_input)
        future_pred_inv = scaler_target.inverse_transform(future_pred)
        future_predictions.append(future_pred_inv[0][0])

        new_row = np.append(future_data[-1][1:], future_pred_inv[0][0])  
        future_data = np.vstack([future_data, new_row])[-time_step:]  

    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 8)]
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_predictions
    })

    print(future_df)

    future_predicted_ma_short = pd.Series(future_predictions).rolling(window=7).mean().iloc[-1]  
    actual_ma_short = data['Close'].rolling(window=7).mean().iloc[-1]
    actual_ma_long = data['Close'].rolling(window=30).mean().iloc[-1]

    future_trend_insight_short = "Uptrend" if future_predicted_ma_short > actual_ma_short else "Downtrend"
    future_trend_insight_long = "Uptrend" if future_predicted_ma_short > actual_ma_long else "Downtrend"

    print(f'Future Trend Insight (7-day MA): {future_trend_insight_short}')
    print(f'Future Trend Insight (30-day MA): {future_trend_insight_long}')

    if backtest_end_date and backtest_start_date:
        all_predictions_inv = np.concatenate([y_pred_train_inv, y_pred_val_inv, y_pred_test_inv]).flatten()

        all_actuals_inv = np.concatenate([y_train_inv, y_val_inv, y_test_inv]).flatten()
        full_data = data.iloc[time_step:]  

        full_df = pd.DataFrame({
            'Open': full_data['Open'],  
            'High': full_data['High'],  
            'Low': full_data['Low'],    
            'Close': all_actuals_inv,   
            'Volume': full_data['Volume'],  
            'Predicted_Close': all_predictions_inv 
        }, index=full_data.index)

        backtest_df = full_df.loc[backtest_start_date:backtest_end_date]
        backtest_results = backtest_strategy(backtest_df)
        return backtest_results

    else: 
        results = {
            'performance_metrics': {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            },
            'volatility': volatility,
            'trend_insight': {
                '7day_MA': future_trend_insight_short,
                '30day_MA': future_trend_insight_long,
            },
            'rsi': rsi,
            'market_cap': market_cap,
            'beta': beta,
            'pe_ratio': pe_ratio,
            'eps': eps,
            'plot_data': {
                "dates": comparison_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                "actual": comparison_df['Actual Closing Price'].tolist(),
                "predicted": comparison_df['Predicted Closing Price'].tolist(),
            }
        }
        return results

def calculate_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mse, rmse, mae, mape, r2

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))
