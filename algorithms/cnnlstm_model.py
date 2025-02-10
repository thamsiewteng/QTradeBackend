import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from keras.callbacks  import EarlyStopping, LearningRateScheduler
from keras.optimizers  import Adam
import random as python_random
import tensorflow as tf
from keras.regularizers import l1_l2
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from algorithms.backtest_strategy import backtest_strategy

def run_cnnlstm(stock_symbol, start_date, backtest_start_date, backtest_end_date):
    
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

    data.dropna(inplace=True)
    features = ['Open', 'High', 'Low', 'Volume']
    target = ['Close']
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(data[features])
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

    lstm_units = 128  
    lstm_l1_reg = 0.0001 
    lstm_l2_reg = 0.0001  

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2, padding='same'),
        LSTM(lstm_units, activation='tanh', return_sequences=False,
            kernel_regularizer=l1_l2(l1=lstm_l1_reg, l2=lstm_l2_reg)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    initial_learning_rate = 0.0001
    optimizer = Adam(learning_rate=initial_learning_rate)

    def scheduler(epoch, lr):
        if epoch < 5:
            return lr
        else:
            return lr * 0.9

    lr_scheduler = LearningRateScheduler(scheduler)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


    callbacks_list = [early_stopping, lr_scheduler]

    history = model.fit(
        X_train, y_train,
        epochs=120,  
        batch_size=32,  
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks_list
    )

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    y_pred_train_inv = scaler_target.inverse_transform(y_pred_train)
    y_pred_val_inv = scaler_target.inverse_transform(y_pred_val)
    y_pred_test_inv = scaler_target.inverse_transform(y_pred_test)

    y_train_inv = scaler_target.inverse_transform(y_train)
    y_val_inv = scaler_target.inverse_transform(y_val)
    y_test_inv = scaler_target.inverse_transform(y_test)

    mae, mse, rmse, mape, r2 = calculate_performance_metrics(y_test_inv.flatten(), y_pred_test_inv.flatten())

    test_dates = data.index[train_size+val_size+time_step:]

    comparison_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Closing Price': y_test_inv.flatten(),
        'Predicted Closing Price': y_pred_test_inv.flatten()
    })
    print(comparison_df.tail(10))

    comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])

    returns = np.diff(y_test_inv.flatten()) / y_test_inv.flatten()[:-1]
    volatility = np.std(returns)

    print(f'Volatility: {volatility}')
    
    future_data = scaled_features[-time_step:]
    future_predictions = []

    for _ in range(7):
        future_input = np.expand_dims(future_data, axis=0)
        future_pred = model.predict(future_input)
        future_pred_inv = scaler_target.inverse_transform(future_pred)
        future_predictions.append(future_pred_inv[0][0])

        new_row = np.append(future_data[-1][1:], scaled_features[-1][0])  
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

def calculate_performance_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mse, rmse, mape, r2

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))
