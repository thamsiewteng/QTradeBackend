import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime, timedelta
from algorithms.backtest_strategy import backtest_strategy

def run_xgboost(stock_symbol, start_date, backtest_start_date, backtest_end_date):
    end_date = datetime.now().strftime('%Y-%m-%d')

    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock = yf.Ticker(stock_symbol)
    info = stock.info

    stock_data['RSI'] = compute_RSI(stock_data['Close'], 14)
    rsi = float(stock_data['RSI'].tail(1).values[0])
    market_cap = info.get('marketCap', 'N/A')
    beta = info.get('beta', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    eps = info.get('trailingEps', 'N/A')

    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['RSI'] = compute_RSI(stock_data['Close'], 14)
    stock_data['MACD'] = compute_MACD(stock_data['Close'])
    stock_data['Volume_osc'] = compute_volume_oscillator(stock_data['Volume'])
    stock_data['Prev_Close'] = stock_data['Close'].shift(1)
    stock_data['Open_Close_diff'] = stock_data['Open'] - stock_data['Close']
    stock_data['High_Low_diff'] = stock_data['High'] - stock_data['Low']
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
    stock_data['5d_Close_pct'] = stock_data['Close'].pct_change(5)
    stock_data['5d_Vol_pct'] = stock_data['Volume'].pct_change(5)
    stock_data.dropna(inplace=True)

    X = stock_data[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volume_osc', 'Prev_Close', 'Open_Close_diff',
                    'High_Low_diff', 'Volume', 'Daily_Return', 'Volume_Change', '5d_Close_pct', '5d_Vol_pct']]
    y = stock_data['Close']

    total_size = len(X)
    train_ratio, val_ratio = 0.7, 0.1
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size + val_size], y.iloc[train_size:train_size + val_size]
    X_test, y_test = X.iloc[train_size + val_size:], y.iloc[train_size + val_size:]

    best_params = {'colsample_bytree': 0.8, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 3,
                   'min_child_weight': 3, 'n_estimators': 300, 'subsample': 0.9}
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_5_features = feature_importances.head(5).index.tolist()
    X_train_selected = X_train[top_5_features]
    X_val_selected = X_val[top_5_features]
    X_test_selected = X_test[top_5_features]
    model.fit(X_train_selected, y_train)

    y_train_pred = model.predict(X_train_selected)
    y_valid_pred = model.predict(X_val_selected)
    y_test_pred = model.predict(X_test_selected)

    mae, mse, rmse, mape, r2 = evaluate_model(model, X_test_selected, y_test, 'Testing')
    y_test_inv = y_test.values.reshape(-1, 1)
    y_pred_test_inv = y_test_pred.reshape(-1, 1)

    volatility = calculate_volatility(y_test_inv)
    print(f'Volatility: {volatility}')

    combined_test_set = pd.DataFrame({
        'ds': X_test.index,  
        'y': y_test,
        'Predicted': y_test_pred
    })

    print(combined_test_set[['ds', 'y', 'Predicted']])

    future_dates = pd.date_range(start=X_test.index[-1] + timedelta(days=1), periods=7, freq='B')
    future_data = pd.DataFrame(index=future_dates)
    future_data['Prev_Close'] = y_test[-1]
    future_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean().iloc[-1]
    future_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean().iloc[-1]
    future_data['RSI'] = compute_RSI(stock_data['Close'], 14).iloc[-1]
    future_data['MACD'] = compute_MACD(stock_data['Close']).iloc[-1]
    future_data['Volume_osc'] = compute_volume_oscillator(stock_data['Volume']).iloc[-1]
    future_data['Open_Close_diff'] = 0
    future_data['High_Low_diff'] = 0
    future_data['Volume'] = stock_data['Volume'].iloc[-1]
    future_data['Daily_Return'] = 0
    future_data['Volume_Change'] = 0
    future_data['5d_Close_pct'] = stock_data['Close'].pct_change(5).iloc[-1]
    future_data['5d_Vol_pct'] = stock_data['Volume'].pct_change(5).iloc[-1]

    future_data = future_data[top_5_features]
    future_predictions = model.predict(future_data)
    
    future_data['Predicted_Close'] = future_predictions
    future_data['SMA_7'] = future_data['Predicted_Close'].rolling(window=7).mean()
    actual_data_with_future = pd.concat([stock_data, future_data])
    actual_data_with_future['SMA_7'] = actual_data_with_future['Close'].rolling(window=7).mean()
    actual_data_with_future['SMA_30'] = actual_data_with_future['Close'].rolling(window=30).mean()

    trend_insight_7d = "Uptrend" if future_data['SMA_7'].iloc[-1] > actual_data_with_future['SMA_7'].iloc[-1] else "Downtrend"
    trend_insight_30d = "Uptrend" if future_data['SMA_7'].iloc[-1] > actual_data_with_future['SMA_30'].iloc[-1] else "Downtrend"

    print (future_dates)
    print (future_data)
    print (trend_insight_7d)
    print (trend_insight_30d)

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
                '7day_MA': trend_insight_7d,
                '30day_MA': trend_insight_30d,
            },
        'rsi': rsi,
        'market_cap': market_cap,
        'beta': beta,
        'pe_ratio': pe_ratio,
        'eps': eps,
        'plot_data': {
            "dates": combined_test_set['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "actual": combined_test_set['y'].tolist(),
            "predicted": combined_test_set['Predicted'].tolist(),
        }
    }
    if backtest_end_date and backtest_start_date:
        predictions = np.concatenate((y_train_pred, y_valid_pred, y_test_pred))
        actual_prices = pd.concat((y_train, y_val, y_test))
        backtest_df = pd.DataFrame({'Predicted_close': predictions, 'close': actual_prices}, index=actual_prices.index)
        backtest_df.sort_index(inplace=True)
        backtest_df = backtest_df.join(stock_data[['Open', 'High', 'Low', 'Volume']], how='left').dropna()
        backtest_df = backtest_df.loc[backtest_start_date:backtest_end_date]
        backtest_results = backtest_strategy(backtest_df)
        return backtest_results

    return results

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def compute_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    exp1 = series.ewm(span=fast_period, adjust=False).mean()
    exp2 = series.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal

def compute_volume_oscillator(volume, short_period=12, long_period=26):
    short_vol_ema = volume.ewm(span=short_period, adjust=False).mean()
    long_vol_ema = volume.ewm(span=long_period, adjust=False).mean()
    vol_osc = (short_vol_ema - long_vol_ema) / long_vol_ema
    return vol_osc * 100

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, X, y, name='Dataset'):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f'{name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, R2: {r2}')
    return mae, mse, rmse, mape, r2

def calculate_volatility(prices):
    returns = np.diff(prices.flatten()) / prices.flatten()[:-1]
    volatility = np.std(returns)
    return volatility

