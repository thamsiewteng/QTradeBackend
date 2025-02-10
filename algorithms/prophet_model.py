from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
from algorithms.backtest_prophet import backtest_prophet

def run_prophet(stock_symbol, start_date, backtest_start_date, backtest_end_date):
     
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['Open'] = stock_data['Open']

    stock = yf.Ticker(stock_symbol)

    info = stock.info
    stock_data['RSI'] = compute_RSI(stock_data['Close'], 14)
    rsi = float(stock_data['RSI'].tail(1).values[0])  
    market_cap = info.get('marketCap', 'N/A')
    beta = info.get('beta', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    eps = info.get('trailingEps', 'N/A')

    stock_data.reset_index(inplace=True)

    stock_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    split_ratio = [0.7, 0.15, 0.15]
    split_points = [int(split_ratio[0] * len(stock_data)),
                    int((split_ratio[0] + split_ratio[1]) * len(stock_data))]

    train_data = stock_data[:split_points[0]]
    val_data = stock_data[split_points[0]:split_points[1]]
    test_data = stock_data[split_points[1]:]

    model = Prophet(
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=0.01,
        holidays_prior_scale=0.01,
        seasonality_mode='additive'
    )

    model.add_regressor('Open')

    train_data['Open'] = train_data['Open'].fillna(method='ffill')
    model.fit(train_data)


    val_future = model.make_future_dataframe(periods=len(val_data))
    val_future = val_future[-len(val_data):]
    val_future['Open'] = val_data['Open'].values
    val_future['Open'] = val_future['Open'].fillna(method='ffill')

    val_forecast = model.predict(val_future)

    test_future = model.make_future_dataframe(periods=len(test_data))
    test_future = test_future[-len(test_data):]
    test_future['Open'] = test_data['Open'].values
    test_future['Open'] = test_future['Open'].fillna(method='ffill')

    test_forecast = model.predict(test_future)

    train_true = train_data['y'].values
    train_pred = model.predict(train_data)['yhat'].values

    val_true = val_data['y'].values
    val_pred = val_forecast['yhat'][-len(val_true):].values

    test_true = test_data['y'].values
    test_pred = test_forecast['yhat'][-len(test_true):].values
    mae, mse, rmse, mape, r2 = calculate_performance_metrics(test_true, test_pred)

    returns = np.diff(test_data['y']) / test_data['y'].iloc[:-1]
    volatility = np.std(returns)

    print(f'Volatility: {volatility:.6f}')

    future_periods = 7
    last_date = stock_data['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods)
    future = pd.DataFrame(future_dates, columns=['ds'])
    future['Open'] = stock_data['Open'].iloc[-1]  
    future_forecast = model.predict(future)

    print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]) 

    predicted_ma_short = future_forecast['yhat'].rolling(window=7).mean().iloc[-1]  
    actual_ma_short = stock_data['y'].rolling(window=7).mean().iloc[-1]
    actual_ma_long = stock_data['y'].rolling(window=30).mean().iloc[-1]

    trend_insight_short = "Uptrend" if predicted_ma_short > actual_ma_short else "Downtrend"
    trend_insight_long = "Uptrend" if predicted_ma_short > actual_ma_long else "Downtrend"

    print(f'Trend Insight (7-day MA): {trend_insight_short}')
    print(f'Trend Insight (30-day MA): {trend_insight_long}')

    combined_test_set = test_data.copy()
    combined_test_set['Predicted'] = test_forecast['yhat'].values[-len(test_data):]
    print(combined_test_set[['ds', 'y', 'Predicted']])

    if backtest_end_date and backtest_start_date:
        train_data['yhat'] = train_pred
        val_data['yhat'] = val_pred
        test_data['yhat'] = test_pred

        frames = [train_data[['ds', 'y', 'yhat']], val_data[['ds', 'y', 'yhat']], test_data[['ds', 'y', 'yhat']]]
        df_backtest = pd.concat(frames)

        df_backtest['ds'] = pd.to_datetime(df_backtest['ds'])

        df_backtest_filtered = df_backtest[(df_backtest['ds'] >= backtest_start_date) & (df_backtest['ds'] <= backtest_end_date)]
        backtest_results = backtest_prophet(df_backtest_filtered)
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
                '7day_MA': trend_insight_short,
                '30day_MA': trend_insight_long,
            },
            'rsi': rsi,
            'market_cap': market_cap,
            'beta': beta,
            'pe_ratio': pe_ratio,
            'eps': eps,
            'plot_data': {
                'dates': combined_test_set['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'actual': combined_test_set['y'].tolist(),
                'predicted': combined_test_set['Predicted'].tolist(),
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
