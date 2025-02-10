from flask import Flask, request, jsonify
from algorithms.prophet_model import run_prophet
from algorithms.xgboost_model import run_xgboost
from algorithms.cnnlstm_model import run_cnnlstm  
from algorithms.rflstm_model import run_rf_lstm
import yfinance as yf

app = Flask(__name__)

@app.route('/predict/<algorithm_name>', methods=['POST'])
def predict(algorithm_name):
    data = request.get_json()
    stock_symbol = data['stockSymbol']
    start_date = data['startDate']
    backtest_start_date = data.get('backtestStartDate')
    backtest_end_date = data.get('backtestEndDate')

    if algorithm_name == 'vJEkT2WjlsZofNu1aPDS':
        results = run_prophet(stock_symbol, start_date, backtest_start_date, backtest_end_date)
    elif algorithm_name == 'IsMcMEJGHBG9NYjIAZbW':
        results = run_xgboost(stock_symbol, start_date, backtest_start_date, backtest_end_date)
    elif algorithm_name == 'WPk9szX4JJJXEFEa0wnE':
        results = run_rf_lstm(stock_symbol, start_date, backtest_start_date, backtest_end_date)
    elif algorithm_name == 'o4Re5dlQUvtXErRqpm82':
        results = run_cnnlstm(stock_symbol, start_date, backtest_start_date, backtest_end_date)
    else:
        return jsonify({'error': 'Algorithm not found'}), 404
    
    return jsonify(results)

@app.route('/stocks_info', methods=['POST'])
def get_stocks_info():
    data = request.get_json()
    tickers = data['tickers']  

    results = []

    for ticker_symbol in tickers:
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            hist = stock.history(period="2d")

            if len(hist) > 1:
                previous_close = hist.iloc[-2]['Close']
                current_price = hist.iloc[-1]['Close']
                change_dollars = current_price - previous_close
                change_percent = (change_dollars / previous_close) * 100
            else:
                current_price = info.get('previousClose', 0)
                change_percent = 0

            results.append({
                'ticker': ticker_symbol,
                'companyName': info.get('longName', ''),
                'currentPrice': current_price,
                'changePercent': change_percent,
            })

        except Exception as e:
            results.append({
                'ticker': ticker_symbol,
                'error': str(e)
            })

    return jsonify(results)

@app.route('/candlestick_data', methods=['POST'])
def get_candlestick_data():
    data = request.get_json()
    ticker_symbol = data['ticker']
    period = data.get('period', '1d')  
    interval = data.get('interval', '1m') 

    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period, interval=interval)

        hist.reset_index(inplace=True)
    
        hist['Datetime'] = hist['Datetime'].dt.strftime('%Y-%m-%d %H:%M')

        candlestick_data = []
        for _, row in hist.iterrows():
            candle = {
                'datetime': row['Datetime'], 
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': round(row['Volume'], 2)  
            }
            candlestick_data.append(candle)
    except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify(candlestick_data)


@app.route('/detailed_stock_info', methods=['POST'])
def get_detailed_stock_info():
    data = request.get_json()
    ticker_symbol = data['ticker']
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        hist = stock.history(period="2d")
        if len(hist) > 1:
            previous_close = hist.iloc[-2]['Close']
            current_close = hist.iloc[-1]['Close']
            change_percent = ((current_close - previous_close) / previous_close) * 100
        else:
            previous_close = info.get('previousClose', 0)
            current_close = info.get('currentPrice', 0)
            if previous_close > 0:
                change_percent = ((current_close - previous_close) / previous_close) * 100
            else:
                change_percent = 0  

        result = {
            'ticker': ticker_symbol,
            'companyName': info.get('longName', ''),
            'open': info.get('open', 0),
            'low': info.get('dayLow', 0),
            'high': info.get('dayHigh', 0),
            'previousClose': previous_close,
            'currentClose': current_close,
            'changePercent': round(change_percent, 2),
            'beta': info.get('beta', 0),
            'marketCap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'eps':  info.get('epsTrailingTwelveMonths', info.get('forwardEps', 0)),
            'peRatio': info.get('trailingPE', 0)
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)