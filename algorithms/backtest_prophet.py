import backtrader as bt

def backtest_prophet(backtest_data):

    backtest_df = backtest_data
    class ProphetData(bt.feeds.PandasData):
        lines = ('yhat',)
        params = (
            ('datetime', 0),
            ('open', 1),
            ('high', 1),
            ('low', 1),
            ('close', 1),
            ('yhat', -1),
            ('volume', -1),
            ('openinterest', -1),
        )

    class ProphetStrategy(bt.Strategy):
        def __init__(self):
            self.predicted_close = self.datas[0].yhat
            self.threshold = 0.8 

        def next(self):
            current_close = self.data.close[0]
            predicted_close = self.predicted_close[0]

            price_difference = predicted_close - current_close

            if not self.position and price_difference > self.threshold:
                self.buy()
            elif self.position and price_difference < -self.threshold:
                self.sell()

    cerebro = bt.Cerebro()

    cerebro.addstrategy(ProphetStrategy)

    data_feed = ProphetData(dataname=backtest_df)
    cerebro.adddata(data_feed)

    cerebro.broker.setcash(10000.0)

    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    cerebro.broker.setcommission(commission=0.1/100)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    strat = results[0]

    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    print('Sharpe Ratio:', sharpe_ratio)

    drawdown_info = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_info['max']['drawdown'] if 'max' in drawdown_info else 0
    print('Drawdown:', max_drawdown)

    returns_info = strat.analyzers.returns.get_analysis()
    annual_return = returns_info['rnorm100'] if 'rnorm100' in returns_info else 0
    print('Annual Return:', annual_return)

    trades = strat.analyzers.tradeanalyzer.get_analysis()
    total_closed_trades = trades.total.closed if 'total' in trades and 'closed' in trades.total else 0
    print('Total Trades:', total_closed_trades)

    if total_closed_trades > 0:
        win_rate = trades.won.total / total_closed_trades
        loss_rate = trades.lost.total / total_closed_trades
    else:
        win_rate = 0
        loss_rate = 0

    print('Win Rate:', win_rate)
    print('Loss Rate:', loss_rate)

    backtest_results = {
        'final_portfolio_value': final_portfolio_value,
        'sharpe_ratio': sharpe_ratio,
        'drawdown': max_drawdown,
        'annual_return': annual_return,
        'total_trades': total_closed_trades,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
    }

    return backtest_results
