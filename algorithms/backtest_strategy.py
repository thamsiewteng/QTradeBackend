import backtrader as bt

def backtest_strategy(backtest_data):
        
        backtest_df = backtest_data

        class StockData(bt.feeds.PandasData):
            lines = ('predicted_close',)
            params = (
                ('datetime', None),
                ('open', 'Open'),
                ('high', 'High'),
                ('low', 'Low'),
                ('close', 'Close'),
                ('predicted_close', 'Predicted_Close'),
                ('volume', 'Volume'),
                ('openinterest', None),
            )

        class Strategy(bt.Strategy):
            params = (
                ('printlog', False),
                ('stop_loss_perc', 0.03),
                ('take_profit_perc', 0.05),
                ('threshold', 0.05),
            )

            def __init__(self):
                self.predicted_close = self.datas[0].predicted_close
                self.data_close = self.datas[0].close
                self.order = None
                self.buy_price = None

            def next(self):
                if not self.position:
                    if self.predicted_close[0] > (1 + self.params.threshold) * self.data_close[0]:
                        self.buy_price = self.data_close[0]
                        self.order = self.buy()
                else:
                    if self.predicted_close[0] < (1 - self.params.threshold) * self.data_close[0]:
                        self.order = self.sell()
                    elif (self.data_close[0] <= self.buy_price * (1 - self.params.stop_loss_perc) or
                        self.data_close[0] >= self.buy_price * (1 + self.params.take_profit_perc)):
                        self.order = self.sell()

        cerebro = bt.Cerebro()

        cerebro.addstrategy(Strategy)

        data_feed = StockData(dataname=backtest_df)
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

