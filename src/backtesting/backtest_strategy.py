"""
Backtest a strategy.

Given a strategy backtest it with historical data.

Following is our backtesting methodology:

1. We should be able to backtest the strategy with a portfolio of a single stock or multiple stocks.
2. We should be able to backtest the strategy over a specific time period or the entire history of the stock.
3. We should start off the strategy at a point where there is at least 2 months of historical data.
4. We should use the closing price of the stock for the backtesting.
5. So start off the portfolio with 100000 USD.
6. Then feed the strategy historical data and let it execute or not execute the trades.
7. Tick ahead to the next day of the historical data. Update the portfolio and the strategy with the new data.
8. Repeat the process until the end of the historical data.
9. Calculate the performance metrics.
10. Print the performance metrics.
"""
