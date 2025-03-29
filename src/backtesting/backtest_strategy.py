"""
Backtest a strategy.

Given a strategy backtest it with historical data.

Following is our backtesting methodology:

1. We should be able to backtest the strategy with a portfolio of a single stock or multiple stocks.
2. We should be able to backtest the strategy over a specific time period or the entire history of the stock.
3. We should start off the strategy at a point where there is at least 2 months of historical data.
4. We should use the closing price of the stock for the backtesting.
5. So start off the portfolio with 10000 USD.
6. Then feed the strategy historical data and let it execute or not execute the trades.
7. Tick ahead to the next day of the historical data. Update the portfolio and the strategy with the new data.
8. Repeat the process until the end of the historical data.
9. Calculate the performance metrics.
10. Print the performance metrics.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Type, Union

import matplotlib.pyplot as plt
import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy
from db_util import get_historical_data


def ensure_utc_tz(dt):
    """Helper function to ensure datetime is in UTC timezone."""
    if dt is None:
        return None

    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            return dt.tz_localize("UTC")
        else:
            return dt.tz_convert("UTC")
    else:  # Python datetime
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Convert existing timezone to UTC
            return dt.astimezone(timezone.utc)


class Backtest:
    """
    A class to backtest trading strategies.

    This class allows backtesting of any strategy that inherits from the Strategy base class
    using historical price data. It can handle single or multiple stocks, different time periods,
    and provides performance metrics and visualization.
    """

    def __init__(
        self,
        strategy_class: Type[Strategy],
        strategy_params: Optional[Dict] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        allow_short_selling: bool = False,
        allow_margin_trading: bool = False,
    ):
        """
        Initialize the Backtest object.

        Parameters:
        -----------
        strategy_class : Type[Strategy]
            The class of the strategy to backtest (not an instance)
        strategy_params : Dict, optional
            Parameters to pass to the strategy
        initial_capital : float, default 10000.0
            Starting capital for the portfolio
        commission : float, default 0.001
            Commission per trade (as a percentage of trade value)
        allow_short_selling : bool, default False
            Whether to allow short selling
        allow_margin_trading : bool, default False
            Whether to allow margin trading
        """
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.commission = commission
        self.allow_short_selling = allow_short_selling
        self.allow_margin_trading = allow_margin_trading

        # These will be initialized during run()
        self.portfolio = None
        self.strategy = None
        self.data = None
        self.benchmark_data = None

    def _prepare_data(
        self,
        tickers: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        db_name: str,
        data_buffer_months: int = 2,
    ) -> pd.DataFrame:
        """
        Fetch and prepare historical data for backtesting.

        Parameters:
        -----------
        tickers : str or List[str]
            Ticker or list of tickers to backtest
        start_date : datetime
            Start date for the backtest
        end_date : datetime
            End date for the backtest
        db_name : str
            Database name to fetch data from
        data_buffer_months : int, default 2
            Number of months of additional data to fetch before start_date for calculations

        Returns:
        --------
        pd.DataFrame
            DataFrame with MultiIndex (Timestamp, Ticker) containing OHLCV data
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Calculate fetch start date with buffer
        buffer_start_date = start_date - timedelta(days=data_buffer_months * 30)
        # Ensure it's UTC
        buffer_start_date = ensure_utc_tz(buffer_start_date)

        # Calculate days to fetch
        days_to_fetch = (end_date - buffer_start_date).days

        all_data = []

        for ticker in tickers:
            try:
                # Fetch data for this ticker
                ticker_data = get_historical_data(
                    ticker_symbol=ticker, db_name=db_name, days=days_to_fetch
                )

                if ticker_data.empty:
                    print(f"Warning: No data found for {ticker}. Skipping.")
                    continue

                # Ensure we have enough history data
                # Make sure buffer_start_date is timezone aware
                buffer_start_date_utc = ensure_utc_tz(buffer_start_date)

                if ticker_data.index[0] > buffer_start_date_utc:
                    print(
                        f"Warning: Data for {ticker} only starts at {ticker_data.index[0].date()}, which is after the buffer start date {buffer_start_date.date()}."
                    )

                # Add Ticker column
                ticker_data["ticker"] = ticker

                # Reset index to prepare for MultiIndex
                ticker_data = ticker_data.reset_index()

                # Append to our list
                all_data.append(ticker_data)

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid data found for any ticker.")

        # Combine all data into one DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)

        # Ensure 'timestamp' is tz-aware (UTC)
        combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])

        # Apply timezone info safely - handle both timezone-naive and aware datetimes
        timestamps = []
        for ts in combined_data["timestamp"]:
            timestamps.append(ensure_utc_tz(ts))

        combined_data["timestamp"] = timestamps

        # Create MultiIndex - rename to capital T Timestamp for compatibility with RSIStrategy
        combined_data = combined_data.set_index(["timestamp", "ticker"])

        # Sort by timestamp and ticker
        combined_data = combined_data.sort_index()

        return combined_data

    def _prepare_benchmark_data(
        self, benchmark_ticker: str, all_data: pd.DataFrame
    ) -> pd.Series:
        """
        Prepare benchmark data series for comparison.

        Parameters:
        -----------
        benchmark_ticker : str
            Ticker to use as benchmark
        all_data : pd.DataFrame
            DataFrame with all historical data

        Returns:
        --------
        pd.Series
            Series with benchmark values
        """
        if benchmark_ticker not in all_data.index.get_level_values("ticker").unique():
            return pd.Series(dtype=float)  # Empty series

        # Get first timestamp in actual backtest (not buffer)
        # Make sure backtest_start_date is tz-aware in the right way
        backtest_start = ensure_utc_tz(self.backtest_start_date)

        try:
            # Get prices for the benchmark ticker
            benchmark_prices = all_data.xs(benchmark_ticker, level="ticker")["close"]

            # Find closest timestamp at or after backtest start
            start_idx = benchmark_prices.index.searchsorted(backtest_start)
            if start_idx >= len(benchmark_prices):
                return pd.Series(dtype=float)  # No data in backtest period

            start_price = benchmark_prices.iloc[start_idx]
            if pd.isna(start_price) or start_price <= 0:
                return pd.Series(dtype=float)  # Invalid start price

            # Calculate shares bought with initial capital
            shares = self.initial_capital / start_price

            # Calculate benchmark values
            benchmark_values = benchmark_prices * shares

            return benchmark_values

        except Exception as e:
            print(f"Error preparing benchmark data: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for detailed debugging
            return pd.Series(dtype=float)

    def run(
        self,
        tickers: Union[str, List[str]],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        db_name: str = "stock_data.db",
        benchmark_ticker: Optional[str] = None,
    ) -> Dict:
        """
        Run the backtest.

        Parameters:
        -----------
        tickers : str or List[str]
            Ticker or list of tickers to backtest
        start_date : datetime
            Start date for the backtest
        end_date : datetime, optional
            End date for the backtest (defaults to current date)
        db_name : str, default "stock_data.db"
            Database name to fetch data from
        benchmark_ticker : str, optional
            Ticker to use as benchmark (typically same as tickers when single ticker)

        Returns:
        --------
        Dict
            Dictionary with backtest results and metrics
        """
        # Set end date to current date if not provided
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Ensure dates are in UTC timezone
        start_date = ensure_utc_tz(start_date)
        end_date = ensure_utc_tz(end_date)

        self.backtest_start_date = start_date
        self.backtest_end_date = end_date

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            allow_short_selling=self.allow_short_selling,
            allow_margin_trading=self.allow_margin_trading,
        )

        # Initialize strategy with parameters
        self.strategy = self.strategy_class(self.portfolio)
        if self.strategy_params:
            self.strategy.set_parameters(**self.strategy_params)

        # Add commission parameter if not already set
        if "commission" not in self.strategy.parameters:
            self.strategy.set_parameters(commission=self.commission)

        # Fetch and prepare data
        print(
            f"Fetching historical data for {tickers} from {start_date} to {end_date}..."
        )
        all_data = self._prepare_data(tickers, start_date, end_date, db_name)
        self.data = all_data

        # Prepare benchmark data if specified
        if benchmark_ticker:
            self.benchmark_data = self._prepare_benchmark_data(
                benchmark_ticker, all_data
            )
        else:
            self.benchmark_data = pd.Series(dtype=float)

        # Get unique timestamps within the actual backtest period (not buffer)
        backtest_timestamps = (
            all_data[
                (all_data.index.get_level_values("timestamp") >= start_date)
                & (all_data.index.get_level_values("timestamp") <= end_date)
            ]
            .index.get_level_values("timestamp")
            .unique()
        )

        # Sort timestamps chronologically
        backtest_timestamps = sorted(backtest_timestamps)

        if not backtest_timestamps:
            raise ValueError(
                f"No data available in the specified date range: {start_date} to {end_date}"
            )

        print(f"Running backtest for {len(backtest_timestamps)} trading days...")

        # Loop through each timestamp
        for i, current_timestamp in enumerate(backtest_timestamps):
            # Report progress every 100 days
            if i % 100 == 0:
                print(
                    f"Processing day {i + 1}/{len(backtest_timestamps)}: {current_timestamp.date()}"
                )

            # Provide data up to and including the current timestamp
            data_slice = all_data[
                all_data.index.get_level_values("timestamp") <= current_timestamp
            ]

            if not data_slice.empty:
                # Get the latest prices for all tickers at this timestamp
                try:
                    latest_data = data_slice.xs(current_timestamp, level="timestamp")
                    # Update portfolio's current prices
                    for ticker, row in latest_data.iterrows():
                        if (
                            isinstance(ticker, str)
                            and "close" in row
                            and pd.notna(row["close"])
                        ):
                            self.portfolio.current_prices[ticker] = row["close"]
                except KeyError:
                    # No data for this exact timestamp
                    pass

                # Execute the strategy for the current day
                self.strategy.execute(data_slice)

                # Ensure portfolio equity history is updated for this timestamp
                # This is needed for plotting, even if no trades were executed
                self.portfolio.update_prices(
                    self.portfolio.current_prices, current_timestamp
                )

        # Calculate and return performance metrics
        metrics = self.get_performance_metrics()

        return metrics

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics for the backtest.

        Returns:
        --------
        Dict
            Dictionary with performance metrics
        """
        if not self.portfolio or not self.strategy:
            return {"error": "Backtest has not been run yet."}

        # Get strategy metrics
        metrics = self.strategy.get_performance_summary()

        # Add backtest-specific information
        metrics.update(
            {
                "start_date": self.backtest_start_date,
                "end_date": self.backtest_end_date,
                "initial_capital": self.initial_capital,
                "final_portfolio_value": self.portfolio.get_value(),
            }
        )

        return metrics

    def plot_results(self, figsize=(12, 8), show_benchmark=True):
        """
        Plot backtest results.

        Parameters:
        -----------
        figsize : tuple, default (12, 8)
            Figure size
        show_benchmark : bool, default True
            Whether to show benchmark comparison if available
        """
        if not hasattr(self, "portfolio") or not self.portfolio.equity_history:
            print("No equity history available to plot.")
            return

        try:
            # Extract strategy timestamps and equity values
            timestamps, equity = zip(*self.portfolio.equity_history, strict=True)

            # Handle timestamps properly for plotting
            pd_timestamps = []
            for ts in timestamps:
                pd_timestamps.append(ensure_utc_tz(ts))

            equity_series = pd.Series(equity, index=pd_timestamps)

            # Create the plot
            plt.figure(figsize=figsize)

            # Plot strategy equity curve
            plt.plot(equity_series.index, equity_series.values, label="Strategy")

            # Plot benchmark if available and requested
            if show_benchmark and not self.benchmark_data.empty:
                # Filter benchmark data to match the strategy's timestamps
                try:
                    filtered_benchmark = self.benchmark_data[
                        (self.benchmark_data.index >= min(pd_timestamps))
                        & (self.benchmark_data.index <= max(pd_timestamps))
                    ]

                    if not filtered_benchmark.empty:
                        # Reindex to align with strategy timestamps
                        aligned_benchmark = filtered_benchmark.reindex(
                            equity_series.index, method="ffill"
                        )
                        plt.plot(
                            aligned_benchmark.index,
                            aligned_benchmark.values,
                            label="Buy & Hold Benchmark",
                            linestyle="--",
                        )
                except Exception as e:
                    print(f"Error plotting benchmark: {e}")

            # Add titles and labels
            plt.title("Portfolio Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.grid(True)

            # Show the plot
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting results: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging

    def print_results(self):
        """Print backtest results and stats."""
        if not self.portfolio or not self.strategy:
            print("Backtest has not been run yet.")
            return

        metrics = self.get_performance_metrics()

        print("\n" + "=" * 50)
        print(f"BACKTEST RESULTS: {self.strategy.name}")
        print("=" * 50)
        print(f"Period: {metrics['start_date'].date()} to {metrics['end_date'].date()}")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")

        profit_loss = metrics["final_portfolio_value"] - metrics["initial_capital"]
        profit_loss_pct = profit_loss / metrics["initial_capital"] * 100
        print(f"Profit/Loss: ${profit_loss:,.2f} ({profit_loss_pct:.2f}%)")

        print("\nPERFORMANCE METRICS:")
        print(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
        print(f"Annualized Return: {metrics.get('annualized_return', 0.0):.2%}")
        print(f"Volatility (Ann.): {metrics.get('volatility', 0.0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0.0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0.0):.2f}")
        print(f"Number of Trades: {metrics.get('num_trades', 0)}")

        print("\nSTRATEGY PARAMETERS:")
        for param, value in self.strategy.parameters.items():
            print(f"  {param}: {value}")

        print("\nFINAL HOLDINGS:")
        if not self.portfolio.holdings:
            print("  No positions held.")
        else:
            for ticker, shares in self.portfolio.holdings.items():
                if shares > 0:
                    current_price = self.portfolio.current_prices.get(ticker, 0)
                    value = shares * current_price
                    print(
                        f"  {ticker}: {shares} shares @ ${current_price:,.2f} (Value: ${value:,.2f})"
                    )

        print(f"\nFinal Cash: ${self.portfolio.cash:,.2f}")
        print("=" * 50)

        # Optional: Print trade summary
        if hasattr(self.portfolio, "trade_history") and self.portfolio.trade_history:
            print("\nTRADE SUMMARY:")
            trades_df = pd.DataFrame(self.portfolio.trade_history)

            # Count buys and sells
            buys = trades_df[trades_df["direction"] == "BUY"]
            sells = trades_df[trades_df["direction"] == "SELL"]

            print(f"  Total Trades: {len(trades_df)}")
            print(f"  Buys: {len(buys)}")
            print(f"  Sells: {len(sells)}")

            # Show 5 most recent trades
            if len(trades_df) > 0:
                print("\nRECENT TRADES:")
                recent_trades = trades_df.sort_values(
                    "timestamp", ascending=False
                ).head(5)
                for _, trade in recent_trades.iterrows():
                    ts = pd.Timestamp(trade["timestamp"]).strftime("%Y-%m-%d")
                    direction = trade["direction"]
                    ticker = trade["ticker"]
                    quantity = abs(trade["quantity"])
                    price = trade["price"]
                    value = abs(trade["value"])
                    print(
                        f"  {ts} - {direction} {quantity} {ticker} @ ${price:.2f} (Value: ${value:.2f})"
                    )


if __name__ == "__main__":
    import importlib.util
    import inspect
    import os
    from datetime import datetime, timedelta

    import click

    @click.command()
    @click.option(
        "--strategy",
        required=True,
        help="Strategy file to use (e.g., 'rsi_strategy.py')",
    )
    @click.option("--ticker", default="SPY", help="Ticker symbol to backtest")
    @click.option(
        "--db-name", default="stock_data.db", help="Database name for historical data"
    )
    @click.option("--months", type=int, default=12, help="Number of months to backtest")
    @click.option(
        "--start-cash",
        type=float,
        default=10000.0,
        help="Initial capital for the portfolio",
    )
    @click.option(
        "--commission",
        type=float,
        default=0.001,
        help="Commission per trade (as a percentage)",
    )
    @click.option("--plot", is_flag=True, help="Plot the backtest results")
    def run_backtest(strategy, ticker, db_name, months, start_cash, commission, plot):
        """Run a strategy backtest with specified parameters."""

        # Dynamically import the strategy class from the specified file
        try:
            # Clean up the strategy filename input
            if not strategy.endswith(".py"):
                strategy += ".py"

            # Construct the path to the strategy file
            strategies_dir = os.path.join("src", "backtesting", "strategies")
            if not os.path.exists(strategies_dir):
                strategies_dir = os.path.join("backtesting", "strategies")
            if not os.path.exists(strategies_dir):
                # Try relative to the current directory
                strategies_dir = os.path.join(
                    os.getcwd(), "src", "backtesting", "strategies"
                )
            if not os.path.exists(strategies_dir):
                strategies_dir = os.path.join(os.getcwd(), "backtesting", "strategies")

            strategy_path = os.path.join(strategies_dir, strategy)

            if not os.path.exists(strategy_path):
                print(
                    f"Error: Strategy file '{strategy}' not found in '{strategies_dir}'"
                )
                return

            # Import the module
            module_name = strategy.replace(".py", "")
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the strategy class in the module
            strategy_class = None
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Strategy)
                    and obj != Strategy
                ):
                    strategy_class = obj
                    break

            if strategy_class is None:
                print(f"Error: No Strategy subclass found in '{strategy}'")
                return

            print(f"Found strategy class: {strategy_class.__name__}")

        except Exception as e:
            print(f"Error loading strategy '{strategy}': {e}")
            import traceback

            traceback.print_exc()
            return

        # Calculate dates - make them timezone-aware (UTC)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months * 30)  # Approximate

        print(
            f"Running {strategy_class.__name__} for {ticker} from {start_date.date()} to {end_date.date()}"
        )

        # Configure strategy-specific parameters based on strategy class
        strategy_params = {}
        if hasattr(strategy_class, "get_default_parameters"):
            strategy_params = strategy_class.get_default_parameters()
        else:
            # Use some reasonable defaults for common strategies
            if "RSI" in strategy_class.__name__:
                strategy_params = {
                    "rsi_period": 14,
                    "overbought_threshold": 70,
                    "oversold_threshold": 30,
                    "max_capital_per_position": 0.9,
                }
            # Add defaults for other strategy types as needed

        # Create and run backtest
        backtest = Backtest(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            initial_capital=start_cash,
            commission=commission,
        )

        try:
            # Run the backtest
            backtest.run(
                tickers=ticker,
                start_date=start_date,
                end_date=end_date,
                db_name=db_name,
                benchmark_ticker=ticker,  # Use same ticker as benchmark
            )

            # Print results
            backtest.print_results()

            # Plot if requested
            if plot:
                backtest.plot_results()

        except Exception as e:
            print(f"Error running backtest: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging

    # Run the command
    run_backtest()
