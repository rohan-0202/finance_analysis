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

from common.df_columns import CLOSE, HIGH, LOW, OPEN, TICKER, TIMESTAMP, VOLUME
from backtesting.portfolio import Portfolio
from backtesting.strategies.strategy_factory import StrategyFactory
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

        # Ensure start_date and end_date are UTC
        start_date = ensure_utc_tz(start_date)
        end_date = ensure_utc_tz(end_date)

        # Calculate fetch start date with buffer
        buffer_start_date = start_date - timedelta(days=data_buffer_months * 30)
        buffer_start_date = ensure_utc_tz(buffer_start_date)

        # Calculate days needed to fetch from buffer_start_date until *today*
        # This ensures we definitely get the data up to 'end_date' using the existing
        # get_historical_data function's logic.
        today = datetime.now(timezone.utc)
        days_to_fetch = (today - buffer_start_date).days + 1  # Add 1 for inclusivity
        if days_to_fetch <= 0:
            days_to_fetch = 1  # Fetch at least one day

        all_data_raw = []

        for ticker in tickers:
            try:
                # Fetch potentially *more* data than needed using the existing function
                ticker_data_raw = get_historical_data(
                    ticker_symbol=ticker, db_name=db_name, days=days_to_fetch
                )

                if ticker_data_raw.empty:
                    # Don't warn here yet, filter first
                    continue

                # --- Filter the fetched data to the *required* range ---
                # This is the core change: filter after fetching
                ticker_data_filtered = ticker_data_raw[
                    (ticker_data_raw.index >= buffer_start_date)
                    & (ticker_data_raw.index <= end_date)
                ].copy()
                # ----------------------------------------------------------

                if ticker_data_filtered.empty:
                    print(
                        f"Warning: No data found for {ticker} in required range {buffer_start_date.date()} to {end_date.date()}. Skipping."
                    )
                    continue

                # Check if the filtered data actually starts after the buffer date
                if ticker_data_filtered.index[0] > buffer_start_date:
                    print(
                        f"Warning: Data for {ticker} starts at {ticker_data_filtered.index[0].date()}, which is within the buffer period but after the intended buffer start {buffer_start_date.date()}."
                    )

                # Add Ticker column
                ticker_data_filtered[TICKER] = ticker

                # Reset index to prepare for MultiIndex
                ticker_data_filtered = ticker_data_filtered.reset_index()

                # Append the *filtered* data to our list
                all_data_raw.append(ticker_data_filtered)

            except ValueError as ve:
                # Handle cases where get_historical_data legitimately finds no data at all
                print(
                    f"Warning: Could not fetch any data for {ticker} (up to {days_to_fetch} days back): {ve}. Skipping."
                )
                continue
            except Exception as e:
                print(f"Error processing data for {ticker}: {e}")
                continue

        if not all_data_raw:
            # Raise error only if NO data was found for ANY ticker in the required range
            raise ValueError(
                "No valid data found for any ticker in the specified date range."
            )

        # Combine all *filtered* data into one DataFrame
        combined_data = pd.concat(all_data_raw, ignore_index=True)

        # Ensure 'timestamp' is tz-aware (UTC)
        # The get_historical_data function should already return UTC timestamps
        # but we double-check and ensure consistency here.
        combined_data[TIMESTAMP] = pd.to_datetime(
            combined_data[TIMESTAMP], errors="coerce", utc=True
        )
        combined_data.dropna(
            subset=[TIMESTAMP], inplace=True
        )  # Drop rows where conversion failed

        if combined_data.empty:
            # If after concatenation and timestamp handling, it's still empty
            raise ValueError(
                "Data became empty after processing timestamps. Check data quality."
            )

        # Create MultiIndex
        combined_data = combined_data.set_index([TIMESTAMP, TICKER])

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
        if benchmark_ticker not in all_data.index.get_level_values(TICKER).unique():
            return pd.Series(dtype=float)  # Empty series

        # Get first timestamp in actual backtest (not buffer)
        # Make sure backtest_start_date is tz-aware in the right way
        backtest_start = ensure_utc_tz(self.backtest_start_date)

        try:
            # Get prices for the benchmark ticker
            benchmark_prices = all_data.xs(benchmark_ticker, level=TICKER)["close"]

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
                (all_data.index.get_level_values(TIMESTAMP) >= start_date)
                & (all_data.index.get_level_values(TIMESTAMP) <= end_date)
            ]
            .index.get_level_values(TIMESTAMP)
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
                all_data.index.get_level_values(TIMESTAMP) <= current_timestamp
            ]

            # Prepare next day's data for trade execution
            next_day_data = {}

            # Check if there's a next timestamp available (not the last day)
            if i < len(backtest_timestamps) - 1:
                next_timestamp = backtest_timestamps[i + 1]

                try:
                    # Get all tickers for the next timestamp
                    next_day_slice = all_data.xs(next_timestamp, level=TIMESTAMP)

                    # Build dictionary of next day OHLC data for each ticker
                    next_day_ohlc = {}
                    for ticker in data_slice.index.get_level_values(TICKER).unique():
                        if ticker in next_day_slice.index:
                            ticker_row = next_day_slice.loc[ticker]
                            next_day_ohlc[ticker] = {
                                "open": ticker_row[OPEN],
                                "high": ticker_row[HIGH],
                                "low": ticker_row[LOW],
                                "close": ticker_row[CLOSE],
                                "volume": ticker_row.get(VOLUME, None),
                            }

                    if next_day_ohlc:
                        next_day_data = next_day_ohlc
                except KeyError:
                    # No data for next timestamp
                    pass

            if not data_slice.empty:
                # Get the latest prices for all tickers at this timestamp
                try:
                    latest_data = data_slice.xs(current_timestamp, level=TIMESTAMP)
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
                self.strategy.execute(data_slice, next_day_data)

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
        # Simply print the string generated by get_results_as_string
        results_str = self.get_results_as_string()
        if results_str:
            print(results_str)
        else:
            print("Backtest has not been run yet or failed to generate results string.")

    def get_results_as_string(self) -> Optional[str]:
        """Generate backtest results and stats as a formatted string."""
        if not self.portfolio or not self.strategy:
            return None  # Indicate that results aren't ready

        metrics = self.get_performance_metrics()
        if "error" in metrics:
            return f"Error retrieving metrics: {metrics['error']}"

        output_lines = []
        separator = "=" * 50

        output_lines.append("\n" + separator)
        output_lines.append(f"BACKTEST RESULTS: {self.strategy.name}")
        output_lines.append(separator)
        output_lines.append(
            f"Period: {metrics['start_date'].date()} to {metrics['end_date'].date()}"
        )
        output_lines.append(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        output_lines.append(
            f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}"
        )

        profit_loss = metrics["final_portfolio_value"] - metrics["initial_capital"]
        profit_loss_pct = (
            profit_loss / metrics["initial_capital"] * 100
            if metrics["initial_capital"] != 0
            else 0
        )
        output_lines.append(
            f"Profit/Loss: ${profit_loss:,.2f} ({profit_loss_pct:.2f}%)"
        )

        output_lines.append("\nPERFORMANCE METRICS:")
        output_lines.append(f"Total Return: {metrics.get('total_return', 0.0):.2%}")
        output_lines.append(
            f"Annualized Return: {metrics.get('annualized_return', 0.0):.2%}"
        )
        output_lines.append(f"Volatility (Ann.): {metrics.get('volatility', 0.0):.2%}")
        output_lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
        output_lines.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0.0):.2f}")
        output_lines.append(f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}")
        output_lines.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0.0):.2f}")
        output_lines.append(f"Number of Trades: {metrics.get('num_trades', 0)}")

        output_lines.append("\nSTRATEGY PARAMETERS:")
        for param, value in self.strategy.parameters.items():
            output_lines.append(f"  {param}: {value}")

        output_lines.append("\nFINAL HOLDINGS:")
        if not self.portfolio.holdings:
            output_lines.append("  No positions held.")
        else:
            for ticker, shares in self.portfolio.holdings.items():
                if shares > 0:
                    current_price = self.portfolio.current_prices.get(ticker, 0)
                    value = shares * current_price
                    output_lines.append(
                        f"  {ticker}: {shares} shares @ ${current_price:,.2f} (Value: ${value:,.2f})"
                    )

        output_lines.append(f"\nFinal Cash: ${self.portfolio.cash:,.2f}")
        output_lines.append(separator)

        # Optional: Add trade summary
        if hasattr(self.portfolio, "trade_history") and self.portfolio.trade_history:
            output_lines.append("\nTRADE SUMMARY:")
            try:
                trades_df = pd.DataFrame(self.portfolio.trade_history)

                # Count buys and sells
                buys = trades_df[trades_df["direction"] == "BUY"]
                sells = trades_df[trades_df["direction"] == "SELL"]

                output_lines.append(f"  Total Trades: {len(trades_df)}")
                output_lines.append(f"  Buys: {len(buys)}")
                output_lines.append(f"  Sells: {len(sells)}")

                # Show 5 most recent trades
                if len(trades_df) > 0:
                    output_lines.append("\nRECENT TRADES:")
                    # Ensure TIMESTAMP column exists and is datetime-like
                    if TIMESTAMP in trades_df.columns:
                        trades_df[TIMESTAMP] = pd.to_datetime(
                            trades_df[TIMESTAMP], errors="coerce"
                        )
                        recent_trades = trades_df.sort_values(
                            TIMESTAMP, ascending=False
                        ).head(5)
                        for _, trade in recent_trades.iterrows():
                            ts_str = (
                                trade[TIMESTAMP].strftime("%Y-%m-%d")
                                if pd.notna(trade[TIMESTAMP])
                                else "N/A"
                            )
                            direction = trade["direction"]
                            ticker = trade[TICKER]
                            quantity = abs(trade["quantity"])
                            price = trade["price"]
                            value = abs(trade["value"])
                            output_lines.append(
                                f"  {ts_str} - {direction} {quantity} {ticker} @ ${price:.2f} (Value: ${value:.2f})"
                            )
                    else:
                        output_lines.append(
                            "  (Timestamp column missing in trade history)"
                        )
            except Exception as trade_err:
                output_lines.append(f"  (Error generating trade summary: {trade_err})")

        return "\n".join(output_lines)


if __name__ == "__main__":
    import os
    from datetime import datetime, timedelta

    import click

    # Read default tickers from nyse_tickers.txt
    def get_default_tickers():
        try:
            # Try different possible locations for the file
            possible_paths = [
                "nyse_tickers.txt",
                os.path.join("src", "nyse_tickers.txt"),
                os.path.join(os.getcwd(), "nyse_tickers.txt"),
                os.path.join(os.getcwd(), "src", "nyse_tickers.txt"),
            ]

            # Try to find and read the file
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        tickers = [
                            line.strip()
                            for line in f
                            if line.strip() and not line.startswith("^")
                        ]
                    print(f"Loaded {len(tickers)} default tickers from {path}")
                    return tickers

            # If file not found, return a small default list
            print(
                "Warning: nyse_tickers.txt not found. Using minimal default ticker list."
            )
            return ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]

        except Exception as e:
            print(f"Error loading default tickers: {e}")
            return ["SPY"]  # Fallback to SPY if there's an error

    @click.group()
    def cli():
        """Backtest trading strategies with historical data."""
        pass

    @cli.command()
    def list_strategies():
        """List all available strategies."""
        strategies = StrategyFactory.list_available_strategies()
        if strategies:
            print("Available strategies:")
            for strategy in strategies:
                print(f"  - {strategy}")
        else:
            print("No strategies found.")

    @cli.command()
    @click.option(
        "--strategy",
        required=True,
        help="Strategy to use (e.g., 'rsi_strategy')",
    )
    @click.option(
        "--ticker",
        help="Comma-separated list of ticker symbols to backtest (e.g., 'AAPL,MSFT,GOOGL'). "
        "If not specified, uses all tickers from nyse_tickers.txt",
    )
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
    @click.option(
        "--benchmark", help="Ticker to use as benchmark (defaults to first ticker)"
    )
    def run(strategy, ticker, db_name, months, start_cash, commission, plot, benchmark):
        """Run a strategy backtest with specified parameters."""

        # Get the strategy class using the factory
        try:
            strategy_class = StrategyFactory.get_strategy_class(strategy)
            print(f"Found strategy class: {strategy_class.__name__}")
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Calculate dates - make them timezone-aware (UTC)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months * 30)  # Approximate

        # Parse ticker option: either comma-separated list or defaults from file
        if ticker:
            # Split comma-separated tickers and strip whitespace
            ticker_list = [t.strip() for t in ticker.split(",") if t.strip()]
        else:
            ticker_list = get_default_tickers()

        if not ticker_list:
            print("Error: No tickers specified and no default tickers found")
            return

        # Set benchmark ticker to first ticker if not specified
        benchmark_ticker = benchmark if benchmark else ticker_list[0]

        print(
            f"Running {strategy_class.__name__} for {len(ticker_list)} tickers from {start_date.date()} to {end_date.date()}"
        )
        # If there are many tickers, just show the first few
        if len(ticker_list) <= 5:
            print(f"Tickers: {', '.join(ticker_list)}")
        else:
            print(
                f"Tickers: {', '.join(ticker_list[:5])}... and {len(ticker_list) - 5} more"
            )

        print(f"Using {benchmark_ticker} as benchmark for performance comparison")

        # Get default parameters for the strategy
        strategy_params = StrategyFactory.get_default_parameters(strategy)

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
                tickers=ticker_list,
                start_date=start_date,
                end_date=end_date,
                db_name=db_name,
                benchmark_ticker=benchmark_ticker,
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

    # Run the command line interface
    cli()
