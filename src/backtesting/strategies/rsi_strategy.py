"""
RSI Strategy

This strategy uses the Relative Strength Index (RSI) to generate buy and sell signals.

The strategy is based on the following principles:

- Buy when the RSI is below 30
- Sell when the RSI is above 70

"""

from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategy import Strategy
from signals.rsi_signal import RSISignal
from signals.signal_factory import SignalFactory

# Assuming RSISignal class might be needed if Factory needs refinement or direct use
# from signals.rsi_signal import RSISignal


class RSIStrategy(Strategy):
    """RSI Strategy implementation."""

    def __init__(self, portfolio: Portfolio):
        super().__init__("RSI Strategy", portfolio)
        # Initialize parameters with defaults
        self.parameters = {
            "rsi_period": 14,
            "overbought_threshold": 70,
            "oversold_threshold": 30,
            "max_capital_per_position": 0.1,
            "commission": 0.0,
        }
        # Create and configure the RSI signal object using parameters
        # Option 1: Assuming SignalFactory can take parameters
        self.rsi_signal: RSISignal = SignalFactory.create_signal(
            "rsi",
            window=self.parameters["rsi_period"],
            overbought=self.parameters["overbought_threshold"],
            oversold=self.parameters["oversold_threshold"],
        )
        # Option 2: If SignalFactory cannot take params, instantiate directly
        # from signals.rsi_signal import RSISignal # requires import
        # self.rsi_signal = RSISignal(
        #     window=self.parameters["rsi_period"],
        #     overbought=self.parameters["overbought_threshold"],
        #     oversold=self.parameters["oversold_threshold"],
        # )

    def set_parameters(self, **kwargs):
        """Set strategy parameters and update signal object."""
        super().set_parameters(**kwargs)
        # Re-configure the rsi_signal instance if relevant parameters changed
        # Assumes the rsi_signal object has mutable attributes like window, overbought, oversold
        if "rsi_period" in kwargs:
            self.rsi_signal.window = self.parameters["rsi_period"]
        if "overbought_threshold" in kwargs:
            self.rsi_signal.overbought = self.parameters["overbought_threshold"]
        if "oversold_threshold" in kwargs:
            self.rsi_signal.oversold = self.parameters["oversold_threshold"]
        # Note: Commission is used directly from self.parameters in place_order
        return self

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on RSI.

        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex DataFrame with ('Timestamp', 'Ticker') index
            and 'close' column. Must contain enough history for RSI calculation.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        signals = {}
        if data.empty or "close" not in data.columns:
            return signals

        # Ensure index is sorted for rolling calculations
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        tickers = data.index.get_level_values("Ticker").unique()
        min_required_data = (
            self.parameters["rsi_period"] + 1
        )  # Need diff, so period+1 points

        overbought = self.parameters["overbought_threshold"]
        oversold = self.parameters["oversold_threshold"]

        for ticker in tickers:
            # Use .loc for potentially non-unique Ticker index slices if needed
            # ticker_data = data.loc[(slice(None), ticker), 'close']
            # Using xs assumes Ticker level is unique per Timestamp or handles non-uniqueness gracefully
            try:
                ticker_data = data.xs(ticker, level="Ticker")["close"]
            except KeyError:
                continue  # Ticker not present in this data slice?

            if len(ticker_data) < min_required_data:
                signals[ticker] = 0.0  # Not enough data
                continue

            # Ensure the RSI calculation uses the current strategy parameter for the window
            self.rsi_signal.window = self.parameters["rsi_period"]
            rsi_series = self.rsi_signal.calculate_rsi(ticker_data)

            if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
                signals[ticker] = 0.0  # RSI calculation failed or latest is NaN
                continue

            latest_rsi = rsi_series.iloc[-1]

            if latest_rsi > overbought:
                signals[ticker] = -1.0  # Sell signal (Overbought)
            elif latest_rsi < oversold:
                signals[ticker] = 1.0  # Buy signal (Oversold)
            else:
                signals[ticker] = 0.0  # Neutral signal

        return signals

    def execute(self, data: pd.DataFrame) -> None:
        """
        Execute the RSI strategy.

        Generates signals and places orders based on the latest data.

        Parameters:
        -----------
        data : pd.DataFrame
            Latest market data (MultiIndex or format expected by generate_signals).
            Should contain data up to the current execution timestamp.
        """
        if data.empty:
            return

        # Ensure data has the expected MultiIndex
        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != [
            "Timestamp",
            "Ticker",
        ]:
            print(
                "Error: Dataframe passed to RSIStrategy.execute does not have ('Timestamp', 'Ticker') MultiIndex."
            )
            # Or handle alternative data formats if necessary
            return

        # Get the latest timestamp from the data provided
        latest_timestamp = data.index.get_level_values("Timestamp").max()
        self.last_update_time = latest_timestamp  # Update strategy timestamp

        # Update portfolio's current prices based on the latest 'close' prices in data
        # Assumes data contains the closing prices for latest_timestamp
        try:
            latest_data_slice = data.xs(latest_timestamp, level="Timestamp")
            for ticker, row in latest_data_slice.iterrows():
                # Check if ticker is a string/expected type, row might be Series
                if (
                    isinstance(ticker, str)
                    and "close" in row
                    and pd.notna(row["close"])
                ):
                    self.portfolio.current_prices[ticker] = row["close"]
        except KeyError:
            # Handle cases where the latest timestamp might not be fully represented?
            # Or log a warning. For backtesting, this slice should typically exist.
            print(
                f"Warning: Could not extract data slice for timestamp {latest_timestamp}"
            )
            # Attempt to get prices from the last available data point for each ticker if needed
            # ... (more robust price update logic might be required depending on data feed)
            pass

        # Generate signals based on the historical data provided (up to latest_timestamp)
        self.current_signals = self.generate_signals(data)

        # Apply risk management (using base implementation for now)
        adjusted_signals = self.apply_risk_management(self.current_signals)

        for ticker, signal in adjusted_signals.items():
            # Check if we have a valid price for the ticker before proceeding
            if (
                ticker not in self.portfolio.current_prices
                or self.portfolio.current_prices[ticker] <= 0
            ):
                continue  # Skip if price is missing or invalid

            # Calculate desired position size based on signal
            target_shares = self.calculate_position_size(ticker, signal)

            # Get current position size
            current_shares = self.portfolio.holdings.get(ticker, 0)

            # Calculate shares to trade to reach the target position
            trade_shares = target_shares - current_shares

            if trade_shares != 0:
                # Place the order using the base class method
                self.place_order(ticker, trade_shares, latest_timestamp)

        # Optional: Update portfolio value after trades (often handled by the backtest loop)
        # self.portfolio.update_value(latest_timestamp)


if __name__ == "__main__":
    # Import necessary modules
    from datetime import datetime, timedelta

    import click  # Import click
    import pandas as pd

    # Import Portfolio if it's not already imported at the top level
    from backtesting.portfolio import Portfolio  # Make sure Portfolio is imported

    # Assuming db_util.py is accessible in the path
    from db_util import get_historical_data

    # --- Configuration --- Default values moved to click options

    @click.command()
    @click.option(
        "--ticker",
        default="SPY",
        help="Ticker symbol to backtest.",
        show_default=True,
    )
    @click.option(
        "--db-name",
        default="stock_data.db",
        help="Database name for historical data.",
        show_default=True,
    )
    @click.option(
        "--months",
        type=int,
        default=12,
        help="Number of months to backtest.",
        show_default=True,
    )
    @click.option(
        "--start-cash",
        type=float,
        default=10000.0,
        help="Initial capital for the portfolio.",
        show_default=True,
    )
    @click.option(
        "--commission",
        type=float,
        default=0.001,
        help="Commission per trade.",
        show_default=True,
    )
    @click.option(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI calculation period.",
        show_default=True,
    )
    @click.option(
        "--overbought",
        type=float,
        default=70.0,
        help="RSI overbought threshold.",
        show_default=True,
    )
    @click.option(
        "--oversold",
        type=float,
        default=30.0,
        help="RSI oversold threshold.",
        show_default=True,
    )
    @click.option(
        "--max-position-pct",
        type=float,
        default=0.9,
        help="Maximum percentage of capital per position.",
        show_default=True,
    )
    def run_backtest(
        ticker,
        db_name,
        months,
        start_cash,
        commission,
        rsi_period,
        overbought,
        oversold,
        max_position_pct,
    ):
        """Runs the RSI strategy backtest for a given ticker and parameters."""

        # --- Setup ---
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months * 30)  # Approximate

        print(f"Starting backtest for {ticker} from {start_date} to {end_date}")

        # Calculate fetch start date (including buffer for initial RSI calculation)
        buffer_days = 30
        fetch_start_date = start_date - timedelta(days=buffer_days)

        # Calculate the total number of days needed from fetch_start_date up to today (end_date)
        total_days_to_fetch = (end_date - fetch_start_date).days

        print(f"Fetching {total_days_to_fetch} days of data ending {end_date}...")

        try:
            # Use the 'days' parameter as defined in db_util.py
            all_data = get_historical_data(
                ticker_symbol=ticker, db_name=db_name, days=total_days_to_fetch
            )
            if all_data is None or all_data.empty:
                raise ValueError("No data fetched.")

            # Convert index to DatetimeIndex if it's not already (db_util does this)
            # Ensure it's timezone-naive or handle timezone consistently if needed
            if all_data.index.tz is not None:
                # Convert to timezone-naive UTC if necessary, or choose appropriate timezone handling
                all_data.index = all_data.index.tz_convert(
                    None
                )  # Example: Make naive UTC

            # Filter data to ensure it doesn't go beyond our desired end_date,
            # although get_historical_data likely stops at the current time anyway.
            all_data = all_data[all_data.index <= pd.Timestamp(end_date).normalize()]

            if all_data.empty:
                raise ValueError(f"No data found up to {end_date}.")

            # Ensure 'close' column exists
            if "close" not in all_data.columns:
                raise ValueError("Data must contain a 'close' column.")

            # Add 'Ticker' level to index for compatibility with strategy methods
            all_data["Ticker"] = ticker
            all_data = all_data.reset_index().rename(columns={"timestamp": "Timestamp"})
            all_data["Timestamp"] = pd.to_datetime(
                all_data["Timestamp"]
            )  # Ensure datetime type
            all_data = all_data.set_index(["Timestamp", "Ticker"])
            all_data = all_data.sort_index()

        except Exception as e:
            print(f"Error fetching or preparing data for {ticker}: {e}")
            return  # Use return instead of exit in a function

        # Create Portfolio and Strategy instances
        portfolio = Portfolio(initial_capital=start_cash)
        strategy = RSIStrategy(portfolio)

        # Set strategy parameters from command-line options
        strategy.set_parameters(
            rsi_period=rsi_period,
            overbought_threshold=overbought,
            oversold_threshold=oversold,
            max_capital_per_position=max_position_pct,
            commission=commission,
        )

        print(f"Initial Portfolio Value: ${portfolio.initial_capital:,.2f}")

        # --- Backtesting Loop ---
        # Get unique timestamps available within the actual backtest period
        # Ensure Timestamps are compatible (e.g., timezone-naive)
        actual_start_timestamp = pd.Timestamp(start_date)  # .normalize() if needed
        actual_end_timestamp = pd.Timestamp(end_date)  # .normalize() if needed

        # Filter all_data to the actual backtest window before getting timestamps
        backtest_data_range = all_data[
            (all_data.index.get_level_values("Timestamp") >= actual_start_timestamp)
            & (all_data.index.get_level_values("Timestamp") <= actual_end_timestamp)
        ]

        backtest_timestamps = backtest_data_range.index.get_level_values(
            "Timestamp"
        ).unique()

        if len(backtest_timestamps) == 0:
            print(
                f"No trading data available between {start_date} and {end_date} after fetching."
            )
            return  # Use return instead of exit in a function

        print(f"Running simulation for {len(backtest_timestamps)} trading days...")

        # Sort timestamps chronologically
        backtest_timestamps = sorted(backtest_timestamps)

        # --- Calculate Buy and Hold Benchmark ---
        buy_hold_values = pd.Series(dtype=float)  # Initialize empty series
        if backtest_timestamps:
            first_day_timestamp = backtest_timestamps[0]
            try:
                # Get the closing price on the first day of the actual backtest period
                first_price = all_data.loc[(first_day_timestamp, ticker), "close"]
                if pd.notna(first_price) and first_price > 0:
                    # Calculate shares bought (ignoring commission for simplicity)
                    buy_hold_shares = start_cash / first_price

                    # Get closing prices for the ticker during the backtest period
                    buy_hold_prices = backtest_data_range.xs(ticker, level="Ticker")[
                        "close"
                    ]

                    # Calculate the value for each day in the backtest period
                    buy_hold_values = buy_hold_prices * buy_hold_shares
                else:
                    print(
                        "Warning: Could not get valid starting price for Buy & Hold benchmark."
                    )

            except KeyError:
                print(
                    f"Warning: Could not find data for {ticker} on {first_day_timestamp} for Buy & Hold benchmark."
                )
            except Exception as e:
                print(f"Warning: Error calculating Buy & Hold benchmark: {e}")
        # --- End Buy and Hold Calculation ---

        for current_timestamp in backtest_timestamps:
            # Provide data up to and including the current day's timestamp
            # Ensure we slice from the original 'all_data' which includes the buffer period
            data_slice = all_data[
                all_data.index.get_level_values("Timestamp") <= current_timestamp
            ]

            if not data_slice.empty:
                # Execute the strategy logic for the current day
                strategy.execute(data_slice)

                # Update portfolio equity history using update_prices
                # It uses the prices already updated within strategy.execute
                # Ensure current_prices is not empty before calling
                if portfolio.current_prices:
                    portfolio.update_prices(portfolio.current_prices, current_timestamp)

        # --- Results ---
        print("\n--- Backtest Finished ---")
        # get_value() calculates based on final cash and holdings * current_prices
        final_value = portfolio.get_value()
        # Use portfolio.initial_capital for consistency in reporting
        print(f"Initial Portfolio Value: ${portfolio.initial_capital:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        # Use portfolio.initial_capital for P/L calculation
        profit_loss = final_value - portfolio.initial_capital
        if portfolio.initial_capital > 0:  # Avoid division by zero
            profit_loss_percent = (profit_loss / portfolio.initial_capital) * 100
        else:
            profit_loss_percent = 0.0
        print(
            f"Profit/Loss:             ${profit_loss:,.2f} ({profit_loss_percent:.2f}%)"
        )
        print("\nFinal Holdings:")
        if not portfolio.holdings:
            print("  No positions held.")
        else:
            for ticker, shares in portfolio.holdings.items():
                if (
                    shares > 0
                ):  # Only show held positions (could also show short positions if needed)
                    current_price = portfolio.current_prices.get(ticker, 0)
                    value = shares * current_price
                    print(
                        f"  {ticker}: {shares} shares @ ${current_price:,.2f} (Value: ${value:,.2f})"
                    )
        print(f"\nFinal Cash: ${portfolio.cash:,.2f}")

        # --- Performance Metrics ---
        print("\n--- Performance Metrics ---")
        metrics = portfolio.get_performance_metrics()
        if metrics:
            print(f"  Total Return:         {metrics.get('total_return', 0.0):.2%}")
            print(
                f"  Annualized Return:    {metrics.get('annualized_return', 0.0):.2%}"
            )
            print(f"  Volatility (Ann.):    {metrics.get('volatility', 0.0):.2%}")
            print(f"  Sharpe Ratio:         {metrics.get('sharpe_ratio', 0.0):.2f}")
            print(f"  Sortino Ratio:        {metrics.get('sortino_ratio', 0.0):.2f}")
            print(f"  Max Drawdown:         {metrics.get('max_drawdown', 0.0):.2%}")
            print(f"  Calmar Ratio:         {metrics.get('calmar_ratio', 0.0):.2f}")
            print(f"  Number of Trades:     {metrics.get('num_trades', 0)}")
            # Note: profit_per_share is a basic calculation in the current Portfolio implementation
            # print(f"  Avg Profit Per Share: ${metrics.get('profit_per_share', 0.0):.2f}")
        else:
            print("  No metrics available (likely no equity history).")

        # Optional: Print trade log if available
        # Check if trade_history exists and is not empty
        if hasattr(portfolio, "trade_history") and portfolio.trade_history:
            print("\n--- Trade Log ---")
            # Limit printing maybe the last N trades if the log is very long
            max_trades_to_print = 20
            start_index = max(0, len(portfolio.trade_history) - max_trades_to_print)
            if len(portfolio.trade_history) > max_trades_to_print:
                print(f"  (Showing last {max_trades_to_print} trades...)")

            for trade in portfolio.trade_history[start_index:]:
                ts = pd.Timestamp(trade["timestamp"]).strftime(
                    "%Y-%m-%d"
                )  # Format timestamp
                direction = trade["direction"]
                quantity = abs(trade["quantity"])  # Use absolute quantity
                value = abs(trade["value"])  # Use absolute value for cost/proceeds
                commission = trade["commission"]
                print(
                    f"  {ts} - {direction.upper()} {quantity} {trade['ticker']} @ ${trade['price']:.2f} (Value: ${value:.2f}, Comm: ${commission:.2f})"
                )

        # --- Plotting ---
        print("\n--- Portfolio Equity Curve ---")
        if hasattr(portfolio, "equity_history") and portfolio.equity_history:
            try:
                # Extract strategy timestamps and equity values
                timestamps, equity = zip(*portfolio.equity_history, strict=True)
                equity_series = pd.Series(equity, index=pd.to_datetime(timestamps))

                # Create the plot
                plt.figure(figsize=(12, 6))
                plt.plot(
                    equity_series.index,
                    equity_series.values,
                    label="Strategy Value",  # Changed label slightly
                )

                # Plot Buy and Hold benchmark if calculated
                if not buy_hold_values.empty:
                    # Ensure index is datetime for plotting alignment
                    buy_hold_values.index = pd.to_datetime(buy_hold_values.index)
                    # Align Buy & Hold data to the strategy's timestamps if necessary
                    # (This handles cases where strategy might not trade every single day)
                    aligned_buy_hold = buy_hold_values.reindex(
                        equity_series.index, method="ffill"
                    )
                    plt.plot(
                        aligned_buy_hold.index,
                        aligned_buy_hold.values,
                        label=f"Buy & Hold {ticker}",
                        linestyle="--",
                    )

                # Add titles and labels
                plt.title(
                    f"Portfolio Value Over Time ({ticker} - RSI Strategy vs Buy & Hold)"
                )  # Updated title
                plt.xlabel("Date")
                plt.ylabel("Portfolio Value ($)")
                plt.legend()  # Updated to show both labels
                plt.grid(True)
                plt.tight_layout()  # Adjust layout

                # Show the plot
                print("Displaying plot...")
                plt.show()
            except Exception as e:
                print(f"Could not generate plot: {e}")
        else:
            print("No equity history available to plot.")

    # Run the click command function
    run_backtest()
