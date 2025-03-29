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
from backtesting.strategies.df_columns import CLOSE, HIGH, LOW, OPEN, TICKER, TIMESTAMP
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
            MultiIndex DataFrame with ('timestamp', 'Ticker') index
            and 'close' column. Must contain enough history for RSI calculation.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        signals = {}
        if data.empty or CLOSE not in data.columns:
            return signals

        # Ensure index is sorted for rolling calculations
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        tickers = data.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["rsi_period"] + 1
        )  # Need diff, so period+1 points

        overbought = self.parameters["overbought_threshold"]
        oversold = self.parameters["oversold_threshold"]

        for ticker in tickers:
            # Use .loc for potentially non-unique Ticker index slices if needed
            # ticker_data = data.loc[(slice(None), ticker), 'close']
            # Using xs assumes Ticker level is unique per timestamp or handles non-uniqueness gracefully
            try:
                ticker_data = data.xs(ticker, level=TICKER)[CLOSE]
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
            TIMESTAMP,
            TICKER,
        ]:
            print(
                "Error: Dataframe passed to RSIStrategy.execute does not have ('timestamp', 'ticker') MultiIndex."
            )
            # Or handle alternative data formats if necessary
            return

        # Get the latest timestamp from the data provided
        latest_timestamp = data.index.get_level_values(TIMESTAMP).max()
        self.last_update_time = latest_timestamp  # Update strategy timestamp

        # Update portfolio's current prices based on the latest 'close' prices in data
        # Assumes data contains the closing prices for latest_timestamp
        try:
            latest_data_slice = data.xs(latest_timestamp, level=TIMESTAMP)
            for ticker, row in latest_data_slice.iterrows():
                # Check if ticker is a string/expected type, row might be Series
                if isinstance(ticker, str) and CLOSE in row and pd.notna(row[CLOSE]):
                    self.portfolio.current_prices[ticker] = row[CLOSE]
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
