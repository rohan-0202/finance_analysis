from typing import Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategies.df_columns import CLOSE, HIGH, LOW, OPEN, TICKER, TIMESTAMP
from backtesting.strategy import Strategy
from signals.signal_factory import SignalFactory


class MACDStrategy(Strategy):
    def __init__(self, portfolio: Portfolio):
        super().__init__("MACD Strategy", portfolio)
        self.parameters = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "max_capital_per_position": 0.9,
            "commission": 0.0,
        }
        self.macd_signal = SignalFactory.create_signal(
            "macd",
            fast_period=self.parameters["fast_period"],
            slow_period=self.parameters["slow_period"],
            signal_period=self.parameters["signal_period"],
        )

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.parameters:
                self.parameters[key] = value

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        if data.empty or CLOSE not in data.columns:
            return signals

        tickers = data.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["slow_period"] + self.parameters["signal_period"]
        )

        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level=TICKER)[CLOSE]
            except KeyError:
                continue  # Ticker not present in this data slice?

            if len(ticker_data) < min_required_data:
                signals[ticker] = 0.0  # Not enough data
                continue

            self.macd_signal.fast_period = self.parameters["fast_period"]
            self.macd_signal.slow_period = self.parameters["slow_period"]
            self.macd_signal.signal_period = self.parameters["signal_period"]

            macd_df = self.macd_signal.calculate_macd(ticker_data)

            if (
                macd_df.empty
                or "MACD" not in macd_df.columns
                or pd.isna(macd_df["MACD"].iloc[-1])
            ):
                signals[ticker] = 0.0  # MACD calculation failed or latest is NaN
                continue

            latest_macd = macd_df["MACD"].iloc[-1]
            print(f"DEBUG: {ticker} - Latest MACD: {latest_macd}")

            if latest_macd > 0:
                signals[ticker] = 1.0  # Buy signal (MACD > 0)
            elif latest_macd < 0:
                signals[ticker] = -1.0  # Sell signal (MACD < 0)
            else:
                signals[ticker] = 0.0  # Neutral signal (MACD = 0)

        return signals

    def execute(self, data: pd.DataFrame) -> None:
        """
        Execute the MACD strategy.

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
                "Error: Dataframe passed to MACDStrategy.execute does not have ('timestamp', 'ticker') MultiIndex."
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
        print(
            f"{latest_timestamp} - Generated Signals: {self.current_signals}"
        )  # DEBUG

        # Apply risk management (using base implementation for now)
        adjusted_signals = self.apply_risk_management(self.current_signals)
        print(f"{latest_timestamp} - Adjusted Signals: {adjusted_signals}")  # DEBUG

        for ticker, signal in adjusted_signals.items():
            print(
                f"{latest_timestamp} - Processing {ticker}, Signal: {signal}"
            )  # DEBUG
            # Check if we have a valid price for the ticker before proceeding
            if (
                ticker not in self.portfolio.current_prices
                or self.portfolio.current_prices[ticker] <= 0
            ):
                print(
                    f"{latest_timestamp} - Skipping {ticker}: Missing or invalid price. Price data: {self.portfolio.current_prices.get(ticker)}"
                )  # DEBUG
                continue  # Skip if price is missing or invalid

            current_price = self.portfolio.current_prices[
                ticker
            ]  # Get price for logging
            print(f"{latest_timestamp} - {ticker} Price: {current_price}")  # DEBUG

            # Calculate desired position size based on signal
            target_shares = self.calculate_position_size(ticker, signal)
            print(
                f"{latest_timestamp} - {ticker} Target Shares: {target_shares}"
            )  # DEBUG

            # Get current position size
            current_shares = self.portfolio.holdings.get(ticker, 0)
            print(
                f"{latest_timestamp} - {ticker} Current Shares: {current_shares}"
            )  # DEBUG

            # Calculate shares to trade to reach the target position
            trade_shares = target_shares - current_shares
            print(
                f"{latest_timestamp} - {ticker} Trade Shares: {trade_shares}"
            )  # DEBUG

            if trade_shares != 0:
                # Place the order using the base class method
                print(
                    f"{latest_timestamp} - Placing order for {ticker}: {trade_shares} shares"
                )  # DEBUG
                self.place_order(ticker, trade_shares, latest_timestamp)
            else:
                print(f"{latest_timestamp} - No trade needed for {ticker}")  # DEBUG

        # Optional: Update portfolio value after trades (often handled by the backtest loop)
        # self.portfolio.update_value(latest_timestamp)
