"""
RSI Strategy

This strategy uses the Relative Strength Index (RSI) to generate buy and sell signals.

The strategy is based on the following principles:

- Buy when the RSI is below 30
- Sell when the RSI is above 70

"""

from typing import Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from common.df_columns import CLOSE, TICKER, TIMESTAMP
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

            # Use the new helper method from RSISignal
            # Pass the 'close' price series for the specific ticker
            rsi_series = self.rsi_signal._calculate_rsi_from_series(ticker_data)

            if (
                rsi_series.empty
                or rsi_series.isna().all()
                or pd.isna(rsi_series.iloc[-1])
            ):
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

    @classmethod
    def get_default_parameters(cls) -> Dict:
        """
        Get default parameters for the RSI strategy.

        Returns:
        --------
        Dict
            Dictionary of default parameters
        """
        return {
            "rsi_period": 14,
            "overbought_threshold": 70,
            "oversold_threshold": 30,
            "max_capital_per_position": 0.9,
            "commission": 0.001,
        }
