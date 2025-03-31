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
from backtesting.strategies.strategyutils.rsi_util import generate_rsi_signal_for_ticker
from backtesting.strategy import Strategy
from common.df_columns import CLOSE, TICKER


class RSIStrategy(Strategy):
    """RSI Strategy implementation."""

    def __init__(self, portfolio: Portfolio):
        super().__init__("RSI Strategy", portfolio)
        # Initialize parameters with defaults
        self.parameters = {
            "rsi_parameters": {
                "rsi_period": 14,
                "overbought_threshold": 70,
                "oversold_threshold": 30,
            },
            "max_capital_per_position": 0.1,
            "commission": 0.0,
        }

    def set_parameters(self, **kwargs):
        """Set strategy parameters and update signal object."""
        super().set_parameters(**kwargs)
        # Re-configure the rsi_signal instance if relevant parameters changed
        # Assumes the rsi_signal object has mutable attributes like window, overbought, oversold
        if "rsi_period" in kwargs:
            self.parameters["rsi_parameters"]["rsi_period"] = self.parameters[
                "rsi_period"
            ]
        if "overbought_threshold" in kwargs:
            self.parameters["rsi_parameters"]["overbought_threshold"] = self.parameters[
                "overbought_threshold"
            ]
        if "oversold_threshold" in kwargs:
            self.parameters["rsi_parameters"]["oversold_threshold"] = self.parameters[
                "oversold_threshold"
            ]
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

            signals[ticker] = generate_rsi_signal_for_ticker(
                ticker_data,
                self.parameters["rsi_parameters"],
            )

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
            "rsi_parameters": {
                "rsi_period": 14,
                "overbought_threshold": 70,
                "oversold_threshold": 30,
            },
            "max_capital_per_position": 0.1,
            "commission": 0.0,
        }
