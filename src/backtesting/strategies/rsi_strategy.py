"""
RSI Strategy

This strategy uses the Relative Strength Index (RSI) to generate buy and sell signals.

The strategy is based on the following principles:

- Buy when the RSI is below 30
- Sell when the RSI is above 70

"""

from typing import Dict, List

from backtesting.portfolio import Portfolio
from backtesting.risk_management.stop_loss_manager import StopLossParameters
from backtesting.strategies.strategyutils.rsi_util import generate_rsi_signal_for_ticker
from backtesting.strategy import DataDict, Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER


class RSIStrategy(Strategy):
    """RSI Strategy implementation."""

    def __init__(self, portfolio: Portfolio):
        super().__init__("RSI Strategy", portfolio)
        # Initialize parameters with defaults
        self.parameters = self.get_default_parameters()

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

    def get_data_requirements(self) -> List[DataRequirement]:
        """RSI strategy only requires ticker data."""
        return [DataRequirement.TICKER]

    def generate_signals(self, data: DataDict) -> Dict[str, float]:
        """
        Generate trading signals based on RSI using the ticker data.

        Parameters:
        -----------
        data : DataDict
            Dictionary containing the required data. Must include
            DataRequirement.TICKER.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        signals = {}
        # Check if Ticker data is present
        if DataRequirement.TICKER not in data:
            print("Error: Ticker data not found in data dictionary for RSIStrategy.")
            return signals

        ticker_data_df = data[DataRequirement.TICKER]

        if ticker_data_df.empty or CLOSE not in ticker_data_df.columns:
            return signals

        # Ensure index is sorted for rolling calculations
        if not ticker_data_df.index.is_monotonic_increasing:
            ticker_data_df = ticker_data_df.sort_index()

        tickers = ticker_data_df.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["rsi_parameters"]["rsi_period"] + 1
        )  # Need diff, so period+1 points

        for ticker in tickers:
            try:
                close_prices = ticker_data_df.xs(ticker, level=TICKER)[CLOSE]
            except KeyError:
                continue  # Ticker not present in this data slice?

            if len(close_prices) < min_required_data:
                signals[ticker] = 0.0  # Not enough data
                continue

            signals[ticker] = generate_rsi_signal_for_ticker(
                close_prices,
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
                "overbought_threshold": 60,
                "oversold_threshold": 40,
            },
            "max_capital_per_position": 0.9,
            "commission": 0.0,
            "use_stop_loss": True,
            "stop_loss_parameters": StopLossParameters.get_defaults(),
        }
