from typing import Any, Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.risk_management.stop_loss_manager import StopLossParameters
from backtesting.strategies.strategyutils.macd_util import (
    generate_macd_signal_for_ticker,
)
from backtesting.strategies.strategyutils.rsi_util import generate_rsi_signal_for_ticker
from backtesting.strategy import Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER


class RSIMACDStrategy(Strategy):
    """
    Combines RSI and MACD signals for trading decisions.

    Trades when either indicator provides a signal, unless they conflict.
    - Buy if RSI signals buy OR MACD signals buy (and the other isn't sell).
    - Sell if RSI signals sell OR MACD signals sell (and the other isn't buy).
    - Neutral if signals conflict (one buy, one sell) or both are neutral.
    """

    def __init__(self, portfolio: Portfolio):
        super().__init__("RSI+MACD Strategy", portfolio)
        # Combine default parameters from both strategies
        self.parameters = self.get_default_parameters()

    def set_parameters(self, **kwargs):
        """Set strategy parameters and update signal objects."""
        super().set_parameters(**kwargs)
        # Re-configure signal objects if relevant parameters changed
        if "rsi_parameters" in kwargs:
            self.parameters["rsi_parameters"] = kwargs["rsi_parameters"]
        if "macd" in kwargs:
            self.parameters["macd"] = kwargs["macd"]
        if "max_capital_per_position" in kwargs:
            self.parameters["max_capital_per_position"] = kwargs[
                "max_capital_per_position"
            ]
        if "commission" in kwargs:
            self.parameters["commission"] = kwargs["commission"]
        return self

    def generate_signals(
        self, data: Dict[DataRequirement, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Generate combined trading signals based on RSI and MACD with nuanced strength.
        This implementation favors aggressive trading while maintaining signal quality.

        Parameters:
        -----------
        data : Dict[DataRequirement, pd.DataFrame]
            Dictionary of data required by the strategy where
            keys are DataRequirement enums and values are DataFrames.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to combined signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        combined_signals = {}

        # Extract ticker data from the data dictionary
        if (
            DataRequirement.TICKER not in data
            or data[DataRequirement.TICKER].empty
            or CLOSE not in data[DataRequirement.TICKER].columns
        ):
            return combined_signals

        # Get the ticker data
        ticker_data = data[DataRequirement.TICKER]

        # Ensure index is sorted for rolling calculations
        if not ticker_data.index.is_monotonic_increasing:
            ticker_data = ticker_data.sort_index()

        tickers = ticker_data.index.get_level_values(TICKER).unique()

        # Get strategy parameters
        dominant_weight = self.parameters.get(
            "dominant_weight", 0.7
        )  # Weight for stronger signal
        secondary_weight = 1.0 - dominant_weight  # Weight for weaker signal
        min_signal_threshold = self.parameters.get(
            "min_signal_threshold", 0.1
        )  # Lower threshold

        for ticker in tickers:
            try:
                ticker_data_series = ticker_data.xs(ticker, level=TICKER)[CLOSE]
            except KeyError:
                combined_signals[ticker] = 0.0
                continue  # Ticker not present

            # Generate individual signals
            rsi_signal = generate_rsi_signal_for_ticker(
                ticker_data_series, self.parameters["rsi_parameters"]
            )
            macd_signal = generate_macd_signal_for_ticker(
                ticker_data_series, self.parameters["macd"]
            )

            # Determine which signal is stronger and order them
            if abs(rsi_signal) >= abs(macd_signal):
                dominant_signal = rsi_signal
                secondary_signal = macd_signal
            else:
                dominant_signal = macd_signal
                secondary_signal = rsi_signal

            # Calculate combined signal - weighted toward the stronger signal
            # but still incorporating both
            combined_signal = (dominant_weight * dominant_signal) + (
                secondary_weight * secondary_signal
            )

            # Handle cases where signals are in opposite directions
            signals_same_direction = (rsi_signal * macd_signal) > 0

            if not signals_same_direction:
                # If signals disagree but one is much stronger than the other,
                # still generate a trade signal in the direction of the stronger one
                if abs(dominant_signal) > 2 * abs(secondary_signal):
                    # Keep 80% of the dominant signal
                    combined_signal = 0.8 * dominant_signal
                # If similar strength but disagreeing, reduce signal (but don't eliminate)
                else:
                    combined_signal *= 0.5
            else:
                # Signals agree - amplify the combined signal
                if combined_signal > 0:
                    combined_signal = min(combined_signal * 1.2, 1.0)
                elif combined_signal < 0:
                    combined_signal = max(combined_signal * 1.2, -1.0)

            # Apply minimum threshold - using a lower threshold to be more aggressive
            if abs(combined_signal) < min_signal_threshold:
                combined_signal = 0.0

            combined_signals[ticker] = combined_signal

        return combined_signals

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Get default parameters combining RSI and MACD.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of default parameters.
        """
        return {
            # RSI defaults
            "rsi_parameters": {
                "rsi_period": 14,
                "overbought_threshold": 60,
                "oversold_threshold": 40,
            },
            # MACD defaults
            "macd": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            # Signal combination parameters
            "dominant_weight": 0.7,  # Weight given to the stronger signal
            "min_signal_threshold": 0.1,  # Lower threshold for more trading opportunities
            # Shared/Risk defaults
            "max_capital_per_position": 0.9,
            "commission": 0.001,
            "use_stop_loss": True,
            "stop_loss_parameters": StopLossParameters.get_defaults(),
        }

    def get_data_requirements(self) -> list[DataRequirement]:
        """RSI+MACD strategy requires ticker data."""
        return [DataRequirement.TICKER]
