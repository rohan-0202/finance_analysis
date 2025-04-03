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
        Generate combined trading signals based on RSI and MACD.

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

            # Combine signals: Trade if either signals, unless they conflict
            signal = 0.0
            if rsi_signal == 1 and macd_signal != -1:
                signal = 1.0
            elif macd_signal == 1 and rsi_signal != -1:
                signal = 1.0
            elif rsi_signal == -1 and macd_signal != 1:
                signal = -1.0
            elif macd_signal == -1 and rsi_signal != 1:
                signal = -1.0
            # else: signal remains 0.0 (Neutral, including conflict case where one is 1 and other is -1)

            combined_signals[ticker] = signal

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
            # Shared/Risk defaults (using RSI's as base, can be adjusted)
            "max_capital_per_position": 0.9,
            "commission": 0.001,
            "use_stop_loss": True,
            "stop_loss_parameters": StopLossParameters.get_defaults(),
        }

    def get_data_requirements(self) -> list[DataRequirement]:
        """RSI+MACD strategy requires ticker data."""
        return [DataRequirement.TICKER]
