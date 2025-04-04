from typing import Dict, TypedDict

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.risk_management.stop_loss_manager import StopLossParameters
from backtesting.risk_management.position_sizing import PositionSizingParameters
from backtesting.strategies.strategyutils.macd_util import (
    MACDParameters,
    generate_macd_signal_for_ticker,
)
from backtesting.strategy import Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER


class MACDStrategyParameters(TypedDict):
    macd: MACDParameters
    max_capital_per_position: float  # Legacy parameter for backward compatibility
    position_sizing_type: str
    position_sizing_parameters: PositionSizingParameters
    commission: float
    use_stop_loss: bool
    stop_loss_parameters: StopLossParameters


class MACDStrategy(Strategy):
    def __init__(self, portfolio: Portfolio):
        super().__init__("MACD Strategy", portfolio)
        self.parameters: MACDStrategyParameters = {
            "macd": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            # Legacy parameter kept for backward compatibility
            "max_capital_per_position": 0.1,
            # New position sizing framework
            "position_sizing_type": "fixed_percentage",
            "position_sizing_parameters": PositionSizingParameters.get_defaults(),
            "commission": 0.0,
            "use_stop_loss": True,
            "stop_loss_parameters": StopLossParameters.get_defaults(),
        }

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key == "macd":
                self.parameters["macd"] = value
            elif key == "max_capital_per_position":
                self.parameters["max_capital_per_position"] = value
            elif key == "position_sizing_type":
                self.parameters["position_sizing_type"] = value
            elif key == "position_sizing_parameters":
                self.parameters["position_sizing_parameters"] = value
            elif key == "commission":
                self.parameters["commission"] = value
            elif key == "use_stop_loss":
                self.parameters["use_stop_loss"] = value
            elif key == "stop_loss_parameters":
                self.parameters["stop_loss_parameters"] = value
        return self

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        # Check if Ticker data is present
        if DataRequirement.TICKER not in data:
            print("Error: Ticker data not found in data dictionary for MACDStrategy.")
            return signals

        ticker_data_df = data[DataRequirement.TICKER]

        if ticker_data_df.empty or CLOSE not in ticker_data_df.columns:
            return signals

        tickers = ticker_data_df.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["macd"]["slow_period"]
            + self.parameters["macd"]["signal_period"]
            + 1
        )

        for ticker in tickers:
            try:
                close_prices = ticker_data_df.xs(ticker, level=TICKER)[CLOSE]
            except KeyError:
                continue  # Ticker not present in this data slice?

            if len(close_prices) < min_required_data:
                signals[ticker] = 0.0  # Not enough data
                continue

            # Calculate MACD using ta library
            signal = generate_macd_signal_for_ticker(
                close_prices, self.parameters["macd"]
            )
            signals[ticker] = signal

        return signals

    def get_data_requirements(self) -> list[DataRequirement]:
        """MACD strategy only requires ticker data."""
        return [DataRequirement.TICKER]
