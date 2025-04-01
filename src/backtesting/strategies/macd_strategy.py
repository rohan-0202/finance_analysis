from typing import Dict, TypedDict

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategies.strategyutils.macd_util import (
    MACDParameters,
    generate_macd_signal_for_ticker,
)
from backtesting.strategy import Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER


class MACDStrategyParameters(TypedDict):
    macd: MACDParameters
    max_capital_per_position: float
    commission: float


class MACDStrategy(Strategy):
    def __init__(self, portfolio: Portfolio):
        super().__init__("MACD Strategy", portfolio)
        self.parameters: MACDStrategyParameters = {
            "macd": {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
            "max_capital_per_position": 0.9,
            "commission": 0.0,
        }

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key == "macd":
                self.parameters["macd"] = value
            elif key == "max_capital_per_position":
                self.parameters["max_capital_per_position"] = value
            elif key == "commission":
                self.parameters["commission"] = value
        return self

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        if data.empty or CLOSE not in data.columns:
            return signals

        tickers = data.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["macd"]["slow_period"]
            + self.parameters["macd"]["signal_period"]
            + 1
        )

        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level=TICKER)[CLOSE]
                if len(ticker_data) < min_required_data:
                    signals[ticker] = 0.0  # Not enough data
                    continue
            except KeyError:
                signals[ticker] = 0.0  # Ticker data not available
                continue

            # Calculate MACD using ta library
            signal = generate_macd_signal_for_ticker(
                ticker_data, self.parameters["macd"]
            )
            signals[ticker] = signal

        return signals

    def get_data_requirements(self) -> list[DataRequirement]:
        """MACD strategy only requires ticker data."""
        return [DataRequirement.TICKER]
