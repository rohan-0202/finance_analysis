from typing import Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from common.df_columns import CLOSE, TICKER, TIMESTAMP
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
        super().set_parameters(**kwargs)
        # Update MACD signal parameters if relevant
        if "fast_period" in kwargs:
            self.macd_signal.fast_period = self.parameters["fast_period"]
        if "slow_period" in kwargs:
            self.macd_signal.slow_period = self.parameters["slow_period"]
        if "signal_period" in kwargs:
            self.macd_signal.signal_period = self.parameters["signal_period"]
        return self

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        if data.empty or CLOSE not in data.columns:
            return signals

        tickers = data.index.get_level_values(TICKER).unique()
        min_required_data = (
            self.parameters["slow_period"] + self.parameters["signal_period"] + 1
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

            macd_df = self.macd_signal.calculate_macd(ticker_data)

            macd_df = macd_df.dropna()

            if (
                macd_df.empty
                or len(macd_df) < 2
                or "macd" not in macd_df.columns
                or "signal" not in macd_df.columns
                or pd.isna(macd_df["macd"].iloc[-1])
                or pd.isna(macd_df["signal"].iloc[-1])
            ):
                signals[ticker] = 0.0
                continue

            latest_macd = macd_df["macd"].iloc[-1]
            latest_signal = macd_df["signal"].iloc[-1]
            prev_macd = macd_df["macd"].iloc[-2]
            prev_signal = macd_df["signal"].iloc[-2]

            if latest_macd > latest_signal and prev_macd <= prev_signal:
                signals[ticker] = 1.0  # Buy signal (MACD crossed above Signal)
            elif latest_macd < latest_signal and prev_macd >= prev_signal:
                signals[ticker] = -1.0  # Sell signal (MACD crossed below Signal)
            else:
                signals[ticker] = 0.0  # Hold signal (No crossover)

        return signals
