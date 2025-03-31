from typing import TypedDict

import pandas as pd
import ta


class MACDParameters(TypedDict):
    fast_period: int
    slow_period: int
    signal_period: int


def generate_macd_signal_for_ticker(
    ticker_data: pd.Series,
    macd_parameters: MACDParameters,
) -> float:
    """
    Generate a MACD signal for a ticker.

    Args:
        ticker_data (pd.Series): The ticker data.
        macd_parameters (MACDParameters): The MACD parameters.

    Returns:
    """
    macd_indicator = ta.trend.MACD(
        close=ticker_data,
        window_slow=macd_parameters["slow_period"],
        window_fast=macd_parameters["fast_period"],
        window_sign=macd_parameters["signal_period"],
    )

    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()

    if (
        macd_line.empty
        or len(macd_line) < 2
        or pd.isna(macd_line.iloc[-1])
        or pd.isna(signal_line.iloc[-1])
    ):
        return 0.0

    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    prev_signal = signal_line.iloc[-2]

    if latest_macd > latest_signal and prev_macd <= prev_signal:
        return 1.0  # Buy signal (MACD crossed above Signal)
    elif latest_macd < latest_signal and prev_macd >= prev_signal:
        return -1.0  # Sell signal (MACD crossed below Signal)
    else:
        return 0.0  # Hold signal (No crossover)
