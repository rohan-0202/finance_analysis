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
    Generate a MACD signal for a ticker with continuous values from -1 to 1.

    Signal strength depends on:
    - MACD/Signal crossovers (primary trigger)
    - Distance between MACD and Signal lines
    - Histogram steepness (rate of change)
    - MACD distance from zero line

    Args:
        ticker_data (pd.Series): The ticker price data.
        macd_parameters (MACDParameters): The MACD parameters.

    Returns:
        float: Signal value between -1 (strong sell) and 1 (strong buy)
    """
    macd_indicator = ta.trend.MACD(
        close=ticker_data,
        window_slow=macd_parameters["slow_period"],
        window_fast=macd_parameters["fast_period"],
        window_sign=macd_parameters["signal_period"],
    )

    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    histogram = macd_indicator.macd_diff()  # This is MACD - Signal

    if (
        macd_line.empty
        or len(macd_line) < 3  # Need at least 3 periods for slope calculations
        or pd.isna(macd_line.iloc[-1])
        or pd.isna(signal_line.iloc[-1])
    ):
        return 0.0

    # Get the recent values
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_histogram = histogram.iloc[-1]

    prev_macd = macd_line.iloc[-2]
    prev_signal = signal_line.iloc[-2]
    prev_histogram = histogram.iloc[-2]

    # Calculate base signal from crossover (as before)
    base_signal = 0.0
    if latest_macd > latest_signal and prev_macd <= prev_signal:
        base_signal = 0.6  # Base buy signal (MACD crossed above Signal)
    elif latest_macd < latest_signal and prev_macd >= prev_signal:
        base_signal = -0.6  # Base sell signal (MACD crossed below Signal)

    # If no crossover, check if MACD is already above/below signal
    # and provide a weaker continuation signal
    elif latest_macd > latest_signal:
        base_signal = 0.2  # Weaker buy signal (MACD already above Signal)
    elif latest_macd < latest_signal:
        base_signal = -0.2  # Weaker sell signal (MACD already below Signal)

    # No signal if barely any difference
    if abs(latest_macd - latest_signal) < 0.0001:
        return 0.0

    # Calculate signal strength modifiers

    # 1. Normalize the distance between MACD and Signal (0 to 0.2)
    # Use average absolute MACD value as a scaling factor
    avg_macd_abs = macd_line.abs().rolling(window=10).mean().iloc[-1]
    if avg_macd_abs == 0:
        avg_macd_abs = 0.0001  # Avoid division by zero

    normalized_distance = (
        min(abs(latest_macd - latest_signal) / avg_macd_abs, 1.0) * 0.2
    )

    # 2. Histogram momentum (change in histogram) (0 to 0.1)
    histogram_momentum = 0.0
    histogram_change = latest_histogram - prev_histogram
    if histogram_change != 0 and latest_histogram != 0:
        # If histogram is increasing in the signal direction, boost the signal
        sign_match = (histogram_change > 0 and base_signal > 0) or (
            histogram_change < 0 and base_signal < 0
        )
        if sign_match:
            histogram_momentum = (
                min(abs(histogram_change / latest_histogram), 1.0) * 0.1
            )

    # 3. MACD distance from zero (0 to 0.1)
    # Further from zero is stronger, but extreme values might mean overextended
    macd_zero_distance = min(abs(latest_macd) / avg_macd_abs, 1.0) * 0.1

    # Combine the modifiers with the base signal
    signal_strength = base_signal

    if base_signal > 0:
        signal_strength += normalized_distance + histogram_momentum + macd_zero_distance
        return min(signal_strength, 1.0)  # Cap at 1.0
    elif base_signal < 0:
        signal_strength -= normalized_distance + histogram_momentum + macd_zero_distance
        return max(signal_strength, -1.0)  # Cap at -1.0
    else:
        return 0.0
