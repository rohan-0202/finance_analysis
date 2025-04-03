from typing import TypedDict

import pandas as pd
import ta

from backtesting.strategies.strategyutils.volatililty_util import VolatilityRegime


# Define the TypedDict for RSI parameters
class RsiParams(TypedDict):
    rsi_period: int
    oversold_threshold: int
    overbought_threshold: int
    position_size_multiplier: float


def get_rsi_parameters_by_volatility(regime: VolatilityRegime) -> RsiParams:
    """
    Get RSI parameters appropriate for the current volatility regime.

    Parameters:
    -----------
    regime : VolatilityRegime
        Current volatility regime

    Returns:
    --------
    RsiParams
        Dictionary of RSI parameters conforming to the RsiParams structure
    """
    params: dict[VolatilityRegime, RsiParams] = {
        VolatilityRegime.LOW: {
            "rsi_period": 14,
            "oversold_threshold": 45,
            "overbought_threshold": 55,
            "position_size_multiplier": 1.2,
        },
        VolatilityRegime.NORMAL: {
            "rsi_period": 14,
            "oversold_threshold": 40,
            "overbought_threshold": 60,
            "position_size_multiplier": 1.0,
        },
        VolatilityRegime.HIGH: {
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "position_size_multiplier": 0.6,
        },
    }

    return params[regime]


def calculate_position_size(
    base_position_size: float,
    current_volatility: float,
    target_volatility: float = 0.15,  # 15% annualized volatility as target
    min_size_multiplier: float = 0.25,
    max_size_multiplier: float = 2.0,
) -> float:
    """
    Calculate position size inversely proportional to volatility.

    Parameters:
    -----------
    base_position_size : float
        Base position size (e.g., 0.1 = 10% of portfolio)
    current_volatility : float
        Current annualized volatility
    target_volatility : float, default=0.15
        Target volatility level (e.g., 0.15 = 15% annualized)
    min_size_multiplier : float, default=0.25
        Minimum position size multiplier
    max_size_multiplier : float, default=2.0
        Maximum position size multiplier

    Returns:
    --------
    float
        Adjusted position size
    """
    # Calculate the volatility-based multiplier
    if current_volatility <= 0:
        multiplier = max_size_multiplier
    else:
        multiplier = target_volatility / current_volatility

    # Clamp the multiplier to reasonable bounds
    multiplier = max(min_size_multiplier, min(max_size_multiplier, multiplier))

    # Calculate and return the adjusted position size
    return base_position_size * multiplier


def generate_rsi_signal_for_ticker(
    ticker_data: pd.Series,
    rsi_parameters: RsiParams,
) -> float:
    """
    Generate an RSI signal for a ticker with continuous strength between -1 and 1.

    The signal strength is based on:
    - How far RSI is from normal range boundaries
    - Distance from neutral (50) for values within the normal range

    Args:
        ticker_data (pd.Series): The ticker data.
        rsi_parameters (RsiParams): The RSI parameters.

    Returns:
        float: Signal value between -1 (strongest sell) and 1 (strongest buy)
    """
    if ticker_data.empty or len(ticker_data) < rsi_parameters["rsi_period"]:
        # Return 0 (neutral) if not enough data
        return 0.0

    # Use the ta library to calculate RSI
    rsi_indicator = ta.momentum.RSIIndicator(
        close=ticker_data, window=rsi_parameters["rsi_period"], fillna=False
    )
    rsi_series = rsi_indicator.rsi()

    if rsi_series.empty or rsi_series.isna().all() or pd.isna(rsi_series.iloc[-1]):
        return 0.0  # RSI calculation failed or latest is NaN

    latest_rsi = rsi_series.iloc[-1]

    # Get the thresholds from parameters
    oversold = rsi_parameters["oversold_threshold"]
    overbought = rsi_parameters["overbought_threshold"]

    # Calculate the signal strength based on RSI value
    if latest_rsi <= oversold:
        # Buy signal (Oversold) - scale from 0.5 to 1.0 based on how low RSI is
        # 0 RSI (theoretical min) = 1.0 signal, oversold threshold = 0.5 signal
        signal_strength = 0.5 + 0.5 * (oversold - latest_rsi) / oversold
        return min(signal_strength, 1.0)  # Cap at 1.0

    elif latest_rsi >= overbought:
        # Sell signal (Overbought) - scale from -0.5 to -1.0 based on how high RSI is
        # 100 RSI (theoretical max) = -1.0 signal, overbought threshold = -0.5 signal
        signal_strength = -0.5 - 0.5 * (latest_rsi - overbought) / (100 - overbought)
        return max(signal_strength, -1.0)  # Cap at -1.0

    else:
        # Neutral range - provide smaller signal based on distance from 50 (neutral RSI)
        # Scale from -0.5 to 0.5 based on position within the neutral band
        # Above 50 = negative signal (potential sell), Below 50 = positive signal (potential buy)
        neutral_point = 50
        max_neutral_signal = 0.5  # Maximum strength for signals within neutral range

        # Calculate normalized position within neutral band (-1 to 1)
        if latest_rsi > neutral_point:
            # Above 50 - slight sell signal
            normalized_position = (latest_rsi - neutral_point) / (
                overbought - neutral_point
            )
            return -max_neutral_signal * min(normalized_position, 1.0)
        else:
            # Below 50 - slight buy signal
            normalized_position = (neutral_point - latest_rsi) / (
                neutral_point - oversold
            )
            return max_neutral_signal * min(normalized_position, 1.0)
