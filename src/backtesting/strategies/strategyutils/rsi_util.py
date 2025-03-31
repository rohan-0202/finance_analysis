from typing import TypedDict

import numpy as np
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
    Generate an RSI signal for a ticker.

    Args:
        ticker_data (pd.Series): The ticker data.
        rsi_parameters (RsiParams): The RSI parameters.

    Returns:
    """
    if ticker_data.empty or len(ticker_data) < rsi_parameters["rsi_period"]:
        # Return an empty series or series of NaNs matching the input index length
        # Ensure index is preserved for alignment later
        return pd.Series(
            [np.nan] * len(ticker_data), index=ticker_data.index, dtype=float
        )

    # Use the ta library to calculate RSI
    # fillna=False prevents filling initial NaNs required for window calculation
    rsi_indicator = ta.momentum.RSIIndicator(
        close=ticker_data, window=rsi_parameters["rsi_period"], fillna=False
    )
    rsi_series = rsi_indicator.rsi()

    if rsi_series.empty or rsi_series.isna().all() or pd.isna(rsi_series.iloc[-1]):
        return 0.0  # RSI calculation failed or latest is NaN

    latest_rsi = rsi_series.iloc[-1]

    if latest_rsi > rsi_parameters["overbought_threshold"]:
        return -1.0  # Sell signal (Overbought)
    elif latest_rsi < rsi_parameters["oversold_threshold"]:
        return 1.0  # Buy signal (Oversold)
    else:
        return 0.0  # Neutral signal
