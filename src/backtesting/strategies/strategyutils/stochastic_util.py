"""
Stochastic Oscillator Utility Functions

This module provides utility functions for calculating the Stochastic Oscillator
and generating trading signals based on it.

The Stochastic Oscillator is a momentum indicator that compares a security's closing price
to its price range over a specific period. It's particularly useful for identifying
overbought and oversold conditions in range-bound markets, making it ideal for
mean reversion strategies.

Key components:
- %K: The main line representing the current price relative to the high-low range
- %D: The signal line (typically a 3-period moving average of %K)
"""

from typing import Dict, Tuple, TypedDict, Optional

import pandas as pd
import ta


class StochasticParameters(TypedDict):
    """Parameters for Stochastic Oscillator calculations and signal generation."""

    k_period: int  # Look-back period for %K
    d_period: int  # SMA period for calculating %D
    smooth_k: int  # Smoothing for %K (1 = no smoothing)
    overbought: float  # Threshold for overbought condition (typically 80)
    oversold: float  # Threshold for oversold condition (typically 20)
    signal_threshold: float  # Magnitude required for signal (0-1)


def get_default_stochastic_parameters() -> StochasticParameters:
    """
    Get default parameters for Stochastic Oscillator calculations.

    Returns:
    --------
    StochasticParameters
        Dictionary of default Stochastic parameters
    """
    return {
        "k_period": 14,  # Standard lookback period
        "d_period": 3,  # Standard signal line smoothing
        "smooth_k": 3,  # Standard K smoothing
        "overbought": 80.0,  # Standard overbought threshold
        "oversold": 20.0,  # Standard oversold threshold
        "signal_threshold": 0.3,  # Minimum deviation from threshold for signal
    }


def calculate_stochastic(
    prices: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D) for a price series.

    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame containing 'high', 'low', and 'close' columns
    k_period : int, default=14
        Look-back period for %K calculation
    d_period : int, default=3
        Period for the signal line %D calculation
    smooth_k : int, default=3
        Smoothing period for %K

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        (%K, %D) series
    """
    if not all(col in prices.columns for col in ["high", "low", "close"]):
        raise ValueError("Price data must contain 'high', 'low', and 'close' columns")

    # Use TA-Lib integration in ta library
    stoch_indicator = ta.momentum.StochasticOscillator(
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        window=k_period,
        smooth_window=smooth_k,
        fillna=False,
    )

    # Calculate %K and %D
    k = stoch_indicator.stoch()
    d = k.rolling(window=d_period).mean()

    return k, d


def generate_stochastic_signal(
    k_series: pd.Series, d_series: pd.Series, params: StochasticParameters
) -> float:
    """
    Generate a trading signal based on Stochastic Oscillator values.

    This function generates signals based on:
    1. Overbought/oversold conditions
    2. %K and %D crossovers
    3. Divergence from extreme levels

    Parameters:
    -----------
    k_series : pd.Series
        The %K series of the Stochastic Oscillator
    d_series : pd.Series
        The %D series of the Stochastic Oscillator
    params : StochasticParameters
        Parameters for signal generation

    Returns:
    --------
    float
        Signal value: -1 (sell), 0 (neutral), or 1 (buy)
    """
    if k_series.empty or d_series.empty or len(k_series) < 2 or len(d_series) < 2:
        return 0.0

    # Get the latest values
    latest_k = k_series.iloc[-1]
    latest_d = d_series.iloc[-1]
    prev_k = k_series.iloc[-2]
    prev_d = d_series.iloc[-2]

    # Check for NaN values
    if pd.isna(latest_k) or pd.isna(latest_d) or pd.isna(prev_k) or pd.isna(prev_d):
        return 0.0

    signal = 0.0
    overbought = params["overbought"]
    oversold = params["oversold"]
    signal_threshold = params["signal_threshold"]

    # Check for extreme overbought/oversold conditions with confirmation
    if latest_k > overbought and latest_d > overbought:
        # Overbought condition - stronger signal when further from threshold
        signal_strength = min((latest_k - overbought) / (100 - overbought), 1.0)
        if signal_strength >= signal_threshold:
            signal = -1.0 * signal_strength
    elif latest_k < oversold and latest_d < oversold:
        # Oversold condition - stronger signal when further from threshold
        signal_strength = min((oversold - latest_k) / oversold, 1.0)
        if signal_strength >= signal_threshold:
            signal = 1.0 * signal_strength

    # Check for crossovers, which can generate additional signals or strengthen existing ones
    k_d_crossover = prev_k <= prev_d and latest_k > latest_d  # Bullish crossover
    d_k_crossover = prev_k >= prev_d and latest_k < latest_d  # Bearish crossover

    # Adjust signal based on crossovers
    if k_d_crossover and latest_k < overbought:
        # If %K crosses above %D and not overbought, bullish signal
        crossover_strength = 0.5 if signal == 0 else 0.25
        signal = max(signal, crossover_strength)
    elif d_k_crossover and latest_k > oversold:
        # If %K crosses below %D and not oversold, bearish signal
        crossover_strength = -0.5 if signal == 0 else -0.25
        signal = min(signal, crossover_strength)

    return signal


def generate_stochastic_signal_for_ticker(
    ohlc_data: pd.DataFrame, params: Optional[Dict] = None
) -> float:
    """
    Calculate Stochastic Oscillator and generate trading signals for a ticker.

    Parameters:
    -----------
    ohlc_data : pd.DataFrame
        DataFrame containing at minimum 'high', 'low', and 'close' columns
    params : Dict, optional
        Parameters for Stochastic calculation and signal generation, defaults
        to standard parameters if None

    Returns:
    --------
    float
        Signal value: -1 (sell), 0 (neutral), or 1 (buy)
    """
    # Get default parameters if none provided
    if params is None:
        params = get_default_stochastic_parameters()
    else:
        # Fill in any missing parameters with defaults
        default_params = get_default_stochastic_parameters()
        for key, value in default_params.items():
            if key not in params:
                params[key] = value

    # Ensure we have enough data
    min_required = max(params["k_period"], params["d_period"], params["smooth_k"]) + 1
    if len(ohlc_data) < min_required:
        return 0.0

    try:
        # Calculate Stochastic Oscillator
        k_series, d_series = calculate_stochastic(
            ohlc_data,
            k_period=params["k_period"],
            d_period=params["d_period"],
            smooth_k=params["smooth_k"],
        )

        # Generate signal
        return generate_stochastic_signal(k_series, d_series, params)
    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")
        return 0.0


def adapt_stochastic_parameters_to_volatility(
    base_params: StochasticParameters,
    current_volatility: float,
    long_term_volatility: float,
) -> StochasticParameters:
    """
    Adapt Stochastic Oscillator parameters based on the current market volatility.

    Parameters:
    -----------
    base_params : StochasticParameters
        Base parameters to adjust
    current_volatility : float
        Current market volatility (standard deviation of returns)
    long_term_volatility : float
        Long-term average volatility (for comparison)

    Returns:
    --------
    StochasticParameters
        Adjusted parameters
    """
    # Copy the base parameters
    adjusted_params = base_params.copy()

    # Calculate volatility ratio
    vol_ratio = (
        current_volatility / long_term_volatility if long_term_volatility > 0 else 1.0
    )

    # Adjust overbought/oversold thresholds based on volatility
    if vol_ratio > 1.5:  # High volatility
        # Widen thresholds in high volatility (more extreme values)
        adjusted_params["overbought"] = min(base_params["overbought"] + 5, 90)
        adjusted_params["oversold"] = max(base_params["oversold"] - 5, 10)
        # Increase signal threshold to reduce false signals
        adjusted_params["signal_threshold"] = min(
            base_params["signal_threshold"] + 0.1, 0.5
        )
    elif vol_ratio < 0.75:  # Low volatility
        # Narrow thresholds in low volatility
        adjusted_params["overbought"] = max(base_params["overbought"] - 5, 70)
        adjusted_params["oversold"] = min(base_params["oversold"] + 5, 30)
        # Decrease signal threshold to capture more subtle movements
        adjusted_params["signal_threshold"] = max(
            base_params["signal_threshold"] - 0.1, 0.2
        )

    return adjusted_params
