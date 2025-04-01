"""
Bollinger Bands Utility Functions

This module provides utility functions for calculating Bollinger Bands and
generating trading signals based on Bollinger Bands.
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np


def calculate_bollinger_bands(prices: pd.Series, window: int, num_std: float, use_ema: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for a price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Series of price data
    window : int
        Window size for the moving average
    num_std : float
        Number of standard deviations for the bands
    use_ema : bool
        Whether to use EMA instead of SMA for the middle band
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series, pd.Series]
        (middle_band, upper_band, lower_band)
    """
    if len(prices) < window:
        return None, None, None
        
    # Calculate middle band (using EMA or SMA)
    if use_ema:
        middle_band = prices.ewm(span=window, min_periods=window).mean()
        # Calculate a rolling standard deviation based on the EMA
        rolling_std = prices.rolling(window=window).std()
    else:
        # Calculate middle band (simple moving average)
        middle_band = prices.rolling(window=window).mean()
        # Calculate standard deviation
        rolling_std = prices.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return middle_band, upper_band, lower_band


def generate_bollinger_bands_signal_for_ticker(prices: pd.Series, params: Dict) -> float:
    """
    Generate trading signal based on Bollinger Bands for a single ticker.
    
    Parameters:
    -----------
    prices : pd.Series
        Series of price data for a ticker
    params : Dict
        Dictionary containing parameters for Bollinger Bands:
        - window: Window size for the moving average (default: 20)
        - num_std: Number of standard deviations for the bands (default: 2.0)
        - use_ema: Whether to use EMA instead of SMA (default: False)
        
    Returns:
    --------
    float
        Signal value: -1 (sell), 0 (neutral), or 1 (buy)
    """
    window = params.get("window", 20)
    num_std = params.get("num_std", 2.0)
    use_ema = params.get("use_ema", False)
    
    if len(prices) < window:
        return 0.0  # Not enough data
        
    # Calculate Bollinger Bands
    middle_band, upper_band, lower_band = calculate_bollinger_bands(prices, window, num_std, use_ema)
    
    if middle_band is None:
        return 0.0
    
    # Get the most recent values
    current_price = prices.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_middle = middle_band.iloc[-1]
    
    # Calculate percent B which measures where price is relative to the bands (0-100%)
    # This can be used for graduated signals rather than discrete buy/sell
    band_width = current_upper - current_lower
    if band_width == 0:
        percent_b = 0.5  # Avoid division by zero
    else:
        percent_b = (current_price - current_lower) / band_width
    
    # Get previous values to detect crossings
    if len(prices) > 1:
        prev_price = prices.iloc[-2]
        prev_lower = lower_band.iloc[-2] if len(lower_band) > 1 else None
        prev_upper = upper_band.iloc[-2] if len(upper_band) > 1 else None
        
        # Check for band crossings (stronger signals)
        crossing_lower = prev_price >= prev_lower and current_price < current_lower if prev_lower is not None else False
        crossing_upper = prev_price <= prev_upper and current_price > current_upper if prev_upper is not None else False
        
        if crossing_lower:
            return 1.0  # Strong buy on crossing lower band
        if crossing_upper:
            return -1.0  # Strong sell on crossing upper band
    
    # Generate signal based on band position
    if current_price <= current_lower:
        # Oversold condition (below lower band)
        return 1.0  # Strong buy signal
    elif current_price >= current_upper:
        # Overbought condition (above upper band)
        return -1.0  # Strong sell signal
    
    # Distance from middle band (percent of band width)
    distance_from_middle = abs(current_price - current_middle) / band_width
    
    # Position relative to bands for graduated signals
    if percent_b < 0.3:
        # Near lower band but not below it - strength based on distance from middle
        return 0.5 + 0.5 * distance_from_middle  # Moderate to strong buy signal (0.5-1.0)
    elif percent_b > 0.7:
        # Near upper band but not above it - strength based on distance from middle
        return -0.5 - 0.5 * distance_from_middle  # Moderate to strong sell signal (-0.5 to -1.0)
    else:
        # Strong mean reversion zone - return to middle expected
        if current_price < current_middle:
            # Price below middle, expect rise towards middle
            return 0.2  # Weak buy signal
        elif current_price > current_middle:
            # Price above middle, expect fall towards middle
            return -0.2  # Weak sell signal
        else:
            # At the middle
            return 0.0  # Neutral 