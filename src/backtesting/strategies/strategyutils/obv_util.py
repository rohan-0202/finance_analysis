"""
On-Balance Volume (OBV) Utility Functions

This module provides utility functions for calculating On-Balance Volume (OBV)
and generating trading signals based on it. OBV is a momentum indicator that uses
volume flow to predict changes in stock price.
"""

from typing import Dict, Optional

import pandas as pd
import ta


def calculate_obv(close_prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    Parameters:
    -----------
    close_prices : pd.Series
        Series of closing prices
    volume : pd.Series
        Series of volume data corresponding to the prices

    Returns:
    --------
    pd.Series
        The calculated OBV series, or an empty Series if input is invalid.
    """
    if close_prices.empty or volume.empty or len(close_prices) != len(volume):
        print("Warning: Close prices or volume data is empty or lengths mismatch.")
        return pd.Series(dtype=float)

    try:
        obv_indicator = ta.volume.OnBalanceVolumeIndicator(
            close=close_prices, volume=volume, fillna=False
        )
        obv_series = obv_indicator.on_balance_volume()
        return obv_series
    except Exception as e:
        print(f"Error calculating OBV: {e}")
        return pd.Series(dtype=float)


def generate_obv_signal_for_ticker(
    close_prices: pd.Series, volume: pd.Series, params: Optional[Dict] = None
) -> float:
    """
    Generate a trading signal based on OBV crossing its moving average.

    Signal Logic:
    - Buy Signal (1.0): OBV crosses above its Simple Moving Average (SMA).
    - Sell Signal (-1.0): OBV crosses below its Simple Moving Average (SMA).
    - Neutral (0.0): No crossover or insufficient data.

    Parameters:
    -----------
    close_prices : pd.Series
        Series of closing prices for the ticker
    volume : pd.Series
        Series of volume data for the ticker
    params : Dict, optional
        Dictionary containing parameters for signal generation:
        - sma_window: Window size for the OBV's moving average (default: 20)

    Returns:
    --------
    float
        Signal value: -1.0 (sell), 0.0 (neutral), or 1.0 (buy)
    """
    # Set default parameters if none provided
    if params is None:
        params = {}
    sma_window = params.get("sma_window", 20)

    # Ensure we have enough data for OBV calculation and SMA
    if close_prices.empty or volume.empty or len(close_prices) < sma_window + 1:
        return 0.0

    # Calculate OBV
    obv_series = calculate_obv(close_prices, volume)
    if obv_series.empty or len(obv_series) < sma_window + 1:
        return 0.0 # OBV calculation failed or not enough data

    # Calculate SMA of OBV
    try:
        obv_sma = obv_series.rolling(window=sma_window).mean()
    except Exception as e:
        print(f"Error calculating OBV SMA: {e}")
        return 0.0

    # Check for sufficient data after SMA calculation
    if len(obv_sma) < 2 or obv_sma.isna().all():
         return 0.0

    # Get the latest and previous values
    latest_obv = obv_series.iloc[-1]
    prev_obv = obv_series.iloc[-2]
    latest_sma = obv_sma.iloc[-1]
    prev_sma = obv_sma.iloc[-2]

    # Check for NaN values in the relevant period
    if pd.isna(latest_obv) or pd.isna(prev_obv) or pd.isna(latest_sma) or pd.isna(prev_sma):
        return 0.0

    # Check for crossovers
    if prev_obv <= prev_sma and latest_obv > latest_sma:
        return 1.0  # Bullish crossover: OBV crossed above its SMA
    elif prev_obv >= prev_sma and latest_obv < latest_sma:
        return -1.0  # Bearish crossover: OBV crossed below its SMA
    else:
        return 0.0  # No crossover
