from enum import Enum

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """Enumeration of possible volatility regimes."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


def calculate_volatility(
    price_series: pd.Series, window: int = 20, annualize: bool = True
) -> pd.Series:
    """
    Calculate the rolling volatility of a price series.

    Parameters:
    -----------
    price_series : pd.Series
        Series of price data
    window : int, default=20
        Rolling window for volatility calculation (e.g., 20 days)
    annualize : bool, default=True
        Whether to annualize the volatility

    Returns:
    --------
    pd.Series
        Series of volatility values
    """
    # Calculate log returns
    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    # Calculate rolling standard deviation of returns
    volatility = log_returns.rolling(window=window).std()

    # Annualize if requested (√252 for daily data)
    if annualize:
        volatility = volatility * np.sqrt(252)

    return volatility


def determine_volatility_regime(
    current_volatility: float,
    historical_volatilities: pd.Series,
    low_threshold: float = 0.5,
    high_threshold: float = 1.5,
) -> VolatilityRegime:
    """
    Determine the current volatility regime based on historical data.

    Parameters:
    -----------
    current_volatility : float
        Current volatility value
    historical_volatilities : pd.Series
        Series of historical volatility values
    low_threshold : float, default=0.5
        Threshold for low volatility as fraction of median
    high_threshold : float, default=1.5
        Threshold for high volatility as fraction of median

    Returns:
    --------
    VolatilityRegime
        The current volatility regime
    """
    # Calculate the historical median volatility
    median_volatility = historical_volatilities.median()

    # Determine regime
    if current_volatility < low_threshold * median_volatility:
        return VolatilityRegime.LOW
    elif current_volatility > high_threshold * median_volatility:
        return VolatilityRegime.HIGH
    else:
        return VolatilityRegime.NORMAL


def get_volatility_regime(
    price_series: pd.Series,
    window: int = 20,
    lookback_period: int = 252,
    low_threshold: float = 0.5,
    high_threshold: float = 1.5,
) -> VolatilityRegime:
    """
    Combined function to calculate volatility and determine the current regime.

    Parameters:
    -----------
    price_series : pd.Series
        Series of price data
    window : int, default=20
        Rolling window for volatility calculation
    lookback_period : int, default=252
        Period for historical comparison (e.g., 252 trading days = 1 year)
    low_threshold : float, default=0.5
        Threshold for low volatility as fraction of median
    high_threshold : float, default=1.5
        Threshold for high volatility as fraction of median

    Returns:
    --------
    VolatilityRegime
        The current volatility regime
    """
    # Ensure we have enough data
    if len(price_series) < window + 1:
        return VolatilityRegime.NORMAL  # Default if not enough data

    # Calculate volatility
    volatility_series = calculate_volatility(price_series, window)

    # Get the current volatility (most recent value)
    current_volatility = volatility_series.iloc[-1]

    # Get historical volatilities for comparison
    lookback_length = min(lookback_period, len(volatility_series) - 1)
    historical_volatilities = volatility_series.iloc[-lookback_length - 1 : -1]

    # Determine and return the regime
    return determine_volatility_regime(
        current_volatility, historical_volatilities, low_threshold, high_threshold
    )


if __name__ == "__main__":
    from db_util import get_historical_data

    # Get historical data for a ticker
    ticker_data = get_historical_data("AAPL", days=365)

    # Extract closing prices
    close_prices = ticker_data["close"]

    # Determine the volatility regime
    regime = get_volatility_regime(close_prices)

    print(f"Current volatility regime for AAPL: {regime}")

    # Get the volatility series for plotting or further analysis
    volatility_series = calculate_volatility(close_prices)
