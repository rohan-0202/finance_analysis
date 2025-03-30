from backtesting.strategies.strategyutils.volatililty_utils import VolatilityRegime


def get_rsi_parameters(regime: VolatilityRegime) -> dict:
    """
    Get RSI parameters appropriate for the current volatility regime.

    Parameters:
    -----------
    regime : VolatilityRegime
        Current volatility regime

    Returns:
    --------
    dict
        Dictionary of RSI parameters
    """
    params = {
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
