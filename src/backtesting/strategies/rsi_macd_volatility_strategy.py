import pandas as pd
from loguru import logger

from backtesting.portfolio import Portfolio
from backtesting.strategies.rsi_macd_strategy import RSIMACDStrategy
from backtesting.strategies.strategyutils.rsi_util import (
    get_rsi_parameters_by_volatility,
)
from backtesting.strategies.strategyutils.volatililty_util import (
    get_volatility_regime,
)


class RSIMACDVolatilityStrategy(RSIMACDStrategy):
    """
    Combines RSI and MACD signals for trading decisions but also adjusts
    position size (TODO) and rsi parameters based on volatility regime.

    Trades when either indicator provides a signal, unless they conflict.
    - Buy if RSI signals buy OR MACD signals buy (and the other isn't sell).
    - Sell if RSI signals sell OR MACD signals sell (and the other isn't buy).
    - Neutral if signals conflict (one buy, one sell) or both are neutral.
    - Adjust position size based on volatility regime.
    - Adjust rsi parameters based on volatility regime.


    """

    def __init__(self, portfolio: Portfolio):
        super().__init__(portfolio)
        logger.info("Initializing RSI+MACD+Volatility Strategy")
        self.name = "RSI+MACD+Volatility Strategy"

    def _generate_rsi_signal(self, ticker_data: pd.Series, ticker: str) -> float:
        """Generates the RSI signal for a single ticker."""
        volatility_regime = get_volatility_regime(ticker_data)
        rsi_parameters = get_rsi_parameters_by_volatility(volatility_regime)
        # logger.info(f"RSI parameters: {rsi_parameters}")
        self.set_parameters(**rsi_parameters)

        return super()._generate_rsi_signal(ticker_data, ticker)
