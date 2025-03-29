from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from backtesting.portfolio import Portfolio


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Concrete strategy implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, name: str, portfolio: Portfolio):
        """
        Initialize the strategy.

        Parameters:
        -----------
        name : str
            Name of the strategy
        portfolio : Portfolio
            Portfolio object
        """
        self.name = name
        self.portfolio = portfolio
        self.parameters = {}  # Strategy parameters
        self.current_signals = {}  # Current trading signals for each ticker
        self.last_update_time = None  # Timestamp of last strategy update

    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)
        return self

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals for each ticker based on the data.

        Parameters:
        -----------
        data : pd.DataFrame
            Market data containing price and other information

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (typically -1 to 1)
            where -1 is strong sell, 0 is neutral, and 1 is strong buy
        """
        pass

    @abstractmethod
    def execute(self, data: pd.DataFrame) -> None:
        """
        Execute the strategy on the given data.

        This is the main method that processes data, generates signals,
        and executes trades based on those signals.

        Parameters:
        -----------
        data : pd.DataFrame
            Market data for the current time period
        """
        pass

    def calculate_position_size(self, ticker: str, signal: float) -> int:
        """
        Calculate position size for a trade based on the signal strength.

        Parameters:
        -----------
        ticker : str
            The ticker symbol
        signal : float
            Signal strength, typically between -1 and 1

        Returns:
        --------
        int
            Number of shares to buy or sell
        """
        # Default implementation based on fixed position sizing
        # Subclasses can override with more sophisticated logic
        max_capital_per_position = self.parameters.get("max_capital_per_position", 0.1)
        portfolio_value = self.portfolio.get_value()
        max_position = max_capital_per_position * portfolio_value

        current_price = self.portfolio.current_prices.get(ticker, 0)
        if current_price <= 0:
            return 0

        # Base position size on signal strength
        position_value = max_position * abs(signal)
        shares = int(position_value / current_price)

        # Negative for sell, positive for buy
        return shares if signal > 0 else -shares

    def place_order(self, ticker: str, quantity: int, timestamp: datetime) -> bool:
        """
        Place an order to buy or sell a security.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Quantity (positive for buy, negative for sell)
        timestamp : datetime
            Current timestamp

        Returns:
        --------
        bool
            True if order was executed successfully, False otherwise
        """
        if quantity == 0:
            return False

        price = self.portfolio.current_prices.get(ticker)
        if price is None:
            return False

        commission = self.parameters.get("commission", 0.0)

        try:
            # Check if trade is valid before executing
            if quantity > 0 and not self.portfolio.canbuy(
                ticker, quantity, price, commission
            ):
                return False
            if quantity < 0 and not self.portfolio.cansell(ticker, abs(quantity)):
                return False

            self.portfolio.execute_trade(ticker, quantity, price, timestamp, commission)
            return True
        except ValueError:
            return False

    def apply_risk_management(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk management rules to adjust trading signals.

        Parameters:
        -----------
        signals : Dict[str, float]
            Original trading signals

        Returns:
        --------
        Dict[str, float]
            Adjusted trading signals after risk management
        """
        # Default implementation - subclasses can implement custom risk rules
        return signals

    def update(self, data: pd.DataFrame) -> None:
        """
        Update strategy state with new data without executing trades.

        Parameters:
        -----------
        data : pd.DataFrame
            New market data
        """
        if data.empty:
            return

        self.last_update_time = data.index[-1]
        self.current_signals = self.generate_signals(data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the strategy's performance.

        Returns:
        --------
        Dict[str, Any]
            Performance metrics and statistics
        """
        metrics = self.portfolio.get_performance_metrics()

        # Add strategy-specific metrics
        metrics.update(
            {
                "strategy_name": self.name,
                "parameters": self.parameters,
                "last_update": self.last_update_time,
            }
        )

        return metrics
