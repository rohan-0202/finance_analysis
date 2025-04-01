from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from common.df_columns import CLOSE, TICKER, TIMESTAMP
from common.ohlc import OHLCData


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

    def execute(self, data: pd.DataFrame, next_day_data: dict[str, OHLCData]) -> None:
        """
        Execute the strategy on the given data.

        This is the main method that processes data, generates signals,
        and executes trades based on those signals.

        Parameters:
        -----------
        data : pd.DataFrame
            Market data for the current time period
        next_day_data : dict[str, OHLCData]
            Market data for the next day. This is required because we cannot place any order
            derived from the signal at the current day's close price. So a realistic simulation
            should use the next day's prices for the order. This data is not used in the strategy calculation.
        """
        if data.empty:
            return

        # Ensure data has the expected MultiIndex
        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != [
            TIMESTAMP,
            TICKER,
        ]:
            print(
                f"Error: Dataframe passed to {self.name}.execute does not have ('timestamp', 'ticker') MultiIndex."
            )
            return

        # Get the latest timestamp from the data provided
        latest_timestamp = data.index.get_level_values(TIMESTAMP).max()
        self.last_update_time = latest_timestamp  # Update strategy timestamp

        # Update portfolio's current prices
        self._update_current_prices(data, latest_timestamp)

        # Generate signals based on the historical data provided
        self.current_signals = self.generate_signals(data)

        # Apply risk management
        adjusted_signals = self.apply_risk_management(self.current_signals)

        # Execute trades based on adjusted signals
        for ticker, signal in adjusted_signals.items():
            if (
                ticker not in self.portfolio.current_prices
                or self.portfolio.current_prices[ticker] <= 0
            ):
                continue  # Skip if price is missing or invalid

            # Calculate desired position size based on signal
            target_shares = self.calculate_position_size(ticker, signal)

            # Get current position size
            current_shares = self.portfolio.holdings.get(ticker, 0)

            # Calculate shares to trade to reach the target position
            trade_shares = target_shares - current_shares

            if trade_shares != 0:
                # Place the order using the base class method
                self.place_order(
                    ticker, trade_shares, latest_timestamp, next_day_data.get(ticker)
                )

    def _update_current_prices(self, data: pd.DataFrame, timestamp) -> None:
        """
        Update the portfolio's current prices from the data at the given timestamp.

        Parameters:
        -----------
        data : pd.DataFrame
            Market data containing price information
        timestamp : datetime
            The timestamp to use for extracting latest prices
        """
        try:
            latest_data_slice = data.xs(timestamp, level=TIMESTAMP)
            for ticker, row in latest_data_slice.iterrows():
                if isinstance(ticker, str) and CLOSE in row and pd.notna(row[CLOSE]):
                    self.portfolio.current_prices[ticker] = row[CLOSE]
        except KeyError:
            print(
                f"Warning: Could not extract data slice for timestamp {timestamp} in {self.name}"
            )

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

    def place_order(
        self, ticker: str, quantity: int, timestamp: datetime, next_day_data: OHLCData
    ) -> bool:
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

        if next_day_data is None:
            return False

        price = next_day_data["open"]
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
