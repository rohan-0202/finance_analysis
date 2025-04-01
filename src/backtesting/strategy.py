from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, TypeAlias

import pandas as pd

from backtesting.portfolio import Portfolio
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER, TIMESTAMP
from common.ohlc import OHLCData

# Type alias for the data dictionary passed to strategies
DataDict: TypeAlias = Dict[DataRequirement, pd.DataFrame]


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
    def get_data_requirements(self) -> List[DataRequirement]:
        """
        Declare the types of data required by the strategy.

        Returns:
        --------
        List[DataRequirement]
            A list of DataRequirement enums indicating the needed data.
        """
        pass

    @abstractmethod
    def generate_signals(self, data: DataDict) -> Dict[str, float]:
        """
        Generate trading signals for each ticker based on the provided data dictionary.

        Parameters:
        -----------
        data : DataDict
            A dictionary where keys are DataRequirement enums and values are
            DataFrames containing the corresponding data.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (typically -1 to 1)
            where -1 is strong sell, 0 is neutral, and 1 is strong buy
        """
        pass

    def execute(self, data: DataDict, next_day_data: dict[str, OHLCData]) -> None:
        """
        Execute the strategy using the provided data dictionary.

        Parameters:
        -----------
        data : DataDict
            Dictionary containing all the required data for the strategy.
        next_day_data : dict[str, OHLCData]
            Market data for the next day, used for order execution price.
        """
        # Check if the primary ticker data is available
        if DataRequirement.TICKER not in data or data[DataRequirement.TICKER].empty:
            print(f"Warning: Ticker data missing or empty for {self.name}.execute.")
            return

        ticker_data = data[DataRequirement.TICKER]

        # Ensure ticker data has the expected MultiIndex
        if not isinstance(ticker_data.index, pd.MultiIndex) or list(
            ticker_data.index.names
        ) != [
            TIMESTAMP,
            TICKER,
        ]:
            print(
                f"Error: Ticker data passed to {self.name}.execute does not have ('timestamp', 'ticker') MultiIndex."
            )
            return

        # Get the latest timestamp from the ticker data
        latest_timestamp = ticker_data.index.get_level_values(TIMESTAMP).max()
        self.last_update_time = latest_timestamp  # Update strategy timestamp

        # Update portfolio's current prices using Ticker data
        self._update_current_prices(ticker_data, latest_timestamp)

        # Generate signals using the full data dictionary provided
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

    def _update_current_prices(self, ticker_data: pd.DataFrame, timestamp) -> None:
        """
        Update the portfolio's current prices from the ticker data at the given timestamp.

        Parameters:
        -----------
        ticker_data : pd.DataFrame
            Market data containing price information (assumed to be the TICKER data)
        timestamp : datetime
            The timestamp to use for extracting latest prices
        """
        try:
            latest_data_slice = ticker_data.xs(timestamp, level=TIMESTAMP)
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

    def update(self, data: DataDict) -> None:
        """
        Update strategy state with new data without executing trades.

        Parameters:
        -----------
        data : DataDict
            New market data dictionary.
        """
        if DataRequirement.TICKER not in data or data[DataRequirement.TICKER].empty:
            return

        ticker_data = data[DataRequirement.TICKER]
        self.last_update_time = ticker_data.index.get_level_values(TIMESTAMP).max()
        # We still generate signals here based on potentially multiple data types
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
