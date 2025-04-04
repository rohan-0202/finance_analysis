"""
Position Sizing Strategies
=========================

This module provides various position sizing strategies for trading:

1. Fixed Percentage: Allocates a fixed percentage of portfolio to each position
2. Equal Risk: Allocates capital based on equal risk per position
3. Kelly Criterion: Uses the Kelly formula to optimize position size
4. Volatility Based: Adjusts position size based on asset volatility
5. Signal Proportional: Allocates more capital to stronger signals
"""

from abc import ABC, abstractmethod
from typing import Dict, TypedDict, Optional, Type


from backtesting.portfolio import Portfolio


class PositionSizingParameters(TypedDict, total=False):
    """
    Parameters for configuring position sizing behavior.

    Attributes:
    -----------
    max_position_size_pct : float
        Maximum position size as a percentage of portfolio (default: 0.1 or 10%)
    min_position_size_pct : float
        Minimum position size as a percentage of portfolio (default: 0.01 or 1%)
    volatility_lookback : int
        Number of periods to use for volatility calculation (default: 20)
    kelly_fraction : float
        Fraction of Kelly to use (default: 0.5 or half-Kelly)
    winrate_lookback : int
        Number of trades to use for win rate calculation (default: 20)
    """

    max_position_size_pct: float
    min_position_size_pct: float
    volatility_lookback: int
    kelly_fraction: float
    winrate_lookback: int

    @classmethod
    def get_defaults(cls) -> "PositionSizingParameters":
        """Returns default parameters for position sizing."""
        return {
            "max_position_size_pct": 0.1,
            "min_position_size_pct": 0.01,
            "volatility_lookback": 20,
            "kelly_fraction": 0.5,
            "winrate_lookback": 20,
        }


class PositionSizingStrategy(ABC):
    """Base class for all position sizing strategies."""

    def __init__(
        self,
        portfolio: Portfolio,
        parameters: Optional[PositionSizingParameters] = None,
    ):
        """
        Initialize the position sizing strategy.

        Parameters:
        -----------
        portfolio : Portfolio
            The portfolio object
        parameters : PositionSizingParameters, optional
            Parameters for the position sizing strategy
        """
        self.portfolio = portfolio
        self.parameters = parameters or PositionSizingParameters.get_defaults()

    @abstractmethod
    def calculate_position_size(
        self, ticker: str, signal: float, signals: Dict[str, float]
    ) -> int:
        """
        Calculate the position size for a trade.

        Parameters:
        -----------
        ticker : str
            The ticker symbol
        signal : float
            Signal strength, typically between -1 and 1
        signals : Dict[str, float]
            All current signals for context

        Returns:
        --------
        int
            Number of shares to hold (positive for long, negative for short, 0 for no position)
        """
        pass

    def _handle_position_reversal(
        self, ticker: str, signal: float, current_shares: int
    ) -> Optional[int]:
        """
        Handle cases where the signal is in the opposite direction of current position.

        Parameters:
        -----------
        ticker : str
            The ticker symbol
        signal : float
            Signal strength
        current_shares : int
            Current position size

        Returns:
        --------
        Optional[int]
            Number of shares to trade if position should be reversed or closed,
            None if normal calculation should proceed
        """
        # No opposing signal case
        if not (
            (current_shares > 0 and signal < 0) or (current_shares < 0 and signal > 0)
        ):
            return None

        # For opposing signals, determine how much to close based on signal strength
        if (
            abs(signal) > 0.5
        ):  # If signal is strong, close position and potentially reverse
            # Get parameters for potential reversal
            max_position_pct = self.parameters.get("max_position_size_pct", 0.1)
            portfolio_value = self.portfolio.get_value()
            current_price = self.portfolio.current_prices.get(ticker, 0)

            if current_price <= 0:
                return -current_shares  # Just close the position if price is invalid

            # Calculate new position in the opposite direction
            position_value = portfolio_value * max_position_pct * abs(signal)
            new_shares = int(position_value / current_price)
            new_position = -new_shares if signal < 0 else new_shares

            # This will close the current position and open a new one in the opposite direction
            return new_position - current_shares
        else:
            # For weaker opposing signals, just close the position
            return -current_shares


class FixedPercentageStrategy(PositionSizingStrategy):
    """Allocates a fixed percentage of portfolio to each position."""

    def calculate_position_size(
        self, ticker: str, signal: float, signals: Dict[str, float]
    ) -> int:
        """Calculate position size based on a fixed percentage of portfolio."""
        if signal == 0:
            return 0

        # Get current position and check for reversal
        current_shares = self.portfolio.get_position(ticker)
        reversal_shares = self._handle_position_reversal(ticker, signal, current_shares)
        if reversal_shares is not None:
            return reversal_shares

        max_position_pct = self.parameters.get("max_position_size_pct", 0.1)
        min_position_pct = self.parameters.get("min_position_size_pct", 0.01)

        # Scale position size between min and max based on signal strength
        position_pct = min_position_pct + (max_position_pct - min_position_pct) * abs(
            signal
        )

        # Calculate monetary value of position
        portfolio_value = self.portfolio.get_value()
        position_value = portfolio_value * position_pct

        # Calculate number of shares
        current_price = self.portfolio.current_prices.get(ticker, 0)
        if current_price <= 0:
            return 0

        # Calculate shares and apply signal direction
        shares = int(position_value / current_price)
        return shares if signal > 0 else -shares


class EqualRiskStrategy(PositionSizingStrategy):
    """Allocates capital based on equal risk per position."""

    def calculate_position_size(
        self, ticker: str, signal: float, signals: Dict[str, float]
    ) -> int:
        """Calculate position size based on equal risk allocation."""
        if signal == 0 or abs(signal) < 0.1:  # Ignore very weak signals
            return 0

        # Get current position and check for reversal
        current_shares = self.portfolio.get_position(ticker)
        reversal_shares = self._handle_position_reversal(ticker, signal, current_shares)
        if reversal_shares is not None:
            return reversal_shares

        # Get portfolio parameters
        portfolio_value = self.portfolio.get_value()
        max_risk_per_trade = (
            self.parameters.get("max_position_size_pct", 0.1) * portfolio_value
        )

        # Scale risk based on signal strength
        risk_amount = max_risk_per_trade * abs(signal)

        # Get current price
        current_price = self.portfolio.current_prices.get(ticker, 0)
        if current_price <= 0:
            return 0

        # Estimate a stop loss level (simplified)
        # In practice, this would use actual stop loss levels from a stop loss manager
        stop_loss_pct = 0.05  # Default 5% stop loss
        price_distance = current_price * stop_loss_pct

        # Calculate position size based on risk
        if price_distance > 0:
            shares = int(risk_amount / price_distance)
            return shares if signal > 0 else -shares

        return 0


class VolatilityBasedStrategy(PositionSizingStrategy):
    """Adjusts position size based on asset volatility."""

    def calculate_position_size(
        self, ticker: str, signal: float, signals: Dict[str, float]
    ) -> int:
        """Calculate position size inversely proportional to volatility."""
        if signal == 0:
            return 0

        # Get current position and check for reversal
        current_shares = self.portfolio.get_position(ticker)
        reversal_shares = self._handle_position_reversal(ticker, signal, current_shares)
        if reversal_shares is not None:
            return reversal_shares

        # Get maximum position size percentage
        max_position_pct = self.parameters.get("max_position_size_pct", 0.1)
        portfolio_value = self.portfolio.get_value()

        # This is a simplified implementation
        # In practice, you would calculate actual historical volatility
        # based on price data over a lookback period
        # Here we assume volatility data is available elsewhere

        # For now, use a simple proxy for volatility
        # (in a real implementation, this would use actual volatility metrics)
        volatility_factor = 1.0  # Default factor, 1.0 means no adjustment

        # Adjust position size inversely to volatility
        # Higher volatility = smaller position
        adjusted_position_pct = max_position_pct * abs(signal) / volatility_factor

        # Calculate position value and shares
        position_value = portfolio_value * adjusted_position_pct
        current_price = self.portfolio.current_prices.get(ticker, 0)

        if current_price <= 0:
            return 0

        shares = int(position_value / current_price)
        return shares if signal > 0 else -shares


class SignalProportionalStrategy(PositionSizingStrategy):
    """Allocates more capital to stronger signals, considering all current signals."""

    def calculate_position_size(
        self, ticker: str, signal: float, signals: Dict[str, float]
    ) -> int:
        """Calculate position size proportional to signal strength relative to all signals."""
        if signal == 0:
            return 0

        # Get current position and check for reversal
        current_shares = self.portfolio.get_position(ticker)
        reversal_shares = self._handle_position_reversal(ticker, signal, current_shares)
        if reversal_shares is not None:
            return reversal_shares

        # For non-opposing signals or no current position, continue with standard calculation
        # Get parameters
        max_position_pct = self.parameters.get("max_position_size_pct", 0.1)
        portfolio_value = self.portfolio.get_value()

        # Calculate total absolute signal strength across all tickers
        total_signal_strength = sum(abs(s) for s in signals.values() if abs(s) > 0.1)

        if total_signal_strength == 0:
            return 0

        # Allocate portfolio based on this signal's proportion of total signal strength
        signal_proportion = abs(signal) / total_signal_strength
        position_pct = max_position_pct * signal_proportion

        # Calculate target position value
        position_value = portfolio_value * position_pct
        current_price = self.portfolio.current_prices.get(ticker, 0)

        if current_price <= 0:
            return 0

        # Calculate target number of shares based on signal direction
        target_shares = int(position_value / current_price)
        return target_shares if signal > 0 else -target_shares


class PositionSizingFactory:
    """Factory class for creating position sizing strategies."""

    _strategies: Dict[str, Type[PositionSizingStrategy]] = {
        "fixed_percentage": FixedPercentageStrategy,
        "equal_risk": EqualRiskStrategy,
        "volatility_based": VolatilityBasedStrategy,
        "signal_proportional": SignalProportionalStrategy,
    }

    @classmethod
    def get_strategy(
        cls,
        strategy_type: str,
        portfolio: Portfolio,
        parameters: Optional[PositionSizingParameters] = None,
    ) -> PositionSizingStrategy:
        """
        Create a position sizing strategy instance.

        Parameters:
        -----------
        strategy_type : str
            Type of strategy to create
        portfolio : Portfolio
            Portfolio instance
        parameters : PositionSizingParameters, optional
            Strategy parameters

        Returns:
        --------
        PositionSizingStrategy
            Instance of the requested strategy

        Raises:
        -------
        ValueError
            If the strategy type is not supported
        """
        if strategy_type not in cls._strategies:
            valid_strategies = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown position sizing strategy '{strategy_type}'. "
                f"Valid strategies are: {valid_strategies}"
            )

        strategy_class = cls._strategies[strategy_type]
        return strategy_class(portfolio, parameters)

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[PositionSizingStrategy]):
        """
        Register a new strategy type.

        Parameters:
        -----------
        name : str
            Name to register the strategy under
        strategy_class : Type[PositionSizingStrategy]
            Strategy class to register
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def adjust_parameters_for_universe(
        cls, parameters: Optional[PositionSizingParameters], ticker_count: int
    ) -> PositionSizingParameters:
        """
        Adjust position sizing parameters based on the number of tickers in the universe.

        This helps ensure efficient capital utilization during backtesting by scaling
        the position size parameters based on the number of securities.

        Parameters:
        -----------
        parameters : Optional[PositionSizingParameters]
            Original position sizing parameters or None
        ticker_count : int
            Number of tickers in the trading universe

        Returns:
        --------
        PositionSizingParameters
            Adjusted parameters for the given universe size
        """
        # Start with default parameters if none provided
        params = (
            parameters.copy() if parameters else PositionSizingParameters.get_defaults()
        )

        # Set minimum and maximum allowed position size percentages
        min_allowed_pct = 0.1  # Minimum 10% of portfolio
        max_allowed_pct = 0.95  # Maximum 95% of portfolio

        # Calculate target position size based on universe size (1/n allocation)
        target_position_pct = (
            1.0 / ticker_count if ticker_count > 0 else max_allowed_pct
        )

        # Constrain between the minimum and maximum allowed values
        target_position_pct = max(
            min(target_position_pct, max_allowed_pct), min_allowed_pct
        )

        # Update the parameters
        params["max_position_size_pct"] = target_position_pct

        # Adjust min position size to be proportional but smaller
        params["min_position_size_pct"] = max(target_position_pct * 0.5, 0.01)

        return params
