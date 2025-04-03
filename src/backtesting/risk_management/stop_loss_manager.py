"""
Stop Loss Management Algorithm
==============================

This module implements a comprehensive trailing stop loss system with adaptive features:

Key Components:
--------------
1. Basic Trailing Stop Loss:
   - For long positions: Stop level is set below entry price and moves up as price increases
   - For short positions: Stop level is set above entry price and moves down as price decreases
   - Default stop distance is 5% from price (configurable via 'trailing_stop_pct' parameter)

2. Adaptive Stop Loss Widening (Martingale-inspired):
   - After consecutive losses, stop distance can be automatically widened
   - Controlled by parameters:
     * 'use_martingale': Enable/disable feature
     * 'stop_widening_factor': How much to widen stop (default 25% per loss)
     * 'max_martingale_rounds': Maximum number of widening rounds (default 3)
   - Formula: adjusted_stop_pct = base_stop_pct * (1 + (loss_count * widening_factor))

3. Position Management:
   - Tracks entry prices for each position
   - Maintains current stop loss levels for all active positions
   - Returns appropriate exit signals (-1 for long exits, 1 for short exits)
   - Automatically handles position tracking and stop level clearing

Usage Flow:
----------
1. Initialize with portfolio and parameters
2. For new positions: record entry price with update_entry_price()
3. Track consecutive losses with update_consecutive_losses() if using adaptive stops
4. Periodically call check_stop_loss() with current prices to:
   - Initialize stops for new positions
   - Update trailing stops as prices move favorably
   - Get exit signals when stops are triggered
"""

import logging
from typing import Dict, Optional, TypedDict

from backtesting.portfolio import Portfolio

# Configure logging
logger = logging.getLogger(__name__)


class StopLossParameters(TypedDict, total=False):
    """
    Parameters for configuring the StopLossManager behavior.

    Attributes:
    -----------
    trailing_stop_pct : float
        Percentage distance for trailing stops (default: 0.05 or 5%)
    stop_widening_factor : float
        Factor to widen stops after consecutive losses (default: 0.25 or 25%)
    use_martingale : bool
        Enable adaptive stop widening after losses (default: False)
    max_martingale_rounds : int
        Maximum number of widening rounds (default: 3)
    """

    trailing_stop_pct: float
    stop_widening_factor: float
    use_martingale: bool
    max_martingale_rounds: int

    @classmethod
    def get_defaults(cls) -> "StopLossParameters":
        """
        Returns a StopLossParameters instance with default values.

        Returns:
        --------
        StopLossParameters
            Instance with default parameter values
        """
        return {
            "trailing_stop_pct": 0.01,
            "stop_widening_factor": 0.25,
            "use_martingale": False,
            "max_martingale_rounds": 3,
        }


class StopLossManager:
    """Manages stop-loss levels for trading positions."""

    def __init__(self, portfolio: Portfolio, parameters: StopLossParameters):
        """
        Initialize the StopLossManager.

        Parameters:
        -----------
        portfolio : Portfolio
            The portfolio object containing holdings and prices.
        parameters : StopLossParameters
            Strategy parameters for configuring stop-loss behavior.
            Default values will be used for any missing parameters.
        """
        self.portfolio = portfolio
        self.parameters = parameters
        self.stop_losses: Dict[str, float] = {}  # {ticker: stop_price}
        self.entry_prices: Dict[
            str, float
        ] = {}  # {ticker: entry_price} # Needed for initial SL calc
        self.consecutive_losses: Dict[str, int] = {}  # Needed for stop widening

    def update_entry_price(self, ticker: str, entry_price: float):
        """Record the entry price for a new position."""
        self.entry_prices[ticker] = entry_price

    def update_consecutive_losses(self, ticker: str, losses: int):
        """Update the consecutive loss count for a ticker."""
        self.consecutive_losses[ticker] = losses

    def check_stop_loss(self, ticker: str, current_price: float) -> Optional[float]:
        """
        Check if a trailing stop loss should be triggered for a ticker.

        Parameters:
        -----------
        ticker: str
            Ticker symbol
        current_price: float
            Current price of the ticker

        Returns:
        --------
        Optional[float]
            Signal value (-1 for long exit, 1 for short exit) if stop loss triggered,
            None otherwise.
        """
        # If we don't have a position, clear stale data and return
        if (
            ticker not in self.portfolio.holdings
            or self.portfolio.holdings[ticker] == 0
        ):
            if ticker in self.stop_losses:
                del self.stop_losses[ticker]
            if ticker in self.entry_prices:
                # Keep entry price? Maybe needed if position re-entered quickly.
                # For now, let's clear it when position is closed.
                # Let the strategy manage clearing entry prices on exit.
                pass  # Don't clear entry price here
            return None

        position = self.portfolio.holdings[ticker]
        is_long = position > 0

        # Initialize stop loss if it's a new position for the manager
        if ticker not in self.stop_losses and ticker in self.entry_prices:
            self._initialize_stop_loss(ticker, is_long)
            # Don't exit on the same bar we initialize
            return None

        # If stop loss wasn't initialized (e.g., entry price missing), cannot check
        if ticker not in self.stop_losses:
            # print(f"Warning: Stop loss check skipped for {ticker}, missing stop_loss level.")
            return None

        # Update trailing stop and check if triggered
        return self._update_and_check_trailing_stop(ticker, current_price, is_long)

    def _initialize_stop_loss(self, ticker: str, is_long: bool):
        """Set the initial stop loss for a new position."""
        trailing_stop_pct = self.parameters.get(
            "trailing_stop_pct", 0.05
        )  # Default if not set
        stop_widening_factor = self.parameters.get("stop_widening_factor", 0.25)
        use_martingale_widening = self.parameters.get(
            "use_martingale", False
        )  # Check if stop widening is enabled
        max_martingale_rounds = self.parameters.get("max_martingale_rounds", 3)

        # Adjust stop distance based on consecutive losses if enabled
        if use_martingale_widening:
            loss_count = self.consecutive_losses.get(ticker, 0)
            if loss_count > 0 and loss_count <= max_martingale_rounds:
                trailing_stop_pct *= 1 + (loss_count * stop_widening_factor)

        entry_price = self.entry_prices[ticker]
        if is_long:
            self.stop_losses[ticker] = entry_price * (1 - trailing_stop_pct)
        else:
            self.stop_losses[ticker] = entry_price * (1 + trailing_stop_pct)
        # print(f"Initialized Stop Loss for {ticker} at {self.stop_losses[ticker]:.2f} (Entry: {entry_price:.2f}, Long: {is_long})")

    def _update_and_check_trailing_stop(
        self, ticker: str, current_price: float, is_long: bool
    ) -> Optional[float]:
        """Update the trailing stop loss level and check if it's hit."""
        trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.05)
        current_stop = self.stop_losses[ticker]

        if is_long:
            # Move stop up if price rises
            potential_new_stop = current_price * (1 - trailing_stop_pct)
            if potential_new_stop > current_stop:
                self.stop_losses[ticker] = potential_new_stop
                # print(f"Updated Long Stop Loss for {ticker} to {potential_new_stop:.2f}")

            # Check if stop was hit
            if current_price <= self.stop_losses[ticker]:
                # print(f"Stop Loss HIT for LONG {ticker} at {current_price:.2f} (Stop: {self.stop_losses[ticker]:.2f})")
                logger.info(
                    f"STOP LOSS TRIGGERED: LONG position for {ticker} at price {current_price:.2f} (stop level: {self.stop_losses[ticker]:.2f})"
                )
                self.clear_stop_loss_data(ticker)  # Clear state on exit
                return -1.0  # Exit signal for long position
        else:
            # Move stop down if price falls
            potential_new_stop = current_price * (1 + trailing_stop_pct)
            if potential_new_stop < current_stop:
                self.stop_losses[ticker] = potential_new_stop
                # print(f"Updated Short Stop Loss for {ticker} to {potential_new_stop:.2f}")

            # Check if stop was hit
            if current_price >= self.stop_losses[ticker]:
                # print(f"Stop Loss HIT for SHORT {ticker} at {current_price:.2f} (Stop: {self.stop_losses[ticker]:.2f})")
                logger.info(
                    f"STOP LOSS TRIGGERED: SHORT position for {ticker} at price {current_price:.2f} (stop level: {self.stop_losses[ticker]:.2f})"
                )
                self.clear_stop_loss_data(ticker)  # Clear state on exit
                return 1.0  # Exit signal for short position

        return None

    def clear_stop_loss_data(self, ticker: str):
        """Clear stop loss and related data for a specific ticker, typically on position exit."""
        if ticker in self.stop_losses:
            del self.stop_losses[ticker]
        # if ticker in self.entry_prices: # Let strategy manage entry price clearing
        #     del self.entry_prices[ticker]

    def clear_all_stop_losses(self):
        """Clear all stop loss data."""
        self.stop_losses.clear()
        self.entry_prices.clear()  # Clear entry prices as well
        # Don't clear consecutive losses here, managed by strategy logic
