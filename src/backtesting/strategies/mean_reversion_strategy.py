"""
Mean Reversion Strategy

This strategy uses Bollinger Bands to identify mean reversion opportunities.
The strategy is based on the following principles:

- Buy when price is below the lower Bollinger Band (oversold condition)
- Sell when price is above the upper Bollinger Band (overbought condition)
- Hold when price is between the bands

Bollinger Bands consist of:
- A middle band (simple moving average)
- An upper band (middle band + standard deviations)
- A lower band (middle band - standard deviations)
"""

from typing import Dict, List, Optional

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategies.strategyutils.bollinger_bands_util import generate_bollinger_bands_signal_for_ticker
from backtesting.strategy import DataDict, Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER


class MeanReversionStrategy(Strategy):
    """Mean Reversion Strategy implementation using Bollinger Bands."""

    def __init__(self, portfolio: Portfolio):
        super().__init__("Mean Reversion Strategy", portfolio)
        # Initialize parameters with defaults
        self.parameters = self.get_default_parameters()
        # Dictionary to track stop losses for each position
        self.stop_losses = {}
        # Track previous close prices for trailing stops
        self.prev_prices = {}
        # Track where we entered positions
        self.entry_prices = {}

    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        super().set_parameters(**kwargs)
        # Update bollinger_bands_parameters if specific parameters are changed
        if "window" in kwargs:
            self.parameters["bollinger_bands_parameters"]["window"] = kwargs["window"]
        if "num_std" in kwargs:
            self.parameters["bollinger_bands_parameters"]["num_std"] = kwargs["num_std"]
        if "max_capital_per_position" in kwargs:
            self.parameters["max_capital_per_position"] = kwargs["max_capital_per_position"]
        if "trailing_stop_pct" in kwargs:
            self.parameters["trailing_stop_pct"] = kwargs["trailing_stop_pct"]
        if "use_ema" in kwargs:
            self.parameters["bollinger_bands_parameters"]["use_ema"] = kwargs["use_ema"]
        if "profit_target_pct" in kwargs:
            self.parameters["profit_target_pct"] = kwargs["profit_target_pct"]
        return self

    def get_data_requirements(self) -> List[DataRequirement]:
        """Mean Reversion strategy only requires ticker data."""
        return [DataRequirement.TICKER]

    def generate_signals(self, data: DataDict) -> Dict[str, float]:
        """
        Generate trading signals based on Bollinger Bands using the ticker data.

        Parameters:
        -----------
        data : DataDict
            Dictionary containing the required data. Must include
            DataRequirement.TICKER.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        signals = {}
        
        # Check if Ticker data is present
        if DataRequirement.TICKER not in data:
            print("Error: Ticker data not found in data dictionary for MeanReversionStrategy.")
            return signals

        ticker_data_df = data[DataRequirement.TICKER]

        if ticker_data_df.empty or CLOSE not in ticker_data_df.columns:
            return signals

        # Ensure index is sorted for rolling calculations
        if not ticker_data_df.index.is_monotonic_increasing:
            ticker_data_df = ticker_data_df.sort_index()

        tickers = ticker_data_df.index.get_level_values(TICKER).unique()
        window = self.parameters["bollinger_bands_parameters"]["window"]
        
        for ticker in tickers:
            try:
                close_prices = ticker_data_df.xs(ticker, level=TICKER)[CLOSE]
                current_price = close_prices.iloc[-1]
                
                # Update previous prices
                if ticker not in self.prev_prices:
                    self.prev_prices[ticker] = current_price
                
                # Check if we need to exit based on trailing stop
                signal = self._check_stop_loss(ticker, current_price)
                if signal is not None:
                    signals[ticker] = signal
                    continue
                
                # Check profit target if we have a position
                signal = self._check_profit_target(ticker, current_price)
                if signal is not None:
                    signals[ticker] = signal
                    continue
                
                # If no stop or profit target triggered, calculate regular signal
                if len(close_prices) < window:
                    signals[ticker] = 0.0  # Not enough data
                    continue

                # Get the regular Bollinger Band signal
                signal = generate_bollinger_bands_signal_for_ticker(
                    close_prices,
                    self.parameters["bollinger_bands_parameters"],
                )
                
                # If we get a signal at all, make it stronger (more aggressive)
                if signal > 0:
                    signal = min(signal * 1.5, 1.0)  # Scale up buy signals
                elif signal < 0:
                    signal = max(signal * 1.5, -1.0)  # Scale up sell signals
                
                signals[ticker] = signal
                
                # Store previous price for next iteration
                self.prev_prices[ticker] = current_price
                
            except KeyError:
                continue  # Ticker not present in this data slice?

        return signals
        
    def _check_stop_loss(self, ticker: str, current_price: float) -> Optional[float]:
        """
        Check if a trailing stop loss should be triggered.
        
        Parameters:
        -----------
        ticker: str
            Ticker symbol
        current_price: float
            Current price of the ticker
            
        Returns:
        --------
        Optional[float]
            Signal value if stop loss triggered, None otherwise
        """
        # If we don't have a position, no stop loss to check
        if ticker not in self.portfolio.holdings or self.portfolio.holdings[ticker] == 0:
            # Clear any stale stop loss data
            if ticker in self.stop_losses:
                del self.stop_losses[ticker]
            if ticker in self.entry_prices:
                del self.entry_prices[ticker]
            return None
            
        # Get current position direction (long/short)
        position = self.portfolio.holdings[ticker]
        is_long = position > 0
        
        # If this is a new position, set initial stop loss
        if ticker not in self.stop_losses:
            self.entry_prices[ticker] = current_price
            trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.05)
            if is_long:
                self.stop_losses[ticker] = current_price * (1 - trailing_stop_pct)
            else:
                self.stop_losses[ticker] = current_price * (1 + trailing_stop_pct)
            return None
            
        # Update trailing stop if price moved in our favor
        trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.05)
        if is_long:
            # For long positions, move stop up as price rises
            potential_new_stop = current_price * (1 - trailing_stop_pct)
            if potential_new_stop > self.stop_losses[ticker]:
                self.stop_losses[ticker] = potential_new_stop
                
            # Check if stop was hit
            if current_price <= self.stop_losses[ticker]:
                # Stop loss hit, exit position
                return -1.0
        else:
            # For short positions, move stop down as price falls
            potential_new_stop = current_price * (1 + trailing_stop_pct)
            if potential_new_stop < self.stop_losses[ticker]:
                self.stop_losses[ticker] = potential_new_stop
                
            # Check if stop was hit
            if current_price >= self.stop_losses[ticker]:
                # Stop loss hit, exit position
                return 1.0
                
        return None
        
    def _check_profit_target(self, ticker: str, current_price: float) -> Optional[float]:
        """
        Check if profit target has been reached.
        
        Parameters:
        -----------
        ticker: str
            Ticker symbol
        current_price: float
            Current price of the ticker
            
        Returns:
        --------
        Optional[float]
            Signal value if profit target reached, None otherwise
        """
        # If we don't have a position or entry price, nothing to check
        if (ticker not in self.portfolio.holdings or 
            self.portfolio.holdings[ticker] == 0 or
            ticker not in self.entry_prices):
            return None
            
        # Get current position direction and profit target
        position = self.portfolio.holdings[ticker]
        is_long = position > 0
        profit_target_pct = self.parameters.get("profit_target_pct", 0.15)
        
        # Check if profit target reached
        entry_price = self.entry_prices[ticker]
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= profit_target_pct:
                return -1.0  # Sell signal to take profit
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= profit_target_pct:
                return 1.0  # Buy signal to cover short
                
        return None

    def apply_risk_management(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk management rules to adjust trading signals.
        
        For mean reversion, we might want to adjust signals based on:
        - Overall market conditions
        - Position concentration
        - Volatility
        
        Parameters:
        -----------
        signals : Dict[str, float]
            Original trading signals
            
        Returns:
        --------
        Dict[str, float]
            Adjusted trading signals after risk management
        """
        # Simple implementation - could be enhanced with more sophisticated rules
        adjusted_signals = {}
        
        # Get current positions
        current_positions = {ticker: shares for ticker, shares in self.portfolio.holdings.items() if shares != 0}
        
        # Calculate maximum number of open positions
        max_positions = self.parameters.get("max_open_positions", 4)
        current_position_count = len(current_positions)
        
        for ticker, signal in signals.items():
            # If signal is to exit, always allow it
            if (ticker in current_positions and 
                ((signal < 0 and current_positions[ticker] > 0) or
                 (signal > 0 and current_positions[ticker] < 0))):
                adjusted_signals[ticker] = signal
                continue
                
            # If we already have too many positions, only allow exits
            if current_position_count >= max_positions and ticker not in current_positions:
                adjusted_signals[ticker] = 0.0
                continue
                
            # For existing positions, adjust signal strength based on current position
            if ticker in current_positions and abs(current_positions[ticker]) > 0:
                # If we already have a position and signal is in same direction, reduce it
                current_position_sign = 1 if current_positions[ticker] > 0 else -1
                if (signal > 0 and current_position_sign > 0) or (signal < 0 and current_position_sign < 0):
                    # Don't add to existing positions for mean reversion
                    adjusted_signals[ticker] = 0.0  
                else:
                    adjusted_signals[ticker] = signal  # Keep as is for reversing positions
            else:
                # For new positions, keep original signal strength
                adjusted_signals[ticker] = signal
                
        return adjusted_signals

    def calculate_position_size(self, ticker: str, signal: float) -> int:
        """
        Calculate position size for a trade based on the signal strength.

        Override the default implementation with more aggressive sizing.

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
        # More aggressive position sizing
        max_capital_per_position = self.parameters.get("max_capital_per_position", 0.25)
        portfolio_value = self.portfolio.get_value()
        max_position = max_capital_per_position * portfolio_value

        current_price = self.portfolio.current_prices.get(ticker, 0)
        if current_price <= 0:
            return 0

        # Base position size on signal strength with a more aggressive scaling
        position_value = max_position * min(abs(signal) * 1.5, 1.0)
        shares = int(position_value / current_price)

        # Negative for sell, positive for buy
        return shares if signal > 0 else -shares

    @classmethod
    def get_default_parameters(cls) -> Dict:
        """
        Get default parameters for the Mean Reversion strategy.

        Returns:
        --------
        Dict
            Dictionary of default parameters
        """
        return {
            "bollinger_bands_parameters": {
                "window": 15,  # Shorter window (was 20)
                "num_std": 1.8,  # Tighter bands (was 2.0)
                "use_ema": True,  # Use EMA instead of SMA for more responsiveness
            },
            "max_capital_per_position": 0.25,  # Risk 25% of portfolio per position (was 10%)
            "commission": 0.0,
            "trailing_stop_pct": 0.05,  # 5% trailing stop
            "profit_target_pct": 0.15,  # 15% profit target
            "max_open_positions": 4,  # Maximum number of open positions
        } 