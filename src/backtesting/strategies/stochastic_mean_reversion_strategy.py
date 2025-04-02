"""
Stochastic Mean Reversion Strategy

This strategy uses the Stochastic Oscillator to identify mean reversion opportunities.
The strategy is based on the following principles:

- Buy when the Stochastic Oscillator is in oversold territory and starting to rise
- Sell when the Stochastic Oscillator is in overbought territory and starting to fall
- Use %K and %D crossovers for additional confirmation

The Stochastic Oscillator is particularly well-suited for mean reversion as it:
1. Has defined overbought/oversold boundaries (usually 80/20)
2. Inherently measures deviation from a price range
3. Oscillates between fixed values (0-100), making signals more consistent

This version has been optimized for higher risk and potentially higher returns.
"""

from typing import Dict, List, Optional

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategies.strategyutils.stochastic_util import (
    generate_stochastic_signal_for_ticker,
    get_default_stochastic_parameters,
    StochasticParameters,
)
from backtesting.strategy import DataDict, Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, HIGH, LOW, OPEN, TICKER


class StochasticMeanReversionStrategy(Strategy):
    """Mean Reversion Strategy implementation using the Stochastic Oscillator with higher risk profile."""

    def __init__(self, portfolio: Portfolio):
        super().__init__("Stochastic Mean Reversion Strategy (High Risk)", portfolio)
        # Initialize parameters with defaults
        self.parameters = self.get_default_parameters()
        # Dictionary to track stop losses for each position
        self.stop_losses = {}
        # Track where we entered positions
        self.entry_prices = {}
        # Track previous signals to detect changes
        self.prev_signals = {}
        # Track consecutive losses per ticker
        self.consecutive_losses = {}
        # Track highest profit seen per position
        self.max_profit_seen = {}

    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        super().set_parameters(**kwargs)
        
        # Update stochastic parameters if specific parameters are changed
        stochastic_param_keys = [
            "k_period", "d_period", "smooth_k", 
            "overbought", "oversold", "signal_threshold"
        ]
        
        for key in stochastic_param_keys:
            if key in kwargs:
                self.parameters["stochastic_parameters"][key] = kwargs[key]
                
        # Update other parameters
        if "max_capital_per_position" in kwargs:
            self.parameters["max_capital_per_position"] = kwargs["max_capital_per_position"]
        if "trailing_stop_pct" in kwargs:
            self.parameters["trailing_stop_pct"] = kwargs["trailing_stop_pct"]
        if "profit_target_pct" in kwargs:
            self.parameters["profit_target_pct"] = kwargs["profit_target_pct"]
        if "max_open_positions" in kwargs:
            self.parameters["max_open_positions"] = kwargs["max_open_positions"]
        if "stop_widening_factor" in kwargs:
            self.parameters["stop_widening_factor"] = kwargs["stop_widening_factor"]
        if "use_martingale" in kwargs:
            self.parameters["use_martingale"] = kwargs["use_martingale"]
        if "martingale_factor" in kwargs:
            self.parameters["martingale_factor"] = kwargs["martingale_factor"]
        if "max_martingale_rounds" in kwargs:
            self.parameters["max_martingale_rounds"] = kwargs["max_martingale_rounds"]
        if "use_trailing_profit" in kwargs:
            self.parameters["use_trailing_profit"] = kwargs["use_trailing_profit"]
        if "trailing_profit_trigger" in kwargs:
            self.parameters["trailing_profit_trigger"] = kwargs["trailing_profit_trigger"]
        if "trailing_profit_distance" in kwargs:
            self.parameters["trailing_profit_distance"] = kwargs["trailing_profit_distance"]
        
        return self

    def get_data_requirements(self) -> List[DataRequirement]:
        """Stochastic Mean Reversion strategy requires ticker data."""
        return [DataRequirement.TICKER]

    def generate_signals(self, data: DataDict) -> Dict[str, float]:
        """
        Generate trading signals based on the Stochastic Oscillator using OHLC data.

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
            print("Error: Ticker data not found in data dictionary for StochasticMeanReversionStrategy.")
            return signals

        ticker_data_df = data[DataRequirement.TICKER]

        if ticker_data_df.empty or not all(col in ticker_data_df.columns for col in [OPEN, HIGH, LOW, CLOSE]):
            print("Error: Required OHLC columns missing from ticker data.")
            return signals

        # Ensure index is sorted for calculations
        if not ticker_data_df.index.is_monotonic_increasing:
            ticker_data_df = ticker_data_df.sort_index()

        tickers = ticker_data_df.index.get_level_values(TICKER).unique()
        
        for ticker in tickers:
            try:
                # Extract OHLC data for this ticker
                ticker_ohlc = ticker_data_df.xs(ticker, level=TICKER)
                current_price = ticker_ohlc[CLOSE].iloc[-1]
                
                # Update profit tracking for trailing profit exits
                self._update_profit_tracking(ticker, current_price)
                
                # Check if we need to exit based on trailing stop
                signal = self._check_stop_loss(ticker, current_price)
                if signal is not None:
                    signals[ticker] = signal
                    continue
                
                # Check trailing profit if enabled
                if self.parameters.get("use_trailing_profit", True):
                    signal = self._check_trailing_profit(ticker, current_price)
                    if signal is not None:
                        signals[ticker] = signal
                        continue
                
                # Check profit target if we have a position
                signal = self._check_profit_target(ticker, current_price)
                if signal is not None:
                    signals[ticker] = signal
                    continue
                
                # Use more aggressive stochastic parameter overrides for this ticker
                aggressive_params = self._get_aggressive_params_for_ticker(ticker)
                
                # Calculate Stochastic signal
                stoch_signal = generate_stochastic_signal_for_ticker(
                    ticker_ohlc,
                    aggressive_params
                )
                
                # Apply signal confirmation logic
                confirmed_signal = self._confirm_signal(ticker, stoch_signal)
                signals[ticker] = confirmed_signal
                
                # Store current signal for next time
                self.prev_signals[ticker] = stoch_signal
                
                # If we're entering a new position, record the entry price
                if confirmed_signal != 0 and (
                    ticker not in self.portfolio.holdings or 
                    self.portfolio.holdings[ticker] == 0
                ):
                    self.entry_prices[ticker] = current_price
                    self.max_profit_seen[ticker] = 0.0
                
            except (KeyError, IndexError) as e:
                print(f"Error processing ticker {ticker}: {e}")
                continue

        return signals
    
    def _get_aggressive_params_for_ticker(self, ticker: str) -> Dict:
        """
        Get more aggressive stochastic parameters based on ticker history.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        Dict
            Adjusted parameters for more aggressive trading
        """
        # Start with default parameters
        aggressive_params = self.parameters["stochastic_parameters"].copy()
        
        # Make parameters more aggressive
        aggressive_params["oversold"] = 30.0  # Less extreme oversold threshold (was 20)
        aggressive_params["overbought"] = 70.0  # Less extreme overbought threshold (was 80)
        aggressive_params["signal_threshold"] = 0.2  # Lower threshold for signal generation (was 0.3)
        
        # If we've had consecutive losses for this ticker, make even more aggressive
        loss_count = self.consecutive_losses.get(ticker, 0)
        if loss_count > 0:
            # Adjust parameters based on loss count, becoming more aggressive
            factor = min(loss_count, 3)  # Cap at 3 levels of adjustment
            aggressive_params["oversold"] = min(aggressive_params["oversold"] + (5 * factor), 40)
            aggressive_params["overbought"] = max(aggressive_params["overbought"] - (5 * factor), 60)
            aggressive_params["signal_threshold"] = max(aggressive_params["signal_threshold"] - (0.05 * factor), 0.1)
        
        return aggressive_params
    
    def _update_profit_tracking(self, ticker: str, current_price: float) -> None:
        """Update max profit seen for trailing profit management."""
        if ticker not in self.portfolio.holdings or self.portfolio.holdings[ticker] == 0:
            return
            
        if ticker not in self.entry_prices:
            return
            
        # Calculate current profit percentage
        position = self.portfolio.holdings[ticker]
        is_long = position > 0
        entry_price = self.entry_prices[ticker]
        
        if is_long:
            current_profit_pct = (current_price - entry_price) / entry_price
        else:
            current_profit_pct = (entry_price - current_price) / entry_price
            
        # Update max profit seen if this is the highest
        if ticker not in self.max_profit_seen or current_profit_pct > self.max_profit_seen[ticker]:
            self.max_profit_seen[ticker] = current_profit_pct
    
    def _confirm_signal(self, ticker: str, current_signal: float) -> float:
        """
        Apply confirmation logic to raw signals to reduce false positives.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        current_signal : float
            Current raw signal from stochastic oscillator
            
        Returns:
        --------
        float
            Confirmed signal value
        """
        # Get previous signal if available
        prev_signal = self.prev_signals.get(ticker, 0.0)
        
        # More aggressive signal handling for higher risk
        
        # If we have a strong signal, amplify it
        if abs(current_signal) > 0.6:
            return current_signal * 1.5 if abs(current_signal * 1.5) <= 1.0 else current_signal
        
        # If signal is changing direction, still consider it but reduce strength
        if current_signal * prev_signal < 0:
            # Direction change - still take signal but at reduced strength
            return current_signal * 0.8
            
        # For weak signals, consider position
        if abs(current_signal) < 0.3:
            # Check if we have a position and this would reverse it
            if ticker in self.portfolio.holdings and self.portfolio.holdings[ticker] != 0:
                position_sign = 1 if self.portfolio.holdings[ticker] > 0 else -1
                if current_signal * position_sign < 0:  # Signal is opposite of position
                    # Strengthen reversal signals for mean reversion
                    return current_signal * 1.2
            
        return current_signal
        
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
            return None
            
        # Get current position direction (long/short)
        position = self.portfolio.holdings[ticker]
        is_long = position > 0
        
        # If this is a new position, set initial stop loss with wider stops for higher risk
        if ticker not in self.stop_losses and ticker in self.entry_prices:
            trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.08)  # Wider stop loss (was 0.05)
            
            # Adjust stop distance based on consecutive losses (martingale-like approach)
            if self.parameters.get("use_martingale", True):
                loss_count = self.consecutive_losses.get(ticker, 0)
                max_rounds = self.parameters.get("max_martingale_rounds", 3)
                if loss_count > 0 and loss_count <= max_rounds:
                    # Widen stop for consecutive losses (more risk)
                    trailing_stop_pct *= (1 + (loss_count * self.parameters.get("stop_widening_factor", 0.25)))
            
            entry_price = self.entry_prices[ticker]
            if is_long:
                self.stop_losses[ticker] = entry_price * (1 - trailing_stop_pct)
            else:
                self.stop_losses[ticker] = entry_price * (1 + trailing_stop_pct)
            return None
        
        # If we don't have a stop loss or entry price, we can't check
        if ticker not in self.stop_losses:
            return None
            
        # Update trailing stop if price moved in our favor
        trailing_stop_pct = self.parameters.get("trailing_stop_pct", 0.08)
        if is_long:
            # For long positions, move stop up as price rises
            potential_new_stop = current_price * (1 - trailing_stop_pct)
            if potential_new_stop > self.stop_losses[ticker]:
                self.stop_losses[ticker] = potential_new_stop
                
            # Check if stop was hit
            if current_price <= self.stop_losses[ticker]:
                # Stop loss hit, exit position
                self._handle_position_exit(ticker, is_profit=False)
                return -1.0
        else:
            # For short positions, move stop down as price falls
            potential_new_stop = current_price * (1 + trailing_stop_pct)
            if potential_new_stop < self.stop_losses[ticker]:
                self.stop_losses[ticker] = potential_new_stop
                
            # Check if stop was hit
            if current_price >= self.stop_losses[ticker]:
                # Stop loss hit, exit position
                self._handle_position_exit(ticker, is_profit=False)
                return 1.0
                
        return None
    
    def _check_trailing_profit(self, ticker: str, current_price: float) -> Optional[float]:
        """
        Check if trailing profit target has been hit.
        
        Parameters:
        -----------
        ticker: str
            Ticker symbol
        current_price: float
            Current price of the ticker
            
        Returns:
        --------
        Optional[float]
            Signal value if trailing profit exit triggered, None otherwise
        """
        # If we don't have position info, can't check
        if (ticker not in self.portfolio.holdings or 
            self.portfolio.holdings[ticker] == 0 or
            ticker not in self.entry_prices or
            ticker not in self.max_profit_seen):
            return None
            
        # Get trigger and distance parameters
        trigger_pct = self.parameters.get("trailing_profit_trigger", 0.1)  # 10% profit to activate
        distance_pct = self.parameters.get("trailing_profit_distance", 0.05)  # 5% pullback to exit
        
        # Check if we've ever hit the trigger profit level
        if self.max_profit_seen[ticker] < trigger_pct:
            return None
            
        # If we're here, we've previously hit the trigger, so check for pullback
        position = self.portfolio.holdings[ticker]
        is_long = position > 0
        entry_price = self.entry_prices[ticker]
        
        # Calculate current profit
        if is_long:
            current_profit_pct = (current_price - entry_price) / entry_price
        else:
            current_profit_pct = (entry_price - current_price) / entry_price
            
        # Check for pullback from maximum profit
        pullback = self.max_profit_seen[ticker] - current_profit_pct
        
        if pullback >= distance_pct:
            # We've had a significant pullback from max profit, exit
            self._handle_position_exit(ticker, is_profit=True)
            return -1.0 if is_long else 1.0
            
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
        profit_target_pct = self.parameters.get("profit_target_pct", 0.25)  # Higher profit target (was 0.15)
        
        # Check if profit target reached
        entry_price = self.entry_prices[ticker]
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= profit_target_pct:
                self._handle_position_exit(ticker, is_profit=True)
                return -1.0  # Sell signal to take profit
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= profit_target_pct:
                self._handle_position_exit(ticker, is_profit=True)
                return 1.0  # Buy signal to cover short
                
        return None
    
    def _handle_position_exit(self, ticker: str, is_profit: bool) -> None:
        """
        Handle position exit bookkeeping for martingale-style position sizing.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        is_profit : bool
            Whether the exit is profitable or a loss
        """
        if not is_profit:
            # Increment consecutive loss counter for this ticker
            self.consecutive_losses[ticker] = self.consecutive_losses.get(ticker, 0) + 1
        else:
            # Reset consecutive loss counter on profit
            self.consecutive_losses[ticker] = 0
            
        # Clear tracking info
        if ticker in self.max_profit_seen:
            del self.max_profit_seen[ticker]

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
        adjusted_signals = {}
        
        # Get current positions
        current_positions = {ticker: shares for ticker, shares in self.portfolio.holdings.items() if shares != 0}
        
        # Calculate maximum number of open positions - higher risk version allows more positions
        max_positions = self.parameters.get("max_open_positions", 6)  # Increased from 4
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
                
            # For existing positions, allow reversal signals but block adding to existing positions
            if ticker in current_positions and abs(current_positions[ticker]) > 0:
                # If we already have a position and signal is in same direction
                current_position_sign = 1 if current_positions[ticker] > 0 else -1
                if (signal > 0 and current_position_sign > 0) or (signal < 0 and current_position_sign < 0):
                    # Don't add to existing positions for mean reversion
                    adjusted_signals[ticker] = 0.0  
                else:
                    # Allow reversal signals at full strength
                    adjusted_signals[ticker] = signal * 1.2  # Boost reversal signals
            else:
                # For new positions, apply martingale-like sizing for tickers with losses
                if self.parameters.get("use_martingale", True) and ticker in self.consecutive_losses:
                    loss_count = self.consecutive_losses[ticker]
                    # Amplify signal after losses (more aggressive entry)
                    if loss_count > 0:
                        factor = 1.0 + (loss_count * self.parameters.get("martingale_factor", 0.25))
                        signal = signal * factor if abs(signal * factor) <= 1.0 else signal
                
                adjusted_signals[ticker] = signal
                
        return adjusted_signals

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
        # More aggressive position sizing
        max_capital_per_position = self.parameters.get("max_capital_per_position", 0.40)  # Increased from 0.25
        portfolio_value = self.portfolio.get_value()
        max_position = max_capital_per_position * portfolio_value

        current_price = self.portfolio.current_prices.get(ticker, 0)
        if current_price <= 0:
            return 0

        # Martingale-style position sizing after losses
        position_multiplier = 1.0
        if self.parameters.get("use_martingale", True) and ticker in self.consecutive_losses:
            loss_count = self.consecutive_losses[ticker]
            max_rounds = self.parameters.get("max_martingale_rounds", 3)
            if loss_count > 0 and loss_count <= max_rounds:
                # Increase position size after losses
                position_multiplier = 1.0 + (loss_count * self.parameters.get("martingale_factor", 0.25))

        # Scale position size by signal strength and any martingale multiplier
        signal_strength = abs(signal)
        position_value = max_position * signal_strength * position_multiplier
        shares = int(position_value / current_price)

        # Negative for sell, positive for buy
        return shares if signal > 0 else -shares

    @classmethod
    def get_default_parameters(cls) -> Dict:
        """
        Get default parameters for the Stochastic Mean Reversion strategy.
        
        Returns:
        --------
        Dict
            Dictionary of default parameters
        """
        # Get standard stochastic parameters
        stoch_params = get_default_stochastic_parameters()
        
        # Modify them to be more aggressive
        stoch_params["oversold"] = 30.0  # Less extreme (was 20)
        stoch_params["overbought"] = 70.0  # Less extreme (was 80)
        stoch_params["signal_threshold"] = 0.2  # More signals (was 0.3)
        
        return {
            "stochastic_parameters": stoch_params,
            "max_capital_per_position": 0.40,  # 40% per position (was 25%)
            "commission": 0.0,
            "trailing_stop_pct": 0.08,  # 8% trailing stop (was 5%)
            "profit_target_pct": 0.25,  # 25% profit target (was 15%)
            "max_open_positions": 6,  # 6 max positions (was 4)
            "use_martingale": True,  # Enable martingale-style position sizing
            "martingale_factor": 0.25,  # Increase position by 25% after each loss
            "max_martingale_rounds": 3,  # Maximum number of martingale increases
            "stop_widening_factor": 0.25,  # Widen stops by 25% per loss
            "use_trailing_profit": True,  # Enable trailing profit targets
            "trailing_profit_trigger": 0.1,  # 10% profit to activate trailing exit
            "trailing_profit_distance": 0.05,  # 5% pullback from max to exit
        } 