"""
OBV Divergence Strategy with RSI Confirmation

This strategy combines On-Balance Volume (OBV) divergence detection with RSI confirmation
to generate trading signals. It looks for:

1. Bullish Divergence: Price makes a lower low while OBV makes a higher low
   (indicates potential upward reversal)
2. Bearish Divergence: Price makes a higher high while OBV makes a lower high
   (indicates potential downward reversal)

After detecting a divergence, it waits for RSI to confirm the momentum shift
before generating a signal.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.risk_management.stop_loss_manager import StopLossParameters
from backtesting.strategies.strategyutils.obv_util import calculate_obv
from backtesting.strategies.strategyutils.rsi_util import generate_rsi_signal_for_ticker
from backtesting.strategy import DataDict, Strategy
from common.data_requirements import DataRequirement
from common.df_columns import CLOSE, TICKER, VOLUME


class OBVDivergenceStrategy(Strategy):
    """
    OBV Divergence Strategy with RSI Confirmation implementation.
    
    This strategy looks for divergences between price and OBV, then confirms
    with RSI momentum shift before generating buy/sell signals.
    """

    def __init__(self, portfolio: Portfolio):
        super().__init__("OBV Divergence Strategy", portfolio)
        # Initialize parameters with defaults
        self.parameters = self.get_default_parameters()
        
    def set_parameters(self, **kwargs):
        """Set strategy parameters and update nested parameters."""
        super().set_parameters(**kwargs)
        
        # Update RSI parameters if provided
        if "rsi_period" in kwargs:
            self.parameters["rsi_parameters"]["rsi_period"] = kwargs["rsi_period"]
        if "rsi_oversold" in kwargs:
            self.parameters["rsi_parameters"]["oversold_threshold"] = kwargs["rsi_oversold"]
        if "rsi_overbought" in kwargs:
            self.parameters["rsi_parameters"]["overbought_threshold"] = kwargs["rsi_overbought"]
            
        # Update OBV parameters if provided
        if "divergence_lookback" in kwargs:
            self.parameters["divergence_lookback"] = kwargs["divergence_lookback"]
        if "price_low_threshold" in kwargs:
            self.parameters["price_low_threshold"] = kwargs["price_low_threshold"]
        if "price_high_threshold" in kwargs:
            self.parameters["price_high_threshold"] = kwargs["price_high_threshold"]
            
        return self

    def get_data_requirements(self) -> List[DataRequirement]:
        """
        Specify the data required by this strategy.
        
        Returns:
        --------
        List[DataRequirement]
            The strategy requires ticker data with volume
        """
        return [DataRequirement.TICKER]

    def detect_divergence(
        self, 
        price_series: pd.Series, 
        obv_series: pd.Series,
        lookback: int = 20
    ) -> Tuple[bool, bool]:
        """
        Detect bullish and bearish divergences between price and OBV.
        
        Parameters:
        -----------
        price_series : pd.Series
            Series of closing prices
        obv_series : pd.Series
            Series of OBV values
        lookback : int, default=20
            Lookback period for detecting divergences
            
        Returns:
        --------
        Tuple[bool, bool]
            (bullish_divergence, bearish_divergence)
        """
        if len(price_series) < lookback or len(obv_series) < lookback:
            return False, False
            
        # Get data for the lookback period
        price_window = price_series[-lookback:]
        obv_window = obv_series[-lookback:]
        
        # Get parameters
        price_low_threshold = self.parameters.get("price_low_threshold", 0.98)
        price_high_threshold = self.parameters.get("price_high_threshold", 1.02)
        
        # Find local extrema
        price_min_idx = price_window.idxmin()
        price_max_idx = price_window.idxmax()
        
        # Check if the minimum/maximum is recent enough (in the last few periods)
        recent_window = 5
        recent_indices = price_window.index[-recent_window:]
        # Fix: Replace .contains() with proper check if index is in the recent window
        price_min_recent = price_min_idx in recent_indices
        price_max_recent = price_max_idx in recent_indices
        
        # If the min/max isn't recent, no divergence to detect yet
        if not (price_min_recent or price_max_recent):
            return False, False
            
        # Get the OBV values corresponding to price extrema
        obv_at_price_min = obv_window.loc[price_min_idx]
        obv_at_price_max = obv_window.loc[price_max_idx]
        
        # Find the second most recent minimum/maximum for comparison
        # Filter out points too close to the most recent extrema
        price_without_recent_min = price_window.drop(price_min_idx)
        price_without_recent_max = price_window.drop(price_max_idx)
        
        bullish_divergence = False
        bearish_divergence = False
        
        # Check for bullish divergence (price makes lower low but OBV makes higher low)
        if price_min_recent and len(price_without_recent_min) > 0:
            prev_min_idx = price_without_recent_min.idxmin()
            prev_price_min = price_window.loc[prev_min_idx]
            prev_obv_at_min = obv_window.loc[prev_min_idx]
            
            # Bullish divergence: price makes a lower low but OBV makes higher low
            if (price_window.loc[price_min_idx] < prev_price_min * price_low_threshold and 
                obv_at_price_min > prev_obv_at_min):
                bullish_divergence = True
                
        # Check for bearish divergence (price makes higher high but OBV makes lower high)
        if price_max_recent and len(price_without_recent_max) > 0:
            prev_max_idx = price_without_recent_max.idxmax()
            prev_price_max = price_window.loc[prev_max_idx]
            prev_obv_at_max = obv_window.loc[prev_max_idx]
            
            # Bearish divergence: price makes a higher high but OBV makes lower high
            if (price_window.loc[price_max_idx] > prev_price_max * price_high_threshold and 
                obv_at_price_max < prev_obv_at_max):
                bearish_divergence = True
                
        return bullish_divergence, bearish_divergence

    def generate_signals(self, data: DataDict) -> Dict[str, float]:
        """
        Generate trading signals based on OBV divergence with RSI confirmation.
        
        Parameters:
        -----------
        data : DataDict
            Dictionary containing the required data. Must include DataRequirement.TICKER
            with close prices and volume.
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        signals = {}
        
        # Check if ticker data is present
        if DataRequirement.TICKER not in data:
            print("Error: Ticker data not found in data dictionary for OBVDivergenceStrategy.")
            return signals
            
        ticker_data_df = data[DataRequirement.TICKER]
        
        # Check if required columns exist
        if (ticker_data_df.empty or 
            CLOSE not in ticker_data_df.columns or 
            VOLUME not in ticker_data_df.columns):
            return signals
            
        # Ensure index is sorted for calculations
        if not ticker_data_df.index.is_monotonic_increasing:
            ticker_data_df = ticker_data_df.sort_index()
            
        tickers = ticker_data_df.index.get_level_values(TICKER).unique()
        
        # Get parameters
        divergence_lookback = self.parameters.get("divergence_lookback", 20)
        rsi_confirmation_threshold = self.parameters.get("rsi_confirmation_threshold", 0.3)
        
        for ticker in tickers:
            try:
                # Get close prices and volume
                ticker_slice = ticker_data_df.xs(ticker, level=TICKER)
                close_prices = ticker_slice[CLOSE]
                volume = ticker_slice[VOLUME]
                
                # Check for minimum required data
                min_required = max(divergence_lookback, 
                                   self.parameters["rsi_parameters"]["rsi_period"] + 1)
                if len(close_prices) < min_required:
                    signals[ticker] = 0.0  # Not enough data
                    continue
                    
                # Calculate OBV
                obv_series = calculate_obv(close_prices, volume)
                if obv_series.empty or len(obv_series) < divergence_lookback:
                    signals[ticker] = 0.0  # OBV calculation failed or insufficient data
                    continue
                    
                # Detect divergence
                bullish_divergence, bearish_divergence = self.detect_divergence(
                    close_prices, obv_series, divergence_lookback
                )
                
                # Get RSI signal for confirmation
                rsi_signal = generate_rsi_signal_for_ticker(
                    close_prices,
                    self.parameters["rsi_parameters"]
                )
                
                # Generate combined signal
                if bullish_divergence and rsi_signal > rsi_confirmation_threshold:
                    # Bullish divergence confirmed by RSI moving up from oversold
                    signals[ticker] = 1.0
                elif bearish_divergence and rsi_signal < -rsi_confirmation_threshold:
                    # Bearish divergence confirmed by RSI moving down from overbought
                    signals[ticker] = -1.0
                else:
                    # No confirmed divergence
                    signals[ticker] = 0.0
                    
            except Exception as e:
                print(f"Error generating signal for {ticker}: {e}")
                signals[ticker] = 0.0
                
        return signals

    @classmethod
    def get_default_parameters(cls) -> Dict:
        """
        Get default parameters for the OBV Divergence Strategy.
        
        Returns:
        --------
        Dict
            Dictionary of default parameters
        """
        return {
            "divergence_lookback": 20,       # Lookback period for divergence detection
            "price_low_threshold": 0.98,     # Threshold for lower low (2% lower)
            "price_high_threshold": 1.02,    # Threshold for higher high (2% higher)
            "rsi_confirmation_threshold": 0.3,  # Minimum RSI signal strength for confirmation
            "rsi_parameters": {
                "rsi_period": 14,
                "oversold_threshold": 40,
                "overbought_threshold": 60,
                "position_size_multiplier": 1.0
            },
            "max_capital_per_position": 0.9,
            "commission": 0.0,
            "use_stop_loss": True,
            "stop_loss_parameters": StopLossParameters.get_defaults()
        }
