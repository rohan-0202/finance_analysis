from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import ta  # Add the ta library import

# --- Add import for df_columns ---
from common import df_columns
from db_util import get_historical_data
from signals.base_signal import BaseSignal, SignalData


class RSISignalData(SignalData):
    """RSI-specific signal data."""

    rsi: float


class RSISignal(BaseSignal):
    """Relative Strength Index (RSI) indicator implementation."""

    def __init__(self, window: int = 14, overbought: int = 70, oversold: int = 30):
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    # Add a helper method to calculate RSI from a given series
    def _calculate_rsi_from_series(self, series: pd.Series) -> pd.Series:
        """Calculate RSI directly from a pandas Series using the configured window."""
        if series.empty or len(series) < self.window:
            # Return an empty series or series of NaNs matching the input index length
            # Ensure index is preserved for alignment later
            return pd.Series([np.nan] * len(series), index=series.index, dtype=float)

        # Use the ta library to calculate RSI
        # fillna=False prevents filling initial NaNs required for window calculation
        rsi_indicator = ta.momentum.RSIIndicator(
            close=series, window=self.window, fillna=False
        )
        rsi_data = rsi_indicator.rsi()
        # ta might return NaNs at the start, which is expected.
        return rsi_data

    def calculate_indicator(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Calculate the RSI for a given ticker.

        Parameters:
        -----------
        ticker_symbol : str
            The stock ticker symbol (e.g., 'AAPL')
        db_name : str, default="stock_data.db"
            The name of the SQLite database file
        days : int, default=365
            Number of days of historical data to use

        Returns:
        --------
        tuple: (price_data, rsi_data)
            - price_data: DataFrame with original price data
            - rsi_data: Series with RSI values
        """
        try:
            # Get historical data
            price_data = get_historical_data(ticker_symbol, db_name, days)

            # --- Data Cleaning Start ---
            if price_data is None or price_data.empty:
                print(f"No historical data found for {ticker_symbol}")
                return None, None

            # Ensure index is datetime and remove NaT values
            price_data.index = pd.to_datetime(price_data.index, errors="coerce")
            initial_count = len(price_data)
            price_data = price_data.loc[price_data.index.notna()]
            if len(price_data) < initial_count:
                print(
                    f"Warning: Removed {initial_count - len(price_data)} rows with invalid timestamps for {ticker_symbol}"
                )

            # Remove any potential remaining duplicate index entries
            if price_data.index.has_duplicates:
                print(
                    f"Warning: Found duplicate timestamps for {ticker_symbol}. Keeping first entry."
                )
                price_data = price_data[~price_data.index.duplicated(keep="first")]

            if price_data.empty:
                print(
                    f"No valid historical data remaining for {ticker_symbol} after cleaning."
                )
                return None, None
            # --- Data Cleaning End ---

            # Check if we have enough data after cleaning
            if len(price_data) < self.window:
                print(
                    f"Insufficient data ({len(price_data)} points) for RSI calculation (window: {self.window}) for {ticker_symbol}"
                )
                return price_data, None  # Return price data but None for RSI

            # Calculate RSI
            # Use the ta library to calculate RSI
            rsi_indicator = ta.momentum.RSIIndicator(
                close=price_data[df_columns.CLOSE], window=self.window
            )
            rsi_data = rsi_indicator.rsi()

            # Drop NaN values caused by RSI calculation (at the beginning of the series)
            rsi_data = rsi_data.dropna()

            # Align price_data with rsi_data AFTER cleaning and RSI calculation
            price_data = price_data.loc[rsi_data.index]

            return price_data, rsi_data

        except Exception as e:
            print(f"Error calculating RSI for {ticker_symbol}: {e}")
            return None, None

    def get_signals(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> List[RSISignalData]:
        """
        Get RSI buy/sell signals for a given ticker.

        Parameters:
        -----------
        ticker_symbol : str
            The stock ticker symbol
        db_name : str, default="stock_data.db"
            The name of the SQLite database file
        days : int, default=365
            Number of days of historical data to use

        Returns:
        --------
        list of RSISignalData: A list of signal events
        """
        price_data, rsi_data = self.calculate_indicator(ticker_symbol, db_name, days)

        if rsi_data is None or len(rsi_data) == 0:
            return []

        # Create a DataFrame for easier processing
        data = pd.DataFrame({"rsi": rsi_data, "close": price_data[df_columns.CLOSE]})

        # Identify buy signals (crossing up through oversold level)
        buy_signals = (data["rsi"] > self.oversold) & (
            data["rsi"].shift() <= self.oversold
        )

        # Identify sell signals (crossing down through overbought level)
        sell_signals = (data["rsi"] < self.overbought) & (
            data["rsi"].shift() >= self.overbought
        )

        # Combine buy and sell signals into a list of dictionaries
        signals: List[RSISignalData] = []

        # Process buy signals
        for date in data[buy_signals].index:
            signals.append(
                cast(
                    RSISignalData,
                    {
                        "date": date,
                        "type": "buy",
                        "rsi": data.loc[date, "rsi"],
                        "price": data.loc[date, "close"],
                    },
                )
            )

        # Process sell signals
        for date in data[sell_signals].index:
            signals.append(
                cast(
                    RSISignalData,
                    {
                        "date": date,
                        "type": "sell",
                        "rsi": data.loc[date, "rsi"],
                        "price": data.loc[date, "close"],
                    },
                )
            )

        # Sort signals by date
        signals.sort(key=lambda x: x["date"])

        return signals

    def get_latest_signal(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Optional[RSISignalData]:
        """
        Get only the most recent RSI signal for a ticker.

        Returns:
        --------
        RSISignalData or None: The most recent signal
        """
        signals = self.get_signals(ticker_symbol, db_name, days)

        # Return the most recent signal if any exist
        if signals:
            return signals[-1]
        return None

    def get_status_text(self, price_data: pd.DataFrame, rsi_data: pd.Series) -> str:
        """
        Generate a text description of the current RSI status.

        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data with at least close prices
        rsi_data : pd.Series
            RSI values

        Returns:
        --------
        str: Text description of RSI status
        """
        if len(rsi_data) < 5:
            return "Insufficient data for RSI analysis"

        # Get current RSI value
        current_rsi = rsi_data.iloc[-1]

        # Generate status text
        status_text = f"Current RSI: {current_rsi:.2f}\n"

        if current_rsi > self.overbought:
            status_text += "Status: OVERBOUGHT - Potential sell signal"
        elif current_rsi < self.oversold:
            status_text += "Status: OVERSOLD - Potential buy signal"
        else:
            status_text += "Status: NEUTRAL"

        return status_text
