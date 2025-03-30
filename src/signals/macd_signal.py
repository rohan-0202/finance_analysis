from datetime import datetime
from typing import List, Optional, Tuple, cast

import pandas as pd
import ta  # Add the ta library import

from common import df_columns
from db_util import get_historical_data
from signals.base_signal import BaseSignal, SignalData


class MACDSignalData(SignalData):
    """MACD-specific signal data."""

    date: datetime
    type: str
    price: float
    macd: float
    signal: float


class MACDSignal(BaseSignal):
    """Moving Average Convergence Divergence (MACD) indicator implementation."""

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate_macd(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate the MACD for a price series.

        Parameters:
        -----------
        series : pd.Series
            Price series (typically close prices)

        Returns:
        --------
        pd.DataFrame: DataFrame containing MACD values, signal line, and histogram
        """
        # Calculate EMAs
        ema_fast = series.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Create DataFrame with results
        macd_data = pd.DataFrame(
            {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram,
            }
        )

        return macd_data

    def calculate_indicator(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Calculate the MACD for a given ticker.

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
        tuple: (price_data, macd_data)
            - price_data: DataFrame with original price data (cleaned)
            - macd_data: DataFrame with MACD values
        """
        try:
            # Get historical data
            price_data = get_historical_data(ticker_symbol, db_name, days)

            # --- Start Data Cleaning ---
            if price_data is None or price_data.empty:
                print(f"No data found for {ticker_symbol}")
                return None, None

            # Ensure index is datetime and remove NaT indices
            price_data.index = pd.to_datetime(price_data.index, errors="coerce")
            price_data = price_data[pd.notna(price_data.index)]

            # Ensure required columns exist and remove rows with NaNs in critical columns
            required_cols = [df_columns.CLOSE]  # Only 'close' is needed for basic MACD
            if not all(col in price_data.columns for col in required_cols):
                print(
                    f"Missing required column ('{df_columns.CLOSE}') for {ticker_symbol}"
                )
                return None, None
            price_data = price_data.dropna(subset=required_cols)

            if price_data.empty:
                print(f"Insufficient valid data after cleaning for {ticker_symbol}")
                return None, None
            # --- End Data Cleaning ---

            # Calculate MACD
            # Use the ta library to calculate MACD
            macd_indicator = ta.trend.MACD(
                close=price_data[df_columns.CLOSE],
                window_fast=self.fast_period,
                window_slow=self.slow_period,
                window_sign=self.signal_period,
            )
            # Retrieve MACD line, signal line, and histogram
            macd = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_hist = macd_indicator.macd_diff()  # Histogram
            macd_data = pd.DataFrame(
                {"macd": macd, "signal": macd_signal, "histogram": macd_hist}
            )

            # Drop NaN values caused by MACD calculation
            macd_data = macd_data.dropna()

            # Align price_data with macd_data to have the same dates
            # Ensure index intersection is handled correctly
            common_index = price_data.index.intersection(macd_data.index)
            price_data = price_data.loc[common_index]
            macd_data = macd_data.loc[common_index]

            if price_data.empty or macd_data.empty:
                print(f"No overlapping data after MACD calculation for {ticker_symbol}")
                return None, None

            return price_data, macd_data

        except Exception as e:
            print(f"Error calculating MACD for {ticker_symbol}: {e}")
            return None, None

    def get_signals(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> List[MACDSignalData]:
        """
        Get MACD buy/sell signals for a given ticker.

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
        list of MACDSignalData: A list of signal events
        """
        price_data, macd_data = self.calculate_indicator(ticker_symbol, db_name, days)

        if macd_data is None or len(macd_data) == 0:
            return []

        # Check data validity after calculation and cleaning
        if (
            price_data is None
            or price_data.empty
            or macd_data is None
            or macd_data.empty
        ):
            return []

        # Identify crossover points
        buy_signals = (macd_data["macd"] > macd_data["signal"]) & (
            macd_data["macd"].shift() <= macd_data["signal"].shift()
        )
        sell_signals = (macd_data["macd"] < macd_data["signal"]) & (
            macd_data["macd"].shift() >= macd_data["signal"].shift()
        )

        # Combine buy and sell signals into a list of dictionaries
        signals: List[MACDSignalData] = []

        # Process buy signals
        for date in macd_data[buy_signals].index:
            # Add explicit check for NaT, although cleaning should prevent this
            if pd.notna(date):
                signals.append(
                    cast(
                        MACDSignalData,
                        {
                            "date": date.to_pydatetime(),  # Convert to standard datetime
                            "type": "buy",
                            "macd": macd_data.loc[date, "macd"],
                            "signal": macd_data.loc[date, "signal"],
                            "price": price_data.loc[date, df_columns.CLOSE],
                        },
                    )
                )

        # Process sell signals
        for date in macd_data[sell_signals].index:
            # Add explicit check for NaT
            if pd.notna(date):
                signals.append(
                    cast(
                        MACDSignalData,
                        {
                            "date": date.to_pydatetime(),  # Convert to standard datetime
                            "type": "sell",
                            "macd": macd_data.loc[date, "macd"],
                            "signal": macd_data.loc[date, "signal"],
                            "price": price_data.loc[date, df_columns.CLOSE],
                        },
                    )
                )

        # Sort signals by date
        signals.sort(key=lambda x: x["date"])

        return signals

    def get_latest_signal(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Optional[MACDSignalData]:
        """
        Get only the most recent MACD signal for a ticker.

        Returns:
        --------
        MACDSignalData or None: The most recent signal
        """
        signals = self.get_signals(ticker_symbol, db_name, days)

        # Return the most recent signal if any exist
        if signals:
            return signals[-1]
        return None

    def get_status_text(self, price_data: pd.DataFrame, macd_data: pd.DataFrame) -> str:
        """
        Generate a text description of the current MACD status.

        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data with at least close prices
        macd_data : pd.DataFrame
            MACD values, signal line, and histogram

        Returns:
        --------
        str: Text description of MACD status
        """
        if len(macd_data) < 5:
            return "Insufficient data for MACD analysis"

        # Get current values
        current_macd = macd_data["macd"].iloc[-1]
        current_signal = macd_data["signal"].iloc[-1]
        current_histogram = macd_data["histogram"].iloc[-1]

        # Generate status text
        status_text = f"Current MACD: {current_macd:.2f}\n"
        status_text += f"Signal Line: {current_signal:.2f}\n"
        status_text += f"Histogram: {current_histogram:.2f}\n"

        # Determine trend
        if current_macd > current_signal:
            if current_histogram > current_histogram.shift(1).iloc[-1]:
                status_text += "Status: BULLISH - Momentum increasing"
            else:
                status_text += "Status: BULLISH - Momentum decreasing"
        else:
            if current_histogram < current_histogram.shift(1).iloc[-1]:
                status_text += "Status: BEARISH - Momentum decreasing"
            else:
                status_text += "Status: BEARISH - Momentum increasing"

        return status_text
