from typing import List, Optional, Tuple, cast

import pandas as pd

from db_util import get_historical_data
from signals.base_signal import BaseSignal, SignalData


class OBVSignalData(SignalData):
    """OBV-specific signal data."""

    obv: float


class OBVSignal(BaseSignal):
    """On-Balance Volume (OBV) indicator implementation."""

    def __init__(self, window: int = 20):
        self.window = window

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV) for a price series.

        Parameters:
        -----------
        close : pd.Series
            Close price series
        volume : pd.Series
            Volume series

        Returns:
        --------
        pd.Series: OBV values
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]  # Initialize with first day's volume
        
        # Calculate OBV based on price movement
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                # Price up, add volume
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                # Price down, subtract volume
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                # Price unchanged, OBV unchanged
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def calculate_indicator(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Calculate the OBV for a given ticker.

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
        tuple: (price_data, obv_data)
            - price_data: DataFrame with original price data
            - obv_data: Series with OBV values
        """
        try:
            # Get historical data
            price_data = get_historical_data(ticker_symbol, db_name, days)
            
            # Calculate OBV
            obv_data = self.calculate_obv(price_data["close"], price_data["volume"])
            
            return price_data, obv_data

        except Exception as e:
            print(f"Error calculating OBV for {ticker_symbol}: {e}")
            return None, None

    def get_signals(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> List[OBVSignalData]:
        """
        Get OBV buy/sell signals for a given ticker based on divergence.

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
        list of OBVSignalData: A list of signal events
        """
        price_data, obv_data = self.calculate_indicator(ticker_symbol, db_name, days)

        if obv_data is None or len(obv_data) == 0:
            return []

        # Create a DataFrame for easier processing
        data = pd.DataFrame({"obv": obv_data, "close": price_data["close"]})
        
        # Calculate moving averages for smoothing
        data["obv_ma"] = data["obv"].rolling(window=self.window).mean()
        data["price_ma"] = data["close"].rolling(window=self.window).mean()
        
        # Calculate rate of change for both price and OBV
        data["obv_roc"] = data["obv_ma"].pct_change(periods=self.window)
        data["price_roc"] = data["price_ma"].pct_change(periods=self.window)
        
        # Drop NaN values
        data = data.dropna()
        
        # Identify bullish divergence (price down, OBV up)
        bullish_divergence = (data["price_roc"] < 0) & (data["obv_roc"] > 0)
        
        # Identify bearish divergence (price up, OBV down)
        bearish_divergence = (data["price_roc"] > 0) & (data["obv_roc"] < 0)
        
        # Filter signals to prevent clustering (only consider signals that are at least window days apart)
        signals: List[OBVSignalData] = []
        last_signal_date = None
        
        # Process buy signals (bullish divergence)
        for date in data[bullish_divergence].index:
            if last_signal_date is None or (date - last_signal_date).days > self.window:
                signals.append(
                    cast(
                        OBVSignalData,
                        {
                            "date": date,
                            "type": "buy",
                            "obv": data.loc[date, "obv"],
                            "price": data.loc[date, "close"],
                        },
                    )
                )
                last_signal_date = date
        
        # Reset for sell signals
        last_signal_date = None
        
        # Process sell signals (bearish divergence)
        for date in data[bearish_divergence].index:
            if last_signal_date is None or (date - last_signal_date).days > self.window:
                signals.append(
                    cast(
                        OBVSignalData,
                        {
                            "date": date,
                            "type": "sell",
                            "obv": data.loc[date, "obv"],
                            "price": data.loc[date, "close"],
                        },
                    )
                )
                last_signal_date = date
        
        # Sort signals by date
        signals.sort(key=lambda x: x["date"])
        
        return signals

    def get_latest_signal(
        self, ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
    ) -> Optional[OBVSignalData]:
        """
        Get only the most recent OBV signal for a ticker.

        Returns:
        --------
        OBVSignalData or None: The most recent signal
        """
        signals = self.get_signals(ticker_symbol, db_name, days)

        # Return the most recent signal if any exist
        if signals:
            return signals[-1]
        return None

    def get_status_text(self, price_data: pd.DataFrame, obv_data: pd.Series) -> str:
        """
        Generate a text description of the current OBV status.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data with at least close prices
        obv_data : pd.Series
            OBV values
            
        Returns:
        --------
        str: Text description of OBV status
        """
        if len(obv_data) < self.window:
            return "Insufficient data for OBV analysis"
        
        # Get recent values
        current_obv = obv_data.iloc[-1]
        prev_obv = obv_data.iloc[-2]
        current_close = price_data["close"].iloc[-1]
        prev_close = price_data["close"].iloc[-2]
        
        # Calculate short-term trends (5-day)
        obv_5d_change = (obv_data.iloc[-1] - obv_data.iloc[-5]) / abs(obv_data.iloc[-5]) * 100
        price_5d_change = (price_data["close"].iloc[-1] - price_data["close"].iloc[-5]) / price_data["close"].iloc[-5] * 100
        
        # Generate status text
        status_text = f"Current OBV: {current_obv:,.0f}, 5-day change: {obv_5d_change:.2f}%\n"
        
        # Check for divergence
        if (obv_5d_change > 0 and price_5d_change < 0):
            status_text += "Status: BULLISH DIVERGENCE - OBV rising while price falling"
        elif (obv_5d_change < 0 and price_5d_change > 0):
            status_text += "Status: BEARISH DIVERGENCE - OBV falling while price rising"
        elif (current_obv > prev_obv and current_close > prev_close):
            status_text += "Status: CONFIRMING UPTREND - Both OBV and price rising"
        elif (current_obv < prev_obv and current_close < prev_close):
            status_text += "Status: CONFIRMING DOWNTREND - Both OBV and price falling"
        else:
            status_text += "Status: NEUTRAL - No clear signal"
        
        return status_text 