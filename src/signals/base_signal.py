from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple, TypedDict

import pandas as pd


class SignalData(TypedDict):
    """Standard format for signal data returned by all signal indicators."""

    date: datetime
    type: str  # 'buy' or 'sell'
    price: float
    # Additional fields can be added in specific implementations


class BaseSignal(ABC):
    """Abstract base class for all technical indicators and signals."""

    @abstractmethod
    def calculate_indicator(
        self, ticker_symbol: str, **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Calculate the technical indicator for a given ticker.

        Returns:
        --------
        tuple: (price_data, indicator_data)
            - price_data: DataFrame with original price data
            - indicator_data: DataFrame or Series with indicator values
        """
        pass

    @abstractmethod
    def get_signals(self, ticker_symbol: str, **kwargs) -> List[SignalData]:
        """
        Get buy/sell signals for a given ticker based on this indicator.

        Returns:
        --------
        list of SignalData: A list of signal events
        """
        pass

    @abstractmethod
    def get_latest_signal(self, ticker_symbol: str, **kwargs) -> Optional[SignalData]:
        """
        Get only the most recent signal for a ticker.

        Returns:
        --------
        SignalData or None: The most recent signal
        """
        pass

    @abstractmethod
    def get_status_text(
        self, price_data: pd.DataFrame, indicator_data: pd.DataFrame
    ) -> str:
        """
        Generate a text description of the current indicator status.

        Returns:
        --------
        str: Text description of indicator status
        """
        pass
