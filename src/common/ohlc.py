from typing import TypedDict, Optional


class OHLCData(TypedDict):
    """Type for OHLC price data."""

    open: float
    high: float
    low: float
    close: float
    volume: Optional[int]
