from enum import Enum, auto


class DataRequirement(Enum):
    """Enum to represent different types of data a strategy might require."""

    TICKER = auto()  # Standard OHLCV price/volume data
    NEWS = auto()  # News sentiment or event data
    OPTIONS = auto()  # Options chain data
    FUNDAMENTALS = auto()  # Fundamental company data
    # Add other data types as needed
