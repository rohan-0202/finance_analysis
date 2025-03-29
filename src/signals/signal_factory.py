from typing import Dict, Type

from signals.base_signal import BaseSignal
# from signals.macd_signal import MACDSignal
# from signals.obv_signal import OBVSignal
from signals.rsi_signal import RSISignal


class SignalFactory:
    """Factory class for creating technical indicator signals."""

    _signal_classes: Dict[str, Type[BaseSignal]] = {
        "rsi": RSISignal,
        # "macd": MACDSignal,
        # "obv": OBVSignal,
    }

    @classmethod
    def register_signal(cls, name: str, signal_class: Type[BaseSignal]) -> None:
        """
        Register a new signal type with the factory.

        Parameters:
        -----------
        name : str
            The name to register the signal under
        signal_class : Type[BaseSignal]
            The signal class to register
        """
        cls._signal_classes[name] = signal_class

    @classmethod
    def create_signal(cls, name: str, **kwargs) -> BaseSignal:
        """
        Create a signal instance by name.

        Parameters:
        -----------
        name : str
            The name of the signal to create
        **kwargs
            Additional arguments to pass to the signal constructor

        Returns:
        --------
        BaseSignal: An instance of the requested signal

        Raises:
        -------
        ValueError: If the signal name is not registered
        """
        if name not in cls._signal_classes:
            raise ValueError(f"Unknown signal type: {name}")

        signal_class = cls._signal_classes[name]
        return signal_class(**kwargs)
