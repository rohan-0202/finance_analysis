import pytest

from signals.base_signal import BaseSignal
from signals.rsi_signal import RSISignal
from signals.signal_factory import SignalFactory


# Define a dummy signal class for testing registration
class DummySignal(BaseSignal):
    @property
    def name(self) -> str:
        """Return the name of the signal."""
        return "dummy_signal"

    def calculate(self, data):
        """Dummy calculation method."""
        # No actual calculation needed for this test
        return "dummy_calculation_result"

    def get_signal(self, data):
        """Dummy signal generation method."""
        # Return a simple dummy signal value
        return 0

    def get_latest_signal(self, data):
        """Dummy latest signal generation method."""
        # Return a simple dummy signal value
        return 0

    def get_status_text(self, data):
        """Dummy status text generation method."""
        # Return a simple dummy status text
        return "dummy_status_text"

    def get_signals(self, data):
        """Dummy signals generation method."""
        # Return a simple dummy signals value
        return [0]

    def calculate_indicator(self, data):
        """Dummy indicator calculation method."""
        # Return a simple dummy indicator value
        return 0


def test_create_rsi_signal():
    """Test that SignalFactory correctly creates an RSISignal instance."""
    # Assuming RSISignal constructor might take arguments in the future,
    # but currently it doesn't require any specific ones for instantiation.
    # If RSISignal required arguments like 'period', they would be passed here.
    # e.g., signal = SignalFactory.create_signal("rsi", period=14)
    signal = SignalFactory.create_signal("rsi")
    assert isinstance(signal, RSISignal)


def test_create_unknown_signal_raises_value_error():
    """Test that SignalFactory raises ValueError for an unregistered signal name."""
    with pytest.raises(ValueError) as excinfo:
        SignalFactory.create_signal("unknown_signal")
    assert "Unknown signal type: unknown_signal" in str(excinfo.value)


def test_register_and_create_signal():
    """Test registering a new signal and then creating it."""
    # Ensure the dummy signal isn't already registered (using its name property)
    dummy_name = DummySignal().name
    with pytest.raises(ValueError):
        SignalFactory.create_signal(dummy_name)

    # Register the new signal using its name property
    SignalFactory.register_signal(dummy_name, DummySignal)

    # Create the newly registered signal
    signal = SignalFactory.create_signal(dummy_name)
    assert isinstance(signal, DummySignal)
    assert signal.name == dummy_name  # Verify name property

    # Clean up: Remove the registered signal (optional, but good practice)
    # Note: Direct manipulation of _signal_classes might be considered fragile.
    # If a dedicated 'unregister' method existed, it would be preferred.
    if dummy_name in SignalFactory._signal_classes:
        del SignalFactory._signal_classes[dummy_name]
