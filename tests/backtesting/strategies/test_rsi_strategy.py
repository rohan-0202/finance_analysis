from unittest.mock import MagicMock, patch

import pytest

# Assuming Portfolio, RSISignal, SignalFactory are importable from these locations
# Adjust paths if necessary based on your project structure
from backtesting.portfolio import Portfolio
from backtesting.strategies.rsi_strategy import RSIStrategy
from signals.rsi_signal import RSISignal
from signals.signal_factory import SignalFactory


@pytest.fixture
def mock_portfolio():
    """Fixture to create a mock Portfolio object."""
    return MagicMock(spec=Portfolio)


# Patch SignalFactory.create_signal for all tests in this module that need it
@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_rsi_strategy_initialization_defaults(mock_create_signal, mock_portfolio):
    """
    Test that RSIStrategy initializes with correct default parameters,
    name, and stores the portfolio.
    """
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance

    # Act
    strategy = RSIStrategy(portfolio=mock_portfolio)

    # Assert
    # Check strategy name
    assert strategy.name == "RSI Strategy"

    # Check portfolio storage
    assert strategy.portfolio is mock_portfolio

    # Check default parameters
    expected_defaults = {
        "rsi_period": 14,
        "overbought_threshold": 70,
        "oversold_threshold": 30,
        "max_capital_per_position": 0.1,
        "commission": 0.0,
    }
    assert strategy.parameters == expected_defaults


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_rsi_strategy_initialization_signal_creation(
    mock_create_signal, mock_portfolio
):
    """
    Test that RSIStrategy calls SignalFactory.create_signal correctly
    during initialization.
    """
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance

    # Act
    strategy = RSIStrategy(portfolio=mock_portfolio)

    # Assert
    # Verify SignalFactory.create_signal was called once
    mock_create_signal.assert_called_once()

    # Verify the arguments passed to create_signal
    call_args, call_kwargs = mock_create_signal.call_args
    assert call_args[0] == "rsi"  # First positional argument is the signal type
    assert call_kwargs == {
        "window": 14,  # Default rsi_period
        "overbought": 70,  # Default overbought_threshold
        "oversold": 30,  # Default oversold_threshold
    }

    # Verify the created signal instance is stored
    assert strategy.rsi_signal is mock_rsi_signal_instance


# --- Tests for set_parameters ---


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_set_parameters_updates_internal_dict(mock_create_signal, mock_portfolio):
    """Test setting parameters updates the strategy's parameter dictionary."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    strategy = RSIStrategy(portfolio=mock_portfolio)
    original_params = strategy.parameters.copy()

    # Act
    strategy.set_parameters(rsi_period=20, commission=0.001)

    # Assert
    assert strategy.parameters["rsi_period"] == 20
    assert strategy.parameters["commission"] == 0.001
    # Check that other params remain unchanged
    assert (
        strategy.parameters["overbought_threshold"]
        == original_params["overbought_threshold"]
    )
    assert (
        strategy.parameters["oversold_threshold"]
        == original_params["oversold_threshold"]
    )
    assert (
        strategy.parameters["max_capital_per_position"]
        == original_params["max_capital_per_position"]
    )


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_set_parameters_updates_signal_object(mock_create_signal, mock_portfolio):
    """
    Test setting RSI-specific parameters updates the rsi_signal object's attributes.
    """
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    # Set initial attributes on the mock signal to check against later
    mock_rsi_signal_instance.window = 14
    mock_rsi_signal_instance.overbought = 70
    mock_rsi_signal_instance.oversold = 30
    mock_create_signal.return_value = mock_rsi_signal_instance

    strategy = RSIStrategy(portfolio=mock_portfolio)

    # Act
    strategy.set_parameters(
        rsi_period=21, overbought_threshold=75, oversold_threshold=25
    )

    # Assert
    # Check that the mock signal object's attributes were updated
    assert mock_rsi_signal_instance.window == 21
    assert mock_rsi_signal_instance.overbought == 75
    assert mock_rsi_signal_instance.oversold == 25


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_set_parameters_unrelated_params_dont_affect_signal(
    mock_create_signal, mock_portfolio
):
    """
    Test that setting non-RSI parameters doesn't change RSI signal attributes.
    """
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    # Set initial attributes on the mock signal
    initial_window = 14
    initial_overbought = 70
    initial_oversold = 30
    mock_rsi_signal_instance.window = initial_window
    mock_rsi_signal_instance.overbought = initial_overbought
    mock_rsi_signal_instance.oversold = initial_oversold
    mock_create_signal.return_value = mock_rsi_signal_instance

    strategy = RSIStrategy(portfolio=mock_portfolio)

    # Act
    strategy.set_parameters(max_capital_per_position=0.05, commission=0.002)

    # Assert
    # Check internal params updated
    assert strategy.parameters["max_capital_per_position"] == 0.05
    assert strategy.parameters["commission"] == 0.002
    # Verify signal attributes remain unchanged
    assert mock_rsi_signal_instance.window == initial_window
    assert mock_rsi_signal_instance.overbought == initial_overbought
    assert mock_rsi_signal_instance.oversold == initial_oversold


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_set_parameters_returns_self(mock_create_signal, mock_portfolio):
    """Test that set_parameters returns the strategy instance (self)."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    strategy = RSIStrategy(portfolio=mock_portfolio)

    # Act
    result = strategy.set_parameters(rsi_period=10)

    # Assert
    assert result is strategy
