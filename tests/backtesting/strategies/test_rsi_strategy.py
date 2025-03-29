from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

# Assuming Portfolio, RSISignal, SignalFactory are importable from these locations
# Adjust paths if necessary based on your project structure
from src.backtesting.portfolio import Portfolio
from src.backtesting.strategies.rsi_strategy import RSIStrategy
from src.signals.rsi_signal import RSISignal


@pytest.fixture
def mock_portfolio():
    """Fixture to create a mock Portfolio object."""
    portfolio = MagicMock(spec=Portfolio)
    # Initialize required attributes
    portfolio.current_prices = {}
    portfolio.holdings = {}
    return portfolio


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


# --- Tests for generate_signals ---

@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_generate_signals_buy_signal(mock_create_signal, mock_portfolio):
    """Test generate_signals produces buy signals when RSI is below oversold threshold."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    # Create a strategy with custom thresholds for testing
    strategy = RSIStrategy(portfolio=mock_portfolio)
    # Set a very short RSI period (2) to work with our limited test data
    strategy.set_parameters(rsi_period=2, oversold_threshold=30)
    
    # Create test data with MultiIndex (Timestamp, Ticker)
    timestamps = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
    tickers = ['AAPL', 'MSFT']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    
    # Create DataFrame with close prices
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0, 250.0, 155.0, 255.0, 160.0, 260.0]}
    )
    
    # Create a simple mock RSI response that always returns a buy signal for AAPL
    def mock_calculate_rsi(series):
        print(f"Inside mock_calculate_rsi with series of length {len(series)}")
        print(f"Series values: {series.values}")
        
        # No matter what, return a buy signal for AAPL and neutral for MSFT
        if series.iloc[0] == 150.0:  # AAPL
            print("This is AAPL data - returning buy signal RSI (25)")
            return pd.Series([float('nan'), 25.0, 25.0], index=timestamps)
        else:  # MSFT
            print("This is MSFT data - returning neutral signal RSI (40)")
            return pd.Series([float('nan'), 40.0, 40.0], index=timestamps)
    
    # Directly patch the strategy's rsi_signal.calculate_rsi method
    strategy.rsi_signal.calculate_rsi = mock_calculate_rsi
    
    # Act
    signals = strategy.generate_signals(data)
    
    print(f"Final signals: {signals}")
    
    # Assert
    assert 'AAPL' in signals
    assert 'MSFT' in signals
    assert signals['AAPL'] == 1.0  # Buy signal (RSI < oversold)
    assert signals['MSFT'] == 0.0  # Neutral signal (RSI between thresholds)


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_generate_signals_sell_signal(mock_create_signal, mock_portfolio):
    """Test generate_signals produces sell signals when RSI is above overbought threshold."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    # Create a strategy with custom thresholds for testing
    strategy = RSIStrategy(portfolio=mock_portfolio)
    # Set a very short RSI period (2) to work with our limited test data
    strategy.set_parameters(rsi_period=2, overbought_threshold=70)
    
    # Create test data with MultiIndex (Timestamp, Ticker)
    timestamps = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
    tickers = ['AAPL', 'MSFT']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    
    # Create DataFrame with close prices
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0, 250.0, 155.0, 255.0, 160.0, 260.0]}
    )
    
    # Mock the RSI calculation to return values above overbought threshold for AAPL
    def mock_calculate_rsi(series):
        # Identify ticker by the first price value instead of length
        if series.iloc[0] == 150.0:  # AAPL data (first price is 150.0)
            return pd.Series([float('nan'), 75.0, 75.0], index=timestamps)
        else:  # MSFT data (first price is 250.0)
            return pd.Series([float('nan'), 65.0, 65.0], index=timestamps)
    
    # Set up the mock RSI signal
    strategy.rsi_signal.calculate_rsi = mock_calculate_rsi
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert 'AAPL' in signals
    assert 'MSFT' in signals
    assert signals['AAPL'] == -1.0  # Sell signal (RSI > overbought)
    assert signals['MSFT'] == 0.0   # Neutral signal (RSI between thresholds)


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_generate_signals_insufficient_data(mock_create_signal, mock_portfolio):
    """Test generate_signals handles insufficient data gracefully."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    strategy.set_parameters(rsi_period=14)  # Need at least 15 data points
    
    # Create test data with only a few data points
    timestamps = pd.DatetimeIndex(['2023-01-01', '2023-01-02'])  # Only 2 data points
    tickers = ['AAPL']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0, 155.0]}
    )
    
    # Mock the RSI calculation to return empty/NaN values due to insufficient data
    mock_rsi_signal_instance.calculate_rsi.return_value = pd.Series([float('nan'), float('nan')], index=timestamps)
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert 'AAPL' in signals
    assert signals['AAPL'] == 0.0  # Neutral signal when insufficient data


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_generate_signals_empty_data(mock_create_signal, mock_portfolio):
    """Test generate_signals handles empty data."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Create empty DataFrame
    data = pd.DataFrame()
    
    # Act
    signals = strategy.generate_signals(data)
    
    # Assert
    assert isinstance(signals, dict)
    assert len(signals) == 0  # Empty signals dictionary


# --- Tests for execute ---

@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_execute_places_buy_orders(mock_create_signal, mock_portfolio):
    """Test execute places buy orders when a buy signal is generated."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Configure the portfolio mock
    mock_portfolio.current_prices = {'AAPL': 150.0}
    mock_portfolio.holdings = {'AAPL': 0}  # No current holdings
    mock_portfolio.get_value.return_value = 10000.0
    
    # Create test data
    timestamps = pd.DatetimeIndex(['2023-01-01'])
    tickers = ['AAPL']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0]}
    )
    
    # Mock generate_signals to return a buy signal for AAPL
    strategy.generate_signals = MagicMock(return_value={'AAPL': 1.0})  # Buy signal
    
    # Act
    strategy.execute(data)
    
    # Assert
    # Check that place_order was called for AAPL with a positive quantity
    mock_portfolio.execute_trade.assert_called_once()
    call_args = mock_portfolio.execute_trade.call_args[0]
    assert call_args[0] == 'AAPL'  # Ticker
    assert call_args[1] > 0  # Positive quantity for buy
    assert call_args[2] == 150.0  # Price
    assert call_args[3] == timestamps[0]  # Timestamp


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_execute_places_sell_orders(mock_create_signal, mock_portfolio):
    """Test execute places sell orders when a sell signal is generated."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Configure the portfolio mock
    mock_portfolio.current_prices = {'AAPL': 150.0}
    mock_portfolio.holdings = {'AAPL': 10}  # Existing holdings
    mock_portfolio.get_value.return_value = 10000.0
    mock_portfolio.cansell.return_value = True
    
    # Create test data
    timestamps = pd.DatetimeIndex(['2023-01-01'])
    tickers = ['AAPL']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0]}
    )
    
    # Mock generate_signals to return a sell signal for AAPL
    strategy.generate_signals = MagicMock(return_value={'AAPL': -1.0})  # Sell signal
    
    # Act
    strategy.execute(data)
    
    # Assert
    # Check that place_order was called for AAPL with a negative quantity
    mock_portfolio.execute_trade.assert_called_once()
    call_args = mock_portfolio.execute_trade.call_args[0]
    assert call_args[0] == 'AAPL'  # Ticker
    assert call_args[1] < 0  # Negative quantity for sell
    assert call_args[2] == 150.0  # Price
    assert call_args[3] == timestamps[0]  # Timestamp


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_execute_no_action_on_neutral_signal(mock_create_signal, mock_portfolio):
    """Test execute doesn't place orders when signals are neutral."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Configure the portfolio mock
    mock_portfolio.current_prices = {'AAPL': 150.0}
    mock_portfolio.holdings = {'AAPL': 0}
    
    # Create test data
    timestamps = pd.DatetimeIndex(['2023-01-01'])
    tickers = ['AAPL']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0]}
    )
    
    # Mock generate_signals to return a neutral signal for AAPL
    strategy.generate_signals = MagicMock(return_value={'AAPL': 0.0})  # Neutral signal
    
    # Mock the calculate_position_size method to return 0
    strategy.calculate_position_size = MagicMock(return_value=0)
    
    # Act
    strategy.execute(data)
    
    # Assert
    # Check that place_order was not called
    mock_portfolio.execute_trade.assert_not_called()


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_execute_empty_data(mock_create_signal, mock_portfolio):
    """Test execute handles empty data gracefully."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Create empty DataFrame
    data = pd.DataFrame()
    
    # Act
    strategy.execute(data)
    
    # Assert
    # Check that no signals were generated and no trades were executed
    mock_portfolio.execute_trade.assert_not_called()


@patch("backtesting.strategies.rsi_strategy.SignalFactory.create_signal")
def test_execute_updates_last_update_time(mock_create_signal, mock_portfolio):
    """Test execute updates the last_update_time property."""
    # Arrange
    mock_rsi_signal_instance = MagicMock(spec=RSISignal)
    mock_create_signal.return_value = mock_rsi_signal_instance
    
    strategy = RSIStrategy(portfolio=mock_portfolio)
    
    # Create test data
    timestamps = pd.DatetimeIndex(['2023-01-01'])
    tickers = ['AAPL']
    index = pd.MultiIndex.from_product([timestamps, tickers], names=['Timestamp', 'Ticker'])
    data = pd.DataFrame(
        index=index,
        data={'close': [150.0]}
    )
    
    # Mock generate_signals to return any signal
    strategy.generate_signals = MagicMock(return_value={'AAPL': 0.0})
    
    # Act
    strategy.execute(data)
    
    # Assert
    # Check that last_update_time was updated
    assert strategy.last_update_time == timestamps[0]
