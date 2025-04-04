"""
Test script for position sizing strategies
"""

import pandas as pd
from datetime import datetime, timedelta

from backtesting.portfolio import Portfolio
from backtesting.risk_management.position_sizing import (
    PositionSizingFactory,
    PositionSizingParameters,
)
from backtesting.strategies.strategy_factory import StrategyFactory


def test_position_sizing_strategies():
    """Test the different position sizing strategies."""

    # Create a portfolio with some initial capital
    portfolio = Portfolio(initial_capital=100000.0)

    # Add some current prices to the portfolio
    portfolio.current_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOG": 2500.0,
        "AMZN": 3200.0,
    }

    # Create signals for testing
    signals = {
        "AAPL": 0.8,  # Strong buy
        "MSFT": 0.3,  # Weak buy
        "GOOG": -0.6,  # Medium sell
        "AMZN": -0.2,  # Weak sell
    }

    # Test each position sizing strategy
    strategies = [
        "fixed_percentage",
        "equal_risk",
        "volatility_based",
        "signal_proportional",
    ]

    for strategy_name in strategies:
        # Create a strategy instance with default parameters
        position_sizing = PositionSizingFactory.get_strategy(strategy_name, portfolio)

        # Calculate position sizes for each ticker
        for ticker, signal in signals.items():
            position_size = position_sizing.calculate_position_size(
                ticker, signal, signals
            )
            assert isinstance(position_size, int)


def test_position_reversal():
    """Test the position reversal functionality in position sizing strategies."""

    # Create a portfolio with some initial capital
    portfolio = Portfolio(initial_capital=100000.0)

    # Add some current prices to the portfolio
    portfolio.current_prices = {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOG": 2500.0,
        "AMZN": 3200.0,
    }

    # Add some existing positions to the portfolio
    portfolio.holdings = {
        "AAPL": 100,  # Long position
        "MSFT": -50,  # Short position
        "GOOG": 0,  # No position
        "AMZN": 20,  # Long position
    }

    # Create signals for testing
    signals = {
        "AAPL": -0.7,  # Strong sell (opposed to current long)
        "MSFT": 0.6,  # Medium buy (opposed to current short)
        "GOOG": 0.5,  # Medium buy (no current position)
        "AMZN": -0.3,  # Weak sell (opposed to current long)
    }

    # Test each position sizing strategy
    strategies = [
        "fixed_percentage",
        "equal_risk",
        "volatility_based",
        "signal_proportional",
    ]

    expected_behaviors = {
        "AAPL": lambda size: size < 0
        and abs(size) >= 100,  # Should close long and potentially go short
        "MSFT": lambda size: size > 0
        and size >= 50,  # Should close short and potentially go long
        "GOOG": lambda size: size > 0,  # Should open new long position
        "AMZN": lambda size: size == -20,  # Should just close position (weak signal)
    }

    for strategy_name in strategies:
        # Create a strategy instance with default parameters
        position_sizing = PositionSizingFactory.get_strategy(strategy_name, portfolio)

        print(f"Testing {strategy_name} strategy for position reversal:")
        # Calculate position sizes for each ticker
        for ticker, signal in signals.items():
            position_size = position_sizing.calculate_position_size(
                ticker, signal, signals
            )
            print(
                f"  {ticker}: Current={portfolio.holdings.get(ticker, 0)}, Signal={signal}, New Size={position_size}"
            )

            # Assert the expected behavior
            assert expected_behaviors[ticker](
                position_size
            ), f"Unexpected position size for {ticker}: {position_size}"


def test_strategy_with_position_sizing():
    """Test a trading strategy with the new position sizing framework."""

    # Create a portfolio with some initial capital
    portfolio = Portfolio(initial_capital=100000.0)

    # Create an RSI strategy
    rsi_strategy = StrategyFactory.create_strategy("rsi_strategy", portfolio)

    # Configure the strategy to use different position sizing strategies
    position_sizing_types = [
        "fixed_percentage",
        "equal_risk",
        "signal_proportional",
    ]

    # Create some fake price data for testing
    start_date = datetime.now() - timedelta(days=100)
    dates = [start_date + timedelta(days=i) for i in range(100)]

    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

    # Create a DataFrame with fake OHLC data
    data = []
    for date in dates:
        for ticker in tickers:
            # Just some arbitrary price movements
            base_price = {"AAPL": 150, "MSFT": 300, "GOOG": 2500, "AMZN": 3200}[ticker]
            day = (date - start_date).days
            price = base_price * (1 + 0.1 * (day % 10) / 10)

            data.append(
                {
                    "timestamp": date,
                    "ticker": ticker,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000000,
                }
            )

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Set the index
    df.set_index(["timestamp", "ticker"], inplace=True)
    df.sort_index(inplace=True)

    # Test each position sizing type with the strategy
    for position_sizing_type in position_sizing_types:
        # Configure the strategy
        rsi_strategy.set_parameters(
            position_sizing_type=position_sizing_type,
            position_sizing_parameters=PositionSizingParameters.get_defaults(),
        )

        # Update current prices
        last_date = dates[-1]
        latest_data = df.xs(last_date, level="timestamp")

        portfolio.current_prices = {
            ticker: row["close"] for ticker, row in latest_data.iterrows()
        }

        # Generate signals (simplified - not actually computing RSI)
        signals = {
            "AAPL": 0.8,
            "MSFT": 0.3,
            "GOOG": -0.6,
            "AMZN": -0.2,
        }

        # Manually initialize the position sizing strategy if it's None
        if rsi_strategy.position_sizing is None:
            rsi_strategy.position_sizing = PositionSizingFactory.get_strategy(
                position_sizing_type,
                portfolio,
                rsi_strategy.parameters.get("position_sizing_parameters"),
            )

        # Calculate position sizes for each ticker
        for ticker, signal in signals.items():
            # Here we would call the strategy's execute method
            # but for testing, we'll just directly calculate position sizes
            position_size = rsi_strategy.position_sizing.calculate_position_size(
                ticker, signal, signals
            )
            assert isinstance(position_size, int)


if __name__ == "__main__":
    # Test position sizing strategies directly
    test_position_sizing_strategies()

    # Test position reversal functionality
    test_position_reversal()

    # Test position sizing with a strategy
    test_strategy_with_position_sizing()
