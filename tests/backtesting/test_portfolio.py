from datetime import datetime

import pandas as pd
import pytest

from src.backtesting.portfolio import Portfolio


class TestPortfolio:
    """Test cases for the Portfolio class."""

    def test_portfolio_initialization(self):
        """Test portfolio initialization with default and custom parameters."""
        # Default initialization
        portfolio = Portfolio()
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert portfolio.allow_short_selling is False
        assert portfolio.allow_margin_trading is False
        assert portfolio.holdings == {}
        assert portfolio.current_prices == {}
        assert portfolio.equity_history == []
        assert portfolio.trade_history == []

        # Custom initialization
        portfolio = Portfolio(
            initial_capital=50000.0, allow_short_selling=True, allow_margin_trading=True
        )
        assert portfolio.initial_capital == 50000.0
        assert portfolio.cash == 50000.0
        assert portfolio.allow_short_selling is True
        assert portfolio.allow_margin_trading is True

    def test_get_position(self):
        """Test getting positions for tickers."""
        portfolio = Portfolio()

        # Position for non-existing ticker
        assert portfolio.get_position("AAPL") == 0

        # Add a position and check
        portfolio.holdings["AAPL"] = 10
        assert portfolio.get_position("AAPL") == 10

    def test_get_value(self):
        """Test portfolio value calculation."""
        portfolio = Portfolio(initial_capital=10000.0)

        # Initially only cash
        assert portfolio.get_value() == 10000.0

        # Add positions and prices
        portfolio.holdings["AAPL"] = 10
        portfolio.holdings["MSFT"] = 5
        portfolio.current_prices["AAPL"] = 150.0
        portfolio.current_prices["MSFT"] = 200.0

        # Expected value: cash + position values
        expected_value = 10000.0 + (10 * 150.0) + (5 * 200.0)
        assert portfolio.get_value() == expected_value

    def test_update_prices(self):
        """Test updating prices and equity history."""
        portfolio = Portfolio()
        now = datetime.now()

        # Update prices
        portfolio.update_prices({"AAPL": 150.0, "MSFT": 200.0}, now)
        assert portfolio.current_prices["AAPL"] == 150.0
        assert portfolio.current_prices["MSFT"] == 200.0

        # Check equity history
        assert len(portfolio.equity_history) == 1
        assert portfolio.equity_history[0][0] == now
        assert portfolio.equity_history[0][1] == 100000.0  # Only cash

        # Add positions
        portfolio.holdings["AAPL"] = 10

        # Update prices again
        later = datetime.now()
        portfolio.update_prices({"AAPL": 160.0}, later)
        assert portfolio.current_prices["AAPL"] == 160.0
        assert portfolio.current_prices["MSFT"] == 200.0  # Unchanged

        # Check equity history
        assert len(portfolio.equity_history) == 2
        assert portfolio.equity_history[1][0] == later
        assert portfolio.equity_history[1][1] == 100000.0 + (10 * 160.0)

    def test_execute_trade_buy(self):
        """Test executing buy trades."""
        portfolio = Portfolio()
        now = datetime.now()

        # Execute buy trade
        portfolio.execute_trade("AAPL", 10, 150.0, now)

        # Check position and cash
        assert portfolio.get_position("AAPL") == 10
        assert portfolio.cash == 100000.0 - (10 * 150.0)

        # Check trade history
        assert len(portfolio.trade_history) == 1
        trade = portfolio.trade_history[0]
        assert trade["timestamp"] == now
        assert trade["ticker"] == "AAPL"
        assert trade["quantity"] == 10
        assert trade["price"] == 150.0
        assert trade["value"] == 10 * 150.0
        assert trade["direction"] == "BUY"

    def test_execute_trade_sell(self):
        """Test executing sell trades."""
        portfolio = Portfolio()
        now = datetime.now()

        # Add initial position
        portfolio.holdings["AAPL"] = 10

        # Execute sell trade
        portfolio.execute_trade("AAPL", -5, 160.0, now)

        # Check position and cash
        assert portfolio.get_position("AAPL") == 5
        assert portfolio.cash == 100000.0 + (5 * 160.0)

        # Check trade history
        assert len(portfolio.trade_history) == 1
        trade = portfolio.trade_history[0]
        assert trade["direction"] == "SELL"

    def test_execute_trade_with_commission(self):
        """Test executing trades with commission fees."""
        portfolio = Portfolio()
        now = datetime.now()

        # Execute trade with commission
        commission = 9.99
        portfolio.execute_trade("AAPL", 10, 150.0, now, commission)

        # Check cash (reduced by trade value + commission)
        assert portfolio.cash == 100000.0 - (10 * 150.0) - commission

        # Check trade history
        assert portfolio.trade_history[0]["commission"] == commission

    def test_short_selling_restrictions(self):
        """Test short selling restrictions."""
        # Portfolio without short selling
        portfolio = Portfolio(allow_short_selling=False)
        now = datetime.now()

        # Try to sell without position
        with pytest.raises(ValueError):
            portfolio.execute_trade("AAPL", -5, 150.0, now)

        # Add partial position and try to sell more
        portfolio.holdings["AAPL"] = 3
        with pytest.raises(ValueError):
            portfolio.execute_trade("AAPL", -5, 150.0, now)

        # Portfolio with short selling allowed
        portfolio_short = Portfolio(allow_short_selling=True)
        # This should work
        portfolio_short.execute_trade("AAPL", -5, 150.0, now)
        assert portfolio_short.get_position("AAPL") == -5

    def test_margin_trading_restrictions(self):
        """Test margin trading restrictions."""
        # Portfolio without margin trading
        portfolio = Portfolio(initial_capital=1000.0, allow_margin_trading=False)
        now = datetime.now()

        # Try to buy more than cash allows
        with pytest.raises(ValueError):
            portfolio.execute_trade("AAPL", 10, 150.0, now)  # 1500 > 1000

        # Portfolio with margin trading allowed
        portfolio_margin = Portfolio(initial_capital=1000.0, allow_margin_trading=True)
        # This should work
        portfolio_margin.execute_trade("AAPL", 10, 150.0, now)
        assert portfolio_margin.get_position("AAPL") == 10
        assert portfolio_margin.cash == 1000.0 - (10 * 150.0)

    def test_get_performance_metrics_empty(self):
        """Test performance metrics with empty history."""
        portfolio = Portfolio()
        metrics = portfolio.get_performance_metrics()
        assert metrics["total_return"] == 0.0

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        portfolio = Portfolio(initial_capital=10000.0)

        # Create sample equity history
        start_date = pd.Timestamp("2023-01-01")
        dates = pd.date_range(
            start=start_date, periods=252, freq="B"
        )  # 1 year of business days

        # Simulate growing portfolio
        for i, date in enumerate(dates):
            # Simple growth pattern
            equity = 10000.0 * (1 + 0.0003 * i)  # ~20% annual return
            portfolio.equity_history.append((date.to_pydatetime(), equity))

        # Add some trades
        portfolio.trade_history.append(
            {
                "timestamp": dates[0].to_pydatetime(),
                "ticker": "AAPL",
                "quantity": 10,
                "price": 150.0,
                "value": 1500.0,
                "commission": 9.99,
                "direction": "BUY",
            }
        )

        portfolio.trade_history.append(
            {
                "timestamp": dates[-1].to_pydatetime(),
                "ticker": "AAPL",
                "quantity": -10,
                "price": 180.0,
                "value": 1800.0,
                "commission": 9.99,
                "direction": "SELL",
            }
        )

        # Get metrics
        metrics = portfolio.get_performance_metrics()

        # Basic checks
        assert "total_return" in metrics
        assert "annualized_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "num_trades" in metrics

        # Check values
        assert metrics["total_return"] > 0
        assert metrics["num_trades"] == 2
        assert metrics["profit_per_share"] == 30.0  # 180 - 150

    def test_complex_scenario(self):
        """Test a complex trading scenario."""
        portfolio = Portfolio(initial_capital=10000.0)

        # Day 1: Buy AAPL
        day1 = datetime(2023, 1, 1)
        portfolio.execute_trade("AAPL", 10, 150.0, day1)
        portfolio.update_prices({"AAPL": 150.0}, day1)

        # Day 2: AAPL rises, buy MSFT
        day2 = datetime(2023, 1, 2)
        portfolio.update_prices({"AAPL": 155.0}, day2)
        portfolio.execute_trade("MSFT", 5, 200.0, day2)
        portfolio.update_prices({"AAPL": 155.0, "MSFT": 200.0}, day2)

        # Day 3: Both stocks rise
        day3 = datetime(2023, 1, 3)
        portfolio.update_prices({"AAPL": 160.0, "MSFT": 210.0}, day3)

        # Day 4: AAPL falls, sell half of AAPL
        day4 = datetime(2023, 1, 4)
        portfolio.update_prices({"AAPL": 155.0, "MSFT": 210.0}, day4)
        portfolio.execute_trade("AAPL", -5, 155.0, day4)
        portfolio.update_prices({"AAPL": 155.0, "MSFT": 210.0}, day4)

        # Check final state
        assert portfolio.get_position("AAPL") == 5
        assert portfolio.get_position("MSFT") == 5
        assert len(portfolio.equity_history) == 4  # Initial + 3 updates
        assert len(portfolio.trade_history) == 3  # 3 trades

        # Verify performance metrics
        metrics = portfolio.get_performance_metrics()
        assert metrics["num_trades"] == 3

        # Check final portfolio value
        expected_value = (
            10000.0  # Initial capital
            - (10 * 150.0)  # First AAPL purchase
            - (5 * 200.0)  # MSFT purchase
            + (5 * 155.0)  # AAPL partial sell
            + (5 * 155.0)  # Remaining AAPL value
            + (5 * 210.0)  # MSFT value
        )
        assert portfolio.get_value() == expected_value

    def test_canbuy(self):
        """Test the canbuy method with various scenarios."""
        # Portfolio without margin trading
        portfolio = Portfolio(initial_capital=1000.0, allow_margin_trading=False)

        # Should be able to buy when enough cash
        assert portfolio.canbuy("AAPL", 5, 150.0) is True  # 750 < 1000

        # Should not be able to buy when not enough cash
        assert portfolio.canbuy("AAPL", 10, 150.0) is False  # 1500 > 1000

        # Cannot buy zero or negative quantity
        assert portfolio.canbuy("AAPL", 0, 150.0) is False
        assert portfolio.canbuy("AAPL", -5, 150.0) is False

        # With commission
        assert portfolio.canbuy("AAPL", 6, 150.0, 100.0) is True  # 1000 >= (6*150)+100
        assert portfolio.canbuy("AAPL", 6, 150.0, 101.0) is False  # 1000 < (6*150)+101

        # Portfolio with margin trading
        portfolio_margin = Portfolio(initial_capital=1000.0, allow_margin_trading=True)

        # Should be able to buy even when not enough cash
        assert (
            portfolio_margin.canbuy("AAPL", 10, 150.0) is True
        )  # 1500 > 1000 but margin allowed
        assert (
            portfolio_margin.canbuy("AAPL", 100, 150.0) is True
        )  # 15000 >> 1000 but margin allowed

        # Still cannot buy zero or negative quantity
        assert portfolio_margin.canbuy("AAPL", 0, 150.0) is False
        assert portfolio_margin.canbuy("AAPL", -5, 150.0) is False

    def test_cansell(self):
        """Test the cansell method with various scenarios."""
        # Portfolio without short selling
        portfolio = Portfolio(allow_short_selling=False)

        # Add a position
        portfolio.holdings["AAPL"] = 10

        # Should be able to sell owned shares
        assert portfolio.cansell("AAPL", 5) is True
        assert portfolio.cansell("AAPL", 10) is True

        # Should not be able to sell more than owned
        assert portfolio.cansell("AAPL", 11) is False

        # Should not be able to sell unowned stock
        assert portfolio.cansell("MSFT", 1) is False

        # Cannot sell zero or negative quantity
        assert portfolio.cansell("AAPL", 0) is False
        assert portfolio.cansell("AAPL", -5) is False

        # Portfolio with short selling
        portfolio_short = Portfolio(allow_short_selling=True)

        # Should be able to sell even without position
        assert portfolio_short.cansell("AAPL", 5) is True

        # Should be able to sell more than owned
        portfolio_short.holdings["MSFT"] = 10
        assert portfolio_short.cansell("MSFT", 20) is True

        # Still cannot sell zero or negative quantity
        assert portfolio_short.cansell("MSFT", 0) is False
        assert portfolio_short.cansell("MSFT", -5) is False
