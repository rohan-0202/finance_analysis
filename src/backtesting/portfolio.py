from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


class Portfolio:
    """
    Tracks positions, cash, and performance metrics.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        allow_short_selling: bool = False,
        allow_margin_trading: bool = False,
    ):
        """
        Initialize the portfolio.

        Parameters:
        -----------
        initial_capital : float
            Starting capital
        allow_short_selling : bool
            Whether to allow short selling
        allow_margin_trading : bool
            Whether to allow margin trading
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.allow_short_selling = allow_short_selling
        self.allow_margin_trading = allow_margin_trading
        self.holdings = {}  # {ticker: quantity}
        self.current_prices = {}  # {ticker: price}

        # Performance tracking
        self.equity_history = []  # [(timestamp, equity), ...]
        self.trade_history = []  # List of trade dictionaries

    def get_position(self, ticker: str) -> int:
        """Get current position for a ticker."""
        return self.holdings.get(ticker, 0)

    def get_value(self, timestamp: Optional[datetime] = None) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Parameters:
        -----------
        timestamp : datetime, optional
            Timestamp for the calculation, uses latest if None

        Returns:
        --------
        float
            Portfolio value
        """
        # Sum position values
        position_value = sum(
            self.get_position(ticker) * self.current_prices.get(ticker, 0)
            for ticker in self.holdings
        )
        return self.cash + position_value

    def update_prices(self, ticker_prices: Dict[str, float], timestamp: datetime):
        """
        Update current prices and equity curve.

        Parameters:
        -----------
        ticker_prices : Dict[str, float]
            Dictionary of {ticker: price}
        timestamp : datetime
            Current timestamp
        """
        self.current_prices.update(ticker_prices)

        # Update equity history - check if we already have an entry for this timestamp
        current_equity = self.get_value(timestamp)

        # If there's already an entry for this timestamp, update it instead of appending
        for i, (ts, _) in enumerate(self.equity_history):
            if ts == timestamp:
                self.equity_history[i] = (timestamp, current_equity)
                return

        # Otherwise, append a new entry
        self.equity_history.append((timestamp, current_equity))

    def execute_trade(
        self,
        ticker: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ):
        """
        Execute a trade by updating positions and cash.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Quantity (positive for buy, negative for sell)
        price : float
            Execution price
        timestamp : datetime
            Trade timestamp
        commission : float
            Commission cost

        Raises:
        -------
        ValueError
            If trade violates short selling or margin rules
        """
        # Validate short selling
        current_position = self.get_position(ticker)
        if (
            quantity < 0
            and not self.allow_short_selling
            and current_position + quantity < 0
        ):
            raise ValueError(
                f"Cannot sell {abs(quantity)} shares of {ticker}. Current position: {current_position}. Short selling not allowed."
            )

        # Validate margin trading
        trade_value = quantity * price
        total_cost = trade_value + commission
        if quantity > 0 and not self.allow_margin_trading and self.cash < total_cost:
            raise ValueError(
                f"Insufficient funds to buy {quantity} shares of {ticker} at ${price:.2f}. Required: ${total_cost:.2f}, Available: ${self.cash:.2f}"
            )

        # Update position
        if ticker not in self.holdings:
            self.holdings[ticker] = 0
        self.holdings[ticker] += quantity

        # Update cash
        self.cash -= trade_value + commission

        # Record the trade
        trade = {
            "timestamp": timestamp,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "commission": commission,
            "direction": "BUY" if quantity > 0 else "SELL",
        }
        self.trade_history.append(trade)

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Returns:
        --------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if not self.equity_history:
            return {"total_return": 0.0}

        # Extract time series
        timestamps, equity = zip(*self.equity_history, strict=False)
        equity_series = pd.Series(equity, index=timestamps)

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital

        # Annualized return (assuming 252 trading days)
        days = (timestamps[-1] - timestamps[0]).days
        if days > 0:
            years = days / 365.0
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0

        # Risk metrics
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Maximum drawdown
            rolling_max = equity_series.cummax()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()

            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            sharpe_ratio = (
                np.sqrt(252) * returns.mean() / returns.std()
                if returns.std() > 0
                else 0
            )

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino_ratio = (
                np.sqrt(252) * returns.mean() / downside_returns.std()
                if len(downside_returns) > 0 and downside_returns.std() > 0
                else 0
            )

            # Calmar ratio
            calmar_ratio = (
                annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            )
        else:
            volatility = 0
            max_drawdown = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0

        # Trade statistics
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)

            # Separate buy and sell trades
            buy_trades = trades_df[trades_df["direction"] == "BUY"]
            sell_trades = trades_df[trades_df["direction"] == "SELL"]

            # Count trades
            num_trades = len(trades_df)

            # Calculate profit/loss for completed round trips
            # This is a simplified approach - in a real implementation you would match buys and sells
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                avg_buy_price = buy_trades["price"].mean()
                avg_sell_price = sell_trades["price"].mean()
                profit_per_share = avg_sell_price - avg_buy_price
            else:
                profit_per_share = 0
        else:
            num_trades = 0
            profit_per_share = 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "num_trades": num_trades,
            "profit_per_share": profit_per_share,
        }

    def canbuy(
        self, ticker: str, quantity: int, price: float, commission: float = 0.0
    ) -> bool:
        """
        Check if the portfolio can execute a buy trade.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Quantity to buy (must be positive)
        price : float
            Current price of the asset
        commission : float
            Commission cost for the trade

        Returns:
        --------
        bool
            True if the trade can be executed, False otherwise
        """
        if quantity <= 0:
            return False  # Buy quantity must be positive

        total_cost = quantity * price + commission

        # If margin trading is allowed, we can always buy
        if self.allow_margin_trading:
            return True

        # Otherwise, check if we have enough cash
        return self.cash >= total_cost

    def cansell(self, ticker: str, quantity: int) -> bool:
        """
        Check if the portfolio can execute a sell trade.

        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Quantity to sell (must be positive)

        Returns:
        --------
        bool
            True if the trade can be executed, False otherwise
        """
        if quantity <= 0:
            return False  # Sell quantity must be positive

        current_position = self.get_position(ticker)

        # If short selling is allowed, we can always sell
        if self.allow_short_selling:
            return True

        # Otherwise, check if we have enough shares
        return current_position >= quantity
