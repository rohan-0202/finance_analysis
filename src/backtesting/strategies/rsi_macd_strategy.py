from typing import Any, Dict

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.strategies.df_columns import CLOSE, TICKER, TIMESTAMP
from backtesting.strategy import Strategy
from signals.macd_signal import MACDSignal
from signals.rsi_signal import RSISignal
from signals.signal_factory import SignalFactory


class RSIMACDStrategy(Strategy):
    """
    Combines RSI and MACD signals for trading decisions.

    Trades when either indicator provides a signal, unless they conflict.
    - Buy if RSI signals buy OR MACD signals buy (and the other isn't sell).
    - Sell if RSI signals sell OR MACD signals sell (and the other isn't buy).
    - Neutral if signals conflict (one buy, one sell) or both are neutral.
    """

    def __init__(self, portfolio: Portfolio):
        super().__init__("RSI+MACD Strategy", portfolio)
        # Combine default parameters from both strategies
        self.parameters = self.get_default_parameters()

        # Create signal objects using factory and initial parameters
        self.rsi_signal: RSISignal = SignalFactory.create_signal(
            "rsi",
            window=self.parameters["rsi_period"],
            overbought=self.parameters["overbought_threshold"],
            oversold=self.parameters["oversold_threshold"],
        )
        self.macd_signal: MACDSignal = SignalFactory.create_signal(
            "macd",
            fast_period=self.parameters["fast_period"],
            slow_period=self.parameters["slow_period"],
            signal_period=self.parameters["signal_period"],
        )

    def set_parameters(self, **kwargs):
        """Set strategy parameters and update signal objects."""
        super().set_parameters(**kwargs)
        # Re-configure signal objects if relevant parameters changed
        if "rsi_period" in kwargs:
            self.rsi_signal.window = self.parameters["rsi_period"]
        if "overbought_threshold" in kwargs:
            self.rsi_signal.overbought = self.parameters["overbought_threshold"]
        if "oversold_threshold" in kwargs:
            self.rsi_signal.oversold = self.parameters["oversold_threshold"]
        if "fast_period" in kwargs:
            self.macd_signal.fast_period = self.parameters["fast_period"]
        if "slow_period" in kwargs:
            self.macd_signal.slow_period = self.parameters["slow_period"]
        if "signal_period" in kwargs:
            self.macd_signal.signal_period = self.parameters["signal_period"]
        return self

    def _generate_rsi_signal(self, ticker_data: pd.Series, ticker: str) -> float:
        """Generates the RSI signal for a single ticker."""
        min_required_data = self.parameters["rsi_period"] + 1
        if len(ticker_data) < min_required_data:
            return 0.0

        # Ensure the RSI calculation uses the current strategy parameter for the window
        self.rsi_signal.window = self.parameters["rsi_period"]
        self.rsi_signal.overbought = self.parameters["overbought_threshold"]
        self.rsi_signal.oversold = self.parameters["oversold_threshold"]

        # Call the correct helper method to calculate RSI from the series
        rsi_series = self.rsi_signal._calculate_rsi_from_series(ticker_data)

        if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
            return 0.0  # RSI calculation failed or latest is NaN

        latest_rsi = rsi_series.iloc[-1]

        if latest_rsi > self.parameters["overbought_threshold"]:
            return -1.0  # Sell signal (Overbought)
        elif latest_rsi < self.parameters["oversold_threshold"]:
            return 1.0  # Buy signal (Oversold)
        else:
            return 0.0  # Neutral signal

    def _generate_macd_signal(self, ticker_data: pd.Series, ticker: str) -> float:
        """Generates the MACD signal for a single ticker."""
        min_required_data = (
            self.parameters["slow_period"] + self.parameters["signal_period"] + 1
        )
        if len(ticker_data) < min_required_data:
            return 0.0

        # Ensure MACD calculation uses current parameters
        self.macd_signal.fast_period = self.parameters["fast_period"]
        self.macd_signal.slow_period = self.parameters["slow_period"]
        self.macd_signal.signal_period = self.parameters["signal_period"]

        macd_df = self.macd_signal.calculate_macd(ticker_data)
        macd_df = macd_df.dropna()

        if (
            macd_df.empty
            or len(macd_df) < 2
            or "macd" not in macd_df.columns
            or "signal" not in macd_df.columns
            or pd.isna(macd_df["macd"].iloc[-1])
            or pd.isna(macd_df["signal"].iloc[-1])
        ):
            return 0.0

        latest_macd = macd_df["macd"].iloc[-1]
        latest_signal = macd_df["signal"].iloc[-1]
        prev_macd = macd_df["macd"].iloc[-2]
        prev_signal = macd_df["signal"].iloc[-2]

        if latest_macd > latest_signal and prev_macd <= prev_signal:
            return 1.0  # Buy signal (MACD crossed above Signal)
        elif latest_macd < latest_signal and prev_macd >= prev_signal:
            return -1.0  # Sell signal (MACD crossed below Signal)
        else:
            return 0.0  # Hold signal (No crossover)

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate combined trading signals based on RSI and MACD.

        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex DataFrame with ('timestamp', 'Ticker') index
            and 'close' column.

        Returns:
        --------
        Dict[str, float]
            Dictionary mapping tickers to combined signal values (-1 Sell, 0 Neutral, 1 Buy)
        """
        combined_signals = {}
        if data.empty or CLOSE not in data.columns:
            return combined_signals

        # Ensure index is sorted for rolling calculations
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        tickers = data.index.get_level_values(TICKER).unique()

        for ticker in tickers:
            try:
                ticker_data = data.xs(ticker, level=TICKER)[CLOSE]
            except KeyError:
                combined_signals[ticker] = 0.0
                continue  # Ticker not present

            # Generate individual signals
            rsi_signal = self._generate_rsi_signal(ticker_data, ticker)
            macd_signal = self._generate_macd_signal(ticker_data, ticker)

            # Combine signals: Trade if either signals, unless they conflict
            signal = 0.0
            if rsi_signal == 1 and macd_signal != -1:
                signal = 1.0
            elif macd_signal == 1 and rsi_signal != -1:
                signal = 1.0
            elif rsi_signal == -1 and macd_signal != 1:
                signal = -1.0
            elif macd_signal == -1 and rsi_signal != 1:
                signal = -1.0
            # else: signal remains 0.0 (Neutral, including conflict case where one is 1 and other is -1)

            combined_signals[ticker] = signal

        return combined_signals

    def execute(self, data: pd.DataFrame) -> None:
        """
        Execute the RSI+MACD strategy.

        Generates signals and places orders based on the latest data.

        Parameters:
        -----------
        data : pd.DataFrame
            Latest market data (MultiIndex expected).
        """
        if data.empty:
            return

        if not isinstance(data.index, pd.MultiIndex) or list(data.index.names) != [
            TIMESTAMP,
            TICKER,
        ]:
            print(
                f"Error: Dataframe passed to {self.name}.execute does not have ('timestamp', 'ticker') MultiIndex."
            )
            return

        latest_timestamp = data.index.get_level_values(TIMESTAMP).max()
        self.last_update_time = latest_timestamp

        # Update current prices in the portfolio
        try:
            latest_data_slice = data.xs(latest_timestamp, level=TIMESTAMP)
            for ticker, row in latest_data_slice.iterrows():
                if isinstance(ticker, str) and CLOSE in row and pd.notna(row[CLOSE]):
                    self.portfolio.current_prices[ticker] = row[CLOSE]
        except KeyError:
            print(
                f"Warning: Could not extract data slice for timestamp {latest_timestamp} in {self.name}"
            )
            pass  # Continue execution, potentially with stale prices

        # Generate combined signals using the historical data provided
        self.current_signals = self.generate_signals(data)

        # Apply risk management (using base implementation for now)
        adjusted_signals = self.apply_risk_management(self.current_signals)

        # Execute trades based on adjusted signals
        for ticker, signal in adjusted_signals.items():
            if (
                ticker not in self.portfolio.current_prices
                or self.portfolio.current_prices[ticker] <= 0
            ):
                continue  # Skip if price is missing or invalid

            target_shares = self.calculate_position_size(ticker, signal)
            current_shares = self.portfolio.holdings.get(ticker, 0)
            trade_shares = target_shares - current_shares

            if trade_shares != 0:
                self.place_order(ticker, trade_shares, latest_timestamp)

        # Optional: Update portfolio value if needed here (often done in backtest loop)
        # self.portfolio.update_value(latest_timestamp)

    @classmethod
    def get_default_parameters(cls) -> Dict[str, Any]:
        """
        Get default parameters combining RSI and MACD.

        Returns:
        --------
        Dict[str, Any]
            Dictionary of default parameters.
        """
        return {
            # RSI defaults
            "rsi_period": 14,
            "overbought_threshold": 60,
            "oversold_threshold": 40,
            # MACD defaults
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            # Shared/Risk defaults (using RSI's as base, can be adjusted)
            "max_capital_per_position": 0.9,
            "commission": 0.001,
        }
