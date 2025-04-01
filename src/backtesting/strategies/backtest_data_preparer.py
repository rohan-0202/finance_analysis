from datetime import datetime, timedelta, timezone
from typing import Dict, List, Union

import pandas as pd

from backtesting.strategy import DataRequirement
from common.df_columns import (
    ASK,
    BID,
    EXPIRATION_DATE,
    IMPLIED_VOLATILITY,
    LAST_PRICE,
    LAST_UPDATED,
    OPEN_INTEREST,
    OPTION_TYPE,
    STRIKE,
    TICKER,
    TIMESTAMP,
    VOLUME,
)
from common.tz_util import ensure_utc_tz
from db_util import get_historical_data, get_options_data


class BacktestDataPreparer:
    """
    A class to handle data preparation for backtesting.

    This class is responsible for fetching and preparing all the necessary data
    for backtesting strategies, including ticker data, options data, and benchmark data.
    """

    def __init__(self, strategy):
        """
        Initialize the BacktestDataPreparer.

        Parameters:
        -----------
        strategy : Strategy
            The strategy instance that will be used for backtesting
        """
        self.strategy = strategy

    def prepare_data(
        self,
        tickers: Union[str, List[str]],
        start_date: datetime,
        end_date: datetime,
        db_name: str,
        data_buffer_months: int = 2,
    ) -> Dict[DataRequirement, pd.DataFrame]:
        """
        Fetch and prepare historical data for backtesting based on strategy requirements.

        Parameters:
        -----------
        tickers : str or List[str]
            Ticker or list of tickers to backtest
        start_date : datetime
            Start date for the backtest
        end_date : datetime
            End date for the backtest
        db_name : str
            Database name to fetch data from
        data_buffer_months : int, default 2
            Number of months of additional data to fetch before start_date for calculations

        Returns:
        --------
        Dict[DataRequirement, pd.DataFrame]
            Dictionary mapping data requirements to DataFrames
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Ensure start_date and end_date are UTC
        start_date = ensure_utc_tz(start_date)
        end_date = ensure_utc_tz(end_date)

        # Calculate fetch start date with buffer
        buffer_start_date = start_date - timedelta(days=data_buffer_months * 30)
        buffer_start_date = ensure_utc_tz(buffer_start_date)

        # Calculate days needed to fetch from buffer_start_date until *today*
        # This ensures we definitely get the data up to 'end_date' using the existing
        # get_historical_data function's logic.
        today = datetime.now(timezone.utc)
        days_to_fetch = (today - buffer_start_date).days + 1  # Add 1 for inclusivity
        if days_to_fetch <= 0:
            days_to_fetch = 1  # Fetch at least one day

        # Get data requirements from the strategy
        try:
            data_requirements = self.strategy.get_data_requirements()
        except Exception as e:
            print(f"Warning: Could not get data requirements from strategy: {e}")
            print("Defaulting to ticker data only")
            data_requirements = [DataRequirement.TICKER]

        if not data_requirements:
            print(
                "Warning: Strategy returned empty data requirements. Defaulting to ticker data."
            )
            data_requirements = [DataRequirement.TICKER]

        # Initialize the result dictionary
        result_data = {}

        # Prepare different types of data based on requirements
        for requirement in data_requirements:
            if requirement == DataRequirement.TICKER:
                # Prepare ticker data (OHLCV)
                result_data[requirement] = self._prepare_ticker_data(
                    tickers, buffer_start_date, end_date, days_to_fetch, db_name
                )
            elif requirement == DataRequirement.OPTIONS:
                # Prepare options data
                result_data[requirement] = self._prepare_options_data(
                    tickers, buffer_start_date, end_date, days_to_fetch, db_name
                )
            elif requirement == DataRequirement.NEWS:
                raise NotImplementedError("News data is not yet implemented")
            elif requirement == DataRequirement.FUNDAMENTALS:
                raise NotImplementedError("Fundamentals data is not yet implemented")
            else:
                print(f"Warning: Unknown data requirement {requirement}. Skipping.")

        return result_data

    def _prepare_ticker_data(
        self,
        tickers: List[str],
        buffer_start_date: datetime,
        end_date: datetime,
        days_to_fetch: int,
        db_name: str,
    ) -> pd.DataFrame:
        """
        Prepare ticker (OHLCV) data for backtesting.

        Parameters:
        -----------
        tickers : List[str]
            List of tickers to fetch data for
        buffer_start_date : datetime
            Start date with buffer
        end_date : datetime
            End date for the backtest
        days_to_fetch : int
            Number of days to fetch
        db_name : str
            Database name to fetch data from

        Returns:
        --------
        pd.DataFrame
            DataFrame with MultiIndex (Timestamp, Ticker) containing OHLCV data
        """
        all_data_raw = []

        for ticker in tickers:
            try:
                # Fetch potentially *more* data than needed using the existing function
                ticker_data_raw = get_historical_data(
                    ticker_symbol=ticker, db_name=db_name, days=days_to_fetch
                )

                if ticker_data_raw.empty:
                    # Don't warn here yet, filter first
                    continue

                # Filter the fetched data to the *required* range
                ticker_data_filtered = ticker_data_raw[
                    (ticker_data_raw.index >= buffer_start_date)
                    & (ticker_data_raw.index <= end_date)
                ].copy()

                if ticker_data_filtered.empty:
                    print(
                        f"Warning: No data found for {ticker} in required range {buffer_start_date.date()} to {end_date.date()}. Skipping."
                    )
                    continue

                # Check if the filtered data actually starts after the buffer date
                if ticker_data_filtered.index[0] > buffer_start_date:
                    print(
                        f"Warning: Data for {ticker} starts at {ticker_data_filtered.index[0].date()}, which is within the buffer period but after the intended buffer start {buffer_start_date.date()}."
                    )

                # Add Ticker column
                ticker_data_filtered[TICKER] = ticker

                # Reset index to prepare for MultiIndex
                ticker_data_filtered = ticker_data_filtered.reset_index()

                # Append the *filtered* data to our list
                all_data_raw.append(ticker_data_filtered)

            except ValueError as ve:
                # Handle cases where get_historical_data legitimately finds no data at all
                print(
                    f"Warning: Could not fetch any data for {ticker} (up to {days_to_fetch} days back): {ve}. Skipping."
                )
                continue
            except Exception as e:
                print(f"Error processing data for {ticker}: {e}")
                continue

        if not all_data_raw:
            # Raise error only if NO data was found for ANY ticker in the required range
            raise ValueError(
                "No valid data found for any ticker in the specified date range."
            )

        # Combine all *filtered* data into one DataFrame
        combined_data = pd.concat(all_data_raw, ignore_index=True)

        # Ensure 'timestamp' is tz-aware (UTC)
        combined_data[TIMESTAMP] = pd.to_datetime(
            combined_data[TIMESTAMP], errors="coerce", utc=True
        )
        combined_data.dropna(
            subset=[TIMESTAMP], inplace=True
        )  # Drop rows where conversion failed

        if combined_data.empty:
            # If after concatenation and timestamp handling, it's still empty
            raise ValueError(
                "Data became empty after processing timestamps. Check data quality."
            )

        # Create MultiIndex
        combined_data = combined_data.set_index([TIMESTAMP, TICKER])

        # Sort by timestamp and ticker
        combined_data = combined_data.sort_index()

        return combined_data

    def _prepare_options_data(
        self,
        tickers: List[str],
        buffer_start_date: datetime,
        end_date: datetime,
        days_to_fetch: int,
        db_name: str,
    ) -> pd.DataFrame:
        """
        Prepare options data for backtesting.

        Parameters:
        -----------
        tickers : List[str]
            List of tickers to fetch options data for
        buffer_start_date : datetime
            Start date with buffer
        end_date : datetime
            End date for the backtest
        days_to_fetch : int
            Number of days to fetch
        db_name : str
            Database name to fetch data from

        Returns:
        --------
        pd.DataFrame
            DataFrame with MultiIndex (Timestamp, Ticker) containing options data
        """

        all_data_raw = []

        for ticker in tickers:
            try:
                # Fetch options data using the existing function
                ticker_data_raw = get_options_data(
                    ticker_symbol=ticker, db_name=db_name, days=days_to_fetch
                )

                if ticker_data_raw.empty:
                    continue

                # Add Ticker column
                ticker_data_raw[TICKER] = ticker

                # Reset index to prepare for MultiIndex
                ticker_data_raw = ticker_data_raw.reset_index()

                # Use last_updated as the timestamp for backtesting
                ticker_data_raw[TIMESTAMP] = pd.to_datetime(
                    ticker_data_raw[LAST_UPDATED]
                )

                # Filter the fetched data to the required range
                ticker_data_filtered = ticker_data_raw[
                    (ticker_data_raw[TIMESTAMP] >= buffer_start_date)
                    & (ticker_data_raw[TIMESTAMP] <= end_date)
                ].copy()

                if ticker_data_filtered.empty:
                    print(
                        f"Warning: No options data found for {ticker} in required range {buffer_start_date.date()} to {end_date.date()}. Skipping."
                    )
                    continue

                # Check if the filtered data actually starts after the buffer date
                if ticker_data_filtered[TIMESTAMP].min() > buffer_start_date:
                    print(
                        f"Warning: Options data for {ticker} starts at {ticker_data_filtered[TIMESTAMP].min().date()}, which is within the buffer period but after the intended buffer start {buffer_start_date.date()}."
                    )

                # Append the filtered data to our list
                all_data_raw.append(ticker_data_filtered)

            except ValueError as ve:
                print(
                    f"Warning: Could not fetch any options data for {ticker} (up to {days_to_fetch} days back): {ve}. Skipping."
                )
                continue
            except Exception as e:
                print(f"Error fetching options data for {ticker}: {e}")
                continue

        if not all_data_raw:
            print(
                "Warning: No options data found for any tickers in the specified range."
            )
            # Return empty DataFrame with expected structure
            columns = [
                TIMESTAMP,
                TICKER,
                STRIKE,
                OPTION_TYPE,
                EXPIRATION_DATE,
                LAST_PRICE,
                BID,
                ASK,
                VOLUME,
                OPEN_INTEREST,
                IMPLIED_VOLATILITY,
                LAST_UPDATED,
            ]
            empty_df = pd.DataFrame(columns=columns)
            empty_df = empty_df.set_index([TIMESTAMP, TICKER])
            return empty_df

        # Combine all data
        all_data = pd.concat(all_data_raw, axis=0)

        # Set MultiIndex
        all_data = all_data.set_index([TIMESTAMP, TICKER])

        # Sort index
        all_data = all_data.sort_index()

        return all_data

    def prepare_benchmark_data(
        self,
        benchmark_ticker: str,
        all_data: pd.DataFrame,
        backtest_start_date: datetime,
        initial_capital: float,
    ) -> pd.Series:
        """
        Prepare benchmark data series for comparison.

        Parameters:
        -----------
        benchmark_ticker : str
            Ticker to use as benchmark
        all_data : pd.DataFrame
            DataFrame with all historical data
        backtest_start_date : datetime
            Start date for the backtest
        initial_capital : float
            Initial capital for the portfolio

        Returns:
        --------
        pd.Series
            Series with benchmark values
        """
        if benchmark_ticker not in all_data.index.get_level_values(TICKER).unique():
            return pd.Series(dtype=float)  # Empty series

        # Make sure backtest_start_date is tz-aware in the right way
        backtest_start = ensure_utc_tz(backtest_start_date)

        try:
            # Get prices for the benchmark ticker
            benchmark_prices = all_data.xs(benchmark_ticker, level=TICKER)["close"]

            # Find closest timestamp at or after backtest start
            start_idx = benchmark_prices.index.searchsorted(backtest_start)
            if start_idx >= len(benchmark_prices):
                return pd.Series(dtype=float)  # No data in backtest period

            start_price = benchmark_prices.iloc[start_idx]
            if pd.isna(start_price) or start_price <= 0:
                return pd.Series(dtype=float)  # Invalid start price

            # Calculate shares bought with initial capital
            shares = initial_capital / start_price

            # Calculate benchmark values
            benchmark_values = benchmark_prices * shares

            return benchmark_values

        except Exception as e:
            print(f"Error preparing benchmark data: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for detailed debugging
            return pd.Series(dtype=float)


if __name__ == "__main__":
    data_preparer = BacktestDataPreparer(None)
    df = data_preparer._prepare_options_data(
        tickers=["SPY"],
        buffer_start_date=datetime(2025, 3, 10),
        end_date=datetime(2025, 4, 1),
        days_to_fetch=365,
        db_name="stock_data.db",
    )
    print(df)
