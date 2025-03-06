import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=Warning)


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a price series.

    Parameters:
    -----------
    series : pd.Series
        Price series (typically close prices)
    window : int, default=14
        The window period for RSI calculation

    Returns:
    --------
    pd.Series: RSI values
    """
    # Calculate price changes
    delta = series.diff()

    # Create separate series for gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and average loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_historical_data(
    ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
) -> pd.DataFrame:
    """Fetch historical price data for a ticker from the database."""
    conn = sqlite3.connect(db_name)

    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Query to get data
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM historical_prices
    WHERE ticker = ? AND timestamp >= ?
    ORDER BY timestamp ASC
    """

    # First, get the data without parsing dates
    df = pd.read_sql_query(
        query, conn, params=(ticker_symbol, start_date.strftime("%Y-%m-%d"))
    )

    conn.close()

    if df.empty:
        raise ValueError(f"No historical data found for {ticker_symbol}")

    # Parse the timestamp column manually to avoid timezone issues
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    return df


def calculate_ticker_rsi(
    ticker_symbol: str,
    window: int = 14,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Calculate the RSI for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    window : int, default=14
        The period for RSI calculation
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    tuple: (price_data, rsi_data)
        - price_data: DataFrame with original price data
        - rsi_data: Series with RSI values
    """
    try:
        # Get historical data
        price_data = get_historical_data(ticker_symbol, db_name, days)

        # Calculate RSI
        rsi_data = calculate_rsi(price_data["close"], window)

        # Drop NaN values caused by RSI calculation
        rsi_data = rsi_data.dropna()

        # Align price_data with rsi_data to have the same dates
        price_data = price_data.loc[rsi_data.index]

        return price_data, rsi_data

    except Exception as e:
        print(f"Error calculating RSI for {ticker_symbol}: {e}")
        return None, None


def get_rsi_signals(
    ticker_symbol: str,
    window: int = 14,
    overbought: int = 70,
    oversold: int = 30,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Get RSI buy/sell signals for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    window : int, default=14
        The period for RSI calculation
    overbought : int, default=70
        Level considered overbought (sell signal when crossing down from above)
    oversold : int, default=30
        Level considered oversold (buy signal when crossing up from below)
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    list of dict: A list of signal events, where each signal is a dictionary with:
        - date: datetime of the signal
        - type: 'buy' or 'sell'
        - rsi: RSI value at signal
        - price: Closing price at signal
    """
    price_data, rsi_data = calculate_ticker_rsi(ticker_symbol, window, db_name, days)

    if rsi_data is None or len(rsi_data) == 0:
        return []

    # Create a DataFrame for easier processing
    data = pd.DataFrame({"rsi": rsi_data, "close": price_data["close"]})

    # Identify buy signals (crossing up through oversold level)
    buy_signals = (data["rsi"] > oversold) & (data["rsi"].shift() <= oversold)

    # Identify sell signals (crossing down through overbought level)
    sell_signals = (data["rsi"] < overbought) & (data["rsi"].shift() >= overbought)

    # Combine buy and sell signals into a list of dictionaries
    signals = []

    # Process buy signals
    for date in data[buy_signals].index:
        signals.append(
            {
                "date": date,
                "type": "buy",
                "rsi": data.loc[date, "rsi"],
                "price": data.loc[date, "close"],
            }
        )

    # Process sell signals
    for date in data[sell_signals].index:
        signals.append(
            {
                "date": date,
                "type": "sell",
                "rsi": data.loc[date, "rsi"],
                "price": data.loc[date, "close"],
            }
        )

    # Sort signals by date
    signals.sort(key=lambda x: x["date"])

    return signals


def get_latest_rsi_signal(
    ticker_symbol: str,
    window: int = 14,
    overbought: int = 70,
    oversold: int = 30,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Optional[Dict[str, Any]]:
    """
    Get only the most recent RSI signal for a ticker.

    Returns:
    --------
    dict or None: The most recent signal with date, type, rsi, and price.
                 Returns None if no signals are found.
    """
    signals = get_rsi_signals(
        ticker_symbol, window, overbought, oversold, db_name, days
    )

    # Return the most recent signal if any exist
    if signals:
        return signals[-1]
    return None


if __name__ == "__main__":
    # Prompt user for ticker input
    user_input = (
        input("Enter a ticker symbol (or 'give all' to check all NYSE tickers): ")
        .strip()
        .upper()
    )

    if user_input == "GIVE ALL":
        # Process all tickers from the file
        with open("nyse_tickers.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        print(f"Processing RSI signals for {len(tickers)} tickers...")

        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        no_signals = 0

        # Store results to sort by date later
        results = []

        # Process each ticker
        for ticker in tickers:
            latest_signal = get_latest_rsi_signal(ticker)

            if latest_signal:
                days_ago = (datetime.now().date() - latest_signal["date"].date()).days
                signal_type = latest_signal["type"].upper()

                if signal_type == "BUY":
                    buy_signals += 1
                else:
                    sell_signals += 1

                # Store result with days_ago for sorting
                results.append(
                    {
                        "ticker": ticker,
                        "signal_type": signal_type,
                        "days_ago": days_ago,
                        "price": latest_signal["price"],
                        "rsi": latest_signal["rsi"],
                        "has_signal": True,
                    }
                )
            else:
                no_signals += 1
                # Store tickers with no signals too
                results.append({"ticker": ticker, "has_signal": False})

        # Sort results - first put signals by days_ago (oldest first), then no-signal entries
        sorted_results = sorted(
            results,
            key=lambda x: x.get("days_ago", 0) if x["has_signal"] else -1,
            reverse=True,
        )

        # Print results in order (oldest first, most recent last)
        print("\nRSI Signals (oldest at top, most recent at bottom):")
        for result in sorted_results:
            if result["has_signal"]:
                print(
                    f"{result['ticker']}: {result['signal_type']} signal {result['days_ago']} days ago at ${result['price']:.2f} (RSI: {result['rsi']:.2f})"
                )
            else:
                print(f"{result['ticker']}: No recent RSI signals")

        # Print summary
        print(f"\nSummary:")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"No signals: {no_signals}")

    else:
        # Process single ticker
        ticker = user_input

        # Get RSI data
        price_data, rsi_data = calculate_ticker_rsi(ticker)

        # Show RSI values if available
        if rsi_data is not None:
            print(f"\nRecent RSI values for {ticker}:")
            print(rsi_data.tail())

            # Get the current RSI value
            current_rsi = rsi_data.iloc[-1]
            print(f"\nCurrent RSI: {current_rsi:.2f}")

            # Interpret the current RSI value
            if current_rsi > 70:
                print("Status: OVERBOUGHT - Potential sell signal")
            elif current_rsi < 30:
                print("Status: OVERSOLD - Potential buy signal")
            else:
                print("Status: NEUTRAL")

        # Get the latest signal
        latest_signal = get_latest_rsi_signal(ticker)

        # Display the latest signal if found
        if latest_signal:
            date_str = latest_signal["date"].strftime("%Y-%m-%d")
            signal_type = latest_signal["type"].upper()
            print(f"\nLatest RSI signal for {ticker}:")
            print(f"{date_str}: {signal_type}")
            print(f"RSI: {latest_signal['rsi']:.2f}")
            print(f"Price: ${latest_signal['price']:.2f}")

            # Calculate days since the signal
            days_ago = (datetime.now().date() - latest_signal["date"].date()).days
            print(f"Signal occurred {days_ago} days ago")
        else:
            print(f"\nNo RSI signals found for {ticker} (crossing 30/70 thresholds)")
