import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# lol
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from db_util import get_historical_data

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=Warning)

# Set seaborn style defaults
sns.set_theme()  # Apply the default seaborn theme


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate the Exponential Moving Average for a series."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_macd(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculate the MACD (Moving Average Convergence Divergence) for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    fast_period : int, default=12
        The period for the fast EMA
    slow_period : int, default=26
        The period for the slow EMA
    signal_period : int, default=9
        The period for the signal line (EMA of MACD)
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    tuple: (price_data, macd_data)
        - price_data: DataFrame with original price data
        - macd_data: DataFrame with MACD calculations (macd, signal, histogram)
    """
    try:
        # Get historical data
        price_data = get_historical_data(ticker_symbol, db_name, days)

        # Create a new DataFrame for MACD data with the same index
        macd_data = pd.DataFrame(index=price_data.index)

        # Calculate EMAs
        ema_fast = calculate_ema(price_data["close"], fast_period)
        ema_slow = calculate_ema(price_data["close"], slow_period)

        # Calculate MACD line
        macd_data["macd"] = ema_fast - ema_slow

        # Calculate signal line
        macd_data["signal"] = calculate_ema(macd_data["macd"], signal_period)

        # Calculate histogram
        macd_data["histogram"] = macd_data["macd"] - macd_data["signal"]

        # Drop NaN values caused by EMA calculations
        macd_data = macd_data.dropna()

        # Align price_data with macd_data to have the same dates
        price_data = price_data.loc[macd_data.index]

        return price_data, macd_data

    except Exception as e:
        print(f"Error calculating MACD for {ticker_symbol}: {e}")
        return None, None


def get_macd_signals(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Get MACD buy/sell signals for a given ticker.

    Returns:
    --------
    list of dict: A list of signal events, where each signal is a dictionary with:
        - date: datetime of the signal
        - type: 'buy' or 'sell'
        - macd: MACD value at signal
        - signal: Signal line value at signal
        - price: Closing price at signal
    """
    price_data, macd_data = calculate_macd(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )

    if macd_data is None or macd_data.empty:
        return []

    # Identify crossover points
    buy_signals = (macd_data["macd"] > macd_data["signal"]) & (
        macd_data["macd"].shift() <= macd_data["signal"].shift()
    )
    sell_signals = (macd_data["macd"] < macd_data["signal"]) & (
        macd_data["macd"].shift() >= macd_data["signal"].shift()
    )

    # Combine buy and sell signals into a list of dictionaries
    signals = []

    # Process buy signals
    for date in macd_data[buy_signals].index:
        signals.append(
            {
                "date": date,
                "type": "buy",
                "macd": macd_data.loc[date, "macd"],
                "signal": macd_data.loc[date, "signal"],
                "price": price_data.loc[date, "close"],
            }
        )

    # Process sell signals
    for date in macd_data[sell_signals].index:
        signals.append(
            {
                "date": date,
                "type": "sell",
                "macd": macd_data.loc[date, "macd"],
                "signal": macd_data.loc[date, "signal"],
                "price": price_data.loc[date, "close"],
            }
        )

    # Sort signals by date
    signals.sort(key=lambda x: x["date"])

    return signals


def get_latest_macd_signal(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Optional[Dict[str, Any]]:
    """
    Get only the most recent MACD signal for a ticker.

    Returns:
    --------
    dict or None: The most recent signal with date, type, macd, signal, and price.
                 Returns None if no signals are found.
    """
    signals = get_macd_signals(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )

    # Return the most recent signal if any exist
    if signals:
        return signals[-1]
    return None


def get_signal_stats_text(macd_data: pd.DataFrame) -> str:
    """
    Calculate and format statistics about MACD signals.

    Parameters:
    -----------
    macd_data : pd.DataFrame
        DataFrame with MACD data

    Returns:
    --------
    str
        Formatted text with signal statistics
    """
    # Count crossovers
    bullish_crossovers = 0
    bearish_crossovers = 0

    for i in range(1, len(macd_data)):
        # Bullish crossover (MACD crosses above Signal)
        if (
            macd_data["macd"].iloc[i - 1] < macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] > macd_data["signal"].iloc[i]
        ):
            bullish_crossovers += 1

        # Bearish crossover (MACD crosses below Signal)
        elif (
            macd_data["macd"].iloc[i - 1] > macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] < macd_data["signal"].iloc[i]
        ):
            bearish_crossovers += 1

    # Get current MACD values
    current_macd = macd_data["macd"].iloc[-1] if not macd_data.empty else 0
    current_signal = macd_data["signal"].iloc[-1] if not macd_data.empty else 0
    current_hist = macd_data["histogram"].iloc[-1] if not macd_data.empty else 0

    # Format the text
    stats_text = f"Bullish: {bullish_crossovers}, Bearish: {bearish_crossovers}\n"
    stats_text += f"Current: MACD={current_macd:.2f}, Signal={current_signal:.2f}"

    return stats_text


if __name__ == "__main__":
    # Prompt user for ticker input
    user_input = (
        input("Enter a ticker symbol (or 'give all' to check all NYSE tickers): ")
        .strip()
        .upper()
    )

    if user_input == "GIVE ALL":
        # Process all tickers from the file - no try/except needed since file always exists
        with open("nyse_tickers.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        print(f"Processing MACD signals for {len(tickers)} tickers...")

        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        no_signals = 0

        # Store results to sort by date later
        results = []

        # Process each ticker
        for ticker in tickers:
            latest_signal = get_latest_macd_signal(ticker)

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
        print("\nMACD Signals (oldest at top, most recent at bottom):")
        for result in sorted_results:
            if result["has_signal"]:
                print(
                    f"{result['ticker']}: {result['signal_type']} signal {result['days_ago']} days ago at ${result['price']:.2f}"
                )
            else:
                print(f"{result['ticker']}: No recent MACD signals")

        # Print summary
        print(f"\nSummary:")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"No signals: {no_signals}")

    else:
        # Process single ticker
        ticker = user_input

        # Get MACD data
        price_data, macd_data = calculate_macd(ticker)

        # Show MACD values if available
        if macd_data is not None:
            print(f"\nRecent MACD values for {ticker}:")
            print(macd_data.tail())

        # Get the latest signal
        latest_signal = get_latest_macd_signal(ticker)

        # Display the latest signal if found
        if latest_signal:
            date_str = latest_signal["date"].strftime("%Y-%m-%d")
            signal_type = latest_signal["type"].upper()
            print(f"\nLatest MACD signal for {ticker}:")
            print(f"{date_str}: {signal_type}")
            print(f"MACD: {latest_signal['macd']:.4f}")
            print(f"Signal: {latest_signal['signal']:.4f}")
            print(f"Price: ${latest_signal['price']:.2f}")

            # Calculate days since the signal
            days_ago = (datetime.now().date() - latest_signal["date"].date()).days
            print(f"Signal occurred {days_ago} days ago")
        else:
            print(f"\nNo MACD signals found for {ticker}")

        # Inform user that plotting is now in technical_graphs.py
        print(f"\nTo visualize MACD chart for {ticker}, please run technical_graphs.py")
