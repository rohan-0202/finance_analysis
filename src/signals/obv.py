import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from db_util import get_historical_data

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=Warning)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV) for a price series.

    Parameters:
    -----------
    close : pd.Series
        Close price series
    volume : pd.Series
        Volume series

    Returns:
    --------
    pd.Series: OBV values
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]  # Initialize with first day's volume
    
    # Calculate OBV based on price movement
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            # Price up, add volume
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            # Price down, subtract volume
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            # Price unchanged, OBV unchanged
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_ticker_obv(
    ticker_symbol: str,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Calculate the OBV for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    tuple: (price_data, obv_data)
        - price_data: DataFrame with original price data
        - obv_data: Series with OBV values
    """
    try:
        # Get historical data
        price_data = get_historical_data(ticker_symbol, db_name, days)
        
        # Calculate OBV
        obv_data = calculate_obv(price_data["close"], price_data["volume"])
        
        return price_data, obv_data

    except Exception as e:
        print(f"Error calculating OBV for {ticker_symbol}: {e}")
        return None, None


def get_obv_signals(
    ticker_symbol: str,
    window: int = 20,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Get OBV buy/sell signals for a given ticker based on divergence.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol
    window : int, default=20
        The lookback window for detecting divergences
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    list of dict: A list of signal events, where each signal is a dictionary with:
        - date: datetime of the signal
        - type: 'buy' or 'sell'
        - obv: OBV value at signal
        - price: Closing price at signal
    """
    price_data, obv_data = calculate_ticker_obv(ticker_symbol, db_name, days)

    if obv_data is None or len(obv_data) == 0:
        return []

    # Create a DataFrame for easier processing
    data = pd.DataFrame({"obv": obv_data, "close": price_data["close"]})
    
    # Calculate moving averages for smoothing
    data["obv_ma"] = data["obv"].rolling(window=window).mean()
    data["price_ma"] = data["close"].rolling(window=window).mean()
    
    # Calculate rate of change for both price and OBV
    data["obv_roc"] = data["obv_ma"].pct_change(periods=window)
    data["price_roc"] = data["price_ma"].pct_change(periods=window)
    
    # Drop NaN values
    data = data.dropna()
    
    # Identify bullish divergence (price down, OBV up)
    bullish_divergence = (data["price_roc"] < 0) & (data["obv_roc"] > 0)
    
    # Identify bearish divergence (price up, OBV down)
    bearish_divergence = (data["price_roc"] > 0) & (data["obv_roc"] < 0)
    
    # Filter signals to prevent clustering (only consider signals that are at least window days apart)
    signals = []
    last_signal_date = None
    
    # Process buy signals (bullish divergence)
    for date in data[bullish_divergence].index:
        if last_signal_date is None or (date - last_signal_date).days > window:
            signals.append(
                {
                    "date": date,
                    "type": "buy",
                    "obv": data.loc[date, "obv"],
                    "price": data.loc[date, "close"],
                }
            )
            last_signal_date = date
    
    # Reset for sell signals
    last_signal_date = None
    
    # Process sell signals (bearish divergence)
    for date in data[bearish_divergence].index:
        if last_signal_date is None or (date - last_signal_date).days > window:
            signals.append(
                {
                    "date": date,
                    "type": "sell",
                    "obv": data.loc[date, "obv"],
                    "price": data.loc[date, "close"],
                }
            )
            last_signal_date = date
    
    # Sort signals by date
    signals.sort(key=lambda x: x["date"])
    
    return signals


def get_latest_obv_signal(
    ticker_symbol: str,
    window: int = 20,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Optional[Dict[str, Any]]:
    """
    Get only the most recent OBV signal for a ticker.

    Returns:
    --------
    dict or None: The most recent signal with date, type, obv, and price.
                 Returns None if no signals are found.
    """
    signals = get_obv_signals(ticker_symbol, window, db_name, days)

    # Return the most recent signal if any exist
    if signals:
        return signals[-1]
    return None


def get_obv_status_text(price_data: pd.DataFrame, obv_data: pd.Series) -> str:
    """
    Generate a text description of the current OBV status.
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data with at least close prices
    obv_data : pd.Series
        OBV values
        
    Returns:
    --------
    str: Text description of OBV status
    """
    if len(obv_data) < 20:
        return "Insufficient data for OBV analysis"
    
    # Get recent values
    current_obv = obv_data.iloc[-1]
    prev_obv = obv_data.iloc[-2]
    current_close = price_data["close"].iloc[-1]
    prev_close = price_data["close"].iloc[-2]
    
    # Calculate short-term trends (5-day)
    obv_5d_change = (obv_data.iloc[-1] - obv_data.iloc[-5]) / abs(obv_data.iloc[-5]) * 100
    price_5d_change = (price_data["close"].iloc[-1] - price_data["close"].iloc[-5]) / price_data["close"].iloc[-5] * 100
    
    # Generate status text
    status_text = f"Current OBV: {current_obv:,.0f}, 5-day change: {obv_5d_change:.2f}%\n"
    
    # Check for divergence
    if (obv_5d_change > 0 and price_5d_change < 0):
        status_text += "BULLISH DIVERGENCE: OBV rising while price falling"
    elif (obv_5d_change < 0 and price_5d_change > 0):
        status_text += "BEARISH DIVERGENCE: OBV falling while price rising"
    elif (current_obv > prev_obv and current_close > prev_close):
        status_text += "CONFIRMING UPTREND: Both OBV and price rising"
    elif (current_obv < prev_obv and current_close < prev_close):
        status_text += "CONFIRMING DOWNTREND: Both OBV and price falling"
    else:
        status_text += "NEUTRAL: No clear signal"
    
    return status_text


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

        print(f"Processing OBV signals for {len(tickers)} tickers...")

        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        no_signals = 0

        # Store results to sort by date later
        results = []

        # Process each ticker
        for ticker in tickers:
            latest_signal = get_latest_obv_signal(ticker)

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
                        "obv": latest_signal["obv"],
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
        print("\nOBV Signals (oldest at top, most recent at bottom):")
        for result in sorted_results:
            if result["has_signal"]:
                print(
                    f"{result['ticker']}: {result['signal_type']} signal {result['days_ago']} days ago at ${result['price']:.2f} (OBV: {result['obv']:,.0f})"
                )
            else:
                print(f"{result['ticker']}: No recent OBV signals")

        # Print summary
        print("\nSummary:")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"No signals: {no_signals}")

    else:
        # Process single ticker
        ticker = user_input

        # Get OBV data
        price_data, obv_data = calculate_ticker_obv(ticker)

        # Show OBV values if available
        if obv_data is not None:
            print(f"\nRecent OBV values for {ticker}:")
            print(obv_data.tail())

            # Get the current status
            status_text = get_obv_status_text(price_data, obv_data)
            print(f"\n{status_text}")

        # Get the latest signal
        latest_signal = get_latest_obv_signal(ticker)

        # Display the latest signal if found
        if latest_signal:
            date_str = latest_signal["date"].strftime("%Y-%m-%d")
            signal_type = latest_signal["type"].upper()
            print(f"\nLatest OBV signal for {ticker}:")
            print(f"{date_str}: {signal_type}")
            print(f"OBV: {latest_signal['obv']:,.0f}")
            print(f"Price: ${latest_signal['price']:.2f}")

            # Calculate days since the signal
            days_ago = (datetime.now().date() - latest_signal["date"].date()).days
            print(f"Signal occurred {days_ago} days ago")
        else:
            print(f"\nNo OBV divergence signals found for {ticker}") 