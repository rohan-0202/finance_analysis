import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_ema(series, span):
    """Calculate the Exponential Moving Average for a series."""
    return series.ewm(span=span, adjust=False).mean()


def get_historical_data(ticker_symbol, db_name="stock_data.db", days=365):
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
        query, 
        conn, 
        params=(ticker_symbol, start_date.strftime('%Y-%m-%d'))
    )
    
    conn.close()
    
    if df.empty:
        raise ValueError(f"No historical data found for {ticker_symbol}")
    
    # Parse the timestamp column manually to avoid timezone issues
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df


def calculate_macd(ticker_symbol, fast_period=12, slow_period=26, signal_period=9, 
                  db_name="stock_data.db", days=365):
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
        ema_fast = calculate_ema(price_data['close'], fast_period)
        ema_slow = calculate_ema(price_data['close'], slow_period)
        
        # Calculate MACD line
        macd_data['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        macd_data['signal'] = calculate_ema(macd_data['macd'], signal_period)
        
        # Calculate histogram
        macd_data['histogram'] = macd_data['macd'] - macd_data['signal']
        
        # Drop NaN values caused by EMA calculations
        macd_data = macd_data.dropna()
        
        # Align price_data with macd_data to have the same dates
        price_data = price_data.loc[macd_data.index]
        
        return price_data, macd_data
    
    except Exception as e:
        print(f"Error calculating MACD for {ticker_symbol}: {e}")
        return None, None


def get_macd_signals(ticker_symbol, fast_period=12, slow_period=26, signal_period=9,
                    db_name="stock_data.db", days=365):
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
    buy_signals = (macd_data['macd'] > macd_data['signal']) & (macd_data['macd'].shift() <= macd_data['signal'].shift())
    sell_signals = (macd_data['macd'] < macd_data['signal']) & (macd_data['macd'].shift() >= macd_data['signal'].shift())
    
    # Combine buy and sell signals into a list of dictionaries
    signals = []
    
    # Process buy signals
    for date in macd_data[buy_signals].index:
        signals.append({
            'date': date,
            'type': 'buy',
            'macd': macd_data.loc[date, 'macd'],
            'signal': macd_data.loc[date, 'signal'],
            'price': price_data.loc[date, 'close']
        })
    
    # Process sell signals
    for date in macd_data[sell_signals].index:
        signals.append({
            'date': date,
            'type': 'sell',
            'macd': macd_data.loc[date, 'macd'],
            'signal': macd_data.loc[date, 'signal'],
            'price': price_data.loc[date, 'close']
        })
    
    # Sort signals by date
    signals.sort(key=lambda x: x['date'])
    
    return signals


def get_latest_macd_signal(ticker_symbol, fast_period=12, slow_period=26, signal_period=9,
                         db_name="stock_data.db", days=365):
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


if __name__ == "__main__":
    # Prompt user for ticker input
    user_input = input("Enter a ticker symbol (or 'give all' to check all NYSE tickers): ").strip().upper()
    
    if user_input == "GIVE ALL":
        # Process all tickers from the file - no try/except needed since file always exists
        with open("nyse_tickers.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        
        print(f"Processing MACD signals for {len(tickers)} tickers...")
        
        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        no_signals = 0
        
        # Process each ticker
        for ticker in tickers:
            latest_signal = get_latest_macd_signal(ticker)
            
            if latest_signal:
                days_ago = (datetime.now().date() - latest_signal['date'].date()).days
                signal_type = latest_signal['type'].upper()
                
                if signal_type == "BUY":
                    buy_signals += 1
                else:
                    sell_signals += 1
                    
                print(f"{ticker}: {signal_type} signal {days_ago} days ago at ${latest_signal['price']:.2f}")
            else:
                no_signals += 1
                print(f"{ticker}: No recent MACD signals")
        
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
            date_str = latest_signal['date'].strftime('%Y-%m-%d')
            signal_type = latest_signal['type'].upper()
            print(f"\nLatest MACD signal for {ticker}:")
            print(f"{date_str}: {signal_type}")
            print(f"MACD: {latest_signal['macd']:.4f}")
            print(f"Signal: {latest_signal['signal']:.4f}")
            print(f"Price: ${latest_signal['price']:.2f}")
            
            # Calculate days since the signal
            days_ago = (datetime.now().date() - latest_signal['date'].date()).days
            print(f"Signal occurred {days_ago} days ago")
        else:
            print(f"\nNo MACD signals found for {ticker}")
