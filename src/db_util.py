import sqlite3
from datetime import datetime, timedelta

import pandas as pd


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
