import random
import sqlite3
import time
from datetime import datetime, timedelta
from functools import wraps

import click
import yfinance as yf

from retry_utils import retry_on_rate_limit


@retry_on_rate_limit()
def save_ticker_info(ticker_symbol, db_name="stock_data.db"):
    """Save essential ticker information to the database."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO ticker_info 
        (ticker, company_name, sector, industry, market_cap, currency, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ticker_symbol,
                info.get("shortName", None),
                info.get("sector", None),
                info.get("industry", None),
                info.get("marketCap", None),
                info.get("currency", None),
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()
        print(f"Basic info for {ticker_symbol} saved to database.")
    except Exception as e:
        print(f"Error saving ticker info for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_historical_data(
    ticker_symbol, period="2y", interval="1d", db_name="stock_data.db"
):
    """Save historical price data to the database."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period=period, interval=interval)

        conn = sqlite3.connect(db_name)

        # Prepare data for insertion
        data_to_insert = []
        for date_idx, row in history.iterrows():
            data_to_insert.append(
                (
                    ticker_symbol,
                    date_idx.to_pydatetime(),  # Convert Timestamp to datetime
                    row.get("Open", None),
                    row.get("High", None),
                    row.get("Low", None),
                    row.get("Close", None),
                    row.get("Volume", None),
                    row.get("Dividends", None),
                )
            )

        # Insert data
        cursor = conn.cursor()
        cursor.executemany(
            """
        INSERT OR REPLACE INTO historical_prices 
        (ticker, date, open, high, low, close, volume, dividends)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data_to_insert,
        )

        conn.commit()
        conn.close()
        print(
            f"Historical data for {ticker_symbol} ({period}, {interval}) saved to database."
        )
    except Exception as e:
        print(f"Error saving historical data for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_financial_metrics(ticker_symbol, db_name="stock_data.db"):
    """Save key financial metrics to the database."""
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Get financial data
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        quarterly_income = ticker.quarterly_income_stmt
        quarterly_balance = ticker.quarterly_balance_sheet

        # Get key statistics
        info = ticker.info

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Process annual financials
        if not income_stmt.empty and not balance_sheet.empty:
            for date in income_stmt.columns:
                # Calculate key metrics
                revenue = (
                    income_stmt.loc["Total Revenue", date]
                    if "Total Revenue" in income_stmt.index
                    else None
                )
                net_income = (
                    income_stmt.loc["Net Income", date]
                    if "Net Income" in income_stmt.index
                    else None
                )

                # Find closest balance sheet date
                balance_date = (
                    min(balance_sheet.columns, key=lambda x: abs((x - date).days))
                    if balance_sheet.columns.size > 0
                    else None
                )

                if balance_date is not None:
                    total_equity = (
                        balance_sheet.loc["Total Stockholder Equity", balance_date]
                        if "Total Stockholder Equity" in balance_sheet.index
                        else None
                    )
                    total_debt = (
                        balance_sheet.loc["Total Debt", balance_date]
                        if "Total Debt" in balance_sheet.index
                        else None
                    )
                    debt_to_equity = (
                        (total_debt / total_equity)
                        if (total_equity and total_debt)
                        else None
                    )
                else:
                    debt_to_equity = None

                # Calculate other metrics
                shares_outstanding = info.get("sharesOutstanding", None)
                eps = (
                    (net_income / shares_outstanding)
                    if (net_income and shares_outstanding)
                    else None
                )
                current_price = info.get("currentPrice", None)
                pe_ratio = (current_price / eps) if (current_price and eps) else None

                operating_income = (
                    income_stmt.loc["Operating Income", date]
                    if "Operating Income" in income_stmt.index
                    else None
                )
                operating_margin = (
                    (operating_income / revenue)
                    if (operating_income and revenue)
                    else None
                )

                roe = (
                    (net_income / total_equity)
                    if (net_income and total_equity)
                    else None
                )

                cursor.execute(
                    """
                INSERT OR REPLACE INTO financial_metrics 
                (ticker, date, is_quarterly, revenue, earnings, eps, pe_ratio, debt_to_equity, operating_margin, roe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date.to_pydatetime()
                        if hasattr(date, "to_pydatetime")
                        else date,
                        False,  # Not quarterly
                        revenue,
                        net_income,
                        eps,
                        pe_ratio,
                        debt_to_equity,
                        operating_margin,
                        roe,
                    ),
                )

        # Process quarterly financials
        if not quarterly_income.empty and not quarterly_balance.empty:
            for date in quarterly_income.columns:
                # Extract similar metrics for quarterly data
                # (Similar calculations as above, but for quarterly statements)
                revenue = (
                    quarterly_income.loc["Total Revenue", date]
                    if "Total Revenue" in quarterly_income.index
                    else None
                )
                net_income = (
                    quarterly_income.loc["Net Income", date]
                    if "Net Income" in quarterly_income.index
                    else None
                )

                # Find closest balance sheet date
                balance_date = (
                    min(quarterly_balance.columns, key=lambda x: abs((x - date).days))
                    if quarterly_balance.columns.size > 0
                    else None
                )

                if balance_date is not None:
                    total_equity = (
                        quarterly_balance.loc["Total Stockholder Equity", balance_date]
                        if "Total Stockholder Equity" in quarterly_balance.index
                        else None
                    )
                    total_debt = (
                        quarterly_balance.loc["Total Debt", balance_date]
                        if "Total Debt" in quarterly_balance.index
                        else None
                    )
                    debt_to_equity = (
                        (total_debt / total_equity)
                        if (total_equity and total_debt)
                        else None
                    )
                else:
                    debt_to_equity = None

                # Calculate other metrics
                shares_outstanding = info.get("sharesOutstanding", None)
                eps = (
                    (net_income / shares_outstanding)
                    if (net_income and shares_outstanding)
                    else None
                )
                current_price = info.get("currentPrice", None)
                pe_ratio = (current_price / eps) if (current_price and eps) else None

                operating_income = (
                    quarterly_income.loc["Operating Income", date]
                    if "Operating Income" in quarterly_income.index
                    else None
                )
                operating_margin = (
                    (operating_income / revenue)
                    if (operating_income and revenue)
                    else None
                )

                roe = (
                    (net_income / total_equity)
                    if (net_income and total_equity)
                    else None
                )

                cursor.execute(
                    """
                INSERT OR REPLACE INTO financial_metrics 
                (ticker, date, is_quarterly, revenue, earnings, eps, pe_ratio, debt_to_equity, operating_margin, roe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date.to_pydatetime()
                        if hasattr(date, "to_pydatetime")
                        else date,
                        True,  # Is quarterly
                        revenue,
                        net_income,
                        eps,
                        pe_ratio,
                        debt_to_equity,
                        operating_margin,
                        roe,
                    ),
                )

        conn.commit()
        conn.close()
        print(f"Financial metrics for {ticker_symbol} saved to database.")
    except Exception as e:
        print(f"Error saving financial metrics for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_options_data(ticker_symbol, db_name="stock_data.db"):
    """Save options data for the next 6 months to the database."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Get available expiration dates
        all_dates = ticker.options

        if not all_dates:
            print(f"No options data available for {ticker_symbol}")
            return

        # Filter to include only expirations within the next 6 months
        six_months_later = (datetime.now() + timedelta(days=180)).timestamp()
        dates = [
            date
            for date in all_dates
            if datetime.strptime(date, "%Y-%m-%d").timestamp() <= six_months_later
        ]

        if not dates:
            print(f"No options data within 6 months available for {ticker_symbol}")
            return

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        for date in dates:
            options = ticker.option_chain(date)

            # Filter options to only include those with decent liquidity
            calls = options.calls[options.calls["volume"] > 10]
            puts = options.puts[options.puts["volume"] > 10]

            # Process calls
            for _, row in calls.iterrows():
                cursor.execute(
                    """
                INSERT OR REPLACE INTO options_data 
                (ticker, expiration_date, option_type, strike, last_price, bid, ask, 
                volume, open_interest, implied_volatility, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date,
                        "call",
                        row.get("strike", None),
                        row.get("lastPrice", None),
                        row.get("bid", None),
                        row.get("ask", None),
                        row.get("volume", None),
                        row.get("openInterest", None),
                        row.get("impliedVolatility", None),
                        datetime.now(),
                    ),
                )

            # Process puts
            for _, row in puts.iterrows():
                cursor.execute(
                    """
                INSERT OR REPLACE INTO options_data 
                (ticker, expiration_date, option_type, strike, last_price, bid, ask, 
                volume, open_interest, implied_volatility, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date,
                        "put",
                        row.get("strike", None),
                        row.get("lastPrice", None),
                        row.get("bid", None),
                        row.get("ask", None),
                        row.get("volume", None),
                        row.get("openInterest", None),
                        row.get("impliedVolatility", None),
                        datetime.now(),
                    ),
                )

        conn.commit()
        conn.close()
        print(f"Options data for {ticker_symbol} saved to database.")
    except Exception as e:
        print(f"Error saving options data for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_recent_news(ticker_symbol, db_name="stock_data.db"):
    """Save only recent news (last 7 days) to the database."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news

        if not news:
            print(f"No news data available for {ticker_symbol}")
            return

        # Filter to only include news from the last 7 days
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_news = [
            item
            for item in news
            if datetime.fromtimestamp(item.get("providerPublishTime", 0))
            >= recent_cutoff
        ]

        if not recent_news:
            print(f"No recent news available for {ticker_symbol}")
            return

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        for item in recent_news:
            cursor.execute(
                """
            INSERT OR IGNORE INTO recent_news 
            (ticker, title, publish_date)
            VALUES (?, ?, ?)
            """,
                (
                    ticker_symbol,
                    item.get("title", None),
                    datetime.fromtimestamp(item.get("providerPublishTime", 0)),
                ),
            )

        conn.commit()
        conn.close()
        print(f"Recent news for {ticker_symbol} saved to database.")
    except Exception as e:
        print(f"Error saving recent news for {ticker_symbol}: {e}")
        raise


def save_all_data_for_ticker(ticker_symbol, db_name="stock_data.db"):
    """Save all essential trading data for a given ticker to the database."""
    print(f"Saving essential trading data for {ticker_symbol}...")
    try:
        save_ticker_info(ticker_symbol, db_name)
        save_historical_data(ticker_symbol, period="2y", db_name=db_name)
        save_financial_metrics(ticker_symbol, db_name)
        save_options_data(ticker_symbol, db_name)
        save_recent_news(ticker_symbol, db_name)
        print(
            f"All essential trading data for {ticker_symbol} has been saved to {db_name}."
        )
        return True
    except Exception as e:
        print(f"Failed to complete data collection for {ticker_symbol}: {e}")
        return False


@click.command()
@click.option("--tickers-file", default="nyse_tickers.txt", help="File with ticker symbols")
@click.option("--limit", default=None, type=int, help="Limit tickers to process")
@click.option("--delay", default=2.0, type=float, help="Delay between tickers")
@click.option("--resume", default=None, help="Resume from a ticker")
def main(tickers_file, limit, delay, resume):
    """Process stock data for tickers."""
    # Read tickers
    try:
        with open(tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {tickers_file}")
    except FileNotFoundError:
        print(f"{tickers_file} not found. Using default tickers.")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    # Apply limit
    if limit is not None and limit < len(tickers):
        tickers = tickers[:limit]

    # Resume point
    start_idx = 0
    if resume and resume.upper() in tickers:
        start_idx = tickers.index(resume.upper())
        print(f"Resuming from {resume}")

    # Process tickers
    succeeded, failed = 0, 0
    for i, ticker in enumerate(tickers[start_idx:], start_idx):
        try:
            print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
            if save_all_data_for_ticker(ticker):
                succeeded += 1
            else:
                failed += 1
            if i < len(tickers) - 1:
                time.sleep(delay)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            failed += 1

    print(f"\nProcessing complete! Succeeded: {succeeded}, Failed: {failed}")


if __name__ == "__main__":
    main()
