import random
import sqlite3
import time
from datetime import datetime, timedelta
from functools import wraps

import click
import yfinance as yf

from retry_utils import retry_on_rate_limit


def parse_db_datetime(date_str):
    """Parse datetime strings from the database into datetime objects, handling various formats."""
    if not date_str:
        return None

    if isinstance(date_str, datetime):
        return date_str

    try:
        # Try parsing with microseconds
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            # Try without microseconds
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # Try ISO format
                return datetime.fromisoformat(
                    date_str.replace("Z", "+00:00") if "Z" in date_str else date_str
                )
            except ValueError:
                # Try just the date
                return datetime.strptime(date_str, "%Y-%m-%d")


@retry_on_rate_limit()
def save_ticker_info(ticker_symbol, db_name="stock_data.db"):
    """Save essential ticker information to the database."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check when the ticker info was last updated
        cursor.execute(
            "SELECT last_updated FROM ticker_info WHERE ticker = ?", (ticker_symbol,)
        )
        result = cursor.fetchone()

        # Only update if we don't have data or it's older than 7 days
        if result and result[0]:
            last_updated = parse_db_datetime(result[0])
            if last_updated and (datetime.now() - last_updated).days < 7:
                print(
                    f"Ticker info for {ticker_symbol} already updated within the last 7 days. Skipping."
                )
                conn.close()
                return

        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

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
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check for the latest date in the database for this ticker
        cursor.execute(
            "SELECT MAX(timestamp) FROM historical_prices WHERE ticker = ?",
            (ticker_symbol,),
        )
        latest_date_result = cursor.fetchone()[0]

        # If we have data, get data only after that date
        if latest_date_result:
            latest_date = parse_db_datetime(latest_date_result)
            if latest_date:
                # Add one day to avoid duplicates
                start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
                print(
                    f"Fetching {ticker_symbol} historical data from {start_date} onwards"
                )

                # Use start parameter instead of period
                ticker = yf.Ticker(ticker_symbol)
                history = ticker.history(start=start_date, interval=interval)

                if history.empty:
                    print(
                        f"No new historical data for {ticker_symbol} since {latest_date}"
                    )
                    conn.close()
                    return
            else:
                # If parsing failed, fetch data for the specified period
                print(
                    f"Failed to parse latest date for {ticker_symbol}, fetching {period} of data"
                )
                ticker = yf.Ticker(ticker_symbol)
                history = ticker.history(period=period, interval=interval)
        else:
            # If no data exists, fetch data for the specified period
            print(f"No existing data for {ticker_symbol}, fetching {period} of data")
            ticker = yf.Ticker(ticker_symbol)
            history = ticker.history(period=period, interval=interval)

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
        if data_to_insert:
            cursor.executemany(
                """
            INSERT OR REPLACE INTO historical_prices 
            (ticker, timestamp, open, high, low, close, volume, dividends)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                data_to_insert,
            )

            conn.commit()
            print(
                f"{len(data_to_insert)} new records of historical data for {ticker_symbol} saved to database."
            )
        else:
            print(f"No new historical data to insert for {ticker_symbol}")

        conn.close()
    except Exception as e:
        print(f"Error saving historical data for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_financial_metrics(ticker_symbol, db_name="stock_data.db"):
    """Save key financial metrics to the database."""
    try:
        # Get the latest financial dates from the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check the latest annual and quarterly data
        cursor.execute(
            "SELECT MAX(timestamp) FROM financial_metrics WHERE ticker = ? AND is_quarterly = 0",
            (ticker_symbol,),
        )
        latest_annual_result = cursor.fetchone()[0]

        cursor.execute(
            "SELECT MAX(timestamp) FROM financial_metrics WHERE ticker = ? AND is_quarterly = 1",
            (ticker_symbol,),
        )
        latest_quarterly_result = cursor.fetchone()[0]

        # Convert to datetime if not None
        latest_annual = (
            parse_db_datetime(latest_annual_result) if latest_annual_result else None
        )
        latest_quarterly = (
            parse_db_datetime(latest_quarterly_result)
            if latest_quarterly_result
            else None
        )

        ticker = yf.Ticker(ticker_symbol)

        # Get financial data
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        quarterly_income = ticker.quarterly_income_stmt
        quarterly_balance = ticker.quarterly_balance_sheet

        # Get key statistics
        info = ticker.info

        # Process annual financials
        if not income_stmt.empty and not balance_sheet.empty:
            for date in income_stmt.columns:
                # Skip if we already have this data
                date_dt = (
                    date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
                )
                if latest_annual and date_dt <= latest_annual:
                    continue

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
                (ticker, timestamp, is_quarterly, revenue, earnings, eps, pe_ratio, debt_to_equity, operating_margin, roe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date_dt,
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
                # Skip if we already have this data
                date_dt = (
                    date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
                )
                if latest_quarterly and date_dt <= latest_quarterly:
                    continue

                # Extract similar metrics for quarterly data
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
                (ticker, timestamp, is_quarterly, revenue, earnings, eps, pe_ratio, debt_to_equity, operating_margin, roe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker_symbol,
                        date_dt,
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
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check the latest options update date
        cursor.execute(
            "SELECT MAX(last_updated) FROM options_data WHERE ticker = ?",
            (ticker_symbol,),
        )
        latest_update_result = cursor.fetchone()[0]

        # If we updated options data within the last day, skip
        if latest_update_result:
            latest_update = parse_db_datetime(latest_update_result)
            if latest_update and (datetime.now() - latest_update).days < 1:
                print(
                    f"Options data for {ticker_symbol} already updated within the last day. Skipping."
                )
                conn.close()
                return

        ticker = yf.Ticker(ticker_symbol)
        # Get available expiration dates
        all_dates = ticker.options

        if not all_dates:
            print(f"No options data available for {ticker_symbol}")
            conn.close()
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
            conn.close()
            return

        # Track options inserted/updated
        options_count = 0

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
                options_count += 1

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
                options_count += 1

        conn.commit()
        conn.close()
        print(
            f"{options_count} options records for {ticker_symbol} saved/updated in database."
        )
    except Exception as e:
        print(f"Error saving options data for {ticker_symbol}: {e}")
        raise


@retry_on_rate_limit()
def save_recent_news(ticker_symbol, db_name="stock_data.db"):
    """Save only recent news (last 7 days) to the database."""
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check the latest news date
        cursor.execute(
            "SELECT MAX(publish_date) FROM recent_news WHERE ticker = ?",
            (ticker_symbol,),
        )
        latest_news_date_result = cursor.fetchone()[0]

        latest_news_date = None
        if latest_news_date_result:
            latest_news_date = parse_db_datetime(latest_news_date_result)
            if latest_news_date:
                print(
                    f"Found news data for {ticker_symbol} up to {latest_news_date.strftime('%Y-%m-%d')}"
                )

        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news

        if not news:
            print(f"No news data available for {ticker_symbol}")
            conn.close()
            return

        # Filter to only include news from the last 7 days
        recent_cutoff = datetime.now() - timedelta(days=7)

        # If we have data in the database, only get newer news
        if latest_news_date and latest_news_date > recent_cutoff:
            recent_cutoff = latest_news_date

        recent_news = [
            item
            for item in news
            if datetime.fromtimestamp(item.get("providerPublishTime", 0))
            > recent_cutoff
        ]

        if not recent_news:
            print(f"No new news available for {ticker_symbol}")
            conn.close()
            return

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
        print(
            f"{len(recent_news)} new news items for {ticker_symbol} saved to database."
        )
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
@click.option(
    "--tickers-file", default="nyse_tickers.txt", help="File with ticker symbols"
)
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
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

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
            print(f"[{i + 1}/{len(tickers)}] Processing {ticker}...")
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
