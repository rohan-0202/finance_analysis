#!/usr/bin/env python
"""
Analyze Ticker History - Query and display ticker data from the database

This tool lets you view sentiment and news data for a ticker over a specific timeframe.
"""

import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("analyze_ticker_history")


class TickerHistoryAnalyzer:
    """Analyze historical ticker data from the database."""

    def __init__(self, db_path: str = os.getenv("DB_PATH", "finance_news.db")):
        """
        Initialize the Ticker History Analyzer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        # Ensure database exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        logger.info(f"Initialized with database: {db_path}")

    def get_ticker_analyses(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Get all analyses for a ticker within the specified timeframe.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for timeframe (None for no start limit)
            end_date: End date for timeframe (None for no end limit)

        Returns:
            List of analysis dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT * FROM analyses 
                WHERE ticker = ?
                """
                params = [ticker]

                # Add date filters if provided
                if start_date:
                    query += " AND analyzed_at >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND analyzed_at <= ?"
                    params.append(end_date.isoformat())

                # Order by date
                query += " ORDER BY analyzed_at"

                cursor.execute(query, params)
                analyses = [dict(row) for row in cursor.fetchall()]

                # Get related data for each analysis
                for analysis in analyses:
                    analysis_id = analysis["id"]

                    # Get key points
                    cursor.execute(
                        "SELECT point FROM key_points WHERE analysis_id = ?",
                        (analysis_id,),
                    )
                    analysis["key_points"] = [row["point"] for row in cursor.fetchall()]

                    # Get risk factors
                    cursor.execute(
                        "SELECT factor FROM risk_factors WHERE analysis_id = ?",
                        (analysis_id,),
                    )
                    analysis["risk_factors"] = [
                        row["factor"] for row in cursor.fetchall()
                    ]

                    # Get catalysts
                    cursor.execute(
                        "SELECT catalyst FROM catalysts WHERE analysis_id = ?",
                        (analysis_id,),
                    )
                    analysis["catalysts"] = [
                        row["catalyst"] for row in cursor.fetchall()
                    ]

                logger.info(
                    f"Found {len(analyses)} analyses for {ticker} in specified timeframe"
                )
                return analyses

        except Exception as e:
            logger.error(f"Error getting analyses for {ticker}: {e}")
            return []

    def get_ticker_news(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get news articles for a ticker within the specified timeframe.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for timeframe (None for no start limit)
            end_date: End date for timeframe (None for no end limit)
            limit: Maximum number of articles to return

        Returns:
            List of news article dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT * FROM news_articles 
                WHERE ticker = ?
                """
                params = [ticker]

                # Add date filters if provided
                if start_date:
                    query += " AND publish_date >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND publish_date <= ?"
                    params.append(end_date.isoformat())

                # Order by date and limit
                query += f" ORDER BY publish_date DESC LIMIT {limit}"

                cursor.execute(query, params)
                news = [dict(row) for row in cursor.fetchall()]

                logger.info(
                    f"Found {len(news)} news articles for {ticker} in specified timeframe"
                )
                return news

        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []

    def get_sentiment_trend(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[datetime, str, str]]:
        """
        Get sentiment trend for a ticker within the specified timeframe.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for timeframe (None for no start limit)
            end_date: End date for timeframe (None for no end limit)

        Returns:
            List of (date, sentiment, confidence) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                SELECT analyzed_at, sentiment, confidence FROM analyses 
                WHERE ticker = ?
                """
                params = [ticker]

                # Add date filters if provided
                if start_date:
                    query += " AND analyzed_at >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND analyzed_at <= ?"
                    params.append(end_date.isoformat())

                # Order by date
                query += " ORDER BY analyzed_at"

                cursor.execute(query, params)
                trend_data = []

                for row in cursor.fetchall():
                    # Convert ISO format date string to datetime object
                    try:
                        analyzed_at = datetime.fromisoformat(row[0])
                    except ValueError:
                        # Handle non-standard ISO format (e.g., with Z at the end)
                        analyzed_at = datetime.fromisoformat(
                            row[0].replace("Z", "+00:00")
                        )

                    sentiment = row[1]
                    confidence = row[2]
                    trend_data.append((analyzed_at, sentiment, confidence))

                return trend_data

        except Exception as e:
            logger.error(f"Error getting sentiment trend for {ticker}: {e}")
            return []

    def analyze_sentiment_distribution(self, analyses: List[Dict]) -> Dict:
        """
        Get the distribution of sentiment in analyses.

        Args:
            analyses: List of analysis dictionaries

        Returns:
            Dictionary with sentiment distribution
        """
        # Count sentiment distribution
        sentiment_counts = Counter()
        for analysis in analyses:
            sentiment = analysis.get("sentiment", "neutral")
            sentiment_counts[sentiment] += 1

        # Format results
        total_analyses = len(analyses)
        sentiment_distribution = {
            "bullish": f"{sentiment_counts['bullish']} ({sentiment_counts['bullish'] * 100 / total_analyses:.1f}%)"
            if total_analyses > 0
            else "0 (0%)",
            "neutral": f"{sentiment_counts['neutral']} ({sentiment_counts['neutral'] * 100 / total_analyses:.1f}%)"
            if total_analyses > 0
            else "0 (0%)",
            "bearish": f"{sentiment_counts['bearish']} ({sentiment_counts['bearish'] * 100 / total_analyses:.1f}%)"
            if total_analyses > 0
            else "0 (0%)",
        }

        return sentiment_distribution

    def display_ticker_report(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Display a report for a ticker on the console.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for timeframe (None for no start limit)
            end_date: End date for timeframe (None for no end limit)
        """
        # Get all the data
        analyses = self.get_ticker_analyses(ticker, start_date, end_date)
        news = self.get_ticker_news(ticker, start_date, end_date)
        # trend_data = self.get_sentiment_trend(ticker, start_date, end_date)

        # Generate sentiment distribution
        sentiment_distribution = self.analyze_sentiment_distribution(analyses)

        # Print the report
        print(f"=== TICKER HISTORY REPORT: {ticker} ===")
        print("")

        # Timeframe
        print("TIMEFRAME:")
        if start_date:
            print(f"  From: {start_date.strftime('%Y-%m-%d')}")
        else:
            print("  From: Beginning of records")

        if end_date:
            print(f"  To: {end_date.strftime('%Y-%m-%d')}")
        else:
            print("  To: Present")
        print("")

        # Summary statistics
        print("SUMMARY:")
        print(f"  Total Analyses: {len(analyses)}")
        print(f"  Total News Articles: {len(news)}")
        print("")

        # Sentiment distribution
        print("SENTIMENT DISTRIBUTION:")
        for sentiment, count in sentiment_distribution.items():
            print(f"  {sentiment.capitalize()}: {count}")
        print("")

        # Latest analyses
        print("LATEST ANALYSES:")
        for analysis in analyses[-3:]:
            analysis_date = datetime.fromisoformat(
                analysis["analyzed_at"].replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
            print(
                f"  {analysis_date} - Sentiment: {analysis['sentiment']}, Confidence: {analysis['confidence']}"
            )
            print(f"    Summary: {analysis['summary'][:100]}...")
            if analysis["key_points"]:
                print(f"    Key Point: {analysis['key_points'][0]}")
            print("")

        # Latest news
        print("LATEST NEWS:")
        for article in news[:5]:
            if "publish_date" in article and article["publish_date"]:
                try:
                    pub_date = datetime.fromisoformat(
                        article["publish_date"].replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                except ValueError:
                    pub_date = article["publish_date"]
            else:
                pub_date = "Unknown date"

            print(f"  {pub_date} - {article['title']}")
            print(f"    Source: {article['source']}")
            if article.get("url"):
                print(f"    URL: {article['url']}")
            print("")


@click.group()
def cli():
    """
    Analyze ticker history from the finance news database.

    This tool lets you view sentiment and news for a ticker over time.
    """
    pass


@cli.command("report")
@click.argument("ticker")
@click.option("--days", default=30, type=int, help="Number of days to look back")
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (format: YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (format: YYYY-MM-DD)",
)
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
def display_report(ticker, days, start_date, end_date, db_path):
    """Display a report for a ticker."""
    ticker = ticker.upper()

    # Calculate start date if days is provided and start_date is not
    if not start_date and days > 0:
        start_date = datetime.now() - timedelta(days=days)

    click.echo(f"Generating report for {ticker}...")
    if start_date:
        click.echo(f"From {start_date.strftime('%Y-%m-%d')}")
    if end_date:
        click.echo(f"To {end_date.strftime('%Y-%m-%d')}")

    analyzer = TickerHistoryAnalyzer(db_path=db_path)
    analyzer.display_ticker_report(ticker, start_date, end_date)


@cli.command("news")
@click.argument("ticker")
@click.option("--days", default=7, type=int, help="Number of days to look back")
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (format: YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (format: YYYY-MM-DD)",
)
@click.option(
    "--limit", default=10, type=int, help="Maximum number of news articles to display"
)
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
def list_news(ticker, days, start_date, end_date, limit, db_path):
    """List news articles for a ticker."""
    ticker = ticker.upper()

    # Calculate start date if days is provided and start_date is not
    if not start_date and days > 0:
        start_date = datetime.now() - timedelta(days=days)

    click.echo(f"Listing news for {ticker}...")

    analyzer = TickerHistoryAnalyzer(db_path=db_path)
    news = analyzer.get_ticker_news(ticker, start_date, end_date, limit=limit)

    if not news:
        click.echo(f"No news found for {ticker} in the specified timeframe.")
        return

    click.echo(f"\nFound {len(news)} news articles for {ticker}:")
    for i, article in enumerate(news):
        if "publish_date" in article and article["publish_date"]:
            try:
                pub_date = datetime.fromisoformat(
                    article["publish_date"].replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
            except ValueError:
                pub_date = article["publish_date"]
        else:
            pub_date = "Unknown date"

        click.echo(f"\n[{i + 1}] {article['title']}")
        click.echo(f"Date: {pub_date} | Source: {article['source']}")
        if article.get("url"):
            click.echo(f"URL: {article['url']}")
        if article.get("summary"):
            click.echo(f"Summary: {article['summary'][:150]}...")


@cli.command("sentiment")
@click.argument("ticker")
@click.option("--days", default=30, type=int, help="Number of days to look back")
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (format: YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (format: YYYY-MM-DD)",
)
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
def display_sentiment(ticker, days, start_date, end_date, db_path):
    """Display sentiment history for a ticker."""
    ticker = ticker.upper()

    # Calculate start date if days is provided and start_date is not
    if not start_date and days > 0:
        start_date = datetime.now() - timedelta(days=days)

    click.echo(f"Retrieving sentiment history for {ticker}...")

    analyzer = TickerHistoryAnalyzer(db_path=db_path)
    trend_data = analyzer.get_sentiment_trend(ticker, start_date, end_date)

    if not trend_data:
        click.echo(f"No sentiment data found for {ticker} in the specified timeframe.")
        return

    click.echo(f"\nSentiment history for {ticker}:")
    for date, sentiment, confidence in trend_data:
        date_str = date.strftime("%Y-%m-%d")
        click.echo(f"  {date_str}: {sentiment.capitalize()} (confidence: {confidence})")

    # Calculate overall sentiment distribution
    bullish = sum(1 for _, s, _ in trend_data if s == "bullish")
    bearish = sum(1 for _, s, _ in trend_data if s == "bearish")
    neutral = sum(1 for _, s, _ in trend_data if s == "neutral")
    total = len(trend_data)

    click.echo("\nSentiment Distribution:")
    if total > 0:
        click.echo(f"  Bullish: {bullish} ({bullish * 100 / total:.1f}%)")
        click.echo(f"  Neutral: {neutral} ({neutral * 100 / total:.1f}%)")
        click.echo(f"  Bearish: {bearish} ({bearish * 100 / total:.1f}%)")


if __name__ == "__main__":
    cli()
