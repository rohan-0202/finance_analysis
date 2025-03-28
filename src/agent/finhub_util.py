import logging
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


def fetch_news_finnhub(
    ticker: str,
    finnhub_key: str,
    max_results: int = 5,
    days_lookback: int = 7,
) -> List[Dict]:
    """
    Fetch financial news for a ticker from Finnhub.

    Args:
        ticker: Stock ticker symbol
        max_results: Maximum number of results to return

    Returns:
        List of news article dictionaries
    """

    logger.info(f"Fetching news for {ticker} (max: {max_results})")

    # Calculate date range - from X days ago to today
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": finnhub_key,
    }

    try:
        # Remove timeout to prevent truncating requests
        response = requests.get(url, params=params)

        if response.status_code != 200:
            logger.error(f"Error fetching news for {ticker}: {response.status_code}")
            return []

        news_data = response.json()

        # Filter and clean the data
        if news_data and isinstance(news_data, list):
            # Sort by date (newest first) and take top N
            news_data = sorted(
                news_data, key=lambda x: x.get("datetime", 0), reverse=True
            )[:max_results]

            logger.info(f"Retrieved {len(news_data)} news articles for {ticker}")
            return news_data
        else:
            logger.warning(f"No news data found for {ticker}")
            return []

    except Exception as e:
        logger.error(f"Exception fetching news for {ticker}: {e}")
        return []
