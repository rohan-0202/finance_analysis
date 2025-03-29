"""
Finance News Agent - Scans the internet for recent news on NYSE stocks

This agent uses various financial news APIs to gather recent news on stocks
listed in nyse_tickers.txt, then processes them using LMStudio to extract
insights and summarize key information.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from agent.finhub_util import fetch_news_finnhub
from tqdm import tqdm



# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("finance_news_agent")


class FinanceNewsAgent:
    """Agent for fetching and analyzing financial news using LMStudio."""

    def __init__(
        self,
        tickers_file: str,
        lmstudio_url: str,
        finnhub_key: str,
        days_lookback: int,
        db_path: str = "finance_news.db",
        use_db: bool = True,
        max_articles_per_ticker: int = 5,  # Reduced from 10 to 5
        fetch_financial_data: bool = False,  # Turn off financial data fetching by default
        analyze_with_llm: bool = True,
    ):
        """
        Initialize the Finance News Agent.

        Args:
            tickers_file: Path to file containing NYSE ticker symbols
            lmstudio_url: URL for LMStudio API
            finnhub_key: API key for Finnhub
            days_lookback: Number of days to look back for news
            db_path: Path to SQLite database
            use_db: Whether to use database storage
            max_articles_per_ticker: Maximum number of articles to process per ticker
            fetch_financial_data: Whether to fetch additional financial data (slows down processing)
        """
        self.tickers_file = tickers_file
        self.lmstudio_url = lmstudio_url
        self.finnhub_key = finnhub_key
        self.days_lookback = days_lookback
        self.db_path = db_path
        self.use_db = use_db
        self.max_articles_per_ticker = max_articles_per_ticker
        self.fetch_financial_data = fetch_financial_data
        self.analyze_with_llm = analyze_with_llm

        # Load ticker symbols
        self.tickers = self._load_tickers()

        logger.info(f"Initialized with {len(self.tickers)} tickers")
        logger.info(f"Database storage: {'enabled' if use_db else 'disabled'}")
        logger.info(
            f"Financial data fetching: {'enabled' if fetch_financial_data else 'disabled'}"
        )
        logger.info(f"Articles per ticker: {max_articles_per_ticker}")
        logger.info(f"Days lookback: {days_lookback}")

        if self.use_db:
            logger.info(f"Database storage enabled at {db_path}")
            # Ensure the database exists
            if not os.path.exists(db_path):
                logger.warning(
                    f"Database file not found: {db_path}. Run create_news_database.py to initialize it."
                )

    def _load_tickers(self) -> List[str]:
        """Load ticker symbols from file."""
        try:
            # Check if file exists
            if not os.path.exists(self.tickers_file):
                logger.error(f"Tickers file not found: {self.tickers_file}")
                logger.info("Make sure the path to nyse_tickers.txt is correct")
                logger.info("Current working directory: {0}".format(os.getcwd()))
                raise FileNotFoundError(f"Tickers file not found: {self.tickers_file}")

            with open(self.tickers_file, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]

            if not tickers:
                logger.warning(f"No tickers found in {self.tickers_file}")
                return []

            logger.info(f"Loaded {len(tickers)} tickers from {self.tickers_file}")
            return tickers
        except Exception as e:
            logger.error(f"Failed to load tickers: {e}")
            raise

    def fetch_finnhub_financial_data(self, ticker: str) -> Dict:
        """
        Fetch additional financial data from Finnhub for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with financial data
        """
        # If financial data fetching is disabled, return empty dict
        if not self.fetch_financial_data:
            return {}

        financial_data = {"quote": None, "profile": None}

        headers = {"X-Finnhub-Token": self.finnhub_key}

        # Only fetch the most essential data: quote and profile
        # This reduces API calls from 5 to 2 per ticker

        # Quote data (most important for current price)
        try:
            quote_url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker}
            response = requests.get(quote_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["quote"] = response.json()
                logger.info(f"Successfully fetched quote data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching quote data for {ticker}: {e}")

        # Company profile (basic info)
        try:
            profile_url = "https://finnhub.io/api/v1/stock/profile2"
            params = {"symbol": ticker}
            response = requests.get(profile_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["profile"] = response.json()
                logger.info(f"Successfully fetched company profile for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching company profile for {ticker}: {e}")

        return financial_data

    def analyze_with_lmstudio(
        self, news_data: List[Dict], ticker: str, financial_data: Dict = None
    ) -> Dict:
        """
        Analyze financial news using LMStudio API.

        Args:
            news_data: List of news article dictionaries
            ticker: Stock ticker symbol
            financial_data: Additional financial data to include in analysis

        Returns:
            Dictionary with analysis results
        """
        # Basic validation and filtering
        valid_articles = [
            article
            for article in news_data
            if isinstance(article, dict)
            and (article.get("title") or article.get("headline"))
        ][: self.max_articles_per_ticker]

        if not valid_articles:
            logger.warning(f"No valid news articles for {ticker}")
            return self._create_error_response(ticker, "No valid news articles found")

        # Prepare context about the articles for the LLM
        article_summaries = [
            f"{i}. {article.get('title', article.get('headline'))}"
            for i, article in enumerate(valid_articles, 1)
        ]

        context = f"Recent news about {ticker}:\n\n" + "\n".join(article_summaries)

        # Add minimal financial data if available
        if financial_data and financial_data.get("quote"):
            quote = financial_data["quote"]
            context += f"\n\nCurrent price: ${quote.get('c', 'N/A')} (Change: {quote.get('dp', 'N/A')}%)"

        # Create the prompt
        system_prompt = f"You are a financial analyst. Analyze these news headlines for ${ticker} stock."
        user_prompt = f"""{context}

Based on these headlines:
1. Give a 1-2 sentence summary
2. List 2-3 key points
3. State if sentiment is bullish, bearish, or neutral
4. Rate your confidence as high, medium, or low

Format your response in clear sections:
SUMMARY: (your summary)
KEY POINTS:
- (point 1)
- (point 2)
SENTIMENT: (bullish/bearish/neutral)
CONFIDENCE: (high/medium/low)"""

        try:
            # Make the API request to LMStudio
            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30,  # Add timeout to prevent hanging
            )

            response.raise_for_status()  # Raise exception for non-200 status codes

            # Extract the content from the response
            content = response.json().get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            ) or response.json().get("choices", [{}])[0].get("text", "")

            if not content:
                return self._create_error_response(ticker, "Empty response from LLM")

            # Initialize analysis result with defaults
            analysis_result = {
                "ticker": ticker,
                "articles_analyzed": len(valid_articles),
                "summary": "Summary not provided",
                "key_points": [],
                "sentiment": "neutral",
                "confidence": "low",
                "financial_implications": "Limited impact on stock price expected",
            }

            # Define section markers and their end markers
            section_markers = {
                "SUMMARY:": ["KEY POINTS:", "SENTIMENT:", "CONFIDENCE:"],
                "KEY POINTS:": ["SENTIMENT:", "CONFIDENCE:"],
                "SENTIMENT:": ["CONFIDENCE:"],
                "CONFIDENCE:": [],
            }

            # Extract each section
            for section, end_markers in section_markers.items():
                if section in content:
                    start_idx = content.find(section) + len(section)
                    end_idx = len(content)
                    for marker in end_markers:
                        marker_idx = content.find(marker, start_idx)
                        if marker_idx != -1:
                            end_idx = min(end_idx, marker_idx)

                    value = content[start_idx:end_idx].strip()

                    # Process each section type
                    if section == "SUMMARY:":
                        analysis_result["summary"] = value
                    elif section == "KEY POINTS:":
                        analysis_result["key_points"] = [
                            p.strip("- ").strip()
                            for p in value.split("\n")
                            if p.strip().startswith("-")
                        ]
                    elif section == "SENTIMENT:":
                        value = value.lower()
                        if "bull" in value:
                            analysis_result["sentiment"] = "bullish"
                            analysis_result["financial_implications"] = (
                                "Potentially positive impact on stock price"
                            )
                        elif "bear" in value:
                            analysis_result["sentiment"] = "bearish"
                            analysis_result["financial_implications"] = (
                                "Potentially negative impact on stock price"
                            )
                    elif section == "CONFIDENCE:":
                        value = value.lower()
                        if "high" in value:
                            analysis_result["confidence"] = "high"
                        elif any(x in value for x in ["medium", "mod"]):
                            analysis_result["confidence"] = "medium"

            return analysis_result

        except requests.RequestException as e:
            logger.error(f"API request error for {ticker}: {e}")
            return self._create_error_response(ticker, f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return self._create_error_response(ticker, str(e))

    def _create_error_response(self, ticker: str, error_message: str) -> Dict:
        """Create a standardized error response."""
        return {
            "ticker": ticker,
            "error": error_message,
            "summary": "Error during analysis",
            "key_points": [],
            "sentiment": "neutral",
            "confidence": "low",
            "articles_analyzed": 0,
        }

    def _safe_extract(self, data: Dict, key: str, fallbacks: List[str] = None) -> Any:
        """Safely extract a value from a dictionary with fallbacks."""
        if key in data and data[key]:
            return data[key]

        if fallbacks:
            for fallback in fallbacks:
                if fallback in data and data[fallback]:
                    return data[fallback]

        return None

    def _store_ticker_info(
        self, ticker: str, name: Optional[str] = None, industry: Optional[str] = None
    ) -> None:
        """Store or update ticker information in the database."""
        if not self.use_db:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()

                # Use UPSERT (INSERT OR REPLACE) instead of separate INSERT/UPDATE
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO tickers 
                    (ticker, name, industry, last_updated)
                    VALUES (?, 
                            COALESCE(?, (SELECT name FROM tickers WHERE ticker = ?)), 
                            COALESCE(?, (SELECT industry FROM tickers WHERE ticker = ?)),
                            ?)
                """,
                    (ticker, name, ticker, industry, ticker, timestamp),
                )

                conn.commit()
                logger.debug(f"Stored/updated ticker info for {ticker}")

        except Exception as e:
            logger.error(f"Error storing ticker info for {ticker}: {e}")

    def _store_news_articles(self, ticker: str, articles: List[Dict]) -> None:
        """Store news articles in the database."""
        if not self.use_db:
            return

        try:
            # Ensure ticker exists
            self._store_ticker_info(ticker)

            # Store articles
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()

                timestamp = datetime.now().isoformat()

                for article in articles:
                    if not isinstance(article, dict):
                        logger.warning(
                            f"Skipping invalid article format for {ticker}: {type(article)}"
                        )
                        continue

                    # Extract article fields with fallbacks
                    title = article.get("title", article.get("headline", "No title"))

                    # Handle different source formats
                    source = "Unknown"
                    if isinstance(article.get("source"), dict):
                        source = article["source"].get("name", "Unknown Source")
                    elif isinstance(article.get("source_id"), str):
                        source = article.get("source_id")

                    # Handle different date formats
                    publish_date = article.get(
                        "publishedAt",
                        article.get(
                            "published_time", article.get("datetime", timestamp)
                        ),
                    )

                    # Handle different content fields
                    summary = article.get(
                        "description",
                        article.get(
                            "summary",
                            article.get("content", article.get("headline", "")),
                        ),
                    )

                    # Handle different URL fields
                    url = article.get("url", article.get("news_url", ""))

                    # Check if article already exists (by title)
                    cursor.execute(
                        "SELECT id FROM news_articles WHERE ticker = ? AND title = ?",
                        (ticker, title),
                    )
                    if cursor.fetchone():
                        # Skip duplicate articles
                        continue

                    cursor.execute(
                        """INSERT INTO news_articles 
                           (ticker, title, source, publish_date, url, summary, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker, title, source, publish_date, url, summary, timestamp),
                    )

                conn.commit()
                logger.info(f"Stored news articles for {ticker}")

        except Exception as e:
            logger.error(f"Error storing news articles for {ticker}: {e}")

    def _store_financial_data(self, ticker: str, data_type: str, data: Dict) -> None:
        """Store financial data in the database."""
        if not self.use_db or not data:
            return

        try:
            # Ensure ticker exists
            self._store_ticker_info(ticker)

            # Store data
            with sqlite3.connect(self.db_path) as conn:
                # Enable JSON serialization
                conn.execute("PRAGMA foreign_keys = ON")

                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()

                # Serialize data to JSON
                data_json = json.dumps(data)

                cursor.execute(
                    "INSERT INTO financial_data (ticker, fetched_at, data_type, data) VALUES (?, ?, ?, ?)",
                    (ticker, timestamp, data_type, data_json),
                )

                conn.commit()
                logger.debug(f"Stored {data_type} data for {ticker}")

        except Exception as e:
            logger.error(f"Error storing financial data for {ticker}: {e}")

    def _store_analysis_result(self, analysis: Dict) -> None:
        """Store analysis result in the database."""
        if not self.use_db:
            return

        try:
            ticker = analysis.get("ticker")
            if not ticker:
                logger.error("Analysis missing ticker symbol")
                return

            # Extract company info if available
            name = None
            industry = None

            # Store ticker info
            self._store_ticker_info(ticker, name, industry)

            # Store analysis
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()

                # Default timestamp if not in analysis
                analyzed_at = analysis.get("analyzed_at", datetime.now().isoformat())

                # Insert main analysis record
                cursor.execute(
                    """INSERT INTO analyses 
                       (ticker, analyzed_at, summary, financial_implications, 
                        short_term_outlook, medium_term_outlook, sentiment, 
                        sentiment_reasoning, confidence, articles_analyzed, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker,
                        analyzed_at,
                        analysis.get("summary", ""),
                        analysis.get("financial_implications", ""),
                        analysis.get(
                            "short_term_outlook", analysis.get("price_impact", "")
                        ),
                        analysis.get("medium_term_outlook", ""),
                        analysis.get("sentiment", "neutral"),
                        analysis.get("sentiment_reasoning", ""),
                        analysis.get("confidence", "medium"),
                        analysis.get("articles_analyzed", 0),
                        analysis.get("error", ""),
                    ),
                )

                analysis_id = cursor.lastrowid

                # Store key points
                key_points = analysis.get("key_points", [])
                if key_points and isinstance(key_points, list):
                    for point in key_points:
                        cursor.execute(
                            "INSERT INTO key_points (analysis_id, point) VALUES (?, ?)",
                            (analysis_id, point),
                        )

                # Store risk factors
                risk_factors = analysis.get("risk_factors", [])
                if risk_factors and isinstance(risk_factors, list):
                    for factor in risk_factors:
                        cursor.execute(
                            "INSERT INTO risk_factors (analysis_id, factor) VALUES (?, ?)",
                            (analysis_id, factor),
                        )

                # Store catalysts
                catalysts = analysis.get("catalysts", [])
                if catalysts and isinstance(catalysts, list):
                    for catalyst in catalysts:
                        cursor.execute(
                            "INSERT INTO catalysts (analysis_id, catalyst) VALUES (?, ?)",
                            (analysis_id, catalyst),
                        )

                conn.commit()
                logger.info(f"Stored analysis for {ticker}")

        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")

    def _get_latest_analysis(self, ticker: str) -> Optional[Dict]:
        """Get the latest analysis for a ticker from the database."""
        if not self.use_db:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get main analysis
                cursor.execute(
                    """SELECT * FROM analyses 
                       WHERE ticker = ? 
                       ORDER BY analyzed_at DESC LIMIT 1""",
                    (ticker,),
                )

                analysis_row = cursor.fetchone()
                if not analysis_row:
                    return None

                # Convert to dict
                analysis = dict(analysis_row)
                analysis_id = analysis["id"]

                # Get key points
                cursor.execute(
                    "SELECT point FROM key_points WHERE analysis_id = ?", (analysis_id,)
                )
                key_points = [row["point"] for row in cursor.fetchall()]
                analysis["key_points"] = key_points

                # Get risk factors
                cursor.execute(
                    "SELECT factor FROM risk_factors WHERE analysis_id = ?",
                    (analysis_id,),
                )
                risk_factors = [row["factor"] for row in cursor.fetchall()]
                analysis["risk_factors"] = risk_factors

                # Get catalysts
                cursor.execute(
                    "SELECT catalyst FROM catalysts WHERE analysis_id = ?",
                    (analysis_id,),
                )
                catalysts = [row["catalyst"] for row in cursor.fetchall()]
                analysis["catalysts"] = catalysts

                return analysis

        except Exception as e:
            logger.error(f"Error getting latest analysis for {ticker}: {e}")
            return None

    def _get_latest_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get the latest news articles for a ticker from the database."""
        if not self.use_db:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """SELECT * FROM news_articles 
                       WHERE ticker = ? 
                       ORDER BY publish_date DESC LIMIT ?""",
                    (ticker, limit),
                )

                articles = [dict(row) for row in cursor.fetchall()]
                return articles

        except Exception as e:
            logger.error(f"Error getting latest news for {ticker}: {e}")
            return []

    def _get_tickers_with_analysis(self) -> List[str]:
        """Get list of tickers that have analysis results in the database."""
        if not self.use_db:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""SELECT DISTINCT ticker FROM analyses""")

                tickers = [row[0] for row in cursor.fetchall()]
                return tickers

        except Exception as e:
            logger.error(f"Error getting tickers with analysis: {e}")
            return []

    def _get_last_processed_time(self, ticker: str) -> Optional[datetime]:
        """
        Get the datetime when a ticker was last processed by checking the news_articles table.

        Args:
            ticker: Stock ticker symbol

        Returns:
            datetime object of last processing time or None if ticker hasn't been processed
        """
        if not self.use_db:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """SELECT MAX(fetched_at) FROM news_articles 
                       WHERE ticker = ?""",
                    (ticker,),
                )

                result = cursor.fetchone()
                if result and result[0]:
                    # Convert ISO format string to datetime object
                    return datetime.fromisoformat(result[0])
                return None

        except Exception as e:
            logger.error(f"Error getting last processed time for {ticker}: {e}")
            return None

    def process_ticker(self, ticker: str) -> None:
        """
        Process a single ticker - fetch news and analyze it.

        Args:
            ticker: Stock ticker symbol
        """
        logger.info(f"Processing ticker: {ticker}")

        # Fetch news data
        news_data = fetch_news_finnhub(
            ticker, self.finnhub_key, self.max_articles_per_ticker, self.days_lookback
        )

        if not news_data:
            logger.warning(f"No news found for {ticker}")
            return

        # Store news articles in database if enabled
        if self.use_db:
            try:
                self._store_news_articles(ticker, news_data)
            except Exception as e:
                logger.error(f"Error storing news articles in database: {e}")

        # Fetch and store financial data if enabled
        financial_data = {}
        if self.fetch_financial_data:
            financial_data = self.fetch_finnhub_financial_data(ticker)
            if self.use_db and financial_data:
                try:
                    for data_type, data in financial_data.items():
                        if data:
                            self._store_financial_data(ticker, data_type, data)
                except Exception as e:
                    logger.error(f"Error storing financial data in database: {e}")

        # Analyze with LLM if enabled
        if self.analyze_with_llm:
            analysis = self.analyze_with_lmstudio(news_data, ticker, financial_data)
            if self.use_db:
                try:
                    self._store_analysis_result(analysis)
                except Exception as e:
                    logger.error(f"Error storing analysis result in database: {e}")

    def run(self, max_tickers: Optional[int] = None) -> None:
        """
        Run the news analysis on all tickers.

        Args:
            max_tickers: Maximum number of tickers to process (for limiting runtime)
        """
        # Limit the number of tickers if specified
        tickers_to_process = (
            self.tickers[:max_tickers]
            if max_tickers and max_tickers > 0
            else self.tickers
        )
        logger.info(f"Processing {len(tickers_to_process)} tickers")

        for ticker in tqdm(tickers_to_process, desc="Processing tickers"):
            last_processed = self._get_last_processed_time(ticker)
            if (
                last_processed
                and (datetime.now() - last_processed).total_seconds() < 86400
            ):  # 86400 seconds in 24 hours
                logger.info(
                    f"Skipping {ticker} as it was processed in the last 24 hours."
                )
                continue
            time.sleep(3)  # Add a delay of 3 seconds before processing each ticker
            # Remove the ^ symbol from the ticker if it exists. the ^ symbol is used for indices
            # and finnhub does not support the ^ symbol.
            if ticker.startswith("^"):
                ticker = ticker[1:]
            self.process_ticker(ticker)

        logger.info("Completed processing all tickers")
