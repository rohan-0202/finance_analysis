"""
Finance News Agent - Scans the internet for recent news on NYSE stocks

This agent uses various financial news APIs to gather recent news on stocks
listed in nyse_tickers.txt, then processes them using LMStudio to extract
insights and summarize key information.
"""

import os
import sys
import json
import logging
import re  # Added for regex parsing of JSON
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import requests
from tqdm import tqdm
import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("finance_news_agent")

class FinanceNewsAgent:
    """Agent for fetching and analyzing financial news using LMStudio."""
    
    def __init__(
        self, 
        tickers_file: str = os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
        lmstudio_url: str = os.getenv("LMSTUDIO_URL", "http://10.0.0.22:1234/v1"),
        finnhub_key: str = os.getenv("FINNHUB_API_KEY", ""),
        days_lookback: int = int(os.getenv("DAYS_LOOKBACK", "7")),  # Reduced from 14 to 7 days
        db_path: str = os.getenv("DB_PATH", "finance_news.db"),
        use_db: bool = True,
        max_articles_per_ticker: int = 5,  # Reduced from 10 to 5
        fetch_financial_data: bool = False  # Turn off financial data fetching by default
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
        
        # Load ticker symbols
        self.tickers = self._load_tickers()
        
        logger.info(f"Initialized with {len(self.tickers)} tickers")
        logger.info(f"Database storage: {'enabled' if use_db else 'disabled'}")
        logger.info(f"Financial data fetching: {'enabled' if fetch_financial_data else 'disabled'}")
        logger.info(f"Articles per ticker: {max_articles_per_ticker}")
        logger.info(f"Days lookback: {days_lookback}")
        
        if self.use_db:
            logger.info(f"Database storage enabled at {db_path}")
            # Ensure the database exists
            if not os.path.exists(db_path):
                logger.warning(f"Database file not found: {db_path}. Run create_news_database.py to initialize it.")

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
    
    def fetch_news_finnhub(self, ticker: str, max_results: int = None) -> List[Dict]:
        """
        Fetch financial news for a ticker from Finnhub.
        
        Args:
            ticker: Stock ticker symbol
            max_results: Maximum number of results to return
            
        Returns:
            List of news article dictionaries
        """
        if max_results is None:
            max_results = self.max_articles_per_ticker
            
        logger.info(f"Fetching news for {ticker} (max: {max_results})")
        
        # Calculate date range - from X days ago to today
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=self.days_lookback)).strftime("%Y-%m-%d")
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": self.finnhub_key
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
                    news_data, 
                    key=lambda x: x.get("datetime", 0), 
                    reverse=True
                )[:max_results]
                
                logger.info(f"Retrieved {len(news_data)} news articles for {ticker}")
                return news_data
            else:
                logger.warning(f"No news data found for {ticker}")
                return []
                
        except Exception as e:
            logger.error(f"Exception fetching news for {ticker}: {e}")
            return []
            
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
            
        financial_data = {
            "quote": None,
            "profile": None
        }
        
        headers = {
            "X-Finnhub-Token": self.finnhub_key
        }
        
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
    
    def analyze_with_lmstudio(self, news_data: List[Dict], ticker: str, financial_data: Dict = None) -> Dict:
        """
        Analyze financial news using LMStudio API.
        
        Args:
            news_data: List of news article dictionaries
            ticker: Stock ticker symbol
            financial_data: Additional financial data to include in analysis
            
        Returns:
            Dictionary with analysis results
        """
        # Basic validation
        if not news_data:
            logger.warning(f"No news data to analyze for {ticker}")
            return {
                "ticker": ticker,
                "error": "No news data to analyze",
                "summary": "No recent news articles were found for analysis.",
                "articles": [],
                "key_points": [],
                "financial_implications": "N/A",
                "sentiment": "neutral", 
                "confidence": "low"
            }
        
        # Check format and filter any non-dictionary articles
        valid_articles = [article for article in news_data if isinstance(article, dict)]
        
        if len(valid_articles) == 0:
            logger.warning(f"No valid news articles for {ticker}")
            return {
                "ticker": ticker,
                "error": "No valid news articles",
                "summary": "The news data format was invalid for analysis.",
                "articles": [],
                "key_points": [],
                "financial_implications": "N/A",
                "sentiment": "neutral",
                "confidence": "low"
            }
        
        # Take only the most recent articles up to max_articles_per_ticker
        valid_articles = valid_articles[:self.max_articles_per_ticker]
        
        # Prepare context about the articles for the LLM - more concise to reduce tokens
        context = f"Recent news about {ticker}:\n\n"
        
        # Only include essential information from each article to reduce prompt size
        article_summaries = []
        for i, article in enumerate(valid_articles, 1):
            # Extract article information, with fallbacks for different formats
            title = self._safe_extract(article, "headline", fallbacks=["title"]) or "Untitled"
            
            # Only include the title to keep context small and fast to process
            article_summary = f"{i}. {title}"
            article_summaries.append(article_summary)
        
        # Join all article summaries
        context += "\n".join(article_summaries)
        
        # Only include minimal financial data if available
        if financial_data and financial_data.get("quote"):
            quote = financial_data["quote"]
            context += f"\n\nCurrent price: ${quote.get('c', 'N/A')} (Change: {quote.get('dp', 'N/A')}%)"
        
        # Create a very concise prompt focused on speed
        system_prompt = f"You are a financial analyst. Analyze these news headlines for ${ticker} stock."
        
        user_prompt = f"""{context}

Based on these headlines:
1. Give a 1-2 sentence summary
2. List 2-3 key points
3. State if sentiment is bullish, bearish, or neutral
4. Rate your confidence as high, medium, or low

Format as JSON:
{{
  "summary": "brief summary",
  "key_points": ["point 1", "point 2"],
  "sentiment": "bullish/bearish/neutral",
  "confidence": "high/medium/low"
}}"""
        
        # Make the API request to LMStudio with reduced max_tokens
        try:
            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for more consistent responses
                    "max_tokens": 500   # Reduced from 2000 to 500
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Error from LMStudio API: {response.status_code}")
                
                # Try a simpler request format as fallback
                logger.info("Trying simplified request format")
                response = requests.post(
                    f"{self.lmstudio_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "messages": [
                            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code != 200:
                    return self._create_error_response(ticker, f"API Error: {response.status_code}")
            
            # Process the response
            result = response.json()
            
            # Extract content from the response
            content = ""
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                elif "text" in choice:
                    content = choice["text"]
            
            if not content:
                return self._create_error_response(ticker, "Failed to extract content from API response")
            
            # Try to parse JSON from the response content
            try:
                # First, try direct JSON parsing
                analysis_result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from markdown code blocks
                try:
                    # Look for JSON in markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                    
                    if json_match:
                        json_content = json_match.group(1).strip()
                        analysis_result = json.loads(json_content)
                    else:
                        # Try to find anything that looks like a JSON object
                        json_match = re.search(r'({[\s\S]*})', content)
                        if json_match:
                            json_content = json_match.group(1)
                            analysis_result = json.loads(json_content)
                        else:
                            # If all else fails, create a structured response manually
                            analysis_result = self._extract_structured_content(content)
                except Exception as e:
                    analysis_result = self._extract_structured_content(content)
            
            # Ensure all required fields are present (simplified)
            required_fields = ["summary", "key_points", "sentiment", "confidence"]
            
            for field in required_fields:
                if field not in analysis_result:
                    if field == "key_points":
                        analysis_result[field] = []
                    else:
                        analysis_result[field] = "Not provided"
            
            # Get a financial implications from the analysis if needed
            if "financial_implications" not in analysis_result:
                if analysis_result["sentiment"] == "bullish":
                    analysis_result["financial_implications"] = "Potentially positive impact on stock price"
                elif analysis_result["sentiment"] == "bearish":
                    analysis_result["financial_implications"] = "Potentially negative impact on stock price"
                else:
                    analysis_result["financial_implications"] = "Limited impact on stock price expected"
            
            # Add metadata
            analysis_result["ticker"] = ticker
            analysis_result["articles"] = article_summaries
            analysis_result["articles_analyzed"] = len(valid_articles)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing news for {ticker}: {e}")
            return self._create_error_response(ticker, f"Analysis error: {str(e)}")
    
    def _create_error_response(self, ticker: str, error_message: str) -> Dict:
        """Create a standardized error response."""
        return {
            "ticker": ticker,
            "error": error_message,
            "summary": "Error during analysis. See error field for details.",
            "articles": [],
            "key_points": [],
            "financial_implications": "Unable to analyze due to error",
            "price_impact": "unknown",
            "sentiment": "neutral",
            "confidence": "low"
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
    
    def _extract_structured_content(self, text: str) -> Dict:
        """Extract structured content from text when JSON parsing fails."""
        result = {
            "summary": "",
            "key_points": [],
            "financial_implications": "",
            "sentiment": "neutral",
            "confidence": "low"
        }
        
        # Try to extract summary
        summary_match = re.search(r'(?:Summary|SUMMARY)[:\s]*(.*?)(?:\n\n|\n\d\.|\Z)', text, re.DOTALL)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        
        # Try to extract key points
        key_points_section = re.search(r'(?:Key Points|KEY POINTS)[:\s]*(.*?)(?:\n\n|\n\d\.|\Z)', text, re.DOTALL)
        if key_points_section:
            points_text = key_points_section.group(1)
            # Look for bullet points or numbered lists
            points = re.findall(r'(?:^|\n)[•\-*\d)\s]+\s*(.*?)(?=$|\n[•\-*\d)])', points_text, re.DOTALL)
            if points:
                result["key_points"] = [p.strip() for p in points]
            else:
                # If no bullet points found, split by newlines
                result["key_points"] = [p.strip() for p in points_text.split('\n') if p.strip()]
        
        # Try to extract financial implications
        implications_match = re.search(r'(?:Financial Implications|FINANCIAL IMPLICATIONS)[:\s]*(.*?)(?:\n\n|\n\d\.|\Z)', 
                                       text, re.DOTALL)
        if implications_match:
            result["financial_implications"] = implications_match.group(1).strip()
        
        # Try to extract sentiment
        sentiment_match = re.search(r'(?:Sentiment|SENTIMENT)[:\s]*(.*?)(?:\n\n|\n\d\.|\Z)', text, re.DOTALL)
        if sentiment_match:
            sentiment_text = sentiment_match.group(1).lower().strip()
            if 'bull' in sentiment_text:
                result["sentiment"] = "bullish"
            elif 'bear' in sentiment_text:
                result["sentiment"] = "bearish"
            else:
                result["sentiment"] = "neutral"
        
        # Try to extract confidence
        confidence_match = re.search(r'(?:Confidence|CONFIDENCE)[:\s]*(.*?)(?:\n\n|\n\d\.|\Z)', text, re.DOTALL)
        if confidence_match:
            confidence_text = confidence_match.group(1).lower().strip()
            if 'high' in confidence_text:
                result["confidence"] = "high"
            elif 'medium' in confidence_text or 'mod' in confidence_text:
                result["confidence"] = "medium"
            else:
                result["confidence"] = "low"
        
        return result
    
    def _store_ticker_info(self, ticker: str, name: Optional[str] = None, 
                         industry: Optional[str] = None) -> None:
        """Store or update ticker information in the database."""
        if not self.use_db:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if ticker exists
                cursor.execute("SELECT ticker FROM tickers WHERE ticker = ?", (ticker,))
                exists = cursor.fetchone()
                
                timestamp = datetime.now().isoformat()
                
                if exists:
                    # Update existing ticker
                    if name or industry:
                        query = "UPDATE tickers SET last_updated = ?"
                        params = [timestamp]
                        
                        if name:
                            query += ", name = ?"
                            params.append(name)
                        
                        if industry:
                            query += ", industry = ?"
                            params.append(industry)
                        
                        query += " WHERE ticker = ?"
                        params.append(ticker)
                        
                        cursor.execute(query, params)
                else:
                    # Insert new ticker
                    cursor.execute(
                        "INSERT INTO tickers (ticker, name, industry, last_updated) VALUES (?, ?, ?, ?)",
                        (ticker, name, industry, timestamp)
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
                        logger.warning(f"Skipping invalid article format for {ticker}: {type(article)}")
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
                    publish_date = article.get("publishedAt", 
                                  article.get("published_time", 
                                  article.get("datetime", timestamp)))
                    
                    # Handle different content fields
                    summary = article.get("description", 
                              article.get("summary", 
                              article.get("content", article.get("headline", ""))))
                    
                    # Handle different URL fields
                    url = article.get("url", article.get("news_url", ""))
                    
                    # Check if article already exists (by title)
                    cursor.execute("SELECT id FROM news_articles WHERE ticker = ? AND title = ?", (ticker, title))
                    if cursor.fetchone():
                        # Skip duplicate articles
                        continue
                        
                    cursor.execute(
                        """INSERT INTO news_articles 
                           (ticker, title, source, publish_date, url, summary, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker, title, source, publish_date, url, summary, timestamp)
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
                    (ticker, timestamp, data_type, data_json)
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
                        analysis.get("short_term_outlook", analysis.get("price_impact", "")),
                        analysis.get("medium_term_outlook", ""),
                        analysis.get("sentiment", "neutral"),
                        analysis.get("sentiment_reasoning", ""),
                        analysis.get("confidence", "medium"),
                        analysis.get("articles_analyzed", 0),
                        analysis.get("error", "")
                    )
                )
                
                analysis_id = cursor.lastrowid
                
                # Store key points
                key_points = analysis.get("key_points", [])
                if key_points and isinstance(key_points, list):
                    for point in key_points:
                        cursor.execute(
                            "INSERT INTO key_points (analysis_id, point) VALUES (?, ?)",
                            (analysis_id, point)
                        )
                
                # Store risk factors
                risk_factors = analysis.get("risk_factors", [])
                if risk_factors and isinstance(risk_factors, list):
                    for factor in risk_factors:
                        cursor.execute(
                            "INSERT INTO risk_factors (analysis_id, factor) VALUES (?, ?)",
                            (analysis_id, factor)
                        )
                
                # Store catalysts
                catalysts = analysis.get("catalysts", [])
                if catalysts and isinstance(catalysts, list):
                    for catalyst in catalysts:
                        cursor.execute(
                            "INSERT INTO catalysts (analysis_id, catalyst) VALUES (?, ?)",
                            (analysis_id, catalyst)
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
                    (ticker,)
                )
                
                analysis_row = cursor.fetchone()
                if not analysis_row:
                    return None
                
                # Convert to dict
                analysis = dict(analysis_row)
                analysis_id = analysis["id"]
                
                # Get key points
                cursor.execute(
                    "SELECT point FROM key_points WHERE analysis_id = ?",
                    (analysis_id,)
                )
                key_points = [row["point"] for row in cursor.fetchall()]
                analysis["key_points"] = key_points
                
                # Get risk factors
                cursor.execute(
                    "SELECT factor FROM risk_factors WHERE analysis_id = ?",
                    (analysis_id,)
                )
                risk_factors = [row["factor"] for row in cursor.fetchall()]
                analysis["risk_factors"] = risk_factors
                
                # Get catalysts
                cursor.execute(
                    "SELECT catalyst FROM catalysts WHERE analysis_id = ?",
                    (analysis_id,)
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
                    (ticker, limit)
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
                
                cursor.execute(
                    """SELECT DISTINCT ticker FROM analyses"""
                )
                
                tickers = [row[0] for row in cursor.fetchall()]
                return tickers
                
        except Exception as e:
            logger.error(f"Error getting tickers with analysis: {e}")
            return []

    def process_ticker(self, ticker: str) -> Dict:
        """
        Process a single ticker - fetch news and analyze it.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Processing ticker: {ticker}")
        
        # Fetch news data from Finnhub - use the max_articles_per_ticker setting
        news_data = self.fetch_news_finnhub(ticker)
        
        # Fetch financial data only if enabled
        financial_data = {}
        if self.fetch_financial_data:
            financial_data = self.fetch_finnhub_financial_data(ticker)
        
        # Store news articles in database if enabled
        if self.use_db and news_data:
            try:
                self._store_news_articles(ticker, news_data)
            except Exception as e:
                logger.error(f"Error storing news articles in database: {e}")
        
        # Store financial data in database if enabled
        if self.use_db and financial_data:
            try:
                for data_type, data in financial_data.items():
                    if data:
                        self._store_financial_data(ticker, data_type, data)
            except Exception as e:
                logger.error(f"Error storing financial data in database: {e}")
        
        # If we have news, analyze it along with the financial data
        if news_data:
            analysis = self.analyze_with_lmstudio(news_data, ticker, financial_data)
            
            # Store analysis result in database if enabled
            if self.use_db:
                try:
                    self._store_analysis_result(analysis)
                except Exception as e:
                    logger.error(f"Error storing analysis result in database: {e}")
            
            return analysis
        else:
            logger.warning(f"No news found for {ticker}")
            return {"ticker": ticker, "summary": "No news found", "articles": []}
    
    def run(self, output_file: Optional[str] = None, max_tickers: Optional[int] = None) -> Dict[str, Dict]:
        """
        Run the news analysis on all tickers.
        
        Args:
            output_file: Optional path to save results as JSON
            max_tickers: Maximum number of tickers to process (for limiting runtime)
            
        Returns:
            Dictionary mapping ticker symbols to their analysis results
        """
        results = {}
        
        # Limit the number of tickers if specified
        if max_tickers and max_tickers > 0:
            tickers_to_process = self.tickers[:max_tickers]
            logger.info(f"Processing {len(tickers_to_process)} out of {len(self.tickers)} tickers")
        else:
            tickers_to_process = self.tickers
            logger.info(f"Processing all {len(tickers_to_process)} tickers")
        
        for ticker in tqdm(tickers_to_process, desc="Processing tickers"):
            results[ticker] = self.process_ticker(ticker)
            
        logger.info(f"Completed analysis of {len(results)} tickers")
        
        # Save results if output file is specified
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving results to {output_file}: {e}")
        
        return results

if __name__ == "__main__":
    @click.group()
    def cli():
        """Finance News Agent - Analyze financial news for stocks."""
        pass
        
    @cli.command("all")
    @click.option(
        "--tickers-file", 
        default=os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
        help="Path to file with ticker symbols",
        show_default=True
    )
    @click.option(
        "--output-file", 
        default="finance_news_results.json",
        help="Path to save results",
        show_default=True
    )
    @click.option(
        "--db-path", 
        default=os.getenv("DB_PATH", "finance_news.db"),
        help="Path to SQLite database",
        show_default=True
    )
    @click.option(
        "--use-db/--no-db",
        default=True,
        help="Use database storage for results"
    )
    @click.option(
        "--max-tickers", 
        default=0,
        type=int,
        help="Maximum number of tickers to process (0 = no limit)"
    )
    @click.option(
        "--max-articles", 
        default=5,
        type=int,
        help="Maximum number of articles per ticker"
    )
    @click.option(
        "--days-lookback", 
        default=7,
        type=int,
        help="Number of days to look back for news"
    )
    @click.option(
        "--fetch-financials/--no-financials",
        default=False,
        help="Fetch additional financial data (increases runtime)"
    )
    def analyze_all_tickers(tickers_file, output_file, db_path, use_db, max_tickers, 
                           max_articles, days_lookback, fetch_financials):
        """Analyze news for all tickers in the specified file."""
        # Initialize the agent with the specified parameters
        agent = FinanceNewsAgent(
            tickers_file=tickers_file, 
            db_path=db_path,
            use_db=use_db,
            max_articles_per_ticker=max_articles,
            days_lookback=days_lookback,
            fetch_financial_data=fetch_financials
        )
        
        start_time = datetime.now()
        # Run the agent with the specified parameters
        results = agent.run(output_file=output_file, max_tickers=max_tickers)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds() / 60
        
        # Print the results
        click.echo(f"Analyzed {len(results)} tickers in {duration:.1f} minutes.")
        for ticker, result in results.items():
            sentiment = result.get("sentiment", "neutral")
            confidence = result.get("confidence", "low")
            
            # Color-code the sentiment
            if sentiment == "bullish":
                sentiment_str = click.style(sentiment, fg="green")
            elif sentiment == "bearish":
                sentiment_str = click.style(sentiment, fg="red")
            else:
                sentiment_str = click.style(sentiment, fg="yellow")
                
            click.echo(f"{ticker}: {sentiment_str} (confidence: {confidence})")
        
        if output_file:
            click.echo(f"Full results written to {output_file}")
        
        click.echo(f"Total runtime: {duration:.1f} minutes")
    
    @cli.command("ticker")
    @click.argument("ticker")
    @click.option(
        "--tickers-file", 
        default=os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
        help="Path to file with ticker symbols",
        show_default=True
    )
    @click.option(
        "--output-file", 
        help="Optional path to save results",
        default=None
    )
    @click.option(
        "--db-path", 
        default=os.getenv("DB_PATH", "finance_news.db"),
        help="Path to SQLite database",
        show_default=True
    )
    @click.option(
        "--use-db/--no-db",
        default=True,
        help="Use database storage for results"
    )
    @click.option(
        "--max-articles", 
        default=5,
        type=int,
        help="Maximum number of articles to analyze"
    )
    @click.option(
        "--days-lookback", 
        default=7,
        type=int,
        help="Number of days to look back for news"
    )
    @click.option(
        "--fetch-financials/--no-financials",
        default=False,
        help="Fetch additional financial data (increases runtime)"
    )
    def analyze_single_ticker(ticker, tickers_file, output_file, db_path, use_db, 
                             max_articles, days_lookback, fetch_financials):
        """Analyze news for a single ticker."""
        ticker = ticker.upper()  # Convert to uppercase
        
        click.echo(f"Analyzing news for {ticker}...")
        
        agent = FinanceNewsAgent(
            tickers_file=tickers_file,
            db_path=db_path,
            use_db=use_db,
            max_articles_per_ticker=max_articles,
            days_lookback=days_lookback,
            fetch_financial_data=fetch_financials
        )
        
        start_time = datetime.now()
        result = agent.process_ticker(ticker)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Display results
        click.echo("\n===== ANALYSIS RESULTS =====\n")
        click.echo(f"Ticker: {result.get('ticker')}")
        click.echo(f"Summary: {result.get('summary')}")
        
        if 'key_points' in result:
            click.echo("\nKey Points:")
            for point in result.get('key_points', []):
                click.echo(f"- {point}")
        
        if 'financial_implications' in result:
            click.echo(f"\nFinancial Implications: {result.get('financial_implications')}")
        
        sentiment = result.get('sentiment', 'neutral')
        if sentiment == "bullish":
            sentiment_str = click.style(sentiment, fg="green")
        elif sentiment == "bearish":
            sentiment_str = click.style(sentiment, fg="red")
        else:
            sentiment_str = click.style(sentiment, fg="yellow")
            
        click.echo(f"\nSentiment: {sentiment_str}")
        click.echo(f"Confidence: {result.get('confidence', 'low')}")
        
        # Save the full results to a JSON file if output_file specified
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = Path(f"{ticker}_analysis.json")
            
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        click.echo(f"\nFull results saved to {output_path}")
        click.echo(f"Processing time: {duration:.2f} seconds")
    
    # Add database query command
    @cli.command("db-query")
    @click.option(
        "--db-path", 
        default=os.getenv("DB_PATH", "finance_news.db"),
        help="Path to SQLite database",
        show_default=True
    )
    @click.option(
        "--list-tickers", 
        is_flag=True,
        help="List all tickers with analysis in the database"
    )
    @click.option(
        "--ticker", 
        help="Ticker symbol to query analysis for"
    )
    @click.option(
        "--get-news", 
        is_flag=True,
        help="Get latest news for the specified ticker"
    )
    def query_database(db_path, list_tickers, ticker, get_news):
        """Query the finance news database."""
        agent = FinanceNewsAgent(
            db_path=db_path,
            use_db=True
        )
        
        if list_tickers:
            tickers = agent._get_tickers_with_analysis()
            click.echo(f"Tickers with analysis in database ({len(tickers)}):")
            for t in tickers:
                click.echo(f"- {t}")
                
        if ticker:
            ticker = ticker.upper()
            if get_news:
                news = agent._get_latest_news(ticker, limit=5)
                click.echo(f"\nLatest news for {ticker} ({len(news)} articles):")
                for i, article in enumerate(news):
                    click.echo(f"\n[{i+1}] {article.get('title')}")
                    click.echo(f"Source: {article.get('source')} | Date: {article.get('publish_date')}")
                    click.echo(f"URL: {article.get('url')}")
                    click.echo(f"Summary: {article.get('summary')[:100]}...")
            else:
                analysis = agent._get_latest_analysis(ticker)
                if analysis:
                    click.echo(f"\nLatest analysis for {ticker} (from {analysis.get('analyzed_at')}):")
                    click.echo(f"Summary: {analysis.get('summary')[:200]}...")
                    click.echo(f"Sentiment: {analysis.get('sentiment')} | Confidence: {analysis.get('confidence')}")
                    click.echo(f"Key Points: {len(analysis.get('key_points', []))} | Risk Factors: {len(analysis.get('risk_factors', []))}")
                else:
                    click.echo(f"No analysis found for {ticker}")
    
    cli() 