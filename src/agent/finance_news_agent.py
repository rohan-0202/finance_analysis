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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import requests
from tqdm import tqdm

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
        tickers_file: str = "nyse_tickers.txt",
        lmstudio_url: str = "http://10.0.0.22:1234/v1",
        finnhub_key: str = "cviri41r01qult92h1pgcviri41r01qult92h1q0",
        days_lookback: int = 14  # Increased from 7 to 14 to get more historical data
    ):
        """
        Initialize the Finance News Agent.
        
        Args:
            tickers_file: Path to file containing NYSE ticker symbols
            lmstudio_url: URL for LMStudio API
            finnhub_key: API key for Finnhub
            days_lookback: Number of days to look back for news
        """
        self.tickers_file = tickers_file
        self.lmstudio_url = lmstudio_url
        self.finnhub_key = finnhub_key
        self.days_lookback = days_lookback
        self.tickers = self._load_tickers()
        
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
    
    def fetch_news_finnhub(self, ticker: str, max_results: int = 25) -> List[Dict]:
        """
        Fetch news from Finnhub for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_results: Maximum number of results to return
            
        Returns:
            List of news articles as dictionaries
        """
        if not self.finnhub_key:
            logger.warning("Finnhub API key not provided, skipping this source")
            return []
            
        today = datetime.now()
        from_timestamp = int((today - timedelta(days=self.days_lookback)).timestamp())
        to_timestamp = int(today.timestamp())
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": datetime.fromtimestamp(from_timestamp).strftime("%Y-%m-%d"),
            "to": datetime.fromtimestamp(to_timestamp).strftime("%Y-%m-%d"),
            "token": self.finnhub_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json()
            
            if isinstance(articles, dict) and "error" in articles:
                logger.error(f"API Error: {articles.get('error')}")
                return []
                
            # Sort articles by date (newest first)
            if articles and isinstance(articles, list):
                articles.sort(key=lambda x: x.get('datetime', 0), reverse=True)
            
            # Limit the results
            articles = articles[:max_results] if len(articles) > max_results else articles
            
            # Enhance articles with additional data
            for article in articles:
                # Add a clean summary if not present
                if not article.get('summary') and article.get('headline'):
                    article['summary'] = article.get('headline')
                
                # Add source field if not present
                if not article.get('source'):
                    article['source'] = {"name": article.get('source_id', 'Finnhub')}
                    
                # Format datetime for easier reading
                if article.get('datetime'):
                    try:
                        article['publishedAt'] = datetime.fromtimestamp(article['datetime']).isoformat()
                    except:
                        pass
            
            logger.info(f"Found {len(articles)} articles for {ticker} from Finnhub")
            return articles
        except Exception as e:
            logger.error(f"Error fetching news from Finnhub for {ticker}: {e}")
            return []
            
    def fetch_finnhub_financial_data(self, ticker: str) -> Dict:
        """
        Fetch additional financial data from Finnhub for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with financial data
        """
        financial_data = {
            "profile": None,
            "quote": None,
            "recommendation": None,
            "earnings": None,
            "sentiment": None
        }
        
        headers = {
            "X-Finnhub-Token": self.finnhub_key
        }
        
        # Company profile
        try:
            profile_url = f"https://finnhub.io/api/v1/stock/profile2"
            params = {"symbol": ticker}
            response = requests.get(profile_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["profile"] = response.json()
                logger.info(f"Successfully fetched company profile for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching company profile for {ticker}: {e}")
        
        # Quote data
        try:
            quote_url = f"https://finnhub.io/api/v1/quote"
            params = {"symbol": ticker}
            response = requests.get(quote_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["quote"] = response.json()
                logger.info(f"Successfully fetched quote data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching quote data for {ticker}: {e}")
        
        # Recommendation trends
        try:
            rec_url = f"https://finnhub.io/api/v1/stock/recommendation"
            params = {"symbol": ticker}
            response = requests.get(rec_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["recommendation"] = response.json()
                logger.info(f"Successfully fetched recommendation trends for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching recommendation trends for {ticker}: {e}")
        
        # Earnings surprises
        try:
            earnings_url = f"https://finnhub.io/api/v1/stock/earnings"
            params = {"symbol": ticker}
            response = requests.get(earnings_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["earnings"] = response.json()
                logger.info(f"Successfully fetched earnings data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching earnings data for {ticker}: {e}")
        
        # Social sentiment
        try:
            sentiment_url = f"https://finnhub.io/api/v1/stock/social-sentiment"
            params = {"symbol": ticker, "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")}
            response = requests.get(sentiment_url, params=params, headers=headers)
            if response.status_code == 200 and response.json():
                financial_data["sentiment"] = response.json()
                logger.info(f"Successfully fetched social sentiment data for {ticker}")
        except Exception as e:
            logger.error(f"Error fetching social sentiment data for {ticker}: {e}")
        
        return financial_data
    
    def analyze_with_lmstudio(self, news_data: List[Dict], ticker: str, financial_data: Dict = None) -> Dict:
        """
        Analyze news data using LMStudio.
        
        Args:
            news_data: List of news articles to analyze
            ticker: Stock ticker symbol
            financial_data: Additional financial data for context
            
        Returns:
            Dictionary with analysis results
        """
        if not news_data:
            logger.warning(f"No news data to analyze for {ticker}")
            return {"ticker": ticker, "summary": "No recent news found", "articles": []}
            
        # Prepare data for LMStudio
        context = f"Analyze the following financial news and data about {ticker} stock:\n\n"
        
        # Add financial context if available
        if financial_data:
            context += "FINANCIAL OVERVIEW:\n"
            
            if financial_data.get("profile"):
                profile = financial_data["profile"]
                context += f"Company Name: {profile.get('name', ticker)}\n"
                context += f"Industry: {profile.get('finnhubIndustry', 'Unknown')}\n"
                context += f"Market Cap: ${profile.get('marketCapitalization', 'Unknown')} billion\n"
                context += f"IPO Date: {profile.get('ipo', 'Unknown')}\n"
                
            if financial_data.get("quote"):
                quote = financial_data["quote"]
                context += f"Current Price: ${quote.get('c', 'Unknown')}\n"
                context += f"Daily Change: {quote.get('dp', 'Unknown')}%\n"
                context += f"52-Week High: ${quote.get('h', 'Unknown')}\n"
                context += f"52-Week Low: ${quote.get('l', 'Unknown')}\n"
                
            if financial_data.get("recommendation") and financial_data["recommendation"]:
                rec = financial_data["recommendation"][0] if isinstance(financial_data["recommendation"], list) and financial_data["recommendation"] else {}
                context += f"Analyst Recommendations: Buy: {rec.get('buy', 'Unknown')}, Hold: {rec.get('hold', 'Unknown')}, Sell: {rec.get('sell', 'Unknown')}\n"
                
            if financial_data.get("earnings") and financial_data["earnings"]:
                earnings = financial_data["earnings"]
                context += "Recent Earnings Surprises:\n"
                for i, quarter in enumerate(earnings[:3]):  # Show last 3 quarters
                    if i < len(earnings):
                        context += f"- {quarter.get('period', 'Unknown')}: Actual ${quarter.get('actual', 'Unknown')} vs Estimate ${quarter.get('estimate', 'Unknown')}\n"
                
            context += "\n"
        
        # Filter out any non-dictionary entries
        valid_articles = []
        for item in news_data:
            if isinstance(item, dict):
                valid_articles.append(item)
            else:
                logger.warning(f"Skipping invalid article format for {ticker}: {type(item)}")
        
        # If no valid articles remain after filtering
        if not valid_articles:
            logger.warning(f"No valid articles to analyze for {ticker}")
            return {"ticker": ticker, "summary": "No properly formatted news found", "articles": []}
        
        # Format and add article content, limiting to top 8 articles (increased from 5)
        for i, article in enumerate(valid_articles[:8]):
            try:
                # Safe extraction of article fields with fallbacks
                title = article.get("title", article.get("headline", "No title"))
                
                # Handle different source formats
                source = "Unknown"
                if isinstance(article.get("source"), dict):
                    source = article["source"].get("name", "Unknown Source")
                elif isinstance(article.get("source_id"), str):
                    source = article.get("source_id")
                
                # Handle different date formats
                date = article.get("publishedAt", 
                        article.get("published_time", 
                        article.get("datetime", "Unknown date")))
                
                # Handle different content fields
                description = article.get("description", 
                               article.get("summary", 
                               article.get("content", article.get("headline", ""))))
                
                # Handle different URL fields
                url = article.get("url", article.get("news_url", ""))
                
                context += f"ARTICLE {i+1}:\n"
                context += f"Title: {title}\n"
                context += f"Source: {source}\n"
                context += f"Date: {date}\n"
                context += f"URL: {url}\n"
                context += f"Summary: {description}\n\n"
            except Exception as e:
                logger.warning(f"Error formatting article {i+1} for {ticker}: {e}")
                continue
            
        # Create enhanced prompt for LMStudio
        prompt = f"""{context}

Based on these news articles and financial data about {ticker}, please provide:
1. A detailed summary of the key information and trends (2-3 paragraphs)
2. Important financial implications for investors
3. Potential impact on stock price in the short term (1-3 months) and medium term (6-12 months)
4. The overall sentiment (positive, negative, or neutral) with reasoning
5. Key risk factors for this stock
6. Major catalysts or upcoming events that could affect the stock

Format your response as JSON with the following structure:
{{
  "summary": "detailed summary of the news and financial data",
  "key_points": ["point 1", "point 2", "point 3", ...],
  "financial_implications": "explanation of financial implications",
  "short_term_outlook": "potential impact on stock price in 1-3 months",
  "medium_term_outlook": "potential impact on stock price in 6-12 months",
  "sentiment": "positive/negative/neutral",
  "sentiment_reasoning": "explanation for the sentiment rating",
  "risk_factors": ["risk 1", "risk 2", "risk 3", ...],
  "catalysts": ["catalyst 1", "catalyst 2", ...],
  "confidence": "high/medium/low"
}}
"""
        
        try:
            # Call LMStudio API
            headers = {
                "Content-Type": "application/json"
            }
            
            # Modified payload for LMStudio compatibility
            payload = {
                # Remove model parameter to let LMStudio use default model
                "messages": [
                    {"role": "system", "content": "You are a financial analyst expert specialized in stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
                # Remove response_format which might not be supported by LMStudio
            }
            
            logger.info(f"Sending request to LMStudio at {self.lmstudio_url}/chat/completions")
            response = requests.post(
                f"{self.lmstudio_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # Adding a timeout to prevent hanging
            )
            
            # Log response details for debugging
            logger.info(f"LMStudio API status code: {response.status_code}")
            
            # For LMStudio API errors, try a simpler request
            if response.status_code == 400:
                logger.warning("Initial request failed, trying a simpler version...")
                
                # Try a simpler request format
                simpler_payload = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                }
                
                response = requests.post(
                    f"{self.lmstudio_url}/chat/completions",
                    headers=headers,
                    json=simpler_payload,
                    timeout=60
                )
                
                logger.info(f"Simplified request status code: {response.status_code}")
            
            # Handle unsuccessful responses
            if response.status_code != 200:
                logger.error(f"LMStudio API error: {response.status_code} - {response.text[:200]}")
                return {
                    "ticker": ticker,
                    "error": f"LMStudio API returned status {response.status_code}",
                    "summary": "Analysis failed due to API error",
                    "key_points": ["API error occurred"],
                    "financial_implications": "Unable to analyze due to technical error",
                    "price_impact": "Unknown due to technical error",
                    "sentiment": "neutral",
                    "confidence": "low"
                }
                
            response.raise_for_status()
            result = response.json()
            
            # Better error handling for API response
            if "choices" not in result or not result["choices"]:
                logger.error(f"Invalid response format from LMStudio API: {result}")
                return {
                    "ticker": ticker,
                    "error": "Invalid API response format",
                    "summary": "Unable to analyze news due to API response format issues",
                    "articles": [],
                    "key_points": ["API format error"],
                    "financial_implications": "Unable to analyze due to API error",
                    "price_impact": "Unknown due to API error",
                    "sentiment": "neutral",
                    "confidence": "low"
                }
            
            # Extract content safely, handling different response formats
            content = ""
            try:
                if "choices" in result and result["choices"]:
                    choice = result["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice and isinstance(choice["message"], dict):
                            content = choice["message"].get("content", "")
                        elif "text" in choice:
                            content = choice.get("text", "")
                    elif isinstance(choice, str):
                        content = choice
            except Exception as e:
                logger.error(f"Error extracting content from LMStudio response: {e}")
                
            if not content:
                logger.error("Empty content in LMStudio API response")
                return {
                    "ticker": ticker,
                    "error": "Empty content in API response",
                    "summary": "Unable to analyze news due to empty API response",
                    "articles": [],
                    "key_points": ["Empty API response"],
                    "financial_implications": "No data available",
                    "price_impact": "Unknown",
                    "sentiment": "neutral",
                    "confidence": "low"
                }
            
            # Try to parse as JSON with fallback
            try:
                # Try to parse the content as JSON directly
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from text
                logger.warning(f"Failed to parse response as JSON, content: {content[:200]}...")
                
                try:
                    # Try to extract JSON block from markdown code blocks
                    json_block_match = re.search(r'```(?:json)?\s*({\s*".*?"\s*:.*?})\s*```', content, re.DOTALL)
                    if json_block_match:
                        potential_json = json_block_match.group(1)
                        analysis = json.loads(potential_json)
                        logger.info("Successfully extracted JSON from code block")
                    else:
                        # Try to find any JSON-like object in the text
                        json_match = re.search(r'({(?:\s*".*?"\s*:.*?(?:,|}))+})', content, re.DOTALL)
                        if json_match:
                            potential_json = json_match.group(1)
                            analysis = json.loads(potential_json)
                            logger.info("Successfully extracted JSON from content")
                        else:
                            # Create a basic analysis from the text content
                            logger.warning("Could not extract JSON, creating basic analysis from text")
                            analysis = {
                                "summary": content[:500] + "..." if len(content) > 500 else content,
                                "key_points": ["Failed to parse response as JSON"],
                                "financial_implications": "Unable to extract structured data",
                                "price_impact": "Unknown due to processing error",
                                "sentiment": "neutral",
                                "confidence": "low"
                            }
                except Exception as e:
                    logger.error(f"Error during JSON extraction: {e}")
                    analysis = {
                        "summary": "Error processing LMStudio response",
                        "key_points": ["Error during analysis"],
                        "financial_implications": "Analysis failed",
                        "price_impact": "Unknown due to processing error",
                        "sentiment": "neutral",
                        "confidence": "low"
                    }
                
            # Add metadata
            analysis["ticker"] = ticker
            analysis["analyzed_at"] = datetime.now().isoformat()
            analysis["articles_analyzed"] = len(valid_articles)
            
            # Ensure all required fields exist
            if "key_points" not in analysis:
                analysis["key_points"] = ["No specific points identified"]
            
            if "financial_implications" not in analysis:
                analysis["financial_implications"] = "No specific financial implications identified"
                
            if "price_impact" not in analysis:
                analysis["price_impact"] = "No specific price impact identified"
                
            if "sentiment" not in analysis:
                analysis["sentiment"] = "neutral"
                
            if "confidence" not in analysis:
                analysis["confidence"] = "medium"
            
            logger.info(f"Successfully analyzed news for {ticker}")
            return analysis
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error with LMStudio API for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": f"Network error: {str(e)}",
                "summary": "Unable to connect to LMStudio API",
                "articles": []
            }
        except Exception as e:
            logger.error(f"Error analyzing news with LMStudio for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "summary": "Analysis failed",
                "articles": []
            }
    
    def process_ticker(self, ticker: str) -> Dict:
        """
        Process a single ticker - fetch news and analyze it.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Processing ticker: {ticker}")
        
        # Fetch news data from Finnhub (the only API that didn't hit rate limits)
        news_data = self.fetch_news_finnhub(ticker, max_results=25)  # Increased from 10 to 25
        
        # Fetch additional financial data to enrich the analysis
        financial_data = self.fetch_finnhub_financial_data(ticker)
        
        # If we have news, analyze it along with the financial data
        if news_data:
            analysis = self.analyze_with_lmstudio(news_data, ticker, financial_data)
            return analysis
        else:
            logger.warning(f"No news found for {ticker}")
            return {"ticker": ticker, "summary": "No news found", "articles": []}
    
    def run(self, output_file: Optional[str] = None) -> Dict[str, Dict]:
        """
        Run the news analysis on all tickers.
        
        Args:
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary mapping ticker symbols to their analysis results
        """
        results = {}
        
        logger.info(f"Starting analysis of {len(self.tickers)} tickers")
        
        for ticker in tqdm(self.tickers, desc="Processing tickers"):
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Finance News Agent")
    parser.add_argument("--tickers-file", default="nyse_tickers.txt", help="Path to file with ticker symbols")
    parser.add_argument("--output-file", default="finance_news_results.json", help="Path to save results")
    
    args = parser.parse_args()
    
    agent = FinanceNewsAgent(
        tickers_file=args.tickers_file
    )
    
    results = agent.run(output_file=args.output_file)
    print(f"Processed {len(results)} tickers. Results saved to {args.output_file}") 