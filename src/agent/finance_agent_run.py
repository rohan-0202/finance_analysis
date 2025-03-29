import json
import logging
import os
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from finance_news_agent import FinanceNewsAgent
from news_db import create_news_db

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()
lmstudio_url = os.getenv("LMSTUDIO_URL")
logger.info(f"LMStudio URL: {lmstudio_url}")
if not lmstudio_url:
    logger.error("LMStudio URL is not set")
    exit(1)

finnhub_key = os.getenv("FINNHUB_API_KEY")
logger.info(f"Finnhub Key: {finnhub_key}")
if not finnhub_key:
    logger.error("Finnhub Key is not set")
    exit(1)


@click.group()
def cli():
    """Finance News Agent - Analyze financial news for stocks."""
    pass


# region Database query command
# Add database query command
@cli.command("db-query")
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
@click.option(
    "--list-tickers",
    is_flag=True,
    help="List all tickers with analysis in the database",
)
@click.option("--ticker", help="Ticker symbol to query analysis for")
@click.option(
    "--get-news", is_flag=True, help="Get latest news for the specified ticker"
)
def query_database(db_path, list_tickers, ticker, get_news):
    """Query the finance news database."""
    agent = FinanceNewsAgent(db_path=db_path, use_db=True)

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
                click.echo(f"\n[{i + 1}] {article.get('title')}")
                click.echo(
                    f"Source: {article.get('source')} | Date: {article.get('publish_date')}"
                )
                click.echo(f"URL: {article.get('url')}")
                click.echo(f"Summary: {article.get('summary')[:100]}...")
        else:
            analysis = agent._get_latest_analysis(ticker)
            if analysis:
                click.echo(
                    f"\nLatest analysis for {ticker} (from {analysis.get('analyzed_at')}):"
                )
                click.echo(f"Summary: {analysis.get('summary')[:200]}...")
                click.echo(
                    f"Sentiment: {analysis.get('sentiment')} | Confidence: {analysis.get('confidence')}"
                )
                click.echo(
                    f"Key Points: {len(analysis.get('key_points', []))} | Risk Factors: {len(analysis.get('risk_factors', []))}"
                )
            else:
                click.echo(f"No analysis found for {ticker}")


# endregion Database query command


# region analyze all tickers command
@cli.command("all")
@click.option(
    "--tickers-file",
    default=os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
    help="Path to file with ticker symbols",
    show_default=True,
)
@click.option(
    "--output-file",
    default="finance_news_results.json",
    help="Path to save results (deprecated, using database instead)",
    show_default=True,
)
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
@click.option("--use-db/--no-db", default=True, help="Use database storage for results")
@click.option(
    "--max-tickers",
    default=0,
    type=int,
    help="Maximum number of tickers to process (0 = no limit)",
)
@click.option(
    "--max-articles",
    default=5,
    type=int,
    help="Maximum number of articles per ticker",
)
@click.option(
    "--days-lookback",
    default=7,
    type=int,
    help="Number of days to look back for news",
)
@click.option(
    "--fetch-financials/--no-financials",
    default=False,
    help="Fetch additional financial data (increases runtime)",
)
@click.option(
    "--analyze-with-llm/--no-llm",
    default=True,
    help="Analyze with LLM (default: True)",
)
def analyze_all_tickers(
    tickers_file,
    output_file,
    db_path,
    use_db,
    max_tickers,
    max_articles,
    days_lookback,
    fetch_financials,
    analyze_with_llm,
):
    """Analyze news for all tickers in the specified file."""
    # Initialize the agent with the specified parameters
    agent = FinanceNewsAgent(
        tickers_file=tickers_file,
        lmstudio_url=lmstudio_url,
        finnhub_key=finnhub_key,
        days_lookback=days_lookback,
        db_path=db_path,
        use_db=use_db,
        max_articles_per_ticker=max_articles,
        fetch_financial_data=fetch_financials,
        analyze_with_llm=analyze_with_llm,
    )

    start_time = datetime.now()
    # Run the agent with the specified parameters
    agent.run(max_tickers=max_tickers)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    click.echo(f"Total runtime: {duration:.1f} minutes")


# endregion analyze all tickers command


# region analyze single ticker command
@cli.command("ticker")
@click.argument("ticker")
@click.option(
    "--tickers-file",
    default=os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
    help="Path to file with ticker symbols",
    show_default=True,
)
@click.option("--output-file", help="Optional path to save results", default=None)
@click.option(
    "--db-path",
    default=os.getenv("DB_PATH", "finance_news.db"),
    help="Path to SQLite database",
    show_default=True,
)
@click.option("--use-db/--no-db", default=True, help="Use database storage for results")
@click.option(
    "--max-articles",
    default=5,
    type=int,
    help="Maximum number of articles to analyze",
)
@click.option(
    "--days-lookback",
    default=7,
    type=int,
    help="Number of days to look back for news",
)
@click.option(
    "--fetch-financials/--no-financials",
    default=False,
    help="Fetch additional financial data (increases runtime)",
)
@click.option(
    "--analyze-with-llm/--no-llm",
    default=True,
    help="Analyze with LLM (default: True)",
)
def analyze_single_ticker(
    ticker,
    tickers_file,
    output_file,
    db_path,
    use_db,
    max_articles,
    days_lookback,
    fetch_financials,
    analyze_with_llm,
):
    """Analyze news for a single ticker."""
    ticker = ticker.upper()  # Convert to uppercase

    click.echo(f"Analyzing news for {ticker}...")

    agent = FinanceNewsAgent(
        tickers_file=tickers_file,
        lmstudio_url=lmstudio_url,
        finnhub_key=finnhub_key,
        days_lookback=days_lookback,
        db_path=db_path,
        use_db=use_db,
        max_articles_per_ticker=max_articles,
        fetch_financial_data=fetch_financials,
        analyze_with_llm=analyze_with_llm,
    )

    start_time = datetime.now()
    result = agent.process_ticker(ticker)
    duration = (datetime.now() - start_time).total_seconds()
    print(result)

    # Display results
    click.echo("\n===== ANALYSIS RESULTS =====\n")
    click.echo(f"Ticker: {result.get('ticker')}")
    click.echo(f"Summary: {result.get('summary')}")

    if "key_points" in result:
        click.echo("\nKey Points:")
        for point in result.get("key_points", []):
            click.echo(f"- {point}")

    if "financial_implications" in result:
        click.echo(f"\nFinancial Implications: {result.get('financial_implications')}")

    sentiment = result.get("sentiment", "neutral")
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
        output_path = output_file
    else:
        output_dir = Path("news_analysis_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ticker}_analysis.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"\nFull results saved to {output_path}")
    click.echo(f"Processing time: {duration:.2f} seconds")


# endregion analyze single ticker command

if __name__ == "__main__":
    create_news_db()
    cli()
