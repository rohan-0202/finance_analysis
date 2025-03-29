import logging
import os
from datetime import datetime

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
    "--tickers-file",
    default=os.getenv("TICKERS_FILE", "nyse_tickers.txt"),
    help="Path to file with ticker symbols",
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
def query_database(db_path, list_tickers, ticker, get_news, tickers_file):
    """Query the finance news database."""
    agent = agent = FinanceNewsAgent(
        tickers_file=tickers_file,
        lmstudio_url=lmstudio_url,
        finnhub_key=finnhub_key,
        days_lookback=5,
        db_path=db_path,
        use_db=True,
        max_articles_per_ticker=5,
        fetch_financial_data=False,
        analyze_with_llm=False,
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
@click.option(
    "--output-file", help="Optional path to save results (deprecated)", default=None
)
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
    agent.process_ticker(ticker)
    duration = (datetime.now() - start_time).total_seconds()

    # If using database, fetch and display the latest analysis
    if use_db:
        click.echo("\n===== ANALYSIS RESULTS =====\n")

        # Get latest news
        news = agent._get_latest_news(ticker, limit=max_articles)
        if news:
            click.echo(f"Latest news for {ticker} ({len(news)} articles):")
            for i, article in enumerate(news, 1):
                click.echo(f"\n[{i}] {article['title']}")
                click.echo(
                    f"Source: {article['source']} | Date: {article['publish_date']}"
                )
                if article.get("url"):
                    click.echo(f"URL: {article['url']}")
                if article.get("summary"):
                    click.echo(f"Summary: {article['summary'][:150]}...")

        # If LLM analysis was enabled, show the analysis results
        if analyze_with_llm:
            analysis = agent._get_latest_analysis(ticker)
            if analysis:
                click.echo("\nAnalysis Summary:")
                click.echo(f"Summary: {analysis['summary']}")

                if analysis.get("key_points"):
                    click.echo("\nKey Points:")
                    for point in analysis["key_points"]:
                        click.echo(f"- {point}")

                sentiment = analysis.get("sentiment", "neutral")
                if sentiment == "bullish":
                    sentiment_str = click.style(sentiment, fg="green")
                elif sentiment == "bearish":
                    sentiment_str = click.style(sentiment, fg="red")
                else:
                    sentiment_str = click.style(sentiment, fg="yellow")

                click.echo(f"\nSentiment: {sentiment_str}")
                click.echo(f"Confidence: {analysis.get('confidence', 'low')}")

                if analysis.get("financial_implications"):
                    click.echo(
                        f"Financial Implications: {analysis['financial_implications']}"
                    )
            else:
                click.echo("\nNo analysis results found in database.")
        else:
            click.echo(
                "\nLLM analysis was disabled. News articles have been stored in the database."
            )

    click.echo(f"\nProcessing time: {duration:.2f} seconds")


# endregion analyze single ticker command

if __name__ == "__main__":
    create_news_db()
    cli()
