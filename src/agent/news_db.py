import os
import sqlite3


def create_news_db(db_name="finance_news.db"):
    """Create an SQLite database for storing financial news and analysis results."""
    # Make sure directory exists
    db_dir = os.path.dirname(db_name)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create tickers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tickers (
        ticker TEXT PRIMARY KEY,
        name TEXT,
        industry TEXT,
        last_updated TIMESTAMP
    )
    """)

    # Create analyses table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        analyzed_at TIMESTAMP,
        summary TEXT,
        financial_implications TEXT,
        short_term_outlook TEXT,
        medium_term_outlook TEXT,
        sentiment TEXT,
        sentiment_reasoning TEXT,
        confidence TEXT,
        articles_analyzed INTEGER,
        error TEXT,
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """)

    # Create key_points table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS key_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER,
        point TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id)
    )
    """)

    # Create risk_factors table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_factors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER,
        factor TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id)
    )
    """)

    # Create catalysts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS catalysts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER,
        catalyst TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id)
    )
    """)

    # Create financial_data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        fetched_at TIMESTAMP,
        data_type TEXT,
        data JSON,
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """)

    # Create news_articles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT,
        source TEXT,
        publish_date TIMESTAMP,
        url TEXT,
        summary TEXT,
        fetched_at TIMESTAMP,
        FOREIGN KEY (ticker) REFERENCES tickers(ticker)
    )
    """)

    conn.commit()
    conn.close()

    print(f"News database '{db_name}' created successfully.")


if __name__ == "__main__":
    create_news_database()
    print("News database setup complete.")
