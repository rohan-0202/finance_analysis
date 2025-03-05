import sqlite3

def create_stock_database(db_name='stock_data.db'):
    """Create an SQLite database with tables for essential stock data for trading decisions."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table for basic ticker information (trimmed)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ticker_info (
        ticker TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        industry TEXT,
        market_cap REAL,
        currency TEXT,
        last_updated TIMESTAMP
    )
    ''')
    
    # Create table for historical price data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date TIMESTAMP,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        dividends REAL,
        UNIQUE(ticker, date)
    )
    ''')
    
    # Create table for key financial metrics (replaces full financial statements)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        date TIMESTAMP,
        is_quarterly BOOLEAN,
        revenue REAL,
        earnings REAL,
        eps REAL,
        pe_ratio REAL,
        debt_to_equity REAL,
        operating_margin REAL,
        roe REAL,
        UNIQUE(ticker, date, is_quarterly)
    )
    ''')
    
    # Create table for options data (focus on active options)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS options_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        expiration_date TIMESTAMP,
        option_type TEXT,  -- 'call' or 'put'
        strike REAL,
        last_price REAL,
        bid REAL,
        ask REAL,
        volume INTEGER,
        open_interest INTEGER,
        implied_volatility REAL,
        last_updated TIMESTAMP,
        UNIQUE(ticker, expiration_date, option_type, strike)
    )
    ''')
    
    # Create table for recent news (last 7 days only)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recent_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT,
        publish_date TIMESTAMP,
        UNIQUE(title)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database '{db_name}' created successfully with essential trading tables.")

if __name__ == "__main__":
    create_stock_database()
    print("Database setup complete.")