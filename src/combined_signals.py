import sys
import os
from datetime import datetime
from typing import Dict, List

# Add the src directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the relevant functions from the RSI and MACD modules
from rsi import get_latest_rsi_signal
from macd import get_latest_macd_signal


def get_combined_buy_signals(
    tickers_file: str = "nyse_tickers.txt",
    db_name: str = "stock_data.db",
    days: int = 365,
    verbose: bool = True,
) -> List[Dict[str, any]]:
    """
    Analyze all tickers from the provided file and return those
    that have both MACD and RSI "buy" signals as their most recent signals.

    Parameters:
    -----------
    tickers_file : str, default="nyse_tickers.txt"
        Path to the file containing ticker symbols
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use
    verbose : bool, default=True
        Whether to print progress information

    Returns:
    --------
    List[Dict[str, any]]: A list of dictionaries containing tickers with buy signals from both indicators
    """
    # Read ticker symbols from file
    try:
        with open(tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Tickers file '{tickers_file}' not found")
        return []

    if verbose:
        print(f"Analyzing {len(tickers)} tickers for combined MACD and RSI signals...")

    # Lists to store results
    combined_buy_signals = []
    macd_buy_only = []
    rsi_buy_only = []
    no_buy_signals = []
    error_tickers = []

    # Track counts for summary
    total_processed = 0
    total_with_macd_buy = 0
    total_with_rsi_buy = 0
    total_with_both = 0

    # Process each ticker
    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(tickers)} tickers processed")

        try:
            # Get latest signals
            latest_macd_signal = get_latest_macd_signal(ticker, db_name=db_name, days=days)
            latest_rsi_signal = get_latest_rsi_signal(ticker, db_name=db_name, days=days)

            # Track whether we have buy signals from each indicator
            has_macd_buy = latest_macd_signal and latest_macd_signal["type"].lower() == "buy"
            has_rsi_buy = latest_rsi_signal and latest_rsi_signal["type"].lower() == "buy"

            # Update counts
            if has_macd_buy:
                total_with_macd_buy += 1
            if has_rsi_buy:
                total_with_rsi_buy += 1

            # Store tickers based on signal results
            if has_macd_buy and has_rsi_buy:
                # Calculate days ago for each signal
                macd_days_ago = (datetime.now().date() - latest_macd_signal["date"].date()).days
                rsi_days_ago = (datetime.now().date() - latest_rsi_signal["date"].date()).days

                combined_buy_signals.append({
                    "ticker": ticker,
                    "macd_signal_date": latest_macd_signal["date"],
                    "macd_days_ago": macd_days_ago,
                    "macd_price": latest_macd_signal["price"],
                    "rsi_signal_date": latest_rsi_signal["date"],
                    "rsi_days_ago": rsi_days_ago,
                    "rsi_price": latest_rsi_signal["price"],
                    "rsi_value": latest_rsi_signal["rsi"]
                })
                total_with_both += 1
            elif has_macd_buy:
                macd_days_ago = (datetime.now().date() - latest_macd_signal["date"].date()).days
                macd_buy_only.append({
                    "ticker": ticker, 
                    "days_ago": macd_days_ago
                })
            elif has_rsi_buy:
                rsi_days_ago = (datetime.now().date() - latest_rsi_signal["date"].date()).days
                rsi_buy_only.append({
                    "ticker": ticker, 
                    "days_ago": rsi_days_ago,
                    "rsi_value": latest_rsi_signal["rsi"]
                })
            else:
                no_buy_signals.append(ticker)

            total_processed += 1

        except Exception as e:
            error_tickers.append({"ticker": ticker, "error": str(e)})
            if verbose:
                print(f"Error processing {ticker}: {e}")

    # Sort results by most recent combined signals first
    combined_buy_signals.sort(key=lambda x: min(x["macd_days_ago"], x["rsi_days_ago"]))

    # Print summary if verbose
    if verbose:
        print("\n--- ANALYSIS SUMMARY ---")
        print(f"Total tickers processed: {total_processed}")
        print(f"Tickers with MACD buy signals: {total_with_macd_buy}")
        print(f"Tickers with RSI buy signals: {total_with_rsi_buy}")
        print(f"Tickers with BOTH MACD and RSI buy signals: {total_with_both}")
        print(f"Tickers with errors during processing: {len(error_tickers)}")

        if combined_buy_signals:
            print("\n--- TICKERS WITH BOTH MACD AND RSI BUY SIGNALS ---")
            for signal in combined_buy_signals:
                print(f"{signal['ticker']}: MACD {signal['macd_days_ago']} days ago, "
                      f"RSI {signal['rsi_days_ago']} days ago (RSI value: {signal['rsi_value']:.2f})")

        if error_tickers and verbose:
            print("\n--- ERROR TICKERS ---")
            for et in error_tickers[:5]:  # Show only the first 5 errors to avoid clutter
                print(f"{et['ticker']}: {et['error']}")
            if len(error_tickers) > 5:
                print(f"...and {len(error_tickers) - 5} more errors")

    return combined_buy_signals


def print_ticker_details(ticker: str, db_name: str = "stock_data.db", days: int = 365):
    """
    Print detailed analysis for a specific ticker including both MACD and RSI information.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol to analyze
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use
    """
    from macd import calculate_macd
    from rsi import calculate_ticker_rsi
    
    print(f"\n=== DETAILED ANALYSIS FOR {ticker} ===")
    
    try:
        # Get MACD data
        price_data_macd, macd_data = calculate_macd(ticker, db_name=db_name, days=days)
        latest_macd_signal = get_latest_macd_signal(ticker, db_name=db_name, days=days)
        
        # Get RSI data
        price_data_rsi, rsi_data = calculate_ticker_rsi(ticker, db_name=db_name, days=days)
        latest_rsi_signal = get_latest_rsi_signal(ticker, db_name=db_name, days=days)
        
        # Print MACD info
        print("\nMACD INFORMATION:")
        if macd_data is not None:
            print("Current MACD values:")
            print(f"MACD: {macd_data['macd'].iloc[-1]:.4f}")
            print(f"Signal: {macd_data['signal'].iloc[-1]:.4f}")
            print(f"Histogram: {macd_data['histogram'].iloc[-1]:.4f}")
            
            if latest_macd_signal:
                days_ago = (datetime.now().date() - latest_macd_signal["date"].date()).days
                print(f"\nLatest MACD signal: {latest_macd_signal['type'].upper()} ({days_ago} days ago)")
                print(f"Price at signal: ${latest_macd_signal['price']:.2f}")
            else:
                print("\nNo recent MACD signals")
        else:
            print("Could not calculate MACD data")
        
        # Print RSI info
        print("\nRSI INFORMATION:")
        if rsi_data is not None:
            current_rsi = rsi_data.iloc[-1]
            print(f"Current RSI: {current_rsi:.2f}")
            
            # Interpret the current RSI value
            if current_rsi > 70:
                print("Status: OVERBOUGHT - Potential sell signal")
            elif current_rsi < 30:
                print("Status: OVERSOLD - Potential buy signal")
            else:
                print("Status: NEUTRAL")
                
            if latest_rsi_signal:
                days_ago = (datetime.now().date() - latest_rsi_signal["date"].date()).days
                print(f"\nLatest RSI signal: {latest_rsi_signal['type'].upper()} ({days_ago} days ago)")
                print(f"RSI at signal: {latest_rsi_signal['rsi']:.2f}")
                print(f"Price at signal: ${latest_rsi_signal['price']:.2f}")
            else:
                print("\nNo recent RSI signals (crossing 30/70 thresholds)")
        else:
            print("Could not calculate RSI data")
            
        # Conclusion
        print("\nCOMBINED ANALYSIS:")
        has_macd_buy = latest_macd_signal and latest_macd_signal["type"].lower() == "buy"
        has_rsi_buy = latest_rsi_signal and latest_rsi_signal["type"].lower() == "buy"
        
        if has_macd_buy and has_rsi_buy:
            print("STRONG BUY SIGNAL - Both MACD and RSI indicate buying opportunity")
        elif has_macd_buy:
            print("MODERATE BUY SIGNAL - MACD indicates buying opportunity but RSI does not confirm")
        elif has_rsi_buy:
            print("MODERATE BUY SIGNAL - RSI indicates buying opportunity but MACD does not confirm")
        else:
            print("NO BUY SIGNALS - Neither MACD nor RSI indicate buying opportunity")
            
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze stocks for combined MACD and RSI buy signals")
    parser.add_argument("-t", "--ticker", help="Analyze a specific ticker")
    parser.add_argument("-d", "--days", type=int, default=365, help="Number of days of historical data to use")
    parser.add_argument("-s", "--silent", action="store_true", help="Run in silent mode (no progress updates)")
    args = parser.parse_args()
    
    if args.ticker:
        # Analyze a specific ticker
        print_ticker_details(args.ticker.upper(), days=args.days)
    else:
        # Get combined buy signals for all tickers
        combined_signals = get_combined_buy_signals(days=args.days, verbose=not args.silent)
        
        # Print a list of tickers with both signals
        if combined_signals:
            print("\nTickers with both MACD and RSI buy signals:")
            for signal in combined_signals:
                print(f"{signal['ticker']}")
        else:
            print("\nNo tickers found with both MACD and RSI buy signals") 