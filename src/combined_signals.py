import sys
import os
from datetime import datetime
from typing import Dict, List

# Add the src directory to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the relevant functions from the RSI, MACD, and OBV modules
from rsi import get_latest_rsi_signal
from macd import get_latest_macd_signal
from obv import get_latest_obv_signal


def get_combined_buy_signals(
    tickers_file: str = "nyse_tickers.txt",
    db_name: str = "stock_data.db",
    days: int = 365,
    verbose: bool = True,
) -> List[Dict[str, any]]:
    """
    Analyze all tickers from the provided file and return those
    that have buy signals from all three indicators (MACD, RSI, OBV).

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
    List[Dict[str, any]]: A list of dictionaries containing tickers with buy signals from all three indicators
    """
    # Read ticker symbols from file
    try:
        with open(tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Tickers file '{tickers_file}' not found")
        return []

    if verbose:
        print(f"Analyzing {len(tickers)} tickers for combined MACD, RSI, and OBV signals...")

    # Lists to store results
    combined_buy_signals = []
    error_tickers = []

    # Track counts for summary
    total_processed = 0
    total_with_macd_buy = 0
    total_with_rsi_buy = 0
    total_with_obv_buy = 0
    total_with_all_three = 0

    # Process each ticker
    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(tickers)} tickers processed")

        try:
            # Get latest signals
            latest_macd_signal = get_latest_macd_signal(ticker, db_name=db_name, days=days)
            latest_rsi_signal = get_latest_rsi_signal(ticker, db_name=db_name, days=days)
            latest_obv_signal = get_latest_obv_signal(ticker, db_name=db_name, days=days)

            # Track whether we have buy signals from each indicator
            has_macd_buy = latest_macd_signal and latest_macd_signal["type"].lower() == "buy"
            has_rsi_buy = latest_rsi_signal and latest_rsi_signal["type"].lower() == "buy"
            has_obv_buy = latest_obv_signal and latest_obv_signal["type"].lower() == "buy"

            # Update counts
            if has_macd_buy:
                total_with_macd_buy += 1
            if has_rsi_buy:
                total_with_rsi_buy += 1
            if has_obv_buy:
                total_with_obv_buy += 1

            # Only include tickers with all three buy signals
            if has_macd_buy and has_rsi_buy and has_obv_buy:
                total_with_all_three += 1
                
                # Calculate days ago for each signal
                macd_days_ago = (datetime.now().date() - latest_macd_signal["date"].date()).days
                rsi_days_ago = (datetime.now().date() - latest_rsi_signal["date"].date()).days
                obv_days_ago = (datetime.now().date() - latest_obv_signal["date"].date()).days
                
                # Calculate the total days for sorting
                total_days_ago = macd_days_ago + rsi_days_ago + obv_days_ago
                
                signal_data = {
                    "ticker": ticker,
                    "macd_signal_date": latest_macd_signal["date"],
                    "macd_days_ago": macd_days_ago,
                    "macd_price": latest_macd_signal["price"],
                    "rsi_signal_date": latest_rsi_signal["date"],
                    "rsi_days_ago": rsi_days_ago,
                    "rsi_price": latest_rsi_signal["price"],
                    "rsi_value": latest_rsi_signal["rsi"],
                    "obv_signal_date": latest_obv_signal["date"],
                    "obv_days_ago": obv_days_ago,
                    "obv_price": latest_obv_signal["price"],
                    "obv_value": latest_obv_signal["obv"],
                    "total_days_ago": total_days_ago,
                }
                
                combined_buy_signals.append(signal_data)

            total_processed += 1

        except Exception as e:
            error_tickers.append({"ticker": ticker, "error": str(e)})
            if verbose:
                print(f"Error processing {ticker}: {e}")

    # Sort results by total days ago (most recent combined signals first)
    combined_buy_signals.sort(key=lambda x: x["total_days_ago"])

    # Print summary if verbose
    if verbose:
        print("\n--- ANALYSIS SUMMARY ---")
        print(f"Total tickers processed: {total_processed}")
        print(f"Tickers with MACD buy signals: {total_with_macd_buy}")
        print(f"Tickers with RSI buy signals: {total_with_rsi_buy}")
        print(f"Tickers with OBV buy signals: {total_with_obv_buy}")
        print(f"Tickers with all three buy signals: {total_with_all_three}")
        print(f"Tickers with errors during processing: {len(error_tickers)}")

        if combined_buy_signals:
            print("\n--- TICKERS WITH ALL THREE BUY SIGNALS (SORTED BY TOTAL RECENCY) ---")
            for signal in combined_buy_signals:
                print(f"{signal['ticker']}: " 
                      f"MACD {signal['macd_days_ago']} days ago, "
                      f"RSI {signal['rsi_days_ago']} days ago (value: {signal['rsi_value']:.2f}), "
                      f"OBV {signal['obv_days_ago']} days ago "
                      f"(Total: {signal['total_days_ago']} days)")

        if error_tickers and verbose:
            print("\n--- ERROR TICKERS ---")
            for et in error_tickers[:5]:  # Show only the first 5 errors to avoid clutter
                print(f"{et['ticker']}: {et['error']}")
            if len(error_tickers) > 5:
                print(f"...and {len(error_tickers) - 5} more errors")

    return combined_buy_signals


def print_ticker_details(ticker: str, db_name: str = "stock_data.db", days: int = 365):
    """
    Print detailed analysis for a specific ticker including MACD, RSI, and OBV information.
    
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
    from obv import calculate_ticker_obv, get_obv_status_text
    
    print(f"\n=== DETAILED ANALYSIS FOR {ticker} ===")
    
    try:
        # Get MACD data
        price_data_macd, macd_data = calculate_macd(ticker, db_name=db_name, days=days)
        latest_macd_signal = get_latest_macd_signal(ticker, db_name=db_name, days=days)
        
        # Get RSI data
        price_data_rsi, rsi_data = calculate_ticker_rsi(ticker, db_name=db_name, days=days)
        latest_rsi_signal = get_latest_rsi_signal(ticker, db_name=db_name, days=days)
        
        # Get OBV data
        price_data_obv, obv_data = calculate_ticker_obv(ticker, db_name=db_name, days=days)
        latest_obv_signal = get_latest_obv_signal(ticker, db_name=db_name, days=days)
        
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
            
        # Print OBV info
        print("\nOBV INFORMATION:")
        if obv_data is not None:
            current_obv = obv_data.iloc[-1]
            print(f"Current OBV: {current_obv:,.0f}")
            
            # Get OBV status text
            if price_data_obv is not None:
                obv_status = get_obv_status_text(price_data_obv, obv_data)
                print(f"Status: {obv_status.split('\n')[1]}")  # Just the status part
                
            if latest_obv_signal:
                days_ago = (datetime.now().date() - latest_obv_signal["date"].date()).days
                print(f"\nLatest OBV signal: {latest_obv_signal['type'].upper()} ({days_ago} days ago)")
                print(f"OBV at signal: {latest_obv_signal['obv']:,.0f}")
                print(f"Price at signal: ${latest_obv_signal['price']:.2f}")
            else:
                print("\nNo recent OBV divergence signals")
        else:
            print("Could not calculate OBV data")
            
        # Conclusion
        print("\nCOMBINED ANALYSIS:")
        has_macd_buy = latest_macd_signal and latest_macd_signal["type"].lower() == "buy"
        has_rsi_buy = latest_rsi_signal and latest_rsi_signal["type"].lower() == "buy"
        has_obv_buy = latest_obv_signal and latest_obv_signal["type"].lower() == "buy"
        
        buy_count = sum([has_macd_buy, has_rsi_buy, has_obv_buy])
        
        if buy_count == 3:
            print("STRONG BUY SIGNAL - All three indicators (MACD, RSI, OBV) indicate buying opportunity")
        elif buy_count == 2:
            indicators = []
            if has_macd_buy: indicators.append("MACD")
            if has_rsi_buy: indicators.append("RSI")
            if has_obv_buy: indicators.append("OBV")
            print(f"MODERATE BUY SIGNAL - Two indicators ({' and '.join(indicators)}) indicate buying opportunity")
        elif buy_count == 1:
            indicator = "MACD" if has_macd_buy else "RSI" if has_rsi_buy else "OBV"
            print(f"WEAK BUY SIGNAL - Only {indicator} indicates buying opportunity")
        else:
            print("NO BUY SIGNALS - None of the indicators show buying opportunity")
            
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze stocks for buy signals from all three technical indicators")
    parser.add_argument("-t", "--ticker", help="Analyze a specific ticker")
    parser.add_argument("-d", "--days", type=int, default=365, help="Number of days of historical data to use")
    parser.add_argument("-s", "--silent", action="store_true", help="Run in silent mode (no progress updates)")
    args = parser.parse_args()
    
    if args.ticker:
        # Analyze a specific ticker
        print_ticker_details(args.ticker.upper(), days=args.days)
    else:
        # Get combined buy signals for all tickers
        get_combined_buy_signals(
            days=args.days, 
            verbose=not args.silent
        )