#!/usr/bin/env python3
import json
import sys
import os
from rsi import calculate_ticker_rsi

def get_rsi_data_for_api(ticker_symbol="SPY", days=180):
    """
    Get RSI data for a ticker and format it for the API.
    
    Returns:
    --------
    dict: A dictionary with RSI data in a format suitable for charting
    """
    try:
        # Use absolute path to the database file
        db_path = os.path.join('/home/rohan/code/finance_analysis', 'stock_data.db')
        
        # Get the RSI data
        price_data, rsi_data = calculate_ticker_rsi(ticker_symbol, window=14, db_name=db_path, days=days)
        
        if price_data is None or rsi_data is None:
            return {"error": f"Failed to calculate RSI for {ticker_symbol}"}
        
        # Format the data for charting
        dates = [d.strftime("%Y-%m-%d") for d in price_data.index]
        
        return {
            "ticker": ticker_symbol,
            "dates": dates,
            "price": price_data["close"].tolist(),
            "rsi": rsi_data.tolist(),
            "overbought_line": [70] * len(dates),  # Overbought threshold line
            "oversold_line": [30] * len(dates)     # Oversold threshold line
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Get ticker from command line arguments or use SPY as default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    # Get the data
    data = get_rsi_data_for_api(ticker)
    
    # Print as JSON so it can be captured by the Node.js process
    print(json.dumps(data)) 