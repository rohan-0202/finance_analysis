#!/usr/bin/env python3
import json
import sys
import os
from datetime import datetime
from macd import calculate_macd

def get_macd_data_for_api(ticker_symbol="SPY", days=180):
    """
    Get MACD data for a ticker and format it for the API.
    
    Returns:
    --------
    dict: A dictionary with MACD data in a format suitable for charting
    """
    try:
        # Use absolute path to the database file
        db_path = os.path.join('/home/rohan/code/finance_analysis', 'stock_data.db')
        
        # Get the MACD data
        price_data, macd_data = calculate_macd(ticker_symbol, fast_period=12, slow_period=26, signal_period=9, db_name=db_path, days=days)
        
        if price_data is None or macd_data is None:
            return {"error": f"Failed to calculate MACD for {ticker_symbol}"}
        
        # Format the data for charting
        dates = [d.strftime("%Y-%m-%d") for d in price_data.index]
        
        return {
            "ticker": ticker_symbol,
            "dates": dates,
            "price": price_data["close"].tolist(),
            "macd": macd_data["macd"].tolist(),
            "signal": macd_data["signal"].tolist(),
            "histogram": macd_data["histogram"].tolist()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Get ticker from command line arguments or use SPY as default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    # Get the data
    data = get_macd_data_for_api(ticker)
    
    # Print as JSON so it can be captured by the Node.js process
    print(json.dumps(data)) 