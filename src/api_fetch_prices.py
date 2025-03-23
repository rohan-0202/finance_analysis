#!/usr/bin/env python3
import json
import sys
import os
from datetime import datetime
from db_util import get_historical_data

def get_price_data_for_api(ticker_symbol="SPY", days=365):
    """
    Get price history data for a ticker and format it for the API.
    
    Returns:
    --------
    dict: A dictionary with price data in a format suitable for charting
    """
    try:
        # Use absolute path to the database file
        db_path = os.path.join('/home/rohan/code/finance_analysis', 'stock_data.db')
        
        # Get the historical price data
        price_data = get_historical_data(ticker_symbol, db_path, days)
        
        if price_data is None or price_data.empty:
            return {"error": f"Failed to get price data for {ticker_symbol}"}
        
        # Calculate a 200-day moving average if we have enough data
        if len(price_data) >= 200:
            price_data['ma200'] = price_data['close'].rolling(window=200).mean()
        else:
            # Use the maximum window size we can if we have less than 200 days
            max_window = min(len(price_data) // 2, 50)  # Use at least some smoothing
            price_data['ma200'] = price_data['close'].rolling(window=max_window).mean() if max_window > 0 else price_data['close']
        
        # Format the data for charting
        dates = [d.strftime("%Y-%m-%d") for d in price_data.index]
        
        return {
            "ticker": ticker_symbol,
            "dates": dates,
            "open": price_data["open"].tolist(),
            "high": price_data["high"].tolist(),
            "low": price_data["low"].tolist(),
            "close": price_data["close"].tolist(),
            "volume": price_data["volume"].tolist(),
            "ma200": price_data["ma200"].fillna(price_data['close']).tolist()  # Fill NaN values at the beginning
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Get ticker from command line arguments or use SPY as default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    
    # Get the data
    data = get_price_data_for_api(ticker)
    
    # Print as JSON so it can be captured by the Node.js process
    print(json.dumps(data)) 