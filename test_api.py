#!/usr/bin/env python3
import requests
import json
import time

print("Waiting 5 seconds for server to start...")
time.sleep(5)

# Test the MACD chart API endpoint
print("\nTesting /api/macd endpoint...")
try:
    response = requests.get("http://localhost:3001/api/macd?ticker=SPY&days=365")
    print(f"Status code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")

# Test the MACD data API endpoint
print("\nTesting /api/macd-data endpoint...")
try:
    response = requests.get("http://localhost:3001/api/macd-data?ticker=SPY&days=365")
    print(f"Status code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")

# Test the RSI API endpoint
print("\nTesting /api/rsi endpoint...")
try:
    response = requests.get("http://localhost:3001/api/rsi?ticker=SPY&days=365")
    print(f"Status code: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {e}")

print("\nAPI testing complete.") 