from signals.signal_factory import SignalFactory

# Create different signal types through the factory
rsi_signal = SignalFactory.create_signal("rsi", window=14)
macd_signal = SignalFactory.create_signal("macd", fast_period=12, slow_period=26)
obv_signal = SignalFactory.create_signal("obv")

# Use the signals with a consistent interface
ticker = "AAPL"

# Get the latest signal for each indicator
rsi_latest = rsi_signal.get_latest_signal(ticker)
macd_latest = macd_signal.get_latest_signal(ticker)
obv_latest = obv_signal.get_latest_signal(ticker)

# Print results
for name, signal in [("RSI", rsi_latest), ("MACD", macd_latest), ("OBV", obv_latest)]:
    if signal:
        print(
            f"{name}: {signal['type']} signal on {signal['date']} at ${signal['price']:.2f}"
        )
    else:
        print(f"{name}: No recent signals")
