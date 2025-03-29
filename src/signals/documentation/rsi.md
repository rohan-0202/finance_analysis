
# Relative Strength Index (RSI): A Comprehensive Guide

## Introduction

The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. Developed by J. Welles Wilder Jr. and introduced in his 1978 book "New Concepts in Technical Trading Systems," RSI has become one of the most popular and widely used technical indicators in financial markets.

## What is RSI?

RSI is a momentum oscillator that ranges from 0 to 100. It provides signals about:

- **Overbought/oversold conditions**: When RSI exceeds 70, the asset may be considered overbought; when it falls below 30, the asset may be considered oversold.
- **Momentum shifts**: Changes in the direction of RSI can signal potential trend reversals.
- **Divergence**: When price makes a new high or low that isn't confirmed by the RSI, it may indicate a weakening trend.

## How RSI is Calculated

RSI is calculated using the following formula:

```
RSI = 100 - (100 / (1 + RS))
```

Where:
- **RS (Relative Strength)** = Average Gain / Average Loss over a specified period (typically 14 periods)
- **Average Gain** = Sum of gains over the specified period / number of periods
- **Average Loss** = Sum of losses over the specified period / number of periods

### Step-by-Step Calculation

1. Calculate price changes (delta) between consecutive periods
2. Separate gains (positive changes) and losses (negative changes)
3. Calculate the average gain and average loss over the specified window
4. Calculate RS (Relative Strength) as average gain divided by average loss
5. Apply the RSI formula to convert RS to a value between 0 and 100

### Implementation in Code

Looking at the implementation in `src/signals/rsi.py`, the calculation is performed in the `calculate_rsi` function:

```python
def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # Calculate price changes
    delta = series.diff()

    # Create separate series for gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and average loss over the window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi
```

Key implementation details:
- The code uses pandas to handle time series data efficiently
- A check for division by zero is included by replacing zero values with a very small number
- The default window period is 14, which is the standard in technical analysis

## Interpreting RSI Values

The RSI oscillates between 0 and 100, with several key levels:

| RSI Value | Interpretation |
|-----------|----------------|
| Above 70  | Overbought - potential sell signal |
| Below 30  | Oversold - potential buy signal |
| 50        | Neutral - neither overbought nor oversold |

### Signal Generation

In the provided code, signals are generated when the RSI crosses these thresholds:

```python
# Identify buy signals (crossing up through oversold level)
buy_signals = (data["rsi"] > oversold) & (data["rsi"].shift() <= oversold)

# Identify sell signals (crossing down through overbought level)
sell_signals = (data["rsi"] < overbought) & (data["rsi"].shift() >= overbought)
```

This means:
- A **buy signal** occurs when RSI crosses upward through the oversold level (typically 30)
- A **sell signal** occurs when RSI crosses downward through the overbought level (typically 70)

## Advanced RSI Concepts

### RSI Divergence

Divergence occurs when the price of an asset and the RSI move in opposite directions:

- **Bullish divergence**: Price makes a lower low, but RSI makes a higher low
- **Bearish divergence**: Price makes a higher high, but RSI makes a lower high

Divergences can signal potential trend reversals.

### RSI Failure Swings

Failure swings are another signal used with RSI:

- **Bullish failure swing**: RSI falls below 30, rises, pulls back but stays above 30, then breaks its previous high
- **Bearish failure swing**: RSI rises above 70, falls, rallies but stays below 70, then breaks its previous low

### Adjusting RSI Parameters

The standard RSI uses a 14-period lookback window and 70/30 thresholds, but these can be adjusted:

- **Shorter windows** (e.g., 5-9 periods) create more volatile RSI values and generate more signals
- **Longer windows** (e.g., 20-25 periods) create smoother RSI values and generate fewer signals
- **Different thresholds** can be used based on market conditions (e.g., 80/20 for trending markets)

## Implementation Details in the Provided Code

The provided Python code offers several functions for working with RSI:

1. `calculate_rsi`: Calculates the RSI values for a price series
2. `calculate_ticker_rsi`: Retrieves historical data for a ticker and calculates its RSI
3. `get_rsi_signals`: Identifies buy/sell signals based on RSI crossovers
4. `get_latest_rsi_signal`: Gets only the most recent signal for a ticker

The code also includes a command-line interface that allows users to:
- Check RSI values and signals for a specific ticker
- Process multiple tickers to find recent RSI signals

## Limitations of RSI

Despite its popularity, RSI has some limitations:

1. **False signals**: RSI can remain in overbought or oversold territory for extended periods during strong trends
2. **Lagging indicator**: As a momentum oscillator, RSI is based on past price movements
3. **Market conditions**: RSI works best in range-bound markets and may be less effective in strong trends

## Best Practices for Using RSI

1. **Combine with other indicators**: Use RSI alongside other technical indicators for confirmation
2. **Consider the timeframe**: RSI signals on higher timeframes (e.g., daily, weekly) tend to be more reliable
3. **Adapt to market conditions**: Adjust RSI parameters based on the volatility and trend of the market
4. **Look for confluence**: The strongest signals occur when multiple technical factors align

## Conclusion

RSI is a powerful momentum oscillator that helps traders identify potential turning points in the market. By measuring the velocity and magnitude of directional price movements, it provides valuable insights into overbought and oversold conditions. The implementation in the provided code offers a practical way to calculate RSI values and generate trading signals based on RSI crossovers.

While RSI is valuable on its own, it's most effective when used as part of a comprehensive trading strategy that includes other technical indicators, fundamental analysis, and sound risk management practices.
