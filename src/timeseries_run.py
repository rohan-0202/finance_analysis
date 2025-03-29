import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

# Import the database utility function
from db_util import get_historical_data

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",  # Use "auto" for device map
    torch_dtype=torch.bfloat16,
)

pipeline2 = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="auto",  # Use "auto" for device map
    torch_dtype=torch.bfloat16,
)


# Fetch historical data for SPY
try:
    # Fetching data for the last 5 years for better context
    df = get_historical_data(ticker_symbol="NVDA", days=365 * 5)
except ValueError as e:
    print(f"Error fetching data: {e}")
    exit()  # Exit if no data is found

# Ensure the DataFrame is sorted by timestamp if not already
df = df.sort_index()

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# Use the 'close' price column
context_series = df["close"]
context = torch.tensor(context_series.values)  # Use .values to get numpy array first
prediction_length = 60  # Predict next 60 trading days (approx 3 months)

# Generate forecast from the first pipeline
forecast1 = pipeline.predict(
    context.unsqueeze(0),
    prediction_length,  # Add batch dimension
)  # shape [num_series, num_samples, prediction_length]

# Generate forecast from the second pipeline
forecast2 = pipeline2.predict(
    context.unsqueeze(0),
    prediction_length,  # Add batch dimension
)  # shape [num_series, num_samples, prediction_length]

# visualize the forecasts
# Create a future date range for the forecast
last_date = df.index[-1]
# Generate future business days for the forecast index
forecast_index = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1), periods=prediction_length
)

# Calculate quantiles for the first forecast
low1, median1, high1 = np.quantile(
    forecast1[0].cpu().numpy(), [0.1, 0.5, 0.9], axis=0
)  # Move forecast to CPU for numpy conversion

# Calculate quantiles for the second forecast
low2, median2, high2 = np.quantile(
    forecast2[0].cpu().numpy(), [0.1, 0.5, 0.9], axis=0
)  # Move forecast to CPU for numpy conversion

plt.figure(figsize=(12, 7))  # Increased figure size slightly
plt.plot(df.index, context_series, color="royalblue", label="Historical SPY Close")

# Plot forecast 1 (small model)
plt.plot(forecast_index, median1, color="tomato", label="Median Forecast (Small Model)")
plt.fill_between(
    forecast_index,
    low1,
    high1,
    color="tomato",
    alpha=0.3,
    label="80% Prediction Interval (Small Model)",
)

# Plot forecast 2 (large model)
plt.plot(
    forecast_index, median2, color="forestgreen", label="Median Forecast (Large Model)"
)
plt.fill_between(
    forecast_index,
    low2,
    high2,
    color="forestgreen",
    alpha=0.3,
    label="80% Prediction Interval (Large Model)",
)

plt.title("SPY Close Price Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
