from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from signals.base_signal import BaseSignal, SignalData

try:
    import torch
    from chronos import ChronosPipeline
except ImportError:
    torch = None


class TimeseriesSignal(BaseSignal):
    def __init__(self, model_name: str, prediction_length: int = 60):
        super().__init__()
        self.prediction_length = prediction_length
        if torch is None:
            raise ImportError(
                "PyTorch is required for TimeseriesSignal. Please install it."
            )

        # Use default model if none provided
        if model_name is None:
            model_name = "amazon/chronos-t5-large"
        self.model_name = model_name

        # Load pipeline with automatic device mapping
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"Error loading Chronos model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to load Chronos model: {e}") from e

    def calculate_signal(self, series: pd.Series) -> Optional[SignalData]:
        """
        Calculate the timeseries forecast for a given price series and generate a signal.

        Args:
            series: A pandas Series of historical prices, indexed by date.

        Returns:
            A SignalData object indicating BUY or HOLD based on forecast vs current price,
            or None if calculation fails or input is invalid.
        """
        if torch is None:
            print("Torch not available, cannot calculate timeseries signal.")
            return None

        if series is None or series.empty:
            print("Input series is empty or None.")
            return None

        # --- Start Data Cleaning ---
        # Ensure index is datetime and remove NaT indices
        try:
            series.index = pd.to_datetime(series.index, errors="coerce")
            series = series[pd.notna(series.index)]
        except Exception as e:
            print(f"Error converting index to datetime or filtering NaT: {e}")
            return None

        # Remove rows with NaN values in the price series itself
        series = series.dropna()

        if series.empty:
            print("Input series is empty after cleaning NaT/NaN.")
            return None
        # --- End Data Cleaning ---

        current_price = series.iloc[-1]
        # Convert cleaned series values to tensor
        context = torch.tensor(series.values, dtype=torch.bfloat16)

        # Ensure context has at least 2 dimensions [batch, sequence_length]
        if context.ndim == 1:
            context = context.unsqueeze(0)

        # Generate forecast
        # forecast shape [num_series, num_samples, prediction_length]
        try:
            forecast_tensor = self.pipeline.predict(
                context,
                self.prediction_length,
            )
        except Exception as e:
            print(f"Error during Chronos prediction: {e}")
            return None

        # Calculate the median forecast across samples
        # Move forecast tensor *result* to CPU for numpy operations
        # Ensure forecast_tensor is not empty and has the expected dimensions before processing
        if (
            forecast_tensor is None
            or forecast_tensor.numel() == 0
            or forecast_tensor.ndim < 3
        ):
            print("Forecast tensor is invalid or empty.")
            return None

        median_forecast = np.quantile(forecast_tensor[0].cpu().numpy(), 0.5, axis=0)

        # Calculate the average of the median forecast values
        average_forecast_price = np.mean(median_forecast)

        # Get the date of the last observation (current date for the signal)
        signal_date = series.index[-1]

        # Check if the average forecast price is greater than the current price
        if average_forecast_price > current_price:
            return SignalData(
                date=signal_date,
                signal_type="BUY",  # Or use a more specific enum/string if defined
                details={
                    "current_price": current_price,
                    "average_forecast_price": average_forecast_price,
                    "prediction_length": self.prediction_length,
                    "model": self.model_name,
                },
            )
        else:
            # Optionally return a HOLD or SELL signal, or None if no action is indicated
            return SignalData(
                date=signal_date,
                signal_type="HOLD",  # Or None if no signal should be generated
                details={
                    "current_price": current_price,
                    "average_forecast_price": average_forecast_price,
                    "prediction_length": self.prediction_length,
                    "model": self.model_name,
                },
            )

    def calculate_indicator(
        self, ticker_symbol: str, **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Calculate the timeseries forecast for a given ticker.
        This method might not directly map to a traditional 'indicator' but
        will prepare the forecast data.
        """
        pass

    def get_signals(self, ticker_symbol: str, **kwargs) -> List[SignalData]:
        """
        Generate buy/sell signals based on the timeseries forecast.
        """
        pass

    def get_latest_signal(self, ticker_symbol: str, **kwargs) -> Optional[SignalData]:
        """
        Get the most recent signal based on the latest forecast.
        """
        pass

    def get_status_text(
        self, price_data: pd.DataFrame, indicator_data: pd.DataFrame
    ) -> str:
        """
        Generate a text description of the current timeseries forecast status.
        """
        pass


if __name__ == "__main__":
    # Import necessary function for fetching data
    # Ensure db_util is accessible from this path or adjust import accordingly
    try:
        from db_util import get_historical_data
    except ImportError:
        print("Could not import get_historical_data. Make sure db_util is in the path.")
        # As a fallback, create dummy data for basic testing
        dates = pd.date_range(
            end=datetime.now(), periods=200, freq="B"
        )  # Approx last 200 business days
        price_data = pd.Series(
            np.random.rand(200) * 100 + 300, index=dates
        )  # Simulate price data
        print("Using dummy data for testing.")
    else:
        # Fetch historical data for SPY (e.g., last 2 years)
        try:
            # Using 500 days for a reasonable context window
            df = get_historical_data(ticker_symbol="SPY", days=500)
            if df is None or df.empty:
                raise ValueError("No data returned for SPY")
            price_data = df["close"]
            print(
                f"Fetched SPY data. Last date: {price_data.index[-1]}, Last close: {price_data.iloc[-1]:.2f}"
            )
        except Exception as e:
            print(f"Error fetching SPY data: {e}")
            print("Using dummy data for testing instead.")
            dates = pd.date_range(end=datetime.now(), periods=200, freq="B")
            price_data = pd.Series(np.random.rand(200) * 100 + 300, index=dates)

    # Instantiate the signal generator - try different models if needed
    # Using 'small' model for potentially faster testing, change if needed
    signal_generator = TimeseriesSignal(
        model_name="amazon/chronos-t5-small", prediction_length=30
    )  # Predict 30 days ahead

    # Calculate the signal based on the fetched price data
    print("Calculating signal...")
    signal = signal_generator.calculate_signal(price_data)
    print(signal)

    # # Print the result
    # if signal:
    #     print("\n--- Signal Calculated ---")
    #     print(f"Date: {signal.date}")
    #     print(f"Type: {signal.signal_type}")
    #     print("Details:")
    #     for key, value in signal.details.items():
    #         if isinstance(value, float):
    #             print(f"  {key}: {value:.2f}")
    #         else:
    #             print(f"  {key}: {value}")
    # else:
    #     print("\nNo signal generated.")
