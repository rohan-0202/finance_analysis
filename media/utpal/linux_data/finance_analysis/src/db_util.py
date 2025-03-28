import pandas as pd

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True) 