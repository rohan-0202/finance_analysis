from datetime import timezone

import pandas as pd


def ensure_utc_tz(dt):
    """Helper function to ensure datetime is in UTC timezone."""
    if dt is None:
        return None

    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            return dt.tz_localize("UTC")
        else:
            return dt.tz_convert("UTC")
    else:  # Python datetime
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Convert existing timezone to UTC
            return dt.astimezone(timezone.utc)
