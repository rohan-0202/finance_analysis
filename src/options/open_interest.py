import pandas as pd


def analyze_open_interest_changes(options_data):
    """Identify options with significant open interest levels.

    This function analyzes open interest (OI) across different option contracts to detect
    potential institutional positioning. High open interest can indicate where major market
    participants have established positions.

    Logic:
    1. Group options data by option type (put/call) and strike price
    2. For each group, extract the most recent data point
    3. Aggregate key data from each group into a list of dictionaries
    4. Convert the list to a DataFrame for analysis
    5. Filter for options with significant open interest (>1000 contracts)

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data with an index of expiration dates
        and columns including 'option_type', 'strike', 'open_interest', and 'volume'

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing options with significant open interest (>1000),
        with columns for expiration date, option type, strike price, open interest and volume
    """
    # Group by expiration date, option type and strike
    groups = options_data.groupby(["option_type", "strike"])

    signals = []
    # Get only the most recent data point for each group
    for name, group in groups:
        if len(group) > 0:
            option_type, strike = name
            latest = group.iloc[-1]
            signals.append(
                {
                    "expiration": latest.name,
                    "option_type": option_type,
                    "strike": strike,
                    "open_interest": latest["open_interest"],
                    "volume": latest["volume"],
                }
            )

    # Convert to DataFrame
    if signals:
        signals_df = pd.DataFrame(signals)
        # Find significant open interest (OI > 1000)
        significant_oi = signals_df[signals_df["open_interest"] > 1000]
        return significant_oi

    return pd.DataFrame()
