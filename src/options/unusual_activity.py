def analyze_unusual_activity(options_data):
    """Identify unusual options activity based on volume vs open interest ratio.

    This function detects potential smart money movements by analyzing the relationship
    between trading volume and open interest. A high volume-to-open-interest ratio
    can indicate unusual activity or institutional interest.

    Logic:
    1. Calculate the ratio of volume to open interest for each option
    2. Replace zero open interest with 1 to avoid division by zero
    3. Filter for options where volume is at least 3x the open interest
    4. Additional filter for minimum volume (>100) to eliminate low liquidity noise

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data with columns for 'volume' and 'open_interest'

    Returns:
    -------
    pandas.DataFrame
        Filtered DataFrame containing only the options with unusual activity,
        including a new 'volume_oi_ratio' column showing the volume to open interest ratio
    """
    # Calculate volume to open interest ratio
    options_data["volume_oi_ratio"] = options_data["volume"] / options_data[
        "open_interest"
    ].replace(0, 1)

    # Consider unusual when volume is at least 3x open interest and volume > 100
    unusual_activity = options_data[
        (options_data["volume_oi_ratio"] > 3) & (options_data["volume"] > 100)
    ]

    return unusual_activity
