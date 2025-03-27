

def calculate_put_call_ratio(options_data):
    """Calculate and analyze put/call volume and open interest ratios.

    This function computes ratios between put and call options to gauge market sentiment.
    High put/call ratios typically indicate bearish sentiment, while low ratios suggest
    bullish sentiment. Extreme values may also signal potential contrarian opportunities.

    Logic:
    1. Group options data by option type (put/call)
    2. Extract separate DataFrames for calls and puts
    3. Calculate total volume and open interest for each option type
    4. Compute the put/call ratio for both volume and open interest
    5. Handle edge cases (no calls or no puts) gracefully
    6. Return both ratios along with underlying metrics for detailed analysis

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data with a column 'option_type' indicating
        'call' or 'put', as well as 'volume' and 'open_interest' columns

    Returns:
    -------
    dict
        Dictionary containing:
        - 'volume_put_call_ratio': Ratio of put volume to call volume
        - 'oi_put_call_ratio': Ratio of put open interest to call open interest
        - 'put_volume': Total put volume
        - 'call_volume': Total call volume
        - 'put_oi': Total put open interest
        - 'call_oi': Total call open interest
        Returns None for ratios if either call or put data is missing
    """
    # Group by option type
    grouped = options_data.groupby("option_type")

    try:
        # Get total volume and open interest for puts and calls
        call_data = grouped.get_group("call")
        put_data = grouped.get_group("put")

        call_volume = call_data["volume"].sum()
        put_volume = put_data["volume"].sum()

        call_oi = call_data["open_interest"].sum()
        put_oi = put_data["open_interest"].sum()

        # Calculate ratios
        volume_pc_ratio = put_volume / call_volume if call_volume > 0 else float("inf")
        oi_pc_ratio = put_oi / call_oi if call_oi > 0 else float("inf")

        return {
            "volume_put_call_ratio": volume_pc_ratio,
            "oi_put_call_ratio": oi_pc_ratio,
            "put_volume": put_volume,
            "call_volume": call_volume,
            "put_oi": put_oi,
            "call_oi": call_oi,
        }
    except KeyError:
        # If there are no puts or calls
        return {
            "volume_put_call_ratio": None,
            "oi_put_call_ratio": None,
            "put_volume": 0,
            "call_volume": 0,
            "put_oi": 0,
            "call_oi": 0,
        }
