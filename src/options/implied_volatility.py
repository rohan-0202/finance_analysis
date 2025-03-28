from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from db_util import get_historical_data


def analyze_implied_volatility(options_data, ticker=None):
    """Analyze implied volatility signals in options data.

    This function performs comprehensive implied volatility analysis to identify
    potential trading opportunities and market sentiment signals. The analysis includes:

    1. IV Percentile/Rank: Identifying where current IV stands relative to its historical range,
       which helps determine if options are relatively expensive or cheap.

    2. Volatility Skew: Examining the difference between put and call IV at the same distance
       from the money, which indicates market sentiment and tail risk expectations.

    3. IV Term Structure: Analyzing how IV changes across different expiration dates,
       which reveals the market's expected volatility over different time horizons.

    4. IV Run-up Detection: Identifying patterns where IV increases substantially,
       which often occurs before significant events like earnings announcements.

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data with an index of expiration dates
        and columns including 'option_type', 'strike', 'implied_volatility',
        'last_price', 'volume', etc.
    ticker : str, optional
        The ticker symbol, used to get historical price data for more accurate ATM option identification.

    Returns:
    -------
    dict
        Dictionary containing the results of various IV analyses:
        - 'iv_percentile': Current IV percentile compared to historical data
        - 'iv_rank': Current IV rank (0-100)
        - 'skew': Volatility skew data showing put-call IV differences
        - 'term_structure': IV values across different expiration dates
        - 'iv_run_up': Information about recent significant IV increases
        - 'iv_signals': List of trading signals derived from IV analysis
    """
    # Initialize results dictionary
    results = {
        "iv_percentile": None,
        "iv_rank": None,
        "skew": None,
        "term_structure": None,
        "iv_run_up": None,
        "iv_signals": [],
    }

    # Check if we have enough data
    if options_data.empty:
        return results

    # Get historical price data if ticker is provided
    current_price = None
    if ticker:
        try:
            historical_data = get_historical_data(ticker)
            if not historical_data.empty:
                # Get the latest close price
                current_price = historical_data["close"].iloc[-1]
        except (ValueError, KeyError):
            # If there's an error getting historical data, continue without it
            pass

    # Calculate IV percentile and rank
    iv_percentile_results = calculate_iv_percentile(options_data, current_price)
    results["iv_percentile"] = iv_percentile_results["percentile"]
    results["iv_rank"] = iv_percentile_results["rank"]

    # Analyze volatility skew
    results["skew"] = analyze_volatility_skew(options_data)

    # Analyze IV term structure
    results["term_structure"] = analyze_iv_term_structure(options_data)

    # Detect IV run-up (potential pre-earnings or pre-event IV increase)
    results["iv_run_up"] = detect_iv_run_up(options_data)

    # Generate signals based on the IV analysis
    results["iv_signals"] = generate_iv_signals(results)

    return results


def calculate_iv_percentile(options_data, current_price=None):
    """Calculate the current implied volatility percentile and rank compared to historical data.

    This function:
    1. Calculates the average IV for at-the-money (ATM) options
    2. Compares current IV levels to their historical range
    3. Calculates both IV percentile and IV rank

    IV Percentile: Percentage of days where IV was lower than current IV
    IV Rank: Current IV's position within its historical range on a scale of 0-100

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data
    current_price : float, optional
        Current price of the underlying asset, used for more accurate ATM option identification

    Returns:
    -------
    dict
        Dictionary containing:
        - 'current_iv': Current average ATM IV
        - 'percentile': IV percentile (0-100)
        - 'rank': IV rank (0-100)
        - 'historical_range': (min_iv, max_iv) tuple
    """
    # Group by date to get the IV for each day
    # Use the last_updated column to group, since we want to analyze IV changes over time
    daily_data = options_data.reset_index().copy()
    daily_data.loc[:, "date"] = pd.to_datetime(daily_data["last_updated"]).dt.date

    # Filter for at-the-money options only (easier to compare)
    # If we have current_price, use it; otherwise, estimate it
    def estimate_price_for_day(day_data, provided_price=None):
        if (
            provided_price is not None
            and day_data["date"].iloc[0] == datetime.now().date()
        ):
            return provided_price

        # Fallback to estimation method if no price is provided or for historical dates
        call_data = day_data[day_data["option_type"] == "call"]
        put_data = day_data[day_data["option_type"] == "put"]

        if call_data.empty or put_data.empty:
            return None

        # Using highest strike with call bid > 0 and lowest strike with put bid > 0
        # as a simple estimation method
        valid_calls = call_data[call_data["bid"] > 0]
        valid_puts = put_data[put_data["bid"] > 0]

        if valid_calls.empty or valid_puts.empty:
            return None

        highest_call = valid_calls["strike"].max()
        lowest_put = valid_puts["strike"].min()

        return (highest_call + lowest_put) / 2

    # Group by date and get estimated price
    daily_price = daily_data.groupby("date").apply(
        estimate_price_for_day, provided_price=current_price
    )

    # Get ATM IV for each day
    atm_iv_history = []

    for date, price in daily_price.items():
        if price is None:
            continue

        day_data = daily_data[daily_data["date"] == date].copy()

        # Find options closest to ATM
        day_data.loc[:, "distance_from_atm"] = abs(day_data["strike"] - price)
        closest_options = day_data.nsmallest(4, "distance_from_atm")

        # Calculate average IV of these ATM options
        if not closest_options.empty:
            atm_iv = closest_options["implied_volatility"].mean()
            atm_iv_history.append((date, atm_iv))

    # Convert to DataFrame for easier analysis
    if atm_iv_history:
        iv_history_df = pd.DataFrame(atm_iv_history, columns=["date", "atm_iv"])

        # Calculate current IV (most recent date)
        current_iv = iv_history_df.iloc[-1]["atm_iv"]

        # Calculate IV percentile
        iv_percentile = (
            sum(iv_history_df["atm_iv"] < current_iv) / len(iv_history_df) * 100
        )

        # Calculate IV rank
        min_iv = iv_history_df["atm_iv"].min()
        max_iv = iv_history_df["atm_iv"].max()

        # Avoid division by zero
        if max_iv == min_iv:
            iv_rank = 50  # Default to middle if all values are the same
        else:
            iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100

        return {
            "current_iv": current_iv,
            "percentile": iv_percentile,
            "rank": iv_rank,
            "historical_range": (min_iv, max_iv),
        }

    return {
        "current_iv": None,
        "percentile": None,
        "rank": None,
        "historical_range": (None, None),
    }


def analyze_volatility_skew(options_data):
    """Analyze the volatility skew between put and call options.

    Volatility skew refers to the difference in implied volatility between put and call options
    at the same distance from the money. A high put-call skew indicates market participants
    are paying more for downside protection, suggesting bearish sentiment or tail risk concerns.

    This function:
    1. Groups options by expiration date
    2. For each expiration, calculates IV for puts and calls at various strikes
    3. Computes the skew as the difference between put and call IV
    4. Identifies noteworthy skew patterns (e.g., unusually high put skew)

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data

    Returns:
    -------
    dict
        Dictionary containing:
        - 'current_skew': Current put-call skew value (positive means puts have higher IV)
        - 'skew_by_expiration': DataFrame of skew values for different expirations
        - 'skew_signal': String indicating if the skew suggests a bullish, bearish, or neutral outlook
    """
    # Initialize result
    result = {
        "current_skew": None,
        "skew_by_expiration": None,
        "skew_signal": "neutral",
    }

    if options_data.empty:
        return result

    # Group by expiration date
    expiration_groups = options_data.groupby(level=0)

    skew_data = []

    for expiration, group in expiration_groups:
        # Separate puts and calls
        calls = group[group["option_type"] == "call"]
        puts = group[group["option_type"] == "put"]

        if calls.empty or puts.empty:
            continue

        # Get unique strikes available in both puts and calls
        common_strikes = set(calls["strike"].unique()) & set(puts["strike"].unique())

        if not common_strikes:
            continue

        # For each common strike, calculate IV difference
        for strike in sorted(common_strikes):
            call_iv = calls[calls["strike"] == strike]["implied_volatility"].iloc[0]
            put_iv = puts[puts["strike"] == strike]["implied_volatility"].iloc[0]

            # Calculate skew (put IV - call IV)
            skew = put_iv - call_iv

            skew_data.append(
                {
                    "expiration": expiration,
                    "strike": strike,
                    "call_iv": call_iv,
                    "put_iv": put_iv,
                    "skew": skew,
                }
            )

    if skew_data:
        skew_df = pd.DataFrame(skew_data)
        result["skew_by_expiration"] = skew_df

        # Calculate average current skew for near-term expirations
        current_date = datetime.now()
        near_term = current_date + timedelta(days=30)

        near_term_skew = skew_df[skew_df["expiration"] <= near_term]

        if not near_term_skew.empty:
            result["current_skew"] = near_term_skew["skew"].mean()

            # Interpret skew
            if result["current_skew"] > 0.05:  # Puts have higher IV
                if result["current_skew"] > 0.1:
                    result["skew_signal"] = "strongly bearish"
                else:
                    result["skew_signal"] = "bearish"
            elif result["current_skew"] < -0.05:  # Calls have higher IV
                if result["current_skew"] < -0.1:
                    result["skew_signal"] = "strongly bullish"
                else:
                    result["skew_signal"] = "bullish"

    return result


def analyze_iv_term_structure(options_data):
    """Analyze the implied volatility term structure across different expiration dates.

    The IV term structure shows how implied volatility varies across different expiration dates.
    A normal term structure has higher IV for longer-dated options, while an inverted term structure
    (higher IV for shorter-dated options) often indicates expected near-term volatility.

    This function:
    1. Groups options by expiration date
    2. Calculates the average IV for ATM options in each expiration
    3. Constructs the IV term structure curve
    4. Identifies if the term structure is normal, flat, or inverted

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data

    Returns:
    -------
    dict
        Dictionary containing:
        - 'term_structure': DataFrame with average IV for each expiration date
        - 'curve_type': String indicating if the curve is 'normal', 'flat', or 'inverted'
        - 'term_structure_signal': Trading signal based on the term structure
    """
    result = {
        "term_structure": None,
        "curve_type": None,
        "term_structure_signal": "neutral",
    }

    if options_data.empty:
        return result

    # Group by expiration date
    expiration_groups = options_data.groupby(level=0)

    term_data = []

    # Calculate average ATM IV for each expiration
    for expiration, group in expiration_groups:
        calls = group[group["option_type"] == "call"].copy()
        puts = group[group["option_type"] == "put"].copy()

        if calls.empty or puts.empty:
            continue

        # Simple method to estimate ATM strikes: use the average of all strikes
        # In practice, you'd want to get the actual stock price
        all_strikes = pd.concat([calls["strike"], puts["strike"]]).unique()
        avg_strike = np.mean(all_strikes)

        # Find options closest to the estimated ATM strike
        calls.loc[:, "distance_from_atm"] = abs(calls["strike"] - avg_strike)
        puts.loc[:, "distance_from_atm"] = abs(puts["strike"] - avg_strike)

        atm_calls = calls.nsmallest(2, "distance_from_atm")
        atm_puts = puts.nsmallest(2, "distance_from_atm")

        # Average IV of ATM calls and puts
        if not atm_calls.empty and not atm_puts.empty:
            atm_call_iv = atm_calls["implied_volatility"].mean()
            atm_put_iv = atm_puts["implied_volatility"].mean()
            avg_atm_iv = (atm_call_iv + atm_put_iv) / 2

            # Calculate days to expiration
            days_to_expiration = (expiration - datetime.now()).days

            term_data.append(
                {
                    "expiration": expiration,
                    "days_to_expiration": max(
                        0, days_to_expiration
                    ),  # Ensure non-negative
                    "avg_iv": avg_atm_iv,
                }
            )

    if term_data:
        # Create term structure DataFrame
        term_df = pd.DataFrame(term_data).sort_values("days_to_expiration")
        result["term_structure"] = term_df

        # Need at least two expirations to determine curve type
        if len(term_df) >= 2:
            # Get short-term (near) and long-term (far) IV
            near_iv = term_df.iloc[0]["avg_iv"]
            far_iv = term_df.iloc[-1]["avg_iv"]

            # Determine curve type
            if near_iv > far_iv * 1.1:  # Near IV is significantly higher
                result["curve_type"] = "inverted"
                result["term_structure_signal"] = "potential near-term volatility"
            elif far_iv > near_iv * 1.1:  # Far IV is significantly higher
                result["curve_type"] = "normal"
                result["term_structure_signal"] = "stable near-term expectations"
            else:
                result["curve_type"] = "flat"
                result["term_structure_signal"] = "neutral"

    return result


def detect_iv_run_up(options_data):
    """Detect significant increases in implied volatility (IV run-up).

    IV run-up often occurs before anticipated events like earnings announcements.
    This function identifies patterns where IV has increased substantially in recent days,
    which can be used to identify potential event-driven trading opportunities.

    # TODO: In the future, enhance this function to incorporate earnings announcement dates
    # to specifically identify pre-earnings IV run-ups versus other catalysts.

    This function:
    1. Tracks IV changes over time for near-term options
    2. Identifies significant increases in IV
    3. Flags potential pre-event IV run-ups

    Parameters:
    ----------
    options_data : pandas.DataFrame
        DataFrame containing options data

    Returns:
    -------
    dict
        Dictionary containing:
        - 'detected': Boolean indicating if an IV run-up was detected
        - 'magnitude': Percentage increase in IV if detected
        - 'run_up_signal': Trading signal based on the IV run-up analysis
    """
    result = {"detected": False, "magnitude": None, "run_up_signal": None}

    if options_data.empty:
        return result

    # Reset index to access the expiration date as a column
    data = options_data.reset_index().copy()

    # Convert last_updated to datetime with better error handling
    # First, check if the column exists
    if "last_updated" not in data.columns:
        # If there's no last_updated column, we can't continue the analysis
        return result

    # Try to convert with robust error handling
    try:
        data.loc[:, "last_updated"] = pd.to_datetime(
            data["last_updated"], errors="coerce"
        )

        # Check if conversion worked and we have valid datetime values
        if data["last_updated"].notna().any():
            # Only proceed if we have some valid datetime values
            # Group by update date to track IV over time
            data.loc[:, "update_date"] = data["last_updated"].dt.date
        else:
            # No valid datetime values
            return result
    except Exception:
        # If conversion fails completely, return without analysis
        return result

    # Focus on near-term options (next 30 days)
    current_date = datetime.now()
    near_term_date = current_date + timedelta(days=30)
    near_term_options = data[data["expiration_date"] <= near_term_date].copy()

    if near_term_options.empty:
        return result

    # Group by update date and calculate average IV
    daily_iv = near_term_options.groupby("update_date")["implied_volatility"].mean()

    # Convert to DataFrame for easier manipulation
    daily_iv_df = daily_iv.reset_index()
    daily_iv_df.columns = ["date", "avg_iv"]

    # Sort by date
    daily_iv_df = daily_iv_df.sort_values("date")

    # Need at least two dates to calculate change
    if len(daily_iv_df) >= 2:
        # Calculate IV change over different periods
        current_iv = daily_iv_df.iloc[-1]["avg_iv"]

        # 1-day change
        if len(daily_iv_df) >= 2:
            prev_day_iv = daily_iv_df.iloc[-2]["avg_iv"]
            one_day_change = (current_iv - prev_day_iv) / prev_day_iv * 100
        else:
            one_day_change = 0

        # 5-day change (or as many days as available)
        lookback = min(5, len(daily_iv_df) - 1)
        if lookback > 0:
            prev_period_iv = daily_iv_df.iloc[-1 - lookback]["avg_iv"]
            period_change = (current_iv - prev_period_iv) / prev_period_iv * 100
        else:
            period_change = 0

        # Detect significant IV run-up
        # A run-up is defined as an increase of 20% or more in 5 days
        # or an increase of 10% or more in 1 day
        significant_runup = (period_change >= 20) or (one_day_change >= 10)

        if significant_runup:
            result["detected"] = True
            result["magnitude"] = max(one_day_change, period_change)

            # Generate signal
            if result["magnitude"] > 30:
                result["run_up_signal"] = (
                    "Strong IV run-up detected - potential pre-earnings or event-driven volatility"
                )
            else:
                result["run_up_signal"] = (
                    "Moderate IV run-up detected - monitor for upcoming events"
                )

    return result


def generate_iv_signals(iv_analysis):
    """Generate trading signals based on the comprehensive IV analysis.

    This function combines the insights from various IV analyses to generate
    actionable trading signals. It considers:
    - IV percentile/rank: Identifies if options are relatively expensive or cheap
    - Volatility skew: Provides insights into market sentiment
    - Term structure: Shows expected volatility across different time horizons
    - IV run-up: Flags potential upcoming events

    Parameters:
    ----------
    iv_analysis : dict
        Dictionary containing results from all IV analyses

    Returns:
    -------
    list
        List of dictionaries, each containing a trading signal with:
        - 'signal': String describing the signal
        - 'type': Signal type ('bullish', 'bearish', or 'neutral')
        - 'source': Which analysis generated the signal
        - 'strength': Signal strength (1-5, with 5 being strongest)
    """
    signals = []

    # Generate signals from IV percentile/rank
    if iv_analysis["iv_percentile"] is not None:
        if iv_analysis["iv_percentile"] > 80:
            signals.append(
                {
                    "signal": f"IV is in the {iv_analysis['iv_percentile']:.1f}th percentile - options are relatively expensive",
                    "type": "bearish",
                    "source": "iv_percentile",
                    "strength": 4,
                }
            )
            signals.append(
                {
                    "signal": "Consider selling premium strategies (e.g., credit spreads, iron condors)",
                    "type": "neutral",
                    "source": "iv_percentile",
                    "strength": 3,
                }
            )
        elif iv_analysis["iv_percentile"] < 20:
            signals.append(
                {
                    "signal": f"IV is in the {iv_analysis['iv_percentile']:.1f}th percentile - options are relatively cheap",
                    "type": "bullish",
                    "source": "iv_percentile",
                    "strength": 4,
                }
            )
            signals.append(
                {
                    "signal": "Consider buying premium strategies (e.g., straddles, strangles)",
                    "type": "neutral",
                    "source": "iv_percentile",
                    "strength": 3,
                }
            )

    # Generate signals from volatility skew
    if iv_analysis["skew"] and iv_analysis["skew"]["current_skew"] is not None:
        skew_value = iv_analysis["skew"]["current_skew"]
        if skew_value > 0.1:
            signals.append(
                {
                    "signal": f"High put-call skew ({skew_value:.2f}) indicates strong demand for downside protection",
                    "type": "bearish",
                    "source": "volatility_skew",
                    "strength": min(
                        5, int(skew_value * 20)
                    ),  # Scale strength based on skew magnitude
                }
            )
        elif skew_value < -0.1:
            signals.append(
                {
                    "signal": f"Negative skew ({skew_value:.2f}) indicates higher call IV than puts - bullish sentiment",
                    "type": "bullish",
                    "source": "volatility_skew",
                    "strength": min(5, int(abs(skew_value) * 20)),
                }
            )

    # Generate signals from term structure
    if iv_analysis["term_structure"] and iv_analysis["term_structure"]["curve_type"]:
        curve_type = iv_analysis["term_structure"]["curve_type"]
        if curve_type == "inverted":
            signals.append(
                {
                    "signal": "Inverted IV term structure suggests expected near-term volatility",
                    "type": "neutral",
                    "source": "term_structure",
                    "strength": 3,
                }
            )
            signals.append(
                {
                    "signal": "Consider strategies that benefit from volatility (e.g., long straddle)",
                    "type": "neutral",
                    "source": "term_structure",
                    "strength": 2,
                }
            )
        elif curve_type == "normal":
            signals.append(
                {
                    "signal": "Normal IV term structure indicates stable short-term expectations",
                    "type": "neutral",
                    "source": "term_structure",
                    "strength": 2,
                }
            )

    # Generate signals from IV run-up
    if iv_analysis["iv_run_up"] and iv_analysis["iv_run_up"]["detected"]:
        magnitude = iv_analysis["iv_run_up"]["magnitude"]
        signals.append(
            {
                "signal": f"IV run-up detected ({magnitude:.1f}% increase) - potential upcoming catalyst",
                "type": "neutral",
                "source": "iv_run_up",
                "strength": min(5, int(magnitude / 10)),
            }
        )
        signals.append(
            {
                "signal": "Consider selling premium strategies ahead of event (e.g., credit spreads)",
                "type": "neutral",
                "source": "iv_run_up",
                "strength": 3,
            }
        )

    return signals
