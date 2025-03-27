import click
import numpy as np
import pandas as pd

from db_util import get_options_data


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


def generate_trading_signal(unusual_activity, significant_oi, pc_ratio):
    """Generate a trading signal (BUY, SELL, or HOLD) based on options analysis results.

    This function implements a multi-factor scoring algorithm that weighs various
    options market indicators to produce an actionable trading signal. The algorithm
    combines sentiment indicators from put/call ratios, unusual activity patterns,
    and institutional positioning through open interest analysis.

    Algorithm Logic:
    ---------------
    1. Scoring System (range -100 to +100, where positive is bullish):
       - Base score starts at 0 (neutral)
       - Each factor adds or subtracts points based on its bullish/bearish implication

    2. Unusual Activity Analysis:
       - Calculates the ratio of unusual call activity versus unusual put activity
       - Heavy call activity (>60%) adds +20 to +30 points (bullish signal)
       - Heavy put activity (>60%) subtracts -20 to -30 points (bearish signal)
       - Extremely one-sided activity (>80%) can add/subtract up to +/-40 points

    3. Open Interest Analysis:
       - Calculates the proportion of significant open interest between calls and puts
       - High call open interest adds +15 to +25 points (bullish positioning)
       - High put open interest subtracts -15 to -25 points (bearish positioning)

    4. Put/Call Ratio Analysis:
       - Volume put/call ratio below 0.7 adds +15 to +35 points (bullish)
       - Volume put/call ratio above 1.2 subtracts -15 to -35 points (bearish)
       - Extreme ratios (>2.0) may be interpreted as contrarian signals (+20 points)
       - Open interest put/call ratio serves as a secondary confirmation

    5. Signal Determination:
       - Final score > 40: BUY signal
       - Final score < -40: SELL signal
       - Otherwise: HOLD signal

    Parameters:
    ----------
    unusual_activity : pandas.DataFrame
        DataFrame containing unusual options activity as returned by analyze_unusual_activity()

    significant_oi : pandas.DataFrame
        DataFrame containing significant open interest data as returned by analyze_open_interest_changes()

    pc_ratio : dict
        Dictionary containing put/call ratio data as returned by calculate_put_call_ratio()

    Returns:
    -------
    dict
        Dictionary containing:
        - 'signal': String indicating 'BUY', 'SELL', or 'HOLD'
        - 'score': Numeric score from the algorithm (-100 to +100)
        - 'factors': Dict of individual factor contributions to the overall score
        - 'explanation': Text explanation of the primary factors affecting the decision
    """
    # Initialize score and factor tracking
    score = 0
    factors = {}
    explanation = []

    # 1. Analyze unusual activity
    if not unusual_activity.empty:
        # Count unusual activity for calls vs puts
        unusual_calls = unusual_activity[unusual_activity["option_type"] == "call"]
        unusual_puts = unusual_activity[unusual_activity["option_type"] == "put"]

        call_count = len(unusual_calls)
        put_count = len(unusual_puts)
        total_count = call_count + put_count

        if total_count > 0:
            call_percentage = call_count / total_count

            # Calculate volume_weighted percentage (gives more importance to higher volume)
            call_volume = (
                unusual_calls["volume"].sum() if not unusual_calls.empty else 0
            )
            put_volume = unusual_puts["volume"].sum() if not unusual_puts.empty else 0
            total_volume = call_volume + put_volume

            if total_volume > 0:
                volume_weighted_call_percentage = call_volume / total_volume

                # Score based on call percentage (volume weighted)
                if volume_weighted_call_percentage > 0.8:
                    activity_score = 40  # Extremely bullish
                    explanation.append(
                        f"Extremely high unusual call activity ({volume_weighted_call_percentage:.1%} of volume)"
                    )
                elif volume_weighted_call_percentage > 0.6:
                    activity_score = 30  # Strongly bullish
                    explanation.append(
                        f"Strong unusual call activity ({volume_weighted_call_percentage:.1%} of volume)"
                    )
                elif volume_weighted_call_percentage > 0.4:
                    activity_score = 10  # Mildly bullish
                    explanation.append(
                        f"Moderate unusual call activity ({volume_weighted_call_percentage:.1%} of volume)"
                    )
                elif volume_weighted_call_percentage < 0.2:
                    activity_score = -40  # Extremely bearish
                    explanation.append(
                        f"Extremely high unusual put activity ({1 - volume_weighted_call_percentage:.1%} of volume)"
                    )
                elif volume_weighted_call_percentage < 0.4:
                    activity_score = -30  # Strongly bearish
                    explanation.append(
                        f"Strong unusual put activity ({1 - volume_weighted_call_percentage:.1%} of volume)"
                    )
                else:
                    activity_score = -10  # Mildly bearish
                    explanation.append(
                        f"Moderate unusual put activity ({1 - volume_weighted_call_percentage:.1%} of volume)"
                    )

                score += activity_score
                factors["unusual_activity"] = activity_score
            else:
                factors["unusual_activity"] = 0
        else:
            factors["unusual_activity"] = 0
    else:
        factors["unusual_activity"] = 0

    # 2. Analyze significant open interest
    if not significant_oi.empty:
        call_oi = significant_oi[significant_oi["option_type"] == "call"]
        put_oi = significant_oi[significant_oi["option_type"] == "put"]

        call_total_oi = call_oi["open_interest"].sum() if not call_oi.empty else 0
        put_total_oi = put_oi["open_interest"].sum() if not put_oi.empty else 0
        total_oi = call_total_oi + put_total_oi

        if total_oi > 0:
            call_oi_percentage = call_total_oi / total_oi

            # Score based on call open interest percentage
            if call_oi_percentage > 0.7:
                oi_score = 25  # Very bullish positioning
                explanation.append(
                    f"Strong bullish institutional positioning ({call_oi_percentage:.1%} of significant OI in calls)"
                )
            elif call_oi_percentage > 0.6:
                oi_score = 15  # Bullish positioning
                explanation.append(
                    f"Bullish institutional positioning ({call_oi_percentage:.1%} of significant OI in calls)"
                )
            elif call_oi_percentage < 0.3:
                oi_score = -25  # Very bearish positioning
                explanation.append(
                    f"Strong bearish institutional positioning ({1 - call_oi_percentage:.1%} of significant OI in puts)"
                )
            elif call_oi_percentage < 0.4:
                oi_score = -15  # Bearish positioning
                explanation.append(
                    f"Bearish institutional positioning ({1 - call_oi_percentage:.1%} of significant OI in puts)"
                )
            else:
                oi_score = 0  # Neutral positioning

            score += oi_score
            factors["open_interest"] = oi_score
        else:
            factors["open_interest"] = 0
    else:
        factors["open_interest"] = 0

    # 3. Analyze put/call ratios
    if pc_ratio["volume_put_call_ratio"] is not None:
        vol_pc_ratio = pc_ratio["volume_put_call_ratio"]
        oi_pc_ratio = pc_ratio["oi_put_call_ratio"]

        # Analyze volume put/call ratio
        if vol_pc_ratio > 2.0:
            # Extremely high put/call ratio might be a contrarian buy signal
            pc_score = 20  # Contrarian bullish
            explanation.append(
                f"Extremely high put/call ratio ({vol_pc_ratio:.2f}) may indicate extreme bearish sentiment (contrarian bullish)"
            )
        elif vol_pc_ratio > 1.5:
            pc_score = -35  # Very bearish
            explanation.append(
                f"Very high put/call ratio ({vol_pc_ratio:.2f}) indicates strong bearish sentiment"
            )
        elif vol_pc_ratio > 1.2:
            pc_score = -15  # Bearish
            explanation.append(
                f"Elevated put/call ratio ({vol_pc_ratio:.2f}) indicates bearish sentiment"
            )
        elif vol_pc_ratio < 0.5:
            pc_score = 35  # Very bullish
            explanation.append(
                f"Very low put/call ratio ({vol_pc_ratio:.2f}) indicates strong bullish sentiment"
            )
        elif vol_pc_ratio < 0.7:
            pc_score = 15  # Bullish
            explanation.append(
                f"Low put/call ratio ({vol_pc_ratio:.2f}) indicates bullish sentiment"
            )
        else:
            pc_score = 0  # Neutral
            explanation.append(f"Neutral put/call ratio ({vol_pc_ratio:.2f})")

        score += pc_score
        factors["put_call_ratio"] = pc_score

        # Use open interest put/call as a secondary confirming indicator
        if oi_pc_ratio > 1.5 and vol_pc_ratio > 1.2:
            score -= 10  # Additional bearish confirmation
            factors["oi_confirmation"] = -10
            explanation.append(
                "Open interest put/call ratio confirms bearish sentiment"
            )
        elif oi_pc_ratio < 0.7 and vol_pc_ratio < 0.7:
            score += 10  # Additional bullish confirmation
            factors["oi_confirmation"] = 10
            explanation.append(
                "Open interest put/call ratio confirms bullish sentiment"
            )
        else:
            factors["oi_confirmation"] = 0
    else:
        factors["put_call_ratio"] = 0
        factors["oi_confirmation"] = 0

    # Ensure score stays within bounds
    score = max(-100, min(100, score))

    # Generate signal based on final score
    if score > 40:
        signal = "BUY"
    elif score < -40:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Create explanation text
    explanation_text = " ".join(explanation)

    return {
        "signal": signal,
        "score": score,
        "factors": factors,
        "explanation": explanation_text,
    }


@click.command()
@click.argument("ticker", type=str)
def main(ticker):
    ticker = ticker.strip().upper()

    # Display indicator data
    print(f"\nAnalyzing options for {ticker}...")

    # Get options data
    options_data = get_options_data(ticker)
    # print("\nSample of options data (first 5 rows):")
    # print(options_data.head())

    # Analyze unusual options activity
    print("\n--- UNUSUAL OPTIONS ACTIVITY ---")
    unusual_activity = analyze_unusual_activity(options_data)
    if not unusual_activity.empty:
        print(f"Found {len(unusual_activity)} instances of unusual options activity:")
        print(
            unusual_activity[
                ["option_type", "strike", "volume", "open_interest", "volume_oi_ratio"]
            ]
        )
    else:
        print("No unusual options activity detected.")

    # Analyze open interest changes
    print("\n--- SIGNIFICANT OPEN INTEREST ---")
    significant_oi = analyze_open_interest_changes(options_data)
    if not significant_oi.empty:
        print(f"Found {len(significant_oi)} options with significant open interest:")
        print(significant_oi.sort_values("open_interest", ascending=False))
    else:
        print("No significant open interest detected.")

    # Calculate put/call ratio
    print("\n--- PUT/CALL RATIO ANALYSIS ---")
    pc_ratio = calculate_put_call_ratio(options_data)

    if pc_ratio["volume_put_call_ratio"] is not None:
        print(f"Volume Put/Call Ratio: {pc_ratio['volume_put_call_ratio']:.2f}")
        print(f"Open Interest Put/Call Ratio: {pc_ratio['oi_put_call_ratio']:.2f}")

        # Interpret the ratios
        if pc_ratio["volume_put_call_ratio"] > 1:
            print(
                f"SIGNAL: High put/call volume ratio ({pc_ratio['volume_put_call_ratio']:.2f}) indicates bearish sentiment"
            )
            if pc_ratio["volume_put_call_ratio"] > 1.5:
                print(
                    "SIGNAL: Extremely high put/call ratio may indicate a potential contrarian buy signal"
                )
        elif pc_ratio["volume_put_call_ratio"] < 0.5:
            print(
                f"SIGNAL: Low put/call volume ratio ({pc_ratio['volume_put_call_ratio']:.2f}) indicates bullish sentiment"
            )

        print(
            f"\nCall Volume: {pc_ratio['call_volume']} | Put Volume: {pc_ratio['put_volume']}"
        )
        print(
            f"Call Open Interest: {pc_ratio['call_oi']} | Put Open Interest: {pc_ratio['put_oi']}"
        )
    else:
        print("Insufficient data to calculate put/call ratios")

    # Generate overall trading signal
    print("\n--- TRADING SIGNAL ---")
    signal_result = generate_trading_signal(unusual_activity, significant_oi, pc_ratio)

    print(f"SIGNAL: {signal_result['signal']}")
    print(f"Score: {signal_result['score']}")
    print("\nFactor Contributions:")
    for factor, contribution in signal_result["factors"].items():
        if contribution != 0:
            print(f"- {factor}: {contribution:+d}")

    print("\nExplanation:")
    print(signal_result["explanation"])


if __name__ == "__main__":
    main()
