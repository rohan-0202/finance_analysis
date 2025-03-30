from typing import Dict, List, Optional, TypedDict, Union

import click
import pandas as pd

from db_util import get_options_data
from options.implied_volatility import analyze_implied_volatility
from options.open_interest import analyze_open_interest_changes
from options.put_call import calculate_put_call_ratio
from options.unusual_activity import analyze_unusual_activity


class AnalysisResult(TypedDict):
    """TypedDict for standardized analysis results across all analysis methods."""

    score: int  # Overall score contribution
    factors: Dict[str, int]  # Individual factor contributions
    explanation: List[str]  # Explanations for the analysis


def analyze_unusual_activity_signal(unusual_activity: pd.DataFrame) -> AnalysisResult:
    """
    Analyze unusual options activity and generate a signal score with explanation.

    Parameters:
    ----------
    unusual_activity : pandas.DataFrame
        DataFrame containing unusual options activity

    Returns:
    -------
    AnalysisResult
        Dictionary containing analysis result with standardized structure
    """
    score = 0
    explanation = []

    if not unusual_activity.empty:
        # Count unusual activity for calls vs puts
        unusual_calls = unusual_activity[unusual_activity["option_type"] == "call"]
        unusual_puts = unusual_activity[unusual_activity["option_type"] == "put"]

        call_count = len(unusual_calls)
        put_count = len(unusual_puts)
        total_count = call_count + put_count

        if total_count > 0:
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

                score = activity_score
                factor = activity_score
            else:
                factor = 0
        else:
            factor = 0
    else:
        factor = 0

    return {
        "score": score,
        "factors": {"unusual_activity": factor},
        "explanation": explanation,
    }


def analyze_open_interest_signal(significant_oi: pd.DataFrame) -> AnalysisResult:
    """
    Analyze significant open interest data and generate a signal score with explanation.

    Parameters:
    ----------
    significant_oi : pandas.DataFrame
        DataFrame containing significant open interest data

    Returns:
    -------
    AnalysisResult
        Dictionary containing analysis result with standardized structure
    """
    score = 0
    factor = 0
    explanation = []

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

            score = oi_score
            factor = oi_score

    return {
        "score": score,
        "factors": {"open_interest": factor},
        "explanation": explanation,
    }


def analyze_put_call_ratio_signal(pc_ratio: Dict) -> AnalysisResult:
    """
    Analyze put/call ratio data and generate a signal score with explanation.

    Parameters:
    ----------
    pc_ratio : dict
        Dictionary containing put/call ratio data as returned by calculate_put_call_ratio()

    Returns:
    -------
    AnalysisResult
        Dictionary containing analysis result with standardized structure
    """
    score = 0
    factors = {}
    explanation = []

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
            oi_score = -10  # Additional bearish confirmation
            explanation.append(
                "Open interest put/call ratio confirms bearish sentiment"
            )
        elif oi_pc_ratio < 0.7 and vol_pc_ratio < 0.7:
            oi_score = 10  # Additional bullish confirmation
            explanation.append(
                "Open interest put/call ratio confirms bullish sentiment"
            )
        else:
            oi_score = 0

        score += oi_score
        factors["oi_confirmation"] = oi_score
    else:
        factors["put_call_ratio"] = 0
        factors["oi_confirmation"] = 0

    return {"score": score, "factors": factors, "explanation": explanation}


def analyze_iv_signal(iv_analysis: Optional[Dict]) -> AnalysisResult:
    """
    Analyze implied volatility data and generate a signal score with explanation.

    Parameters:
    ----------
    iv_analysis : dict or None
        Dictionary containing implied volatility analysis results as returned by analyze_implied_volatility()

    Returns:
    -------
    AnalysisResult
        Dictionary containing analysis result with standardized structure
    """
    score = 0
    factors = {}
    explanation = []

    if iv_analysis is None:
        return {"score": 0, "factors": {}, "explanation": []}

    # 1. IV Percentile/Rank
    if iv_analysis["iv_percentile"] is not None:
        iv_percentile = iv_analysis["iv_percentile"]
        if iv_percentile > 80:
            iv_percentile_score = (
                -15
            )  # High IV is generally bearish (expensive options)
            explanation.append(
                f"High implied volatility (percentile: {iv_percentile:.1f}) suggests expensive options and potential overreaction"
            )
            score += iv_percentile_score
            factors["iv_percentile"] = iv_percentile_score
        elif iv_percentile < 20:
            iv_percentile_score = 15  # Low IV is generally bullish (cheap options)
            explanation.append(
                f"Low implied volatility (percentile: {iv_percentile:.1f}) suggests cheap options and potential complacency"
            )
            score += iv_percentile_score
            factors["iv_percentile"] = iv_percentile_score
        else:
            factors["iv_percentile"] = 0
    else:
        factors["iv_percentile"] = 0

    # 2. Volatility Skew
    if iv_analysis["skew"] and iv_analysis["skew"]["current_skew"] is not None:
        skew_value = iv_analysis["skew"]["current_skew"]
        if skew_value > 0.1:
            skew_score = (
                -25 if skew_value > 0.2 else -15
            )  # High put-call skew is bearish
            explanation.append(
                f"High put-call skew ({skew_value:.2f}) indicates strong demand for downside protection"
            )
            score += skew_score
            factors["volatility_skew"] = skew_score
        elif skew_value < -0.1:
            skew_score = 25 if skew_value < -0.2 else 15  # Negative skew is bullish
            explanation.append(
                f"Negative volatility skew ({skew_value:.2f}) indicates higher call IV than puts - bullish sentiment"
            )
            score += skew_score
            factors["volatility_skew"] = skew_score
        else:
            factors["volatility_skew"] = 0
    else:
        factors["volatility_skew"] = 0

    # 3. IV Run-up Detection
    if iv_analysis["iv_run_up"] and iv_analysis["iv_run_up"]["detected"]:
        magnitude = iv_analysis["iv_run_up"]["magnitude"]
        # IV run-up modifies existing signals rather than providing a direct signal
        # Positive score means bullish, so we amplify that effect
        if score > 20:  # If already bullish
            runup_score = 15  # Amplify bullish signal
            explanation.append(
                f"IV run-up ({magnitude:.1f}%) combined with bullish indicators suggests strong upcoming price movement"
            )
        elif score < -20:  # If already bearish
            runup_score = -15  # Amplify bearish signal
            explanation.append(
                f"IV run-up ({magnitude:.1f}%) combined with bearish indicators suggests strong upcoming price movement"
            )
        else:
            runup_score = 0  # Neutral if overall sentiment is unclear
            explanation.append(
                f"IV run-up ({magnitude:.1f}%) detected - potential upcoming catalyst"
            )

        score += runup_score
        factors["iv_run_up"] = runup_score
    else:
        factors["iv_run_up"] = 0

    # 4. IV Term Structure
    if (
        iv_analysis["term_structure"]
        and iv_analysis["term_structure"]["curve_type"] is not None
    ):
        curve_type = iv_analysis["term_structure"]["curve_type"]
        if curve_type == "inverted":
            # Inverted term structure suggests expectation of near-term volatility
            # This is a confidence modifier more than a direct signal
            if score > 20:  # If already bullish
                term_score = 10  # Stronger conviction in bullish move
                explanation.append(
                    "Inverted IV term structure suggests expected near-term volatility, strengthening bullish conviction"
                )
            elif score < -20:  # If already bearish
                term_score = -10  # Stronger conviction in bearish move
                explanation.append(
                    "Inverted IV term structure suggests expected near-term volatility, strengthening bearish conviction"
                )
            else:
                term_score = 0

            score += term_score
            factors["iv_term_structure"] = term_score
        else:
            factors["iv_term_structure"] = 0
    else:
        factors["iv_term_structure"] = 0

    return {"score": score, "factors": factors, "explanation": explanation}


def generate_trading_signal(
    unusual_activity: pd.DataFrame,
    significant_oi: pd.DataFrame,
    pc_ratio: Dict,
    iv_analysis: Optional[Dict] = None,
) -> Dict[str, Union[str, int, Dict[str, int], str]]:
    """Generate a trading signal (BUY, SELL, or HOLD) based on options analysis results.

    This function implements a multi-factor scoring algorithm that weighs various
    options market indicators to produce an actionable trading signal. The algorithm
    combines sentiment indicators from put/call ratios, unusual activity patterns,
    institutional positioning through open interest analysis, and implied volatility metrics.

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

    5. Implied Volatility Analysis:
       - IV Percentile/Rank: High IV (>80%) subtracts -10 to -15 points (bearish)
       - IV Percentile/Rank: Low IV (<20%) adds +10 to +15 points (bullish)
       - Volatility Skew: High put-call skew subtracts -15 to -25 points (bearish)
       - Volatility Skew: Negative skew adds +15 to +25 points (bullish)
       - IV Run-up: Significant IV increases modify score based on other indicators
       - IV Term Structure: Inverted term structure adds signal confidence modifier

    6. Signal Determination:
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

    iv_analysis : dict, optional
        Dictionary containing implied volatility analysis results as returned by analyze_implied_volatility()

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

    # vix_data = get_historical_data("^VIX")
    # print(vix_data)

    # 1. Analyze unusual activity
    unusual_result = analyze_unusual_activity_signal(unusual_activity)
    score += unusual_result["score"]
    factors.update(unusual_result["factors"])
    explanation.extend(unusual_result["explanation"])

    # 2. Analyze significant open interest
    open_interest_result = analyze_open_interest_signal(significant_oi)
    score += open_interest_result["score"]
    factors.update(open_interest_result["factors"])
    explanation.extend(open_interest_result["explanation"])

    # 3. Analyze put/call ratios
    pc_result = analyze_put_call_ratio_signal(pc_ratio)
    score += pc_result["score"]
    factors.update(pc_result["factors"])
    explanation.extend(pc_result["explanation"])

    # 4. Analyze implied volatility
    iv_result = analyze_iv_signal(iv_analysis)
    score += iv_result["score"]
    factors.update(iv_result["factors"])
    explanation.extend(iv_result["explanation"])

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

    # Analyze implied volatility
    print("\n--- IMPLIED VOLATILITY ANALYSIS ---")
    iv_analysis = analyze_implied_volatility(options_data, ticker)

    # Display IV percentile and rank
    if iv_analysis["iv_percentile"] is not None:
        print(f"IV Percentile: {iv_analysis['iv_percentile']:.1f}")
        print(f"IV Rank: {iv_analysis['iv_rank']:.1f}")

        # Provide interpretation
        if iv_analysis["iv_percentile"] > 80:
            print("SIGNAL: IV is relatively high - options are expensive")
        elif iv_analysis["iv_percentile"] < 20:
            print("SIGNAL: IV is relatively low - options are cheap")
    else:
        print("Insufficient data to calculate IV percentile/rank")

    # Display volatility skew information
    if iv_analysis["skew"] and iv_analysis["skew"]["current_skew"] is not None:
        print(f"\nVolatility Skew: {iv_analysis['skew']['current_skew']:.2f}")
        print(f"Skew Signal: {iv_analysis['skew']['skew_signal']}")
    else:
        print("\nInsufficient data to analyze volatility skew")

    # Display term structure information
    if (
        iv_analysis["term_structure"]
        and iv_analysis["term_structure"]["curve_type"] is not None
    ):
        print(f"\nIV Term Structure: {iv_analysis['term_structure']['curve_type']}")
        print(
            f"Term Structure Signal: {iv_analysis['term_structure']['term_structure_signal']}"
        )
    else:
        print("\nInsufficient data to analyze IV term structure")

    # Display IV run-up information
    if iv_analysis["iv_run_up"] and iv_analysis["iv_run_up"]["detected"]:
        print(
            f"\nIV Run-up Detected: {iv_analysis['iv_run_up']['magnitude']:.1f}% increase"
        )
        print(f"Run-up Signal: {iv_analysis['iv_run_up']['run_up_signal']}")
    else:
        print("\nNo significant IV run-up detected")

    # Display IV-based trading signals
    if iv_analysis["iv_signals"]:
        print("\nIV-Based Trading Signals:")
        for signal in iv_analysis["iv_signals"]:
            print(
                f"- {signal['signal']} (Type: {signal['type']}, Strength: {signal['strength']}/5)"
            )

    # Generate overall trading signal
    print("\n--- TRADING SIGNAL ---")
    signal_result = generate_trading_signal(
        unusual_activity, significant_oi, pc_ratio, iv_analysis
    )

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
