import sqlite3
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=Warning)

# Set seaborn style defaults
sns.set_theme()  # Apply the default seaborn theme


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate the Exponential Moving Average for a series."""
    return series.ewm(span=span, adjust=False).mean()


def get_historical_data(
    ticker_symbol: str, db_name: str = "stock_data.db", days: int = 365
) -> pd.DataFrame:
    """Fetch historical price data for a ticker from the database."""
    conn = sqlite3.connect(db_name)

    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Query to get data
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM historical_prices
    WHERE ticker = ? AND timestamp >= ?
    ORDER BY timestamp ASC
    """

    # First, get the data without parsing dates
    df = pd.read_sql_query(
        query, conn, params=(ticker_symbol, start_date.strftime("%Y-%m-%d"))
    )

    conn.close()

    if df.empty:
        raise ValueError(f"No historical data found for {ticker_symbol}")

    # Parse the timestamp column manually to avoid timezone issues
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Set timestamp as index
    df.set_index("timestamp", inplace=True)

    return df


def plot_macd(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
    save_path: Optional[str] = None,
    theme: str = "darkgrid",
    color_palette: str = "muted",
) -> plt.Figure:
    """
    Plot MACD indicator with price chart for a given ticker symbol.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    fast_period : int, default=12
        The period for the fast EMA
    slow_period : int, default=26
        The period for the slow EMA
    signal_period : int, default=9
        The period for the signal line (EMA of MACD)
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use
    save_path : str, optional
        If provided, save the figure to this path
    theme : str, default="darkgrid"
        The seaborn theme to use for the plot
    color_palette : str, default="muted"
        The seaborn color palette to use

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    """
    # Get MACD data
    price_data, macd_data = calculate_macd(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )

    if price_data is None or macd_data is None:
        print(f"Could not calculate MACD for {ticker_symbol}")
        return None

    # Set plotting style
    sns.set_style(theme)
    color_pal = sns.color_palette(color_palette)

    # Create figure and axis with GridSpec
    fig = plt.figure(figsize=(14, 9), dpi=100)
    gs = GridSpec(2, 1, height_ratios=[2, 1], figure=fig, hspace=0.15)

    # Find MACD crossovers before creating the plots
    crossovers = []
    for i in range(1, len(macd_data)):
        # Bullish crossover (MACD crosses above Signal)
        if (
            macd_data["macd"].iloc[i - 1] < macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] > macd_data["signal"].iloc[i]
        ):
            crossovers.append((macd_data.index[i], "bullish"))

        # Bearish crossover (MACD crosses below Signal)
        elif (
            macd_data["macd"].iloc[i - 1] > macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] < macd_data["signal"].iloc[i]
        ):
            crossovers.append((macd_data.index[i], "bearish"))

    # Price chart
    ax1 = fig.add_subplot(gs[0])
    sns.lineplot(
        x=price_data.index,
        y=price_data["close"],
        color="#1f77b4",  # Standard blue for price
        linewidth=1.8,
        label="Close Price",
        ax=ax1,
    )

    # Add volume in the background with low opacity
    volume_ax = ax1.twinx()
    volume_data = price_data["volume"] if "volume" in price_data.columns else None
    if volume_data is not None:
        # Normalize volume to fit nicely in the background
        volume_height = price_data["close"].max() * 0.3
        normalized_volume = volume_data * (volume_height / volume_data.max())
        volume_ax.fill_between(
            price_data.index,
            normalized_volume,
            alpha=0.15,
            color="#ffb347",  # Light orange for volume
            label="Volume",
        )
        volume_ax.set_yticks([])  # Hide volume axis

    # Plot crossover markers on price chart
    for date, cross_type in crossovers:
        # Get price at this date
        if date in price_data.index:
            price = price_data.loc[date, "close"]

            # Offset based on price range to make markers visible
            price_range = price_data["close"].max() - price_data["close"].min()
            offset = price_range * 0.01

            if cross_type == "bullish":
                # Add green upward triangle below price
                ax1.scatter(
                    date,
                    price - offset,
                    color="green",
                    s=120,
                    marker="^",
                    edgecolor="white",
                    linewidth=1,
                    zorder=5,
                    alpha=0.9,
                    label="_nolegend_",
                )
            else:
                # Add red downward triangle above price
                ax1.scatter(
                    date,
                    price + offset,
                    color="red",
                    s=120,
                    marker="v",
                    edgecolor="white",
                    linewidth=1,
                    zorder=5,
                    alpha=0.9,
                    label="_nolegend_",
                )

    # Create proxy artists for crossover markers legend
    bullish_marker = plt.Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="green",
        markersize=10,
        label="Bullish Signal",
    )
    bearish_marker = plt.Line2D(
        [0],
        [0],
        marker="v",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label="Bearish Signal",
    )

    # Add title and labels with better formatting
    title = f"{ticker_symbol} Price and MACD ({days} days)"
    ax1.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax1.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3)

    # Improve legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    legends = handles1.copy()
    legend_labels = labels1.copy()

    # Add volume to legend if available
    if volume_data is not None:
        handles2, labels2 = volume_ax.get_legend_handles_labels()
        legends.extend(handles2)
        legend_labels.extend(labels2)

    # Add signal markers to legend
    legends.extend([bullish_marker, bearish_marker])
    legend_labels.extend(["Bullish Signal", "Bearish Signal"])

    ax1.legend(
        legends,
        legend_labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        fontsize=10,
    )

    # Format x-axis date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # MACD chart
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Define specific colors for MACD components
    macd_color = "#2ca02c"  # Green
    signal_color = "#d62728"  # Red
    pos_hist_color = "#00cc00"  # Bright green
    neg_hist_color = "#cc0000"  # Bright red

    # Plot histogram first so it's behind the lines
    positive = macd_data["histogram"] > 0
    negative = macd_data["histogram"] <= 0

    # Create a proxy artist for the histogram legend (we'll only show one)
    hist_handle = plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.6)

    # Add histogram bars with improved appearance
    ax2.bar(
        macd_data.index[positive],
        macd_data["histogram"][positive],
        color=pos_hist_color,
        alpha=0.6,
        width=1.5,
    )
    ax2.bar(
        macd_data.index[negative],
        macd_data["histogram"][negative],
        color=neg_hist_color,
        alpha=0.6,
        width=1.5,
    )

    # Plot MACD and signal lines on top of histogram
    macd_line = sns.lineplot(
        x=macd_data.index,
        y=macd_data["macd"],
        color=macd_color,
        linewidth=1.8,
        label="MACD",
        ax=ax2,
    )
    signal_line = sns.lineplot(
        x=macd_data.index,
        y=macd_data["signal"],
        color=signal_color,
        linewidth=1.8,
        label="Signal",
        ax=ax2,
    )

    ax2.set_ylabel("MACD", fontsize=12, fontweight="bold")
    ax2.tick_params(labelsize=10)
    ax2.grid(True, alpha=0.3)

    # Create a cleaner legend with a single entry for histogram
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(hist_handle)
    labels.append("Histogram")
    ax2.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        fontsize=10,
    )

    # Add zero line for MACD
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

    # Add info text with periods used in a nicer box
    info_text = f"Fast: {fast_period}, Slow: {slow_period}, Signal: {signal_period}"
    ax2.annotate(
        info_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
            linewidth=0.5,
        ),
    )

    # Add timestamp for last update
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.98, 0.02, f"Generated: {current_time}", ha="right", fontsize=8, alpha=0.7
    )

    # Add signal statistics
    signal_stats = f"{get_signal_stats_text(macd_data)}"
    ax2.annotate(
        signal_stats,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.8,
            edgecolor="gray",
            linewidth=0.5,
        ),
    )

    # Layout adjustment
    plt.tight_layout()

    # Remove x-axis label from top plot
    ax1.set_xlabel("")

    # Make bottom x-axis tick labels more readable
    ax2.set_xlabel("timestamp", fontsize=12, fontweight="bold")

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def calculate_macd(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculate the MACD (Moving Average Convergence Divergence) for a given ticker.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    fast_period : int, default=12
        The period for the fast EMA
    slow_period : int, default=26
        The period for the slow EMA
    signal_period : int, default=9
        The period for the signal line (EMA of MACD)
    db_name : str, default="stock_data.db"
        The name of the SQLite database file
    days : int, default=365
        Number of days of historical data to use

    Returns:
    --------
    tuple: (price_data, macd_data)
        - price_data: DataFrame with original price data
        - macd_data: DataFrame with MACD calculations (macd, signal, histogram)
    """
    try:
        # Get historical data
        price_data = get_historical_data(ticker_symbol, db_name, days)

        # Create a new DataFrame for MACD data with the same index
        macd_data = pd.DataFrame(index=price_data.index)

        # Calculate EMAs
        ema_fast = calculate_ema(price_data["close"], fast_period)
        ema_slow = calculate_ema(price_data["close"], slow_period)

        # Calculate MACD line
        macd_data["macd"] = ema_fast - ema_slow

        # Calculate signal line
        macd_data["signal"] = calculate_ema(macd_data["macd"], signal_period)

        # Calculate histogram
        macd_data["histogram"] = macd_data["macd"] - macd_data["signal"]

        # Drop NaN values caused by EMA calculations
        macd_data = macd_data.dropna()

        # Align price_data with macd_data to have the same dates
        price_data = price_data.loc[macd_data.index]

        return price_data, macd_data

    except Exception as e:
        print(f"Error calculating MACD for {ticker_symbol}: {e}")
        return None, None


def get_macd_signals(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> List[Dict[str, Any]]:
    """
    Get MACD buy/sell signals for a given ticker.

    Returns:
    --------
    list of dict: A list of signal events, where each signal is a dictionary with:
        - date: datetime of the signal
        - type: 'buy' or 'sell'
        - macd: MACD value at signal
        - signal: Signal line value at signal
        - price: Closing price at signal
    """
    price_data, macd_data = calculate_macd(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )

    if macd_data is None or macd_data.empty:
        return []

    # Identify crossover points
    buy_signals = (macd_data["macd"] > macd_data["signal"]) & (
        macd_data["macd"].shift() <= macd_data["signal"].shift()
    )
    sell_signals = (macd_data["macd"] < macd_data["signal"]) & (
        macd_data["macd"].shift() >= macd_data["signal"].shift()
    )

    # Combine buy and sell signals into a list of dictionaries
    signals = []

    # Process buy signals
    for date in macd_data[buy_signals].index:
        signals.append(
            {
                "date": date,
                "type": "buy",
                "macd": macd_data.loc[date, "macd"],
                "signal": macd_data.loc[date, "signal"],
                "price": price_data.loc[date, "close"],
            }
        )

    # Process sell signals
    for date in macd_data[sell_signals].index:
        signals.append(
            {
                "date": date,
                "type": "sell",
                "macd": macd_data.loc[date, "macd"],
                "signal": macd_data.loc[date, "signal"],
                "price": price_data.loc[date, "close"],
            }
        )

    # Sort signals by date
    signals.sort(key=lambda x: x["date"])

    return signals


def get_latest_macd_signal(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    db_name: str = "stock_data.db",
    days: int = 365,
) -> Optional[Dict[str, Any]]:
    """
    Get only the most recent MACD signal for a ticker.

    Returns:
    --------
    dict or None: The most recent signal with date, type, macd, signal, and price.
                 Returns None if no signals are found.
    """
    signals = get_macd_signals(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )

    # Return the most recent signal if any exist
    if signals:
        return signals[-1]
    return None


def get_signal_stats_text(macd_data: pd.DataFrame) -> str:
    """
    Calculate and format statistics about MACD signals.

    Parameters:
    -----------
    macd_data : pd.DataFrame
        DataFrame with MACD data

    Returns:
    --------
    str
        Formatted text with signal statistics
    """
    # Count crossovers
    bullish_crossovers = 0
    bearish_crossovers = 0

    for i in range(1, len(macd_data)):
        # Bullish crossover (MACD crosses above Signal)
        if (
            macd_data["macd"].iloc[i - 1] < macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] > macd_data["signal"].iloc[i]
        ):
            bullish_crossovers += 1

        # Bearish crossover (MACD crosses below Signal)
        elif (
            macd_data["macd"].iloc[i - 1] > macd_data["signal"].iloc[i - 1]
            and macd_data["macd"].iloc[i] < macd_data["signal"].iloc[i]
        ):
            bearish_crossovers += 1

    # Get current MACD values
    current_macd = macd_data["macd"].iloc[-1] if not macd_data.empty else 0
    current_signal = macd_data["signal"].iloc[-1] if not macd_data.empty else 0
    current_hist = macd_data["histogram"].iloc[-1] if not macd_data.empty else 0

    # Format the text
    stats_text = f"Bullish: {bullish_crossovers}, Bearish: {bearish_crossovers}\n"
    stats_text += f"Current: MACD={current_macd:.2f}, Signal={current_signal:.2f}"

    return stats_text


if __name__ == "__main__":
    # Prompt user for ticker input
    user_input = (
        input("Enter a ticker symbol (or 'give all' to check all NYSE tickers): ")
        .strip()
        .upper()
    )

    if user_input == "GIVE ALL":
        # Process all tickers from the file - no try/except needed since file always exists
        with open("nyse_tickers.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        print(f"Processing MACD signals for {len(tickers)} tickers...")

        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        no_signals = 0

        # Store results to sort by date later
        results = []

        # Process each ticker
        for ticker in tickers:
            latest_signal = get_latest_macd_signal(ticker)

            if latest_signal:
                days_ago = (datetime.now().date() - latest_signal["date"].date()).days
                signal_type = latest_signal["type"].upper()

                if signal_type == "BUY":
                    buy_signals += 1
                else:
                    sell_signals += 1

                # Store result with days_ago for sorting
                results.append(
                    {
                        "ticker": ticker,
                        "signal_type": signal_type,
                        "days_ago": days_ago,
                        "price": latest_signal["price"],
                        "has_signal": True,
                    }
                )
            else:
                no_signals += 1
                # Store tickers with no signals too
                results.append({"ticker": ticker, "has_signal": False})

        # Sort results - first put signals by days_ago (oldest first), then no-signal entries
        sorted_results = sorted(
            results,
            key=lambda x: x.get("days_ago", 0) if x["has_signal"] else -1,
            reverse=True,
        )

        # Print results in order (oldest first, most recent last)
        print("\nMACD Signals (oldest at top, most recent at bottom):")
        for result in sorted_results:
            if result["has_signal"]:
                print(
                    f"{result['ticker']}: {result['signal_type']} signal {result['days_ago']} days ago at ${result['price']:.2f}"
                )
            else:
                print(f"{result['ticker']}: No recent MACD signals")

        # Print summary
        print(f"\nSummary:")
        print(f"Buy signals: {buy_signals}")
        print(f"Sell signals: {sell_signals}")
        print(f"No signals: {no_signals}")

    else:
        # Process single ticker
        ticker = user_input

        # Get MACD data
        price_data, macd_data = calculate_macd(ticker)

        # Show MACD values if available
        if macd_data is not None:
            print(f"\nRecent MACD values for {ticker}:")
            print(macd_data.tail())

        # Get the latest signal
        latest_signal = get_latest_macd_signal(ticker)

        # Display the latest signal if found
        if latest_signal:
            date_str = latest_signal["date"].strftime("%Y-%m-%d")
            signal_type = latest_signal["type"].upper()
            print(f"\nLatest MACD signal for {ticker}:")
            print(f"{date_str}: {signal_type}")
            print(f"MACD: {latest_signal['macd']:.4f}")
            print(f"Signal: {latest_signal['signal']:.4f}")
            print(f"Price: ${latest_signal['price']:.2f}")

            # Calculate days since the signal
            days_ago = (datetime.now().date() - latest_signal["date"].date()).days
            print(f"Signal occurred {days_ago} days ago")
        else:
            print(f"\nNo MACD signals found for {ticker}")

        # Automatically plot the MACD chart for the requested ticker
        print(f"\nPlotting MACD chart for {ticker}...")
        fig = plot_macd(ticker)
        plt.show()
