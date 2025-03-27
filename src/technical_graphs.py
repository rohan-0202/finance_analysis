import warnings
from datetime import datetime
from typing import Optional

import click
# lol
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import functions from macd.py, rsi.py, and obv.py
from macd import (calculate_macd, get_latest_macd_signal, get_macd_crossovers,
                  get_signal_stats_text)
from rsi import calculate_ticker_rsi, get_latest_rsi_signal, get_rsi_signals
from obv import calculate_ticker_obv, get_latest_obv_signal

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=Warning)

# Set seaborn style defaults
sns.set_theme()


def plot_indicators(
    ticker_symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    rsi_window: int = 14,
    overbought: int = 70,
    oversold: int = 30,
    obv_window: int = 20,
    db_name: str = "stock_data.db",
    days: int = 365,
    save_path: Optional[str] = None,
    theme: str = "darkgrid",
    color_palette: str = "muted",
) -> plt.Figure:
    """
    Plot MACD, RSI, and OBV indicators with price chart for a given ticker symbol in a single window.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL')
    fast_period : int, default=12
        The period for the fast EMA in MACD
    slow_period : int, default=26
        The period for the slow EMA in MACD
    signal_period : int, default=9
        The period for the signal line in MACD
    rsi_window : int, default=14
        The period for RSI calculation
    overbought : int, default=70
        Level considered overbought for RSI
    oversold : int, default=30
        Level considered oversold for RSI
    obv_window : int, default=20
        The window for OBV moving average
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
    # Get MACD, RSI, and OBV data
    price_data, macd_data = calculate_macd(
        ticker_symbol, fast_period, slow_period, signal_period, db_name, days
    )
    _, rsi_data = calculate_ticker_rsi(ticker_symbol, rsi_window, db_name, days)
    _, obv_data = calculate_ticker_obv(ticker_symbol, db_name, days)

    if price_data is None or macd_data is None or rsi_data is None or obv_data is None:
        print(f"Could not calculate indicators for {ticker_symbol}")
        return None

    # Setup plot style and figure
    sns.set_style(theme)
    sns.set_palette(color_palette)
    fig = plt.figure(figsize=(14, 14), dpi=100)  # Increased figure height to accommodate OBV panel
    gs = GridSpec(4, 1, height_ratios=[2, 1, 1, 1], figure=fig, hspace=0.15)  # Added 4th panel for OBV

    # Get signals for indicators
    macd_crossovers = get_macd_crossovers(macd_data)

    rsi_signals = get_rsi_signals(
        ticker_symbol, rsi_window, overbought, oversold, db_name, days
    )
    buy_signals = [(s["date"], s["price"]) for s in rsi_signals if s["type"] == "buy"]
    sell_signals = [(s["date"], s["price"]) for s in rsi_signals if s["type"] == "sell"]
    
    # Get OBV signal
    latest_obv_signal = get_latest_obv_signal(ticker_symbol, obv_window, db_name, days)

    # ========== PRICE CHART (TOP PANEL) ==========
    ax1 = fig.add_subplot(gs[0])

    # Plot price line
    sns.lineplot(
        x=price_data.index,
        y=price_data["close"],
        color="#1f77b4",
        linewidth=1.8,
        label="Close Price",
        ax=ax1,
    )

    # Add volume background if available
    volume_ax = ax1.twinx()
    volume_data = price_data.get("volume", None)
    if volume_data is not None:
        volume_height = price_data["close"].max() * 0.3
        normalized_volume = volume_data * (volume_height / volume_data.max())
        volume_ax.fill_between(
            price_data.index,
            normalized_volume,
            alpha=0.15,
            color="#ffb347",
            label="Volume",
        )
        volume_ax.set_yticks([])

    # Plot signal markers
    price_range = price_data["close"].max() - price_data["close"].min()
    offset = price_range * 0.01

    # MACD signals
    for date, signal_type in macd_crossovers:
        if date in price_data.index:
            price = price_data.loc[date, "close"]
            ax1.scatter(
                date,
                price - offset * 2 if signal_type == "bullish" else price + offset * 2,
                color="green" if signal_type == "bullish" else "red",
                s=120,
                marker="^" if signal_type == "bullish" else "v",
                edgecolor="white",
                linewidth=1,
                zorder=5,
                alpha=0.9,
            )

    # RSI signals
    for date, price in buy_signals:
        ax1.scatter(
            date,
            price - offset,
            color="limegreen",
            s=100,
            marker="^",
            edgecolor="black",
            linewidth=1,
            zorder=5,
            alpha=0.9,
        )
    for date, price in sell_signals:
        ax1.scatter(
            date,
            price + offset,
            color="crimson",
            s=100,
            marker="v",
            edgecolor="black",
            linewidth=1,
            zorder=5,
            alpha=0.9,
        )

    # Create legend markers
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="MACD Bullish",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="MACD Bearish",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="limegreen",
            markeredgecolor="black",
            markersize=10,
            label="RSI Buy",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="crimson",
            markeredgecolor="black",
            markersize=10,
            label="RSI Sell",
        ),
    ]

    # Setup price chart formatting
    ax1.set_title(
        f"{ticker_symbol} Price with Technical Indicators ({days} days)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3)

    # Add volume to legend if available
    handles1, labels1 = ax1.get_legend_handles_labels()
    if volume_data is not None:
        handles2, labels2 = volume_ax.get_legend_handles_labels()
        handles1.extend(handles2)
        labels1.extend(labels2)

    # Add all elements to legend
    handles1.extend(legend_elements)
    ax1.legend(
        handles=handles1,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        fontsize=9,
        ncol=2,
    )

    # ========== MACD CHART (MIDDLE PANEL) ==========
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot MACD histogram
    colors = {True: "#00cc00", False: "#cc0000"}  # green for positive, red for negative
    ax2.bar(
        macd_data.index,
        macd_data["histogram"],
        color=[colors[val > 0] for val in macd_data["histogram"]],
        alpha=0.6,
        width=1.5,
    )

    # Plot MACD and signal lines
    sns.lineplot(
        x=macd_data.index,
        y=macd_data["macd"],
        color="#2ca02c",
        linewidth=1.8,
        label="MACD",
        ax=ax2,
    )
    sns.lineplot(
        x=macd_data.index,
        y=macd_data["signal"],
        color="#d62728",
        linewidth=1.8,
        label="Signal",
        ax=ax2,
    )

    # Create legend with histogram
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.6))
    labels.append("Histogram")

    # Setup MACD chart formatting
    ax2.set_ylabel("MACD", fontsize=12, fontweight="bold")
    ax2.legend(
        handles,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        fontsize=9,
    )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax2.grid(True, alpha=0.3)

    # Add MACD info text
    ax2.annotate(
        f"Fast: {fast_period}, Slow: {slow_period}, Signal: {signal_period}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
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

    # Add MACD stats
    ax2.annotate(
        get_signal_stats_text(macd_data),
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

    # ========== RSI CHART (THIRD PANEL) ==========
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot RSI line
    sns.lineplot(
        x=rsi_data.index,
        y=rsi_data,
        color="#2ca02c",
        linewidth=1.8,
        label="RSI",
        ax=ax3,
    )

    # Add threshold lines
    ax3.axhline(
        y=overbought,
        color="#d62728",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Overbought ({overbought})",
    )
    ax3.axhline(
        y=oversold,
        color="#1f77b4",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Oversold ({oversold})",
    )
    ax3.axhline(y=50, color="gray", linestyle="-", alpha=0.3, linewidth=1)

    # Setup RSI chart formatting
    ax3.set_ylabel("RSI", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 100)
    ax3.legend(
        loc="upper left", frameon=True, facecolor="white", framealpha=0.8, fontsize=9
    )
    ax3.grid(True, alpha=0.3)

    # Add RSI info text
    ax3.annotate(
        f"Window: {rsi_window}, Overbought: {overbought}, Oversold: {oversold}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
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

    # Add RSI stats
    latest_value = rsi_data.iloc[-1] if not rsi_data.empty else 0
    status = (
        "Overbought"
        if latest_value >= overbought
        else ("Oversold" if latest_value <= oversold else "Neutral")
    )

    rsi_stats = f"Current RSI: {latest_value:.2f} ({status})\n"
    rsi_stats += f"Signals: {len(buy_signals)} Buy, {len(sell_signals)} Sell"

    latest_signal = get_latest_rsi_signal(
        ticker_symbol, rsi_window, overbought, oversold, db_name, days
    )
    if latest_signal:
        days_ago = (datetime.now().date() - latest_signal["date"].date()).days
        rsi_stats += (
            f"\nLatest: {latest_signal['type'].capitalize()} ({days_ago} days ago)"
        )

    ax3.annotate(
        rsi_stats,
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

    # ========== OBV CHART (BOTTOM PANEL) ==========
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    # Calculate OBV moving average for smoother visualization
    obv_ma = obv_data.rolling(window=obv_window).mean()
    
    # Plot OBV line
    sns.lineplot(
        x=obv_data.index,
        y=obv_data,
        color="#ff7f0e",  # Orange color for OBV
        linewidth=1.8,
        label="OBV",
        ax=ax4,
    )
    
    # Plot OBV moving average
    sns.lineplot(
        x=obv_ma.index,
        y=obv_ma,
        color="#9467bd",  # Purple for MA
        linewidth=1.5,
        linestyle="--",
        label=f"OBV MA ({obv_window})",
        ax=ax4,
    )
    
    # Add horizontal line at zero
    ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    
    # Setup OBV chart formatting
    ax4.set_ylabel("OBV", fontsize=12, fontweight="bold")
    ax4.legend(
        loc="upper left", frameon=True, facecolor="white", framealpha=0.8, fontsize=9
    )
    ax4.grid(True, alpha=0.3)
    
    # Add OBV info text
    ax4.annotate(
        f"Window: {obv_window} (for Moving Average)",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
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
    
    # Add OBV stats
    current_obv = obv_data.iloc[-1] if not obv_data.empty else 0
    obv_change = (current_obv - obv_data.iloc[-6]) / abs(obv_data.iloc[-6]) * 100 if len(obv_data) >= 6 else 0
    
    obv_stats = f"Current OBV: {current_obv:,.0f}\n"
    obv_stats += f"5-day change: {obv_change:.2f}%"
    
    if latest_obv_signal:
        days_ago = (datetime.now().date() - latest_obv_signal["date"].date()).days
        obv_stats += f"\nLatest: {latest_obv_signal['type'].capitalize()} ({days_ago} days ago)"
    
    ax4.annotate(
        obv_stats,
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

    # ========== FINAL FORMATTING ==========
    # Format date ticks
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # Hide x tick labels for top 3 panels
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax3.set_xlabel("")
    ax1.tick_params(axis="x", labelsize=0)  # Hide tick labels for price chart
    ax2.tick_params(axis="x", labelsize=0)  # Hide tick labels for MACD chart
    ax3.tick_params(axis="x", labelsize=0)  # Hide tick labels for RSI chart

    # Configure bottom x-axis
    ax4.set_xlabel("Date", fontsize=12, fontweight="bold")  # OBV panel now has x-axis labels
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add timestamp
    fig.text(
        0.98,
        0.02,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="right",
        fontsize=8,
        alpha=0.7,
    )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# Replace the main function with a Click command
@click.command()
@click.argument("ticker", type=str)
def main(ticker):
    """Analyze and display technical indicators for a stock ticker."""
    ticker = ticker.strip().upper()

    # Display indicator data
    print(f"\nAnalyzing technical indicators for {ticker}...")

    # Get MACD data
    _, macd_data = calculate_macd(ticker)
    if macd_data is not None:
        print("\nMACD values (recent):")
        print(macd_data.tail(3))

        # Show latest MACD signal
        latest_macd = get_latest_macd_signal(ticker)
        if latest_macd:
            print(
                f"Latest signal: {latest_macd['type'].upper()} on "
                f"{latest_macd['date'].strftime('%Y-%m-%d')} "
                f"(MACD: {latest_macd['macd']:.4f}, Signal: {latest_macd['signal']:.4f})"
            )
        else:
            print("No recent MACD signals")

    # Get RSI data
    _, rsi_data = calculate_ticker_rsi(ticker)
    if rsi_data is not None:
        print("\nRSI values (recent):")
        print(rsi_data.tail(3))

        # Show latest RSI signal
        latest_rsi = get_latest_rsi_signal(ticker)
        if latest_rsi:
            print(
                f"Latest signal: {latest_rsi['type'].upper()} on "
                f"{latest_rsi['date'].strftime('%Y-%m-%d')} "
                f"(RSI: {latest_rsi['rsi']:.2f})"
            )
        else:
            print("No recent RSI signals")
    
    # Get OBV data
    _, obv_data = calculate_ticker_obv(ticker)
    if obv_data is not None:
        print("\nOBV values (recent):")
        print(obv_data.tail(3))
        
        # Show latest OBV signal
        latest_obv = get_latest_obv_signal(ticker)
        if latest_obv:
            print(
                f"Latest signal: {latest_obv['type'].upper()} on "
                f"{latest_obv['date'].strftime('%Y-%m-%d')} "
                f"(OBV: {latest_obv['obv']:,.0f})"
            )
        else:
            print("No recent OBV signals")

    # Plot the combined chart
    print(f"\nGenerating technical chart for {ticker}...")
    _ = plot_indicators(ticker)
    plt.show()


if __name__ == "__main__":
    main()
