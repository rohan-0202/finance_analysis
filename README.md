# Finance Analysis

## Table of Contents

- [Finance Analysis](#finance-analysis)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Basic Installation](#basic-installation)
  - [Development Installation](#development-installation)
  - [Optional Extensions](#optional-extensions)
    - [Jupyter Notebook Support](#jupyter-notebook-support)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Linting and Formatting](#linting-and-formatting)
  - [Data Management](#data-management)
- [License](#license)

A comprehensive Python toolkit for financial market analysis, strategy development, and backtesting. This project provides tools for technical analysis, options analysis, backtesting trading strategies, and financial news sentiment analysis.

## Features

- **Technical Indicators**: RSI, MACD, OBV, and other technical indicators
- **Options Analysis**: Unusual activity detection, implied volatility analysis, put/call ratio analysis
- **Backtesting Framework**: Test trading strategies with historical data
- **News Sentiment Analysis**: Analyze financial news for trading insights
- **Time Series Forecasting**: Predict future price movements using advanced models

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finance-analysis.git
cd finance-analysis

# Install with uv
uv pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Optional Extensions

#### Jupyter Notebook Support

```bash
# Install notebook dependencies
uv pip install -e ".[notebook]"

# Set up Jupyter kernel
make install_jupyter_kernel
```

#### Time Series Analysis Support

```bash
# Install time series dependencies
uv pip install -e ".[timeseries]"
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_specific_file.py
```

### Linting and Formatting

```bash
# Run linter
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format
```

### Data Management

```bash
# Get data for all tickers
make get_data

# Get data for a specific ticker
make get_data_for_ticker AAPL
```

## Usage

### Technical Indicators

```python
from signals.signal_factory import SignalFactory

# Create a signal instance
rsi_signal = SignalFactory.create_signal("rsi", window=14)

# Get signals for a ticker
signals = rsi_signal.get_signals("AAPL")

# Get the latest signal
latest_signal = rsi_signal.get_latest_signal("AAPL")
```

### Backtesting Strategies

```python
from datetime import datetime
from backtesting.backtest_strategy import Backtest
from backtesting.strategies.strategy_factory import StrategyFactory

# Get a strategy class
strategy_class = StrategyFactory.get_strategy_class("rsi_strategy")

# Create a backtest
backtest = Backtest(
    strategy_class=strategy_class,
    strategy_params={"rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70},
    initial_capital=10000.0,
)

# Run the backtest
backtest.run(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 1, 1),
    benchmark_ticker="SPY",
)

# Print and plot results
backtest.print_results()
backtest.plot_results()
```

### Options Analysis

```python
from options_analysis import generate_trading_signal
from db_util import get_options_data
from options.implied_volatility import analyze_implied_volatility
from options.open_interest import analyze_open_interest_changes
from options.put_call import calculate_put_call_ratio
from options.unusual_activity import analyze_unusual_activity

# Get options data
options_data = get_options_data("AAPL")

# Analyze options data
unusual_activity = analyze_unusual_activity(options_data)
significant_oi = analyze_open_interest_changes(options_data)
pc_ratio = calculate_put_call_ratio(options_data)
iv_analysis = analyze_implied_volatility(options_data, "AAPL")

# Generate trading signal
signal = generate_trading_signal(unusual_activity, significant_oi, pc_ratio, iv_analysis)
print(f"Signal: {signal['signal']}, Score: {signal['score']}")
print(f"Explanation: {signal['explanation']}")
```

### Financial News Analysis

```python
from agent.finance_news_agent import FinanceNewsAgent

# Initialize agent
agent = FinanceNewsAgent(
    tickers_file="nyse_tickers.txt",
    lmstudio_url="http://localhost:1234",  # Your LLM server URL
    finnhub_key="your_finnhub_key",
    days_lookback=7,
    db_path="finance_news.db",
)

# Process a ticker
agent.process_ticker("AAPL")

# Get analysis results
analysis = agent._get_latest_analysis("AAPL")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Summary: {analysis['summary']}")
```

## License

[License](LICENSE)
