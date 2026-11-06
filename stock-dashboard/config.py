"""Application configuration and global constants.

Defines default ticker symbols, chart intervals, refresh intervals,
technical indicator default parameters, and plot styling constants.
"""

from typing import Dict, List, Tuple

# Default stock tickers to display
DEFAULT_TICKERS: List[str] = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "MSFT"]

# Available time periods for user selection
PERIOD_OPTIONS: List[str] = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]

# Available intervals for user selection
INTERVAL_OPTIONS: List[str] = ["1m", "5m", "15m", "30m", "1h", "1d"]

# Mapping of periods to valid Yahoo Finance intervals:
# - 1m data is restricted to periods <= 7d
# - Intraday (5m, 15m, 30m, 1h) data is restricted to periods <= 60d
# - 1d data is supported across all periods
VALID_INTERVALS_FOR_PERIOD: Dict[str, List[str]] = {
    "1d": ["1m", "5m", "15m", "30m", "1h", "1d"],
    "5d": ["1m", "5m", "15m", "30m", "1h", "1d"],
    "1mo": ["5m", "15m", "30m", "1h", "1d"],
    "3mo": ["1d"],
    "6mo": ["1d"],
    "1y": ["1d"],
    "2y": ["1d"],
}

# Dashboard refresh and data caching parameters (in seconds)
DEFAULT_REFRESH_SECONDS: int = 30
CACHE_TTL_SECONDS: int = 30

# API retry parameters
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 1.0

# Required OHLCV columns
REQUIRED_OHLCV_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

# Default technical indicator configurations
SMA_WINDOWS: Tuple[int, int] = (20, 50)
EMA_WINDOWS: Tuple[int, int] = (12, 26)
RSI_WINDOW: int = 14
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD: float = 2.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
