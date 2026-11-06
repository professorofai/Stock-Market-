"""Data fetching module.

Handles external market data API calls (yfinance), request caching,
rate limiting, network retries, and error handling for market data retrieval.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

# Ensure project root is available on sys.path for direct script execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    CACHE_TTL_SECONDS,
    MAX_RETRIES,
    REQUIRED_OHLCV_COLUMNS,
    RETRY_DELAY_SECONDS,
    VALID_INTERVALS_FOR_PERIOD,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DataFetchError(Exception):
    """Custom exception raised when market data retrieval fails."""

    def __init__(self, message: str, ticker: Optional[str] = None) -> None:
        super().__init__(message)
        self.ticker = ticker


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_data(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV market data for a given ticker symbol.

    Validates period/interval compatibility, retries on network failures,
    and returns a clean DataFrame with a timezone-aware DatetimeIndex.
    Never raises an exception on failure; returns an empty DataFrame instead.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'RELIANCE.NS').
        period: Time range to fetch (e.g. '1d', '5d', '1mo', '1y').
        interval: Bar interval (e.g. '1m', '5m', '15m', '1h', '1d').

    Returns:
        pd.DataFrame containing Open, High, Low, Close, Volume columns with
        a timezone-aware DatetimeIndex, or an empty DataFrame on failure.
    """
    empty_df = pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)
    empty_df.index = pd.DatetimeIndex([], tz="UTC")

    if not ticker or not isinstance(ticker, str):
        logger.error("Invalid ticker argument provided: %s", ticker)
        return empty_df

    clean_ticker = ticker.strip().upper()

    # Validate period and interval compatibility against config
    valid_intervals = VALID_INTERVALS_FOR_PERIOD.get(period)
    effective_interval = interval
    if valid_intervals is None:
        logger.warning(
            "Period '%s' not recognized in VALID_INTERVALS_FOR_PERIOD. Defaulting to '1mo'.",
            period,
        )
        period = "1mo"
        valid_intervals = VALID_INTERVALS_FOR_PERIOD["1mo"]

    if interval not in valid_intervals:
        effective_interval = "1d" if "1d" in valid_intervals else valid_intervals[-1]
        logger.warning(
            "Interval '%s' is incompatible with period '%s' for '%s'. Auto-adjusting to '%s'.",
            interval,
            period,
            clean_ticker,
            effective_interval,
        )

    # Retry loop for resilient fetching
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(clean_ticker)
            df = t.history(period=period, interval=effective_interval, auto_adjust=True)

            if df is None or df.empty:
                logger.warning(
                    "Attempt %d/%d: No data returned for '%s' (period='%s', interval='%s').",
                    attempt,
                    MAX_RETRIES,
                    clean_ticker,
                    period,
                    effective_interval,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            # Ensure required OHLCV columns exist
            missing_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.warning(
                    "Missing expected columns %s for '%s'.", missing_cols, clean_ticker
                )
                return empty_df

            # Retain only required OHLCV columns
            result_df = df[REQUIRED_OHLCV_COLUMNS].copy()

            # Ensure DatetimeIndex is timezone-aware
            if not isinstance(result_df.index, pd.DatetimeIndex):
                result_df.index = pd.to_datetime(result_df.index)

            if result_df.index.tz is None:
                result_df.index = result_df.index.tz_localize("UTC")

            return result_df

        except Exception as exc:
            logger.error(
                "Attempt %d/%d failed while fetching data for '%s': %s",
                attempt,
                MAX_RETRIES,
                clean_ticker,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    logger.error("All %d fetch attempts failed for '%s'. Returning empty DataFrame.", MAX_RETRIES, clean_ticker)
    return empty_df


def fetch_multiple(
    tickers: List[str],
    period: str = "1mo",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for multiple stock tickers.

    Args:
        tickers: List of stock ticker symbols.
        period: Time range to fetch.
        interval: Bar interval.

    Returns:
        Dictionary mapping each ticker symbol to its fetched DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        results[ticker] = fetch_stock_data(ticker=ticker, period=period, interval=interval)
    return results


def get_quote_info(ticker: str) -> Dict[str, Any]:
    """Retrieve key quote metrics and company metadata for a ticker symbol.

    Prioritizes fast_info for low-latency retrieval, falling back to historical
    price bars if fast_info is incomplete or unavailable. Never raises on failure.

    Args:
        ticker: The stock ticker symbol.

    Returns:
        Dictionary containing:
            - current_price (float or None)
            - previous_close (float or None)
            - day_high (float or None)
            - day_low (float or None)
            - volume (int or None)
            - currency (str)
            - company_name (str)
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    default_info: Dict[str, Any] = {
        "current_price": None,
        "previous_close": None,
        "day_high": None,
        "day_low": None,
        "volume": None,
        "currency": "N/A",
        "company_name": clean_ticker,
    }

    if not clean_ticker:
        return default_info

    try:
        t = yf.Ticker(clean_ticker)
        fi = getattr(t, "fast_info", None)

        current_price = None
        previous_close = None
        day_high = None
        day_low = None
        volume = None
        currency = "USD"
        company_name = clean_ticker

        if fi is not None:
            try:
                current_price = getattr(fi, "last_price", None)
                previous_close = getattr(fi, "previous_close", None)
                day_high = getattr(fi, "day_high", None)
                day_low = getattr(fi, "day_low", None)
                volume = getattr(fi, "last_volume", None)
                currency = getattr(fi, "currency", "USD") or "USD"
            except Exception as fi_err:
                logger.debug("fast_info attribute access failed for '%s': %s", clean_ticker, fi_err)

        # Fallback to history if fast_info does not provide a valid current price
        if current_price is None or pd.isna(current_price):
            df_hist = t.history(period="5d", interval="1d")
            if not df_hist.empty and "Close" in df_hist.columns:
                current_price = float(df_hist["Close"].iloc[-1])
                previous_close = (
                    float(df_hist["Close"].iloc[-2])
                    if len(df_hist) >= 2
                    else current_price
                )
                day_high = float(df_hist["High"].iloc[-1]) if "High" in df_hist.columns else current_price
                day_low = float(df_hist["Low"].iloc[-1]) if "Low" in df_hist.columns else current_price
                volume = int(df_hist["Volume"].iloc[-1]) if "Volume" in df_hist.columns else 0

        # Attempt to retrieve human-readable company name from info
        try:
            raw_info = getattr(t, "info", {}) or {}
            company_name = (
                raw_info.get("longName")
                or raw_info.get("shortName")
                or clean_ticker
            )
            if currency == "USD" and "currency" in raw_info:
                currency = raw_info["currency"]
        except Exception:
            company_name = clean_ticker

        return {
            "current_price": float(current_price) if current_price is not None and not pd.isna(current_price) else None,
            "previous_close": float(previous_close) if previous_close is not None and not pd.isna(previous_close) else None,
            "day_high": float(day_high) if day_high is not None and not pd.isna(day_high) else None,
            "day_low": float(day_low) if day_low is not None and not pd.isna(day_low) else None,
            "volume": int(volume) if volume is not None and not pd.isna(volume) else None,
            "currency": str(currency) if currency else "USD",
            "company_name": str(company_name),
        }
    except Exception as exc:
        logger.error("Failed to retrieve quote info for '%s': %s", clean_ticker, exc)
        return default_info


def validate_ticker(ticker: str) -> bool:
    """Validate whether a ticker symbol exists and has accessible market data.

    Args:
        ticker: The stock ticker symbol to validate.

    Returns:
        True if the ticker is valid and returns market data; False otherwise.
    """
    if not ticker or not isinstance(ticker, str):
        return False

    clean_ticker = ticker.strip().upper()
    try:
        t = yf.Ticker(clean_ticker)
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            try:
                price = getattr(fi, "last_price", None)
                if price is not None and not pd.isna(price):
                    return True
            except Exception:
                pass

        # Fallback verification via 1-day history check
        df = t.history(period="1d")
        return not df.empty
    except Exception as exc:
        logger.debug("Ticker validation failed for '%s': %s", clean_ticker, exc)
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Fetching data for RELIANCE.NS")
    print("=" * 60)
    reliance_df = fetch_stock_data("RELIANCE.NS", period="5d", interval="1d")
    print(f"RELIANCE.NS Shape: {reliance_df.shape}")
    print(reliance_df.head())

    print("\n" + "=" * 60)
    print("TEST: Fetching data for AAPL")
    print("=" * 60)
    aapl_df = fetch_stock_data("AAPL", period="5d", interval="1d")
    print(f"AAPL Shape: {aapl_df.shape}")
    print(aapl_df.head())

    print("\n" + "=" * 60)
    print("TEST: Fetching data for invalid ticker 'XXXXFAKE'")
    print("=" * 60)
    fake_df = fetch_stock_data("XXXXFAKE", period="5d", interval="1d")
    print(f"XXXXFAKE Empty? {fake_df.empty} | Shape: {fake_df.shape}")
    print(fake_df)

    print("\n" + "=" * 60)
    print("TEST: Quote Info")
    print("=" * 60)
    print("AAPL Quote Info:", get_quote_info("AAPL"))
    print("XXXXFAKE Quote Info:", get_quote_info("XXXXFAKE"))

    print("\n" + "=" * 60)
    print("TEST: Validation")
    print("=" * 60)
    print("Is AAPL valid?", validate_ticker("AAPL"))
    print("Is XXXXFAKE valid?", validate_ticker("XXXXFAKE"))
