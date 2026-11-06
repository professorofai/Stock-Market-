"""Data processing module.

Handles data cleaning, missing value handling, timezone alignment,
resampling, returns computation, and price normalization with Pandas.
Pure functional design: inputs are never mutated.
"""

import logging
import os
import sys
from typing import Dict

import pandas as pd

# Ensure project root is on sys.path for direct module execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import REQUIRED_OHLCV_COLUMNS  # noqa: E402

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV time-series data.

    Drops rows containing NaN in essential OHLCV columns, eliminates duplicate
    timestamps, sorts monotonically ascending by DatetimeIndex, and coerces
    OHLCV series to standard numeric types.

    Args:
        df: Raw DataFrame containing market price bars.

    Returns:
        Cleaned, deduplicated, sorted pd.DataFrame with numeric dtypes.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)

    # Work on an isolated copy to prevent side effects
    cleaned = df.copy()

    # Verify and enforce DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        try:
            cleaned.index = pd.to_datetime(cleaned.index)
        except Exception as exc:
            logger.error("Failed to parse index as DatetimeIndex: %s", exc)
            return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)

    # Remove duplicate timestamp entries (retaining the first occurrence)
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

    # Sort time index monotonically ascending
    cleaned = cleaned.sort_index()

    # Ensure required numeric columns are present and coerce types
    present_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col in cleaned.columns]
    for col in present_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Drop any row with missing values in the primary price/volume columns
    cleaned = cleaned.dropna(subset=present_cols)

    return cleaned


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and append percentage returns to an OHLCV DataFrame.

    Appends:
      - 'Period Return %': Percentage change in Close price bar-over-bar.
      - 'Daily Return %': Alias for 'Period Return %' for daily intervals.
      - 'Cumulative Return %': Cumulative percentage return rebased to the initial bar.

    Args:
        df: Cleaned OHLCV DataFrame containing a 'Close' column.

    Returns:
        Copy of DataFrame with return percentage columns added.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)

    processed = df.copy()

    # Bar-over-bar percentage change (scaled to percentage points 0-100)
    pct_change = processed["Close"].pct_change() * 100.0
    processed["Period Return %"] = pct_change
    processed["Daily Return %"] = pct_change

    # Cumulative return from the initial bar's Close price
    first_close = processed["Close"].iloc[0]
    if first_close != 0 and not pd.isna(first_close):
        processed["Cumulative Return %"] = ((processed["Close"] / first_close) - 1.0) * 100.0
    else:
        processed["Cumulative Return %"] = 0.0

    return processed


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a higher timeframe.

    Applies standard financial aggregation logic:
      - Open: first
      - High: max
      - Low: min
      - Close: last
      - Volume: sum

    Args:
        df: Input DataFrame with DatetimeIndex and standard OHLCV columns.
        rule: Pandas offset string (e.g. '1W', '1M', '1h', '1D').

    Returns:
        Resampled OHLCV DataFrame, with incomplete/empty aggregation bins dropped.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS)

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("resample_ohlcv requires a DatetimeIndex.")
        return df.copy()

    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    # Aggregate only OHLCV columns present in the input
    active_agg = {col: agg_rules[col] for col in REQUIRED_OHLCV_COLUMNS if col in df.columns}

    try:
        resampled = df.resample(rule).agg(active_agg).dropna(how="any")
        return resampled
    except Exception as exc:
        logger.error("Failed to resample OHLCV data using rule '%s': %s", rule, exc)
        return df.copy()


def normalize_prices(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebase the Close prices of multiple tickers to a common baseline of 100.

    Facilitates comparative multi-stock performance analysis over identical
    time horizons.

    Args:
        df_dict: Mapping of ticker symbol to its respective OHLCV DataFrame.

    Returns:
        pd.DataFrame where columns are ticker symbols and values are rebased
        Close prices (baseline = 100.0), indexed by datetime.
    """
    if not df_dict:
        return pd.DataFrame()

    normalized_series: Dict[str, pd.Series] = {}

    for ticker, df in df_dict.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        clean_series = df["Close"].dropna()
        if clean_series.empty:
            continue

        base_val = clean_series.iloc[0]
        if base_val == 0 or pd.isna(base_val):
            continue

        normalized_series[ticker] = (clean_series / base_val) * 100.0

    if not normalized_series:
        return pd.DataFrame()

    combined_df = pd.DataFrame(normalized_series)
    return combined_df


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Processor unit tests with synthetic data")
    print("=" * 60)

    # Create synthetic test data with duplicates and a NaN
    dates = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 102.0, 105.0, 107.0, None],
            "High": [103.0, 104.0, 104.0, 108.0, 110.0, 112.0],
            "Low": [99.0, 100.0, 100.0, 104.0, 106.0, 108.0],
            "Close": [102.0, 101.0, 101.0, 107.0, 109.0, 111.0],
            "Volume": [1000, 1500, 1500, 1800, 2000, 2200],
        },
        index=[dates[0], dates[1], dates[1], dates[2], dates[3], dates[4]],
    )

    cleaned = clean_data(raw_df)
    print(f"Cleaned shape: {cleaned.shape} (expected 4 rows: 1 duplicate removed, 1 NaN dropped)")
    print(cleaned)

    returns_df = add_returns(cleaned)
    print("\nWith returns:")
    print(returns_df[["Close", "Period Return %", "Cumulative Return %"]])

    resampled = resample_ohlcv(cleaned, "2D")
    print("\nResampled 2D:")
    print(resampled)

    multi_norm = normalize_prices({"STOCK_A": cleaned, "STOCK_B": cleaned * 1.5})
    print("\nNormalized to 100:")
    print(multi_norm)
