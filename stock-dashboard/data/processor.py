"""Data processing module.

Handles data cleaning, missing value handling, timezone alignment,
resampling, returns computation, and price normalization with Pandas.
Pure functional design: inputs are never mutated.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path for direct module execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    REQUIRED_OHLCV_COLUMNS,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)

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


def align_prices_on_common_dates(
    df_dict: Dict[str, pd.DataFrame],
    method: str = "ffill",
) -> pd.DataFrame:
    """Align close prices from multiple tickers with differing exchange trading calendars.

    Converts timestamps to normalized UTC dates to harmonize international markets
    (e.g., US NYSE and Indian NSE) and forward-fills missing session holidays.

    Args:
        df_dict: Mapping of ticker symbol to its respective OHLCV DataFrame.
        method: Alignment fill method ('ffill' or 'inner').

    Returns:
        pd.DataFrame where columns are ticker symbols and index is common UTC dates.
    """
    if not df_dict:
        return pd.DataFrame()

    aligned_series: Dict[str, pd.Series] = {}

    for ticker, df in df_dict.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        series = df["Close"].copy()
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                continue

        # Standardize index to normalized UTC dates for inter-market alignment
        if series.index.tz is not None:
            series.index = series.index.tz_convert("UTC").normalize()
        else:
            series.index = series.index.tz_localize("UTC").normalize()

        # Remove duplicate calendar dates keeping the latest session close
        series = series[~series.index.duplicated(keep="last")]
        aligned_series[ticker] = series

    if not aligned_series:
        return pd.DataFrame()

    combined_df = pd.DataFrame(aligned_series)

    if method == "ffill":
        # Forward fill holiday gaps then drop initial unaligned bars
        combined_df = combined_df.ffill().dropna()
    else:
        combined_df = combined_df.dropna()

    return combined_df


def normalize_prices(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rebase the Close prices of multiple tickers to a common baseline of 100.

    Aligns multi-market trading calendars and rebases each stock's first
    overlapping date value to 100.0.

    Args:
        df_dict: Mapping of ticker symbol to its respective OHLCV DataFrame.

    Returns:
        pd.DataFrame where columns are ticker symbols and values are rebased
        Close prices (baseline = 100.0), indexed by datetime.
    """
    aligned = align_prices_on_common_dates(df_dict, method="ffill")
    if aligned.empty:
        return pd.DataFrame()

    normalized_cols: Dict[str, pd.Series] = {}
    for col in aligned.columns:
        first_val = aligned[col].iloc[0]
        if first_val > 0 and not pd.isna(first_val):
            normalized_cols[col] = (aligned[col] / first_val) * 100.0

    return pd.DataFrame(normalized_cols, index=aligned.index)


def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculate the Maximum Drawdown (MDD) percentage from peak to trough.

    Formula: MDD = min((Price - Running_Max) / Running_Max) * 100

    Args:
        series: Price series (e.g. Close).

    Returns:
        Maximum drawdown as a negative percentage (e.g. -14.2), or 0.0 if empty.
    """
    if series is None or len(series) < 2:
        return 0.0

    clean_s = series.dropna()
    if len(clean_s) < 2:
        return 0.0

    cum_max = clean_s.cummax()
    drawdown = (clean_s - cum_max) / cum_max
    min_dd = float(drawdown.min() * 100.0)
    return min_dd if not pd.isna(min_dd) else 0.0


def calculate_annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate the annualized volatility percentage of fractional or percent returns.

    Formula: Volatility = std(daily_returns) * sqrt(periods_per_year) * 100

    Args:
        returns: Daily percentage or fractional returns series.
        periods_per_year: Annualization factor (default 252 for trading days).

    Returns:
        Annualized volatility percentage (e.g. 24.5 for 24.5%).
    """
    if returns is None or len(returns.dropna()) < 2:
        return 0.0

    clean_ret = returns.dropna()
    # Normalize to fractional returns if series is already in percentage format
    if clean_ret.abs().mean() > 0.5:
        clean_ret = clean_ret / 100.0

    std_val = float(clean_ret.std())
    ann_vol = std_val * np.sqrt(periods_per_year) * 100.0
    return float(ann_vol) if not pd.isna(ann_vol) else 0.0


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate the annualized Sharpe Ratio for a returns series.

    Formula: Sharpe = (mean(excess_returns) / std(excess_returns)) * sqrt(periods_per_year)

    Args:
        returns: Daily percentage or fractional returns series.
        risk_free_rate: Annualized risk-free rate as a decimal (e.g. 0.05 for 5%).
        periods_per_year: Number of trading periods per year (default 252).

    Returns:
        Annualized Sharpe Ratio (e.g. 1.25), or 0.0 if standard deviation is zero.
    """
    if returns is None or len(returns.dropna()) < 2:
        return 0.0

    clean_ret = returns.dropna()
    # Normalize to fractional returns if passed as percentage
    if clean_ret.abs().mean() > 0.5:
        clean_ret = clean_ret / 100.0

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_ret = clean_ret - daily_rf

    std_val = float(excess_ret.std())
    if std_val == 0 or pd.isna(std_val):
        return 0.0

    sharpe = float((excess_ret.mean() / std_val) * np.sqrt(periods_per_year))
    return float(sharpe) if not pd.isna(sharpe) else 0.0


def calculate_risk_return_metrics(
    df_dict: Dict[str, pd.DataFrame],
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Compute Risk and Return summary statistics across multiple tickers.

    Calculates Total Return %, Annualized Volatility %, Max Drawdown %, and Sharpe Ratio.

    Args:
        df_dict: Mapping of ticker symbols to cleaned OHLCV DataFrames.
        risk_free_rate: Annual risk-free rate constant (default from config).

    Returns:
        pd.DataFrame containing formatted risk/return metrics per ticker.
    """
    if not df_dict:
        return pd.DataFrame(columns=["Ticker", "Total Return %", "Annualized Volatility %", "Max Drawdown %", "Sharpe Ratio"])

    rows: List[Dict[str, Any]] = []

    for ticker, df in df_dict.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna()
        if len(close) < 2:
            continue

        total_return = ((close.iloc[-1] / close.iloc[0]) - 1.0) * 100.0
        daily_returns = close.pct_change().dropna()
        vol = calculate_annualized_volatility(daily_returns)
        mdd = calculate_max_drawdown(close)
        sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)

        rows.append(
            {
                "Ticker": ticker,
                "Total Return %": total_return,
                "Annualized Volatility %": vol,
                "Max Drawdown %": mdd,
                "Sharpe Ratio": sharpe,
            }
        )

    return pd.DataFrame(rows)


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
