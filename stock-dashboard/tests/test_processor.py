"""Unit tests for data cleaning, resampling, returns, and risk calculations.

Strictly zero network calls or external I/O.
"""

import numpy as np
import pandas as pd
import pytest

from data.processor import (
    align_prices_on_common_dates,
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_risk_return_metrics,
    calculate_sharpe_ratio,
    clean_data,
    add_returns,
    normalize_prices,
    resample_ohlcv,
)


@pytest.fixture
def dirty_ohlcv() -> pd.DataFrame:
    """Fixture providing raw OHLCV data with duplicate dates and NaN entries."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 102.0, 105.0, np.nan],
            "High": [103.0, 104.0, 104.0, 108.0, 110.0],
            "Low": [99.0, 100.0, 100.0, 104.0, 106.0],
            "Close": [102.0, 101.0, 101.0, 107.0, 109.0],
            "Volume": [1000, 1500, 1500, 1800, 2000],
        },
        index=[dates[0], dates[1], dates[1], dates[2], dates[3]],
    )


def test_clean_data(dirty_ohlcv: pd.DataFrame) -> None:
    """Test deduplication of timestamps and dropping of NaN rows."""
    cleaned = clean_data(dirty_ohlcv)

    # 1 duplicate dropped, 1 row with NaN dropped => 3 rows left
    assert len(cleaned) == 3
    assert not cleaned.index.duplicated().any()
    assert not cleaned.isna().any().any()
    assert cleaned.index.is_monotonic_increasing


def test_clean_data_empty() -> None:
    """Test clean_data returns empty DataFrame with expected columns on empty input."""
    res = clean_data(pd.DataFrame())
    assert res.empty
    assert list(res.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_add_returns() -> None:
    """Test Period Return % and Cumulative Return % calculations."""
    dates = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0, 105.0, 110.0],
            "High": [102.0, 108.0, 115.0],
            "Low": [98.0, 104.0, 109.0],
            "Close": [100.0, 110.0, 121.0],
            "Volume": [1000, 1200, 1500],
        },
        index=dates,
    )

    with_ret = add_returns(df)
    assert "Period Return %" in with_ret.columns
    assert "Cumulative Return %" in with_ret.columns

    # Bar 0: Cumulative is 0.0%, Period Return is NaN
    assert np.isclose(with_ret["Cumulative Return %"].iloc[0], 0.0)
    assert np.isnan(with_ret["Period Return %"].iloc[0])

    # Bar 1: (110 - 100) / 100 = +10.0%
    assert np.isclose(with_ret["Period Return %"].iloc[1], 10.0)
    assert np.isclose(with_ret["Cumulative Return %"].iloc[1], 10.0)

    # Bar 2: (121 - 100) / 100 = +21.0%
    assert np.isclose(with_ret["Cumulative Return %"].iloc[2], 21.0)


def test_resample_ohlcv() -> None:
    """Test standard financial aggregation (first/max/min/last/sum)."""
    dates = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 105.0, 107.0],
            "High": [103.0, 104.0, 108.0, 110.0],
            "Low": [99.0, 101.0, 104.0, 106.0],
            "Close": [102.0, 103.0, 106.0, 109.0],
            "Volume": [1000, 1500, 1800, 2000],
        },
        index=dates,
    )

    resampled = resample_ohlcv(df, rule="2D")
    assert len(resampled) == 2

    # First 2-day bar
    assert resampled["Open"].iloc[0] == 100.0  # first open
    assert resampled["High"].iloc[0] == 104.0  # max high
    assert resampled["Low"].iloc[0] == 99.0   # min low
    assert resampled["Close"].iloc[0] == 103.0 # last close
    assert resampled["Volume"].iloc[0] == 2500 # sum volume


def test_align_prices_and_normalize() -> None:
    """Test date alignment across distinct trading calendars and rebasing to 100."""
    dates_us = pd.date_range("2026-01-01", periods=3, freq="D", tz="America/New_York")
    dates_in = pd.date_range("2026-01-02", periods=3, freq="D", tz="Asia/Kolkata")

    df_us = pd.DataFrame({"Close": [100.0, 105.0, 110.0]}, index=dates_us)
    df_in = pd.DataFrame({"Close": [500.0, 510.0, 520.0]}, index=dates_in)

    aligned = align_prices_on_common_dates({"US_STOCK": df_us, "IN_STOCK": df_in})
    assert not aligned.empty
    assert "US_STOCK" in aligned.columns and "IN_STOCK" in aligned.columns

    norm = normalize_prices({"US_STOCK": df_us, "IN_STOCK": df_in})
    assert np.isclose(norm["US_STOCK"].iloc[0], 100.0)
    assert np.isclose(norm["IN_STOCK"].iloc[0], 100.0)


def test_calculate_max_drawdown() -> None:
    """Test peak-to-trough maximum drawdown calculation."""
    # Prices: 100 -> 120 (peak) -> 90 (trough: -25%) -> 110
    prices = pd.Series([100.0, 120.0, 105.0, 90.0, 110.0])
    mdd = calculate_max_drawdown(prices)
    # Expected: (90 - 120) / 120 * 100 = -25.0%
    assert np.isclose(mdd, -25.0)


def test_calculate_annualized_volatility() -> None:
    """Test annualized standard deviation of returns."""
    daily_returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    vol = calculate_annualized_volatility(daily_returns, periods_per_year=252)
    assert vol > 0.0
    assert isinstance(vol, float)


def test_calculate_sharpe_ratio() -> None:
    """Test Sharpe ratio calculation."""
    returns = pd.Series([0.02, 0.015, 0.01, 0.025, 0.018])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
    assert sharpe > 0.0


def test_calculate_risk_return_metrics() -> None:
    """Test multi-ticker risk/return metrics dataframe generation."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    df1 = pd.DataFrame({"Close": [100.0 + i * 2.0 for i in range(10)]}, index=dates)
    df2 = pd.DataFrame({"Close": [50.0 - i * 1.0 for i in range(10)]}, index=dates)

    metrics = calculate_risk_return_metrics({"STOCK_UP": df1, "STOCK_DOWN": df2})
    assert len(metrics) == 2
    assert set(metrics["Ticker"]) == {"STOCK_UP", "STOCK_DOWN"}
    assert "Total Return %" in metrics.columns
    assert "Max Drawdown %" in metrics.columns
