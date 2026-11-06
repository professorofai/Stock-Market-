"""Unit tests for technical indicator calculation functions using synthetic data.

Strictly zero network requests or third-party TA dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from indicators.technical import (
    bollinger_bands,
    compute_indicator_summary,
    ema,
    macd,
    rsi,
    sma,
    vwap,
)


@pytest.fixture
def synthetic_prices() -> pd.Series:
    """Fixture providing a known price sequence for calculation verification."""
    # 20-element benchmark series from J. Welles Wilder's reference
    return pd.Series(
        [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
            45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        ]
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Fixture providing a 2-day hourly synthetic OHLCV dataset."""
    idx = pd.date_range("2026-01-01 09:30", periods=4, freq="1h").append(
        pd.date_range("2026-01-02 09:30", periods=4, freq="1h")
    )
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0, 103.0, 108.0, 110.0, 109.0, 111.0],
            "High": [102.0, 104.0, 103.0, 105.0, 110.0, 112.0, 111.0, 113.0],
            "Low": [99.0, 101.0, 100.0, 102.0, 107.0, 109.0, 108.0, 110.0],
            "Close": [101.0, 103.0, 102.0, 104.0, 109.0, 111.0, 110.0, 112.0],
            "Volume": [1000, 2000, 1500, 2500, 1000, 2000, 1500, 2500],
        },
        index=idx,
    )


def test_sma_calculation(synthetic_prices: pd.Series) -> None:
    """Test SMA calculation on known values and index alignment."""
    sma_5 = sma(synthetic_prices, window=5)

    assert len(sma_5) == len(synthetic_prices)
    assert sma_5.index.equals(synthetic_prices.index)
    # First 4 values should be NaN
    assert sma_5.iloc[:4].isna().all()
    # Expected at index 4: (44.34 + 44.09 + 44.15 + 43.61 + 44.33) / 5 = 44.104
    assert np.isclose(sma_5.iloc[4], 44.104, atol=1e-4)


def test_sma_short_data() -> None:
    """Test that SMA returns a Series of NaNs when data is shorter than window."""
    short = pd.Series([10.0, 12.0, 14.0])
    result = sma(short, window=5)
    assert len(result) == len(short)
    assert result.isna().all()


def test_ema_calculation(synthetic_prices: pd.Series) -> None:
    """Test EMA calculation using exponential weighting and adjust=False."""
    ema_10 = ema(synthetic_prices, span=10)

    assert len(ema_10) == len(synthetic_prices)
    assert ema_10.index.equals(synthetic_prices.index)
    # Values before span should be masked to NaN
    assert ema_10.iloc[:9].isna().all()
    assert not np.isnan(ema_10.iloc[9])


def test_ema_short_data() -> None:
    """Test EMA handles series shorter than span by returning NaNs."""
    short = pd.Series([10.0, 11.0])
    result = ema(short, span=5)
    assert result.isna().all()


def test_rsi_wilder_benchmark(synthetic_prices: pd.Series) -> None:
    """Test RSI against J. Welles Wilder's textbook benchmark."""
    rsi_14 = rsi(synthetic_prices, period=14)

    assert len(rsi_14) == len(synthetic_prices)
    # First 14 bars must be NaN
    assert rsi_14.iloc[:14].isna().all()
    # Bar 15 (index 14) benchmark value is ~50.66
    assert 50.0 <= rsi_14.iloc[14] <= 51.5
    # All non-NaN RSI values must be bounded between 0 and 100
    valid_rsi = rsi_14.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()


def test_rsi_short_data() -> None:
    """Test that RSI returns all NaNs for insufficient history without raising."""
    short = pd.Series([10.0, 11.0, 12.0, 13.0])
    result = rsi(short, period=14)
    assert result.isna().all()


def test_macd_calculation(synthetic_prices: pd.Series) -> None:
    """Test MACD line, signal line, and histogram mathematical consistency."""
    res = macd(synthetic_prices, fast=5, slow=10, signal=3)

    assert set(res.columns) == {"MACD", "Signal", "Histogram"}
    assert len(res) == len(synthetic_prices)
    # Histogram must strictly equal MACD - Signal
    valid_mask = res["Histogram"].notna()
    expected_hist = res.loc[valid_mask, "MACD"] - res.loc[valid_mask, "Signal"]
    assert np.allclose(res.loc[valid_mask, "Histogram"], expected_hist, atol=1e-6)


def test_macd_short_data() -> None:
    """Test MACD returns all NaNs when data has fewer rows than slow window."""
    short = pd.Series([10.0, 11.0, 12.0])
    res = macd(short, fast=5, slow=10, signal=3)
    assert res.isna().all().all()


def test_bollinger_bands(synthetic_prices: pd.Series) -> None:
    """Test Bollinger Bands upper, middle, lower envelope order."""
    bb = bollinger_bands(synthetic_prices, window=10, num_std=2.0)

    assert set(bb.columns) == {"Middle", "Upper", "Lower"}
    valid = bb.dropna()
    assert (valid["Upper"] >= valid["Middle"]).all()
    assert (valid["Middle"] >= valid["Lower"]).all()


def test_bollinger_short_data() -> None:
    """Test Bollinger Bands return NaNs for data shorter than window."""
    short = pd.Series([10.0, 11.0, 12.0])
    res = bollinger_bands(short, window=20, num_std=2.0)
    assert res.isna().all().all()


def test_vwap_intraday_reset(synthetic_ohlcv: pd.DataFrame) -> None:
    """Test that VWAP resets at the start of each new trading session for intraday bars."""
    vwap_series = vwap(synthetic_ohlcv)

    assert len(vwap_series) == len(synthetic_ohlcv)
    assert vwap_series.index.equals(synthetic_ohlcv.index)

    # First bar of day 1: VWAP must equal typical price of bar 0
    tp_0 = (synthetic_ohlcv["High"].iloc[0] + synthetic_ohlcv["Low"].iloc[0] + synthetic_ohlcv["Close"].iloc[0]) / 3.0
    assert np.isclose(vwap_series.iloc[0], tp_0)

    # First bar of day 2 (bar 4): VWAP must reset and equal typical price of bar 4
    tp_4 = (synthetic_ohlcv["High"].iloc[4] + synthetic_ohlcv["Low"].iloc[4] + synthetic_ohlcv["Close"].iloc[4]) / 3.0
    assert np.isclose(vwap_series.iloc[4], tp_4)


def test_vwap_empty_or_invalid() -> None:
    """Test VWAP handles empty data or missing columns gracefully."""
    empty_df = pd.DataFrame()
    res = vwap(empty_df)
    assert res.empty

    bad_df = pd.DataFrame({"Close": [10.0, 11.0]})
    res2 = vwap(bad_df)
    assert res2.isna().all()


def test_compute_indicator_summary(synthetic_ohlcv: pd.DataFrame) -> None:
    """Test that indicator summary produces non-empty structured interpretation."""
    summary = compute_indicator_summary(synthetic_ohlcv)
    assert isinstance(summary, list)
    assert len(summary) == 3
    for item in summary:
        assert "Indicator" in item
        assert "Current Value" in item
        assert "Interpretation" in item
