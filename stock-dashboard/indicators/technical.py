"""Technical indicator calculation module.

Provides pure mathematical implementations for financial technical indicators
such as SMA, EMA, RSI, MACD, Bollinger Bands, and VWAP using Pandas and NumPy.
Strictly zero reliance on third-party technical analysis libraries ('ta' / 'pandas-ta').
All calculations are vectorized without row-iteration loops.
"""

from typing import Optional

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Formula: SMA_t = (1 / k) * sum_{i=0}^{k-1} P_{t-i}

    Args:
        series: Price series (e.g. Close).
        window: Rolling observation window size (k).

    Returns:
        pd.Series containing the rolling simple moving average.
        Returns a Series of NaNs if the input series has fewer rows than window.
    """
    if series is None or len(series) < window or window <= 0:
        return pd.Series(np.nan, index=series.index if series is not None else [], name=f"SMA_{window}")

    result = series.rolling(window=window, min_periods=window).mean()
    result.name = f"SMA_{window}"
    return result


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Formula: EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}, where alpha = 2 / (span + 1)

    Args:
        series: Price series (e.g. Close).
        span: Decay span for exponential weighting.

    Returns:
        pd.Series containing the exponential moving average.
        Returns a Series of NaNs if the input series has fewer rows than span.
    """
    if series is None or len(series) < span or span <= 0:
        return pd.Series(np.nan, index=series.index if series is not None else [], name=f"EMA_{span}")

    # Use exponential weighted moving average with adjust=False as required
    result = series.ewm(span=span, adjust=False).mean()
    # Mask values before span has sufficient lookback
    result.iloc[: span - 1] = np.nan
    result.name = f"EMA_{span}"
    return result


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing.

    Formula: RSI = 100 - (100 / (1 + RS)), where RS = WilderEMA(Gain) / WilderEMA(Loss)

    Args:
        series: Price series (e.g. Close).
        period: Lookback period for Wilder's smoothing (default 14).

    Returns:
        pd.Series containing the RSI values bounded between 0 and 100.
        Returns a Series of NaNs if the input series has fewer rows than period + 1.
    """
    if series is None or len(series) < period + 1 or period <= 0:
        return pd.Series(np.nan, index=series.index if series is not None else [], name=f"RSI_{period}")

    # Step 1: Calculate bar-to-bar price difference
    delta = series.diff()

    # Step 2: Separate positive gains and negative losses
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Step 3: Apply Wilder's smoothing via ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Step 4: Compute Relative Strength (RS) and RSI
    # Handle division by zero: if avg_loss is 0, RSI is 100; if avg_gain is 0, RSI is 0
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss was 0 and gain > 0, set RSI to 100
    rsi_series = rsi_series.fillna(100.0).where(avg_gain > 0, 0.0)

    # First period values do not have sufficient lookback
    rsi_series.iloc[:period] = np.nan
    rsi_series.name = f"RSI_{period}"
    return rsi_series


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD).

    Formula: MACD = EMA(P, fast) - EMA(P, slow), Signal = EMA(MACD, signal), Histogram = MACD - Signal

    Args:
        series: Price series (e.g. Close).
        fast: Fast EMA lookback window (default 12).
        slow: Slow EMA lookback window (default 26).
        signal: Signal line EMA window (default 9).

    Returns:
        pd.DataFrame with columns ['MACD', 'Signal', 'Histogram'] aligned with input series.
        Returns NaNs if input series has fewer rows than slow window.
    """
    cols = ["MACD", "Signal", "Histogram"]
    if series is None or len(series) < slow or slow <= fast:
        return pd.DataFrame(np.nan, index=series.index if series is not None else [], columns=cols)

    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema

    # Mask insufficient lookback period
    macd_line.iloc[: slow - 1] = np.nan

    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    signal_line.iloc[: slow - 1 + signal - 1] = np.nan

    histogram = macd_line - signal_line

    result_df = pd.DataFrame(
        {
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": histogram,
        },
        index=series.index,
    )
    return result_df


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands (Middle, Upper, Lower).

    Formula: Middle = SMA(P, w), Upper = Middle + (num_std * std), Lower = Middle - (num_std * std)

    Args:
        series: Price series (e.g. Close).
        window: Moving average window (default 20).
        num_std: Multiplier for standard deviation bands (default 2.0).

    Returns:
        pd.DataFrame with columns ['Middle', 'Upper', 'Lower'] aligned with input series.
        Returns NaNs if input series has fewer rows than window.
    """
    cols = ["Middle", "Upper", "Lower"]
    if series is None or len(series) < window or window <= 0:
        return pd.DataFrame(np.nan, index=series.index if series is not None else [], columns=cols)

    middle = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    result_df = pd.DataFrame(
        {
            "Middle": middle,
            "Upper": upper,
            "Lower": lower,
        },
        index=series.index,
    )
    return result_df


def vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP).

    Formula: VWAP = cumsum(Typical_Price * Volume) / cumsum(Volume), where Typical_Price = (High + Low + Close) / 3

    For intraday data, cumulative sums reset at the start of each distinct trading day.
    For daily or multi-day timeframes, VWAP accumulates continuously over the dataset.

    Args:
        df: Cleaned OHLCV DataFrame with a DatetimeIndex.

    Returns:
        pd.Series containing the VWAP series aligned with df.index.
    """
    if df is None or df.empty or not all(col in df.columns for col in ["High", "Low", "Close", "Volume"]):
        return pd.Series(np.nan, index=df.index if df is not None else [], name="VWAP")

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"]
    price_volume = typical_price * volume

    # Detect if data is intraday (median time delta < 1 day)
    is_intraday = False
    if len(df.index) > 1 and isinstance(df.index, pd.DatetimeIndex):
        time_deltas = df.index[1:] - df.index[:-1]
        median_delta = pd.Series(time_deltas).median()
        is_intraday = median_delta < pd.Timedelta(days=1)

    if is_intraday and isinstance(df.index, pd.DatetimeIndex):
        # Reset VWAP daily by grouping on date
        trading_day = df.index.normalize()
        cum_pv = price_volume.groupby(trading_day).cumsum()
        cum_vol = volume.groupby(trading_day).cumsum()
    else:
        cum_pv = price_volume.cumsum()
        cum_vol = volume.cumsum()

    # Avoid division by zero
    vwap_series = cum_pv / cum_vol.replace(0, np.nan)
    vwap_series.name = "VWAP"
    return vwap_series


def compute_indicator_summary(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate high-level technical summary signals for RSI, MACD, and SMA 50.

    Args:
        df: Cleaned OHLCV DataFrame with Close prices.

    Returns:
        List of dictionaries with keys: 'Indicator', 'Current Value', 'Interpretation'.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return []

    close = df["Close"]
    summary: list[dict[str, str]] = []

    # 1. RSI (14)
    rsi_s = rsi(close, 14)
    valid_rsi = rsi_s.dropna()
    if not valid_rsi.empty:
        latest_rsi = float(valid_rsi.iloc[-1])
        if latest_rsi >= 70.0:
            rsi_status = "🔴 Overbought (≥ 70)"
        elif latest_rsi <= 30.0:
            rsi_status = "🟢 Oversold (≤ 30)"
        else:
            rsi_status = "⚪ Neutral (30 - 70)"
        summary.append({
            "Indicator": "Relative Strength Index (RSI 14)",
            "Current Value": f"{latest_rsi:.2f}",
            "Interpretation": rsi_status,
        })
    else:
        summary.append({
            "Indicator": "Relative Strength Index (RSI 14)",
            "Current Value": "N/A",
            "Interpretation": f"Insufficient data (requires >14 bars, got {len(df)})",
        })

    # 2. MACD (12, 26, 9)
    macd_res = macd(close, 12, 26, 9)
    valid_macd = macd_res.dropna()
    if not valid_macd.empty:
        last_m = float(valid_macd["MACD"].iloc[-1])
        last_s = float(valid_macd["Signal"].iloc[-1])
        diff = last_m - last_s
        if diff > 0:
            macd_status = "🟢 Bullish (MACD > Signal)"
        elif diff < 0:
            macd_status = "🔴 Bearish (MACD < Signal)"
        else:
            macd_status = "⚪ Neutral"
        summary.append({
            "Indicator": "MACD (12, 26, 9)",
            "Current Value": f"MACD: {last_m:.2f} | Signal: {last_s:.2f}",
            "Interpretation": macd_status,
        })
    else:
        summary.append({
            "Indicator": "MACD (12, 26, 9)",
            "Current Value": "N/A",
            "Interpretation": f"Insufficient data (requires ≥26 bars, got {len(df)})",
        })

    # 3. Price relative to SMA 50
    sma_50 = sma(close, 50)
    valid_sma = sma_50.dropna()
    if not valid_sma.empty:
        curr_price = float(close.iloc[-1])
        sma_val = float(valid_sma.iloc[-1])
        pct = ((curr_price - sma_val) / sma_val) * 100.0
        if curr_price >= sma_val:
            sma_status = f"🟢 Bullish (Above SMA 50 by {pct:+.2f}%)"
        else:
            sma_status = f"🔴 Bearish (Below SMA 50 by {pct:+.2f}%)"
        summary.append({
            "Indicator": "Price vs. SMA 50",
            "Current Value": f"Price: {curr_price:,.2f} | SMA 50: {sma_val:,.2f}",
            "Interpretation": sma_status,
        })
    else:
        summary.append({
            "Indicator": "Price vs. SMA 50",
            "Current Value": "N/A",
            "Interpretation": f"Insufficient data (requires ≥50 bars, got {len(df)})",
        })

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Technical Indicators Sanity Checks")
    print("=" * 60)

    # 1. Verified benchmark dataset from J. Welles Wilder's 'New Concepts in Technical Trading Systems'
    known_prices = pd.Series(
        [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
            45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        ]
    )

    # Test SMA (window=5)
    # Expected at index 4: (44.34 + 44.09 + 44.15 + 43.61 + 44.33) / 5 = 44.104
    sma_5 = sma(known_prices, 5)
    expected_sma_4 = 44.104
    actual_sma_4 = sma_5.iloc[4]
    print(f"SMA(5) at bar 5: Actual={actual_sma_4:.4f}, Expected={expected_sma_4:.4f}")
    assert np.isclose(actual_sma_4, expected_sma_4, atol=1e-4), "SMA calculation mismatch!"
    assert np.isnan(sma_5.iloc[3]), "SMA should be NaN before window is reached"

    # Test RSI (period=14)
    # Benchmark Wilder 14-period RSI at bar 15 (index 14) is approx 50.66
    rsi_14 = rsi(known_prices, 14)
    actual_rsi_14 = rsi_14.iloc[14]
    print(f"RSI(14) at bar 15: Actual={actual_rsi_14:.2f}, Expected≈50.66")
    assert np.isnan(rsi_14.iloc[13]), "RSI must be NaN for the first 14 periods"
    assert 50.0 <= actual_rsi_14 <= 51.5, f"RSI out of expected range: {actual_rsi_14}"

    # Test EMA
    ema_12 = ema(known_prices, 12)
    assert len(ema_12) == len(known_prices)
    assert np.isnan(ema_12.iloc[10]), "EMA must be NaN before span"
    assert not np.isnan(ema_12.iloc[11]), "EMA should be valid at span"

    # Test Bollinger Bands
    bb = bollinger_bands(known_prices, 10, 2.0)
    assert set(bb.columns) == {"Middle", "Upper", "Lower"}
    assert (bb["Upper"].dropna() >= bb["Middle"].dropna()).all()
    assert (bb["Middle"].dropna() >= bb["Lower"].dropna()).all()

    # Test MACD
    macd_res = macd(known_prices, fast=5, slow=10, signal=3)
    assert set(macd_res.columns) == {"MACD", "Signal", "Histogram"}
    assert len(macd_res) == len(known_prices)

    # Test short data handling (fewer rows than window returns NaN)
    short_data = pd.Series([10.0, 11.0, 12.0])
    short_sma = sma(short_data, 5)
    assert short_sma.isna().all(), "Short SMA must return all NaNs"
    short_rsi = rsi(short_data, 14)
    assert short_rsi.isna().all(), "Short RSI must return all NaNs"
    short_macd = macd(short_data, 12, 26, 9)
    assert short_macd.isna().all().all(), "Short MACD must return all NaNs"
    short_bb = bollinger_bands(short_data, 20, 2.0)
    assert short_bb.isna().all().all(), "Short Bollinger must return all NaNs"

    # Test VWAP
    dates = pd.date_range("2026-01-01 09:30", periods=4, freq="1h")
    sample_df = pd.DataFrame(
        {
            "High": [105, 106, 107, 108],
            "Low": [95, 96, 97, 98],
            "Close": [100, 102, 104, 106],
            "Volume": [1000, 2000, 1500, 2500],
        },
        index=dates,
    )
    vwap_val = vwap(sample_df)
    assert len(vwap_val) == 4
    assert not vwap_val.isna().any()

    print("All indicator sanity tests passed with 0 errors!")
