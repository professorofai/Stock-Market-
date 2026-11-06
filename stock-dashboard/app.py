"""Main Streamlit application entry point.

Handles UI layout, sidebar controls, user input, metric display cards,
technical indicator toggles, and interactive chart presentation.
Does not contain direct data fetching, indicator computations, or business logic.
"""

import os
import sys
from typing import Dict, List

import pandas as pd
import streamlit as st

# Ensure project root is available on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from charts.plots import (  # noqa: E402
    build_bollinger_traces,
    build_ema_traces,
    build_sma_traces,
    build_vwap_trace,
    candlestick_with_volume,
    macd_chart,
    rsi_chart,
)
from config import (  # noqa: E402
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_TICKERS,
    PERIOD_OPTIONS,
    VALID_INTERVALS_FOR_PERIOD,
)
from data.fetcher import fetch_stock_data, get_quote_info, validate_ticker  # noqa: E402
from data.processor import clean_data  # noqa: E402
from indicators.technical import compute_indicator_summary  # noqa: E402

# Currency symbols lookup table
CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "CA$",
    "AUD": "A$",
}


def format_volume(volume: int) -> str:
    """Format volume integer into a human-readable abbreviation (K, M, B).

    Args:
        volume: Raw trading volume number.

    Returns:
        Formatted volume string.
    """
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.2f}B"
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M"
    if volume >= 1_000:
        return f"{volume / 1_000:.2f}K"
    return f"{volume:,}"


def main() -> None:
    """Render the main Streamlit stock market dashboard."""
    st.set_page_config(
        layout="wide",
        page_title="Stock Dashboard",
        page_icon="📈",
    )

    # Initialize ticker list in session state if not already populated
    if "tickers" not in st.session_state:
        st.session_state.tickers = list(DEFAULT_TICKERS)
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = DEFAULT_TICKERS[0]

    # --- Sidebar Controls ---
    st.sidebar.title("📊 Market Controls")

    # Custom Ticker Adder
    with st.sidebar.form(key="custom_ticker_form", clear_on_submit=True):
        custom_input = st.text_input(
            "Add Custom Ticker",
            placeholder="e.g. NVDA, TATAMOTORS.NS",
            help="Enter any valid Yahoo Finance ticker symbol (US, NSE with .NS, BSE with .BO).",
        )
        add_submitted = st.form_submit_button("Add Ticker", use_container_width=True)

        if add_submitted and custom_input:
            candidate = custom_input.strip().upper()
            if candidate in st.session_state.tickers:
                st.session_state.current_ticker = candidate
                st.sidebar.info(f"'{candidate}' is already in your ticker list.")
            else:
                with st.spinner(f"Validating ticker '{candidate}'..."):
                    if validate_ticker(candidate):
                        st.session_state.tickers.append(candidate)
                        st.session_state.current_ticker = candidate
                        st.sidebar.success(f"Added '{candidate}'!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"'{candidate}' could not be verified or has no data.")

    # Ticker Selector
    current_index = (
        st.session_state.tickers.index(st.session_state.current_ticker)
        if st.session_state.current_ticker in st.session_state.tickers
        else 0
    )
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=st.session_state.tickers,
        index=current_index,
        key="ticker_selector",
    )
    st.session_state.current_ticker = selected_ticker

    # Period Selector
    default_period_idx = (
        PERIOD_OPTIONS.index(DEFAULT_PERIOD) if DEFAULT_PERIOD in PERIOD_OPTIONS else 2
    )
    selected_period = st.sidebar.selectbox(
        "Period",
        options=PERIOD_OPTIONS,
        index=default_period_idx,
    )

    # Dynamic Interval Selector (filtered by chosen period)
    valid_intervals = VALID_INTERVALS_FOR_PERIOD.get(selected_period, ["1d"])
    default_interval_idx = (
        valid_intervals.index(DEFAULT_INTERVAL)
        if DEFAULT_INTERVAL in valid_intervals
        else len(valid_intervals) - 1
    )
    selected_interval = st.sidebar.selectbox(
        "Interval",
        options=valid_intervals,
        index=default_interval_idx,
    )

    # Technical Indicators Toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Technical Indicators")
    show_sma = st.sidebar.checkbox("SMA (20, 50)", value=False, help="Simple Moving Averages (20 & 50 period)")
    show_ema = st.sidebar.checkbox("EMA (12, 26)", value=False, help="Exponential Moving Averages (12 & 26 period)")
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=False, help="Bollinger Bands with 2 standard deviations")
    show_vwap = st.sidebar.checkbox("VWAP", value=False, help="Volume Weighted Average Price (resets daily for intraday)")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=False, help="Relative Strength Index oscillator")
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=False, help="Moving Average Convergence Divergence")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # --- Main Content Area ---
    with st.spinner(f"Fetching real-time market data for {selected_ticker}..."):
        raw_df = fetch_stock_data(
            ticker=selected_ticker,
            period=selected_period,
            interval=selected_interval,
        )
        df = clean_data(raw_df)
        quote_info = get_quote_info(selected_ticker)

    # Graceful handling for invalid or empty data
    if df.empty:
        st.error(
            f"⚠️ No market data could be loaded for **{selected_ticker}** "
            f"with period **{selected_period}** and interval **{selected_interval}**."
        )
        st.info("Try selecting a different timeframe (e.g. 1mo / 1d) or another stock ticker.")
        st.stop()

    # Header section
    company_name = quote_info.get("company_name") or selected_ticker
    currency = quote_info.get("currency", "USD")
    curr_sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ")

    st.title(f"{company_name} ({selected_ticker})")

    last_bar_time = df.index[-1].strftime("%Y-%m-%d %H:%M %Z")
    st.caption(
        f"Currency: **{currency}** | Timeframe: **{selected_period}** | Bar: **{selected_interval}** | "
        f"Latest Bar: **{last_bar_time}**"
    )

    # Derive metric card values
    current_price = quote_info.get("current_price")
    if current_price is None or pd.isna(current_price):
        current_price = float(df["Close"].iloc[-1])

    previous_close = quote_info.get("previous_close")
    if previous_close is None or pd.isna(previous_close):
        previous_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price

    day_high = quote_info.get("day_high")
    if day_high is None or pd.isna(day_high):
        day_high = float(df["High"].iloc[-1])

    day_low = quote_info.get("day_low")
    if day_low is None or pd.isna(day_low):
        day_low = float(df["Low"].iloc[-1])

    volume = quote_info.get("volume")
    if volume is None or pd.isna(volume):
        volume = int(df["Volume"].iloc[-1])

    abs_change = current_price - previous_close
    pct_change = (abs_change / previous_close * 100.0) if previous_close else 0.0

    # Metric Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Current Price",
            value=f"{curr_sym}{current_price:,.2f}",
        )
    with col2:
        st.metric(
            label="Day Change",
            value=f"{curr_sym}{abs_change:+,.2f}",
            delta=f"{pct_change:+.2f}%",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Day High",
            value=f"{curr_sym}{day_high:,.2f}",
        )
    with col4:
        st.metric(
            label="Day Low",
            value=f"{curr_sym}{day_low:,.2f}",
        )
    with col5:
        st.metric(
            label="Volume",
            value=format_volume(int(volume)),
        )

    st.markdown("---")

    # Build Technical Overlay Traces for Price Pane
    overlay_traces: List[object] = []
    data_points = len(df)

    if show_sma:
        if data_points < 20:
            st.info(f"ℹ️ Notice: SMA requires at least 20 bars; dataset contains {data_points} bars.")
        overlay_traces.extend(build_sma_traces(df["Close"]))

    if show_ema:
        if data_points < 12:
            st.info(f"ℹ️ Notice: EMA requires at least 12 bars; dataset contains {data_points} bars.")
        overlay_traces.extend(build_ema_traces(df["Close"]))

    if show_bb:
        if data_points < 20:
            st.info(f"ℹ️ Notice: Bollinger Bands require at least 20 bars; dataset contains {data_points} bars.")
        overlay_traces.extend(build_bollinger_traces(df["Close"]))

    if show_vwap:
        vwap_trace = build_vwap_trace(df)
        if vwap_trace is not None:
            overlay_traces.append(vwap_trace)

    # Interactive Candlestick and Volume Chart with Overlays
    main_fig = candlestick_with_volume(df=df, ticker=selected_ticker, overlay_traces=overlay_traces)
    st.plotly_chart(main_fig, use_container_width=True)

    # Separate RSI Panel (rendered only when toggled)
    if show_rsi:
        if data_points < 15:
            st.info(f"ℹ️ Notice: RSI(14) requires at least 15 bars to compute. Current dataset has {data_points} bars.")
        st.plotly_chart(rsi_chart(df, ticker=selected_ticker), use_container_width=True)

    # Separate MACD Panel (rendered only when toggled)
    if show_macd:
        if data_points < 26:
            st.info(f"ℹ️ Notice: MACD requires at least 26 bars for slow EMA. Current dataset has {data_points} bars.")
        st.plotly_chart(macd_chart(df, ticker=selected_ticker), use_container_width=True)

    # Indicator Summary Section
    st.markdown("---")
    st.subheader("📋 Technical Indicator Summary")
    summary_data = compute_indicator_summary(df)
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
