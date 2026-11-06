"""Main Streamlit application entry point.

Handles UI layout, sidebar controls, user input, metric display cards,
technical indicator toggles, live auto-refresh with st.fragment,
and interactive chart presentation.
Does not contain direct data fetching, indicator computations, or business logic.
"""

from datetime import datetime, timezone
import os
import sys
from typing import Dict, List, Optional

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
    DEFAULT_REFRESH_SECONDS,
    DEFAULT_TICKERS,
    INTERVAL_SECONDS_MAP,
    MAX_REFRESH_SECONDS,
    MIN_REFRESH_SECONDS,
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


def render_live_dashboard(
    ticker: str,
    period: str,
    interval: str,
    live_mode: bool,
    refresh_seconds: int,
    show_sma: bool,
    show_ema: bool,
    show_bb: bool,
    show_vwap: bool,
    show_rsi: bool,
    show_macd: bool,
) -> None:
    """Render the live section of the dashboard with partial rerun capabilities.

    Uses st.fragment so auto-refreshes re-render only the metrics, charts,
    and summary table without causing full-page reloads or resetting sidebar widgets.

    Args:
        ticker: Selected stock symbol.
        period: Time range.
        interval: Bar interval.
        live_mode: Whether auto-refresh is active.
        refresh_seconds: Polling interval in seconds.
        show_sma: Toggle for SMA overlay.
        show_ema: Toggle for EMA overlay.
        show_bb: Toggle for Bollinger Bands.
        show_vwap: Toggle for VWAP.
        show_rsi: Toggle for RSI subpanel.
        show_macd: Toggle for MACD subpanel.
    """
    effective_run_every: Optional[int] = refresh_seconds if live_mode else None

    @st.fragment(run_every=effective_run_every)
    def _fragment_content() -> None:
        with st.spinner(f"Updating market data for {ticker}..."):
            raw_df = fetch_stock_data(
                ticker=ticker,
                period=period,
                interval=interval,
            )
            df = clean_data(raw_df)
            quote_info = get_quote_info(ticker)

        # Handle empty dataset cleanly
        if df.empty:
            st.error(
                f"⚠️ No market data could be loaded for **{ticker}** "
                f"with period **{period}** and interval **{interval}**."
            )
            st.info("Try selecting a different timeframe (e.g. 1mo / 1d) or another stock ticker.")
            return

        # Market-closed and stale data detection
        is_market_closed = False
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            latest_bar = df.index[-1]
            if latest_bar.tzinfo is not None:
                now_aware = datetime.now(latest_bar.tzinfo)
                age_seconds = (now_aware - latest_bar).total_seconds()
                is_weekend = now_aware.weekday() >= 5
            else:
                now_utc = datetime.now(timezone.utc)
                age_seconds = (now_utc - pd.to_datetime(latest_bar).tz_localize("UTC")).total_seconds()
                is_weekend = now_utc.weekday() >= 5

            interval_sec = INTERVAL_SECONDS_MAP.get(interval, 86400)
            if is_weekend or age_seconds > (2.0 * interval_sec):
                is_market_closed = True

        # Status badge determination
        last_updated_time = datetime.now().strftime("%H:%M:%S")
        if not live_mode:
            status_badge = "⏸ Paused"
        elif is_market_closed:
            status_badge = "🔴 Market closed or no new data"
        else:
            status_badge = "🟢 Live"

        # Status Bar Header
        status_col1, status_col2 = st.columns([3, 1])
        with status_col1:
            st.markdown(f"**Status:** {status_badge} &nbsp;|&nbsp; **Last updated:** `{last_updated_time}`")
        with status_col2:
            if live_mode:
                st.caption(f"⏱ Auto-refresh: every {refresh_seconds}s")

        if is_market_closed:
            st.warning("⚠️ Market closed — showing last available data.")

        # Header metadata
        company_name = quote_info.get("company_name") or ticker
        currency = quote_info.get("currency", "USD")
        curr_sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ")

        st.title(f"{company_name} ({ticker})")

        last_bar_time = df.index[-1].strftime("%Y-%m-%d %H:%M %Z")
        st.caption(
            f"Currency: **{currency}** | Timeframe: **{period}** | Bar: **{interval}** | "
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
        main_fig = candlestick_with_volume(df=df, ticker=ticker, overlay_traces=overlay_traces)
        st.plotly_chart(main_fig, use_container_width=True)

        # Separate RSI Panel (rendered only when toggled)
        if show_rsi:
            if data_points < 15:
                st.info(f"ℹ️ Notice: RSI(14) requires at least 15 bars to compute. Current dataset has {data_points} bars.")
            st.plotly_chart(rsi_chart(df, ticker=ticker), use_container_width=True)

        # Separate MACD Panel (rendered only when toggled)
        if show_macd:
            if data_points < 26:
                st.info(f"ℹ️ Notice: MACD requires at least 26 bars for slow EMA. Current dataset has {data_points} bars.")
            st.plotly_chart(macd_chart(df, ticker=ticker), use_container_width=True)

        # Indicator Summary Section
        st.markdown("---")
        st.subheader("📋 Technical Indicator Summary")
        summary_data = compute_indicator_summary(df)
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    _fragment_content()


def main() -> None:
    """Render the main Streamlit stock market dashboard."""
    st.set_page_config(
        layout="wide",
        page_title="Stock Dashboard",
        page_icon="📈",
    )

    # Initialize ticker list and state defaults
    if "tickers" not in st.session_state:
        st.session_state.tickers = list(DEFAULT_TICKERS)
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = DEFAULT_TICKERS[0]
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = False
    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = DEFAULT_REFRESH_SECONDS

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
        key="period_selector",
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
        key="interval_selector",
    )

    # Technical Indicators Toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Technical Indicators")
    show_sma = st.sidebar.checkbox("SMA (20, 50)", value=False, key="toggle_sma", help="Simple Moving Averages (20 & 50 period)")
    show_ema = st.sidebar.checkbox("EMA (12, 26)", value=False, key="toggle_ema", help="Exponential Moving Averages (12 & 26 period)")
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=False, key="toggle_bb", help="Bollinger Bands with 2 standard deviations")
    show_vwap = st.sidebar.checkbox("VWAP", value=False, key="toggle_vwap", help="Volume Weighted Average Price (resets daily for intraday)")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=False, key="toggle_rsi", help="Relative Strength Index oscillator")
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=False, key="toggle_macd", help="Moving Average Convergence Divergence")

    # Live Mode & Auto-Refresh Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱ Live Refresh")
    live_mode = st.sidebar.toggle(
        "Live Mode",
        value=st.session_state.live_mode,
        key="live_mode_toggle",
        help="Periodically refresh chart and metrics without full page reload",
    )
    st.session_state.live_mode = live_mode

    refresh_seconds = st.sidebar.slider(
        "Refresh Rate (seconds)",
        min_value=MIN_REFRESH_SECONDS,
        max_value=MAX_REFRESH_SECONDS,
        value=st.session_state.refresh_rate,
        step=5,
        disabled=not live_mode,
        key="refresh_rate_slider",
        help="Polling interval in seconds (minimum 5s)",
    )
    st.session_state.refresh_rate = refresh_seconds

    # Manual Refresh Button
    if st.sidebar.button("🔄 Refresh now", use_container_width=True, key="manual_refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    # --- Render Dashboard Fragment ---
    render_live_dashboard(
        ticker=selected_ticker,
        period=selected_period,
        interval=selected_interval,
        live_mode=live_mode,
        refresh_seconds=refresh_seconds,
        show_sma=show_sma,
        show_ema=show_ema,
        show_bb=show_bb,
        show_vwap=show_vwap,
        show_rsi=show_rsi,
        show_macd=show_macd,
    )


if __name__ == "__main__":
    main()
