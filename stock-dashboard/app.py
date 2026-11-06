"""Main Streamlit application entry point.

Handles UI layout, sidebar controls, user input, metric display cards,
technical indicator toggles, live auto-refresh with st.fragment,
multi-stock watchlist, normalized performance comparison, correlation analysis,
and data/chart exports.
Does not contain direct API calls, indicator computations, or business logic.
"""

from datetime import datetime, timezone
import os
import sys
from typing import Any, Dict, List, Optional

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
    correlation_heatmap,
    macd_chart,
    normalized_comparison_chart,
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
    RISK_FREE_RATE,
    VALID_INTERVALS_FOR_PERIOD,
)
from data.fetcher import (  # noqa: E402
    DataFetchError,
    fetch_multiple,
    fetch_stock_data,
    get_quote_info,
    validate_ticker,
)
from data.processor import (  # noqa: E402
    align_prices_on_common_dates,
    calculate_risk_return_metrics,
    clean_data,
    normalize_prices,
)
from indicators.technical import (  # noqa: E402
    bollinger_bands,
    compute_indicator_summary,
    ema,
    macd,
    rsi,
    sma,
    vwap,
)

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
    """Render the live single-stock section with partial rerun capabilities.

    Uses st.fragment so auto-refreshes re-render only the metrics, charts,
    and summary table without full-page reloads.
    """
    effective_run_every: Optional[int] = refresh_seconds if live_mode else None

    @st.fragment(run_every=effective_run_every)
    def _fragment_content() -> None:
        try:
            with st.spinner(f"Updating market data for {ticker}..."):
                raw_df = fetch_stock_data(
                    ticker=ticker,
                    period=period,
                    interval=interval,
                )
                df = clean_data(raw_df)
                quote_info = get_quote_info(ticker)
        except DataFetchError as dfe:
            st.error(f"⚠️ Network error while fetching data for **{ticker}**: {dfe}")
            st.info("Please check your internet connection or try again shortly.")
            return
        except Exception as exc:
            st.error(f"⚠️ An unexpected error occurred while loading **{ticker}**: {exc}")
            return

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
            f"Latest Bar: **{last_bar_time}** | Source: Yahoo Finance (~15m delay)"
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

        # Export Section
        st.markdown("---")
        st.subheader("📥 Export Data & Interactive Charts")

        # Compile comprehensive export dataset with indicators
        export_df = df.copy()
        export_df["SMA_20"] = sma(df["Close"], 20)
        export_df["SMA_50"] = sma(df["Close"], 50)
        export_df["EMA_12"] = ema(df["Close"], 12)
        export_df["EMA_26"] = ema(df["Close"], 26)
        bb_df = bollinger_bands(df["Close"], 20, 2.0)
        export_df["BB_Upper"] = bb_df["Upper"]
        export_df["BB_Middle"] = bb_df["Middle"]
        export_df["BB_Lower"] = bb_df["Lower"]
        export_df["VWAP"] = vwap(df)
        export_df["RSI_14"] = rsi(df["Close"], 14)
        macd_data = macd(df["Close"], 12, 26, 9)
        export_df["MACD"] = macd_data["MACD"]
        export_df["MACD_Signal"] = macd_data["Signal"]
        export_df["MACD_Hist"] = macd_data["Histogram"]

        csv_bytes = export_df.to_csv().encode("utf-8")
        html_bytes = main_fig.to_html(include_plotlyjs="cdn").encode("utf-8")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Data + Indicators (CSV)",
                data=csv_bytes,
                file_name=f"{ticker}_{period}_{interval}_indicators.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv_btn",
            )
        with col_d2:
            st.download_button(
                label="📊 Download Interactive Chart (HTML)",
                data=html_bytes,
                file_name=f"{ticker}_{period}_{interval}_chart.html",
                mime="text/html",
                use_container_width=True,
                key="download_html_btn",
            )

        # Educational & Methodology Expanders
        st.markdown("---")
        with st.expander("ℹ️ About This Data & Market Latency"):
            st.markdown(
                """
                - **Data Source**: Market data retrieved via Yahoo Finance API (`yfinance`).
                - **Feed Latency**: Real-time for major US equities; up to **15 minutes delayed** for certain global exchanges (e.g. NSE / BSE).
                - **Adjusted Prices**: All OHLC values are split- and dividend-adjusted (`auto_adjust=True`) to maintain mathematical integrity across technical indicators.
                - **Trading Calendars**: Multi-stock analyses harmonize differing regional trading calendars (e.g. US holidays vs. Indian market holidays) via forward-filling.
                """
            )

        with st.expander("📚 Technical Indicator Explanations"):
            st.markdown(
                r"""
                - **SMA (Simple Moving Average)**: The unweighted arithmetic mean over $k$ periods. Smoothes noise to reveal underlying trend.
                - **EMA (Exponential Moving Average)**: Applies exponentially decreasing weights to older data points, prioritizing recent price discovery.
                - **Bollinger Bands**: An envelope plotted at $\pm 2$ standard deviations around a 20-day SMA. Band contraction indicates consolidation; widening indicates surging volatility.
                - **VWAP (Volume Weighted Average Price)**: Cumulative ratio of dollar volume to total share volume. Automatically resets daily on intraday timeframes to benchmark institutional execution quality.
                - **RSI (Relative Strength Index)**: Momentum oscillator bounded from 0 to 100 with Wilder's exponential smoothing. Readouts above 70 indicate overbought conditions; below 30 indicate oversold conditions.
                - **MACD (Moving Average Convergence Divergence)**: Difference between the 12-day and 26-day EMAs, with a 9-day EMA signal trigger line and momentum histogram.
                """
            )

    _fragment_content()


def render_watchlist_tab(
    tickers: List[str],
    period: str,
    interval: str,
) -> None:
    """Render the Watchlist tab with tabular metrics, sparklines, and export."""
    st.subheader("👀 Watchlist Overview")

    if not tickers:
        st.info("Please select or add at least one ticker in the sidebar to populate your watchlist.")
        return

    try:
        with st.spinner("Fetching market quotes for watchlist..."):
            data_dict = fetch_multiple(tickers, period=period, interval=interval)
    except Exception as exc:
        st.error(f"⚠️ Error loading watchlist: {exc}")
        return

    valid_dict: Dict[str, pd.DataFrame] = {t: clean_data(df) for t, df in data_dict.items() if not df.empty}
    failed_tickers = [t for t in tickers if t not in valid_dict]

    if failed_tickers:
        st.warning(f"⚠️ Could not retrieve market data for: {', '.join(failed_tickers)} (skipped)")

    if not valid_dict:
        st.error("No valid market data could be loaded for any selected ticker.")
        return

    rows: List[Dict[str, Any]] = []

    for ticker, df in valid_dict.items():
        info = get_quote_info(ticker)
        curr_price = info.get("current_price")
        if curr_price is None or pd.isna(curr_price):
            curr_price = float(df["Close"].iloc[-1])

        prev_close = info.get("previous_close")
        if prev_close is None or pd.isna(prev_close):
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else curr_price

        day_high = info.get("day_high")
        if day_high is None or pd.isna(day_high):
            day_high = float(df["High"].iloc[-1])

        day_low = info.get("day_low")
        if day_low is None or pd.isna(day_low):
            day_low = float(df["Low"].iloc[-1])

        volume = info.get("volume")
        if volume is None or pd.isna(volume):
            volume = int(df["Volume"].iloc[-1])

        abs_change = curr_price - prev_close
        pct_change = (abs_change / prev_close * 100.0) if prev_close else 0.0

        currency = info.get("currency", "USD")
        curr_sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
        company_name = info.get("company_name", ticker)

        # Extract last 30 closes for sparkline trend
        sparkline_data = df["Close"].dropna().tail(30).tolist()

        rows.append(
            {
                "Ticker": ticker,
                "Company": company_name,
                "Price": f"{curr_sym}{curr_price:,.2f}",
                "Change %": pct_change,
                "Day High": f"{curr_sym}{day_high:,.2f}",
                "Day Low": f"{curr_sym}{day_low:,.2f}",
                "Volume": format_volume(int(volume)),
                "Trend": sparkline_data,
            }
        )

    watchlist_df = pd.DataFrame(rows)

    styled_df = watchlist_df.style.map(
        lambda val: f"color: {'#26a69a' if val >= 0 else '#ef5350'}; font-weight: 600;"
        if isinstance(val, (int, float)) and not pd.isna(val)
        else "",
        subset=["Change %"],
    )

    st.dataframe(
        styled_df,
        column_config={
            "Trend": st.column_config.LineChartColumn("Price Trend (Recent)", width="medium"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Watchlist Export
    st.markdown("---")
    export_watchlist = watchlist_df.drop(columns=["Trend"])
    st.download_button(
        label="📥 Download Watchlist Summary (CSV)",
        data=export_watchlist.to_csv(index=False).encode("utf-8"),
        file_name="watchlist_summary.csv",
        mime="text/csv",
        key="download_watchlist_csv",
    )


def render_compare_tab(
    tickers: List[str],
    period: str,
    interval: str,
) -> None:
    """Render multi-stock normalized comparison, correlation heatmap, and risk metrics."""
    st.subheader("⚖️ Multi-Stock Performance & Correlation")

    if len(tickers) < 2:
        st.info("ℹ️ Please select at least 2 tickers from the sidebar to generate comparative analysis.")
        return

    try:
        with st.spinner("Fetching and harmonizing multi-market comparison data..."):
            data_dict = fetch_multiple(tickers, period=period, interval=interval)
    except Exception as exc:
        st.error(f"⚠️ Error loading comparative data: {exc}")
        return

    valid_dict: Dict[str, pd.DataFrame] = {t: clean_data(df) for t, df in data_dict.items() if not df.empty}
    failed_tickers = [t for t in tickers if t not in valid_dict]

    if failed_tickers:
        st.warning(f"⚠️ Could not load data for: {', '.join(failed_tickers)} (skipped)")

    if len(valid_dict) < 2:
        st.warning("At least 2 valid stocks with active market data are required for comparative charts.")
        return

    # 1. Normalized Performance Chart (rebased to 100)
    norm_df = normalize_prices(valid_dict)
    fig_norm = normalized_comparison_chart(norm_df)
    st.plotly_chart(fig_norm, use_container_width=True)

    st.caption(
        "ℹ️ **Multi-market calendar alignment:** Stocks from different international exchanges "
        "(e.g., US NYSE/NASDAQ and Indian NSE/BSE) are aligned on overlapping trading dates. "
        "Missing exchange holiday sessions are forward-filled."
    )

    st.markdown("---")

    # 2. Correlation Matrix Heatmap of Daily Returns
    aligned_df = align_prices_on_common_dates(valid_dict, method="ffill")
    daily_returns = aligned_df.pct_change().dropna()

    if not daily_returns.empty and len(daily_returns.columns) >= 2:
        corr_matrix = daily_returns.corr()
        col_heat, col_risk = st.columns([1, 1])

        with col_heat:
            st.subheader("🔥 Returns Correlation")
            fig_corr = correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_risk:
            st.subheader("📊 Risk & Return Metrics")
            risk_df = calculate_risk_return_metrics(valid_dict, risk_free_rate=RISK_FREE_RATE)
            st.dataframe(
                risk_df,
                column_config={
                    "Total Return %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Annualized Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Max Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Sharpe Ratio": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Risk-free rate assumed at **{RISK_FREE_RATE * 100:.1f}%** p.a. | "
                "Drawdown measured peak-to-trough."
            )
            st.download_button(
                label="📥 Export Risk Metrics (CSV)",
                data=risk_df.to_csv(index=False).encode("utf-8"),
                file_name="risk_return_comparison.csv",
                mime="text/csv",
                key="download_risk_metrics_csv",
            )
    else:
        st.info("Insufficient overlapping time-series data to compute cross-asset correlation.")


def main() -> None:
    """Render the main Streamlit stock market dashboard with multi-tab layout."""
    st.set_page_config(
        layout="wide",
        page_title="Stock Dashboard",
        page_icon="📈",
    )

    # Minimalist Custom CSS styling for metric cards and polish
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: #1a1c24;
            border: 1px solid #2a2e39;
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetric"]:hover {
            border-color: #26a69a;
            transition: border-color 0.2s ease-in-out;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.85rem;
            color: #9e9e9e;
        }
        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    # Timeframe Controls
    default_period_idx = (
        PERIOD_OPTIONS.index(DEFAULT_PERIOD) if DEFAULT_PERIOD in PERIOD_OPTIONS else 2
    )
    selected_period = st.sidebar.selectbox(
        "Period",
        options=PERIOD_OPTIONS,
        index=default_period_idx,
        key="period_selector",
    )

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

    # Multi-Stock Watchlist & Compare Selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Watchlist & Compare")
    default_watchlist = [t for t in DEFAULT_TICKERS[:4] if t in st.session_state.tickers]
    watchlist_tickers = st.sidebar.multiselect(
        "Select Watchlist Stocks",
        options=st.session_state.tickers,
        default=default_watchlist,
        key="watchlist_multiselect",
        help="Choose 2 to 8 stocks to monitor in the Watchlist and Compare tabs.",
    )

    # Single Stock Specific Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Single Stock Controls")
    current_index = (
        st.session_state.tickers.index(st.session_state.current_ticker)
        if st.session_state.current_ticker in st.session_state.tickers
        else 0
    )
    selected_ticker = st.sidebar.selectbox(
        "Active Stock",
        options=st.session_state.tickers,
        index=current_index,
        key="ticker_selector",
    )
    st.session_state.current_ticker = selected_ticker

    show_sma = st.sidebar.checkbox("SMA (20, 50)", value=False, key="toggle_sma", help="Simple Moving Averages (20 & 50 period)")
    show_ema = st.sidebar.checkbox("EMA (12, 26)", value=False, key="toggle_ema", help="Exponential Moving Averages (12 & 26 period)")
    show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=False, key="toggle_bb", help="Bollinger Bands with 2 standard deviations")
    show_vwap = st.sidebar.checkbox("VWAP", value=False, key="toggle_vwap", help="Volume Weighted Average Price (resets daily for intraday)")
    show_rsi = st.sidebar.checkbox("RSI (14)", value=False, key="toggle_rsi", help="Relative Strength Index oscillator")
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=False, key="toggle_macd", help="Moving Average Convergence Divergence")

    # Live Mode Controls
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

    # --- Main Area Tabs ---
    tab_single, tab_watchlist, tab_compare = st.tabs(
        ["📈 Single Stock", "👀 Watchlist", "⚖️ Compare"]
    )

    with tab_single:
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

    with tab_watchlist:
        render_watchlist_tab(
            tickers=watchlist_tickers,
            period=selected_period,
            interval=selected_interval,
        )

    with tab_compare:
        render_compare_tab(
            tickers=watchlist_tickers,
            period=selected_period,
            interval=selected_interval,
        )

    # Footer Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** Market data is provided by Yahoo Finance and may be delayed up to 15 minutes for certain exchanges. "
        "All calculations, technical indicators, and visualizations are for educational and informational purposes only and do not constitute financial, investment, or trading advice."
    )


if __name__ == "__main__":
    main()
