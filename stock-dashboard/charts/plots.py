"""Interactive chart visualization module.

Constructs Plotly figures including candlesticks, OHLC charts, volume bars,
and technical indicator overlays/subplots with consistent styling.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is available on sys.path for direct module execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    CANDLE_DOWN_COLOR,
    CANDLE_UP_COLOR,
    EMA_WINDOWS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PLOT_TEMPLATE,
    REQUIRED_OHLCV_COLUMNS,
    RSI_WINDOW,
    SMA_WINDOWS,
    VOLUME_DOWN_COLOR,
    VOLUME_UP_COLOR,
)
from indicators.technical import (  # noqa: E402
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    vwap,
)


def get_trading_rangebreaks(df: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
    """Determine rangebreak rules to eliminate weekend and non-trading gaps.

    Args:
        df: DataFrame with DatetimeIndex.
        ticker: Ticker symbol (used to determine market hours).

    Returns:
        List of rangebreak dictionaries for Plotly x-axis configuration.
    """
    rangebreaks: List[Dict[str, Any]] = [dict(bounds=["sat", "mon"])]

    if df is not None and len(df.index) > 1 and isinstance(df.index, pd.DatetimeIndex):
        time_deltas = df.index[1:] - df.index[:-1]
        median_delta = pd.Series(time_deltas).median()
        if median_delta < pd.Timedelta(days=1):
            clean_ticker = ticker.strip().upper() if ticker else ""
            if clean_ticker.endswith((".NS", ".BO")):
                # Indian Exchanges (NSE/BSE): 15:30 to 09:15
                rangebreaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
            else:
                # US Exchanges (NYSE/NASDAQ): 16:00 to 09:30
                rangebreaks.append(dict(bounds=[16.0, 9.5], pattern="hour"))

    return rangebreaks


def build_sma_traces(
    series: pd.Series,
    windows: Tuple[int, ...] = SMA_WINDOWS,
    colors: Optional[Dict[int, str]] = None,
) -> List[go.Scatter]:
    """Construct Plotly Scatter overlay traces for Simple Moving Averages.

    Args:
        series: Price series (e.g. Close).
        windows: Tuple of lookback windows (e.g. 20, 50).
        colors: Optional dictionary mapping window size to hex/rgb color.

    Returns:
        List of go.Scatter traces for SMA lines.
    """
    default_colors = {20: "#FFA726", 50: "#29B6F6", 200: "#AB47BC"}
    active_colors = colors or default_colors
    traces: List[go.Scatter] = []

    for w in windows:
        sma_val = sma(series, window=w)
        if sma_val.isna().all():
            continue
        line_color = active_colors.get(w, "#FFCA28")
        traces.append(
            go.Scatter(
                x=sma_val.index,
                y=sma_val,
                mode="lines",
                name=f"SMA {w}",
                line=dict(color=line_color, width=1.5),
            )
        )
    return traces


def build_ema_traces(
    series: pd.Series,
    spans: Tuple[int, ...] = EMA_WINDOWS,
    colors: Optional[Dict[int, str]] = None,
) -> List[go.Scatter]:
    """Construct Plotly Scatter overlay traces for Exponential Moving Averages.

    Args:
        series: Price series (e.g. Close).
        spans: Tuple of EMA spans (e.g. 12, 26).
        colors: Optional mapping of span to color.

    Returns:
        List of go.Scatter traces for EMA lines.
    """
    default_colors = {12: "#E040FB", 26: "#00E676"}
    active_colors = colors or default_colors
    traces: List[go.Scatter] = []

    for s in spans:
        ema_val = ema(series, span=s)
        if ema_val.isna().all():
            continue
        line_color = active_colors.get(s, "#00E5FF")
        traces.append(
            go.Scatter(
                x=ema_val.index,
                y=ema_val,
                mode="lines",
                name=f"EMA {s}",
                line=dict(color=line_color, width=1.5),
            )
        )
    return traces


def build_bollinger_traces(
    series: pd.Series,
    window: int = BOLLINGER_WINDOW,
    num_std: float = BOLLINGER_STD,
) -> List[go.Scatter]:
    """Construct Plotly Scatter overlay traces for Bollinger Bands.

    Draws upper band, middle SMA band, and lower band with semi-transparent
    fill shading between the outer envelopes.

    Args:
        series: Price series (e.g. Close).
        window: Moving average window (default 20).
        num_std: Multiplier for standard deviation (default 2.0).

    Returns:
        List of 3 go.Scatter traces representing the Bollinger Bands envelope.
    """
    bb_df = bollinger_bands(series, window=window, num_std=num_std)
    if bb_df.isna().all().all():
        return []

    traces: List[go.Scatter] = []

    # 1. Upper Band
    traces.append(
        go.Scatter(
            x=bb_df.index,
            y=bb_df["Upper"],
            mode="lines",
            name=f"BB Upper ({window}, {num_std}σ)",
            line=dict(color="rgba(100, 181, 246, 0.4)", width=1),
            showlegend=True,
        )
    )

    # 2. Lower Band (shaded to upper band via tonexty)
    traces.append(
        go.Scatter(
            x=bb_df.index,
            y=bb_df["Lower"],
            mode="lines",
            name=f"BB Lower ({window}, {num_std}σ)",
            line=dict(color="rgba(100, 181, 246, 0.4)", width=1),
            fill="tonexty",
            fillcolor="rgba(100, 181, 246, 0.08)",
            showlegend=True,
        )
    )

    # 3. Middle Band (SMA line)
    traces.append(
        go.Scatter(
            x=bb_df.index,
            y=bb_df["Middle"],
            mode="lines",
            name=f"BB Middle ({window})",
            line=dict(color="rgba(255, 255, 255, 0.45)", dash="dash", width=1),
            showlegend=True,
        )
    )

    return traces


def build_vwap_trace(df: pd.DataFrame) -> Optional[go.Scatter]:
    """Construct Plotly Scatter overlay trace for VWAP.

    Args:
        df: Cleaned OHLCV DataFrame.

    Returns:
        go.Scatter trace for VWAP, or None if data is insufficient.
    """
    vwap_series = vwap(df)
    if vwap_series.isna().all():
        return None

    return go.Scatter(
        x=vwap_series.index,
        y=vwap_series,
        mode="lines",
        name="VWAP",
        line=dict(color="#FFEB3B", width=1.8),
    )


def candlestick_with_volume(
    df: pd.DataFrame,
    ticker: str,
    overlay_traces: Optional[List[Any]] = None,
) -> go.Figure:
    """Build an interactive 2-row subplot with Candlestick and Volume bars.

    The upper row (75% height) displays price candlesticks and any optional
    technical overlay traces (e.g. moving averages, Bollinger Bands).
    The lower row (25% height) displays volume bars colored by up/down price action.
    Hides non-trading periods and weekends via Plotly rangebreaks to eliminate gaps.

    Args:
        df: Cleaned OHLCV DataFrame with a DatetimeIndex.
        ticker: Ticker symbol for display titling.
        overlay_traces: Optional list of Plotly trace objects to overlay
                        directly onto the primary price pane.

    Returns:
        Plotly Figure object configured with dark styling, unified hover mode,
        and shared x-axes.
    """
    clean_ticker = ticker.strip().upper() if ticker else "UNKNOWN"

    if df is None or df.empty or not all(col in df.columns for col in REQUIRED_OHLCV_COLUMNS):
        fig = go.Figure()
        fig.update_layout(
            template=PLOT_TEMPLATE,
            annotations=[
                dict(
                    text="No market data available for the selected parameters.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#888888"),
                )
            ],
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=500,
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. Primary Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=CANDLE_UP_COLOR,
            increasing_fillcolor=CANDLE_UP_COLOR,
            decreasing_line_color=CANDLE_DOWN_COLOR,
            decreasing_fillcolor=CANDLE_DOWN_COLOR,
        ),
        row=1,
        col=1,
    )

    # 2. Add technical overlays to Row 1
    if overlay_traces:
        for trace in overlay_traces:
            if trace is not None:
                fig.add_trace(trace, row=1, col=1)

    # 3. Volume Bars
    volume_colors = [
        VOLUME_UP_COLOR if close_val >= open_val else VOLUME_DOWN_COLOR
        for close_val, open_val in zip(df["Close"], df["Open"])
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )

    rangebreaks = get_trading_rangebreaks(df, clean_ticker)

    fig.update_xaxes(
        rangebreaks=rangebreaks,
        rangeslider=dict(visible=False),
        showgrid=True,
        gridcolor="#2A2E39",
    )

    fig.update_yaxes(
        title_text="Price",
        row=1,
        col=1,
        showgrid=True,
        gridcolor="#2A2E39",
    )
    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="#2A2E39",
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=50, b=30),
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        title=dict(
            text=f"<b>{clean_ticker}</b> Price Action & Volume",
            x=0.02,
            xanchor="left",
        ),
    )

    return fig


def rsi_chart(df: pd.DataFrame, period: int = RSI_WINDOW, ticker: str = "") -> go.Figure:
    """Build an interactive Relative Strength Index (RSI) panel.

    Displays the RSI oscillator with horizontal reference lines at 70 (Overbought)
    and 30 (Oversold), and a subtle shaded middle neutral zone.

    Args:
        df: Cleaned OHLCV DataFrame.
        period: RSI lookback period (default 14).
        ticker: Ticker symbol for titling.

    Returns:
        Plotly Figure for the RSI panel.
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    fig = go.Figure()

    if df is None or df.empty or "Close" not in df.columns:
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=200,
            annotations=[dict(text="No data for RSI", showarrow=False, font=dict(color="#888"))],
        )
        return fig

    rsi_series = rsi(df["Close"], period=period)

    # 1. Overbought reference line (70)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[70.0] * len(df),
            mode="lines",
            name="Overbought (70)",
            line=dict(color="rgba(239, 83, 80, 0.6)", dash="dash", width=1),
            hoverinfo="skip",
        )
    )

    # 2. Oversold reference line (30) with fill to line 70
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30.0] * len(df),
            mode="lines",
            name="Oversold (30)",
            line=dict(color="rgba(38, 166, 154, 0.6)", dash="dash", width=1),
            fill="tonexty",
            fillcolor="rgba(255, 255, 255, 0.03)",
            hoverinfo="skip",
        )
    )

    # 3. RSI line
    fig.add_trace(
        go.Scatter(
            x=rsi_series.index,
            y=rsi_series,
            mode="lines",
            name=f"RSI ({period})",
            line=dict(color="#AB47BC", width=2),
        )
    )

    rangebreaks = get_trading_rangebreaks(df, clean_ticker)

    fig.update_xaxes(
        rangebreaks=rangebreaks,
        rangeslider=dict(visible=False),
        showgrid=True,
        gridcolor="#2A2E39",
    )
    fig.update_yaxes(
        range=[0, 100],
        tickvals=[30, 50, 70],
        showgrid=True,
        gridcolor="#2A2E39",
        title_text="RSI",
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=30, b=20),
        height=200,
        showlegend=False,
        title=dict(
            text=f"<b>{clean_ticker}</b> Relative Strength Index ({period})",
            font=dict(size=12),
            x=0.02,
        ),
    )

    return fig


def macd_chart(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    ticker: str = "",
) -> go.Figure:
    """Build an interactive Moving Average Convergence Divergence (MACD) panel.

    Displays the MACD line, Signal line, and colored histogram bars.

    Args:
        df: Cleaned OHLCV DataFrame.
        fast: Fast EMA span (default 12).
        slow: Slow EMA span (default 26).
        signal: Signal EMA span (default 9).
        ticker: Ticker symbol for titling.

    Returns:
        Plotly Figure for the MACD panel.
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    fig = go.Figure()

    if df is None or df.empty or "Close" not in df.columns:
        fig.update_layout(
            template=PLOT_TEMPLATE,
            height=220,
            annotations=[dict(text="No data for MACD", showarrow=False, font=dict(color="#888"))],
        )
        return fig

    macd_df = macd(df["Close"], fast=fast, slow=slow, signal=signal)

    # Histogram colors: Green when positive, Red when negative
    hist_colors = [
        "rgba(38, 166, 154, 0.7)" if val >= 0 else "rgba(239, 83, 80, 0.7)"
        for val in macd_df["Histogram"].fillna(0)
    ]

    # 1. Histogram
    fig.add_trace(
        go.Bar(
            x=macd_df.index,
            y=macd_df["Histogram"],
            name="Histogram",
            marker_color=hist_colors,
        )
    )

    # 2. MACD Line
    fig.add_trace(
        go.Scatter(
            x=macd_df.index,
            y=macd_df["MACD"],
            mode="lines",
            name=f"MACD ({fast}, {slow})",
            line=dict(color="#29B6F6", width=1.6),
        )
    )

    # 3. Signal Line
    fig.add_trace(
        go.Scatter(
            x=macd_df.index,
            y=macd_df["Signal"],
            mode="lines",
            name=f"Signal ({signal})",
            line=dict(color="#FFA726", width=1.6),
        )
    )

    rangebreaks = get_trading_rangebreaks(df, clean_ticker)

    fig.update_xaxes(
        rangebreaks=rangebreaks,
        rangeslider=dict(visible=False),
        showgrid=True,
        gridcolor="#2A2E39",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#2A2E39",
        title_text="MACD",
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=30, b=20),
        height=220,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        title=dict(
            text=f"<b>{clean_ticker}</b> MACD ({fast}, {slow}, {signal})",
            font=dict(size=12),
            x=0.02,
        ),
    )

    return fig


if __name__ == "__main__":
    dates = pd.date_range("2026-01-01", periods=60, freq="B", tz="UTC")
    sample_df = pd.DataFrame(
        {
            "Open": [150.0 + i * 0.5 for i in range(60)],
            "High": [152.0 + i * 0.5 for i in range(60)],
            "Low": [149.0 + i * 0.5 for i in range(60)],
            "Close": [151.0 + i * 0.5 for i in range(60)],
            "Volume": [1000000 + i * 10000 for i in range(60)],
        },
        index=dates,
    )

    # Test overlays
    overlays = []
    overlays.extend(build_sma_traces(sample_df["Close"]))
    overlays.extend(build_ema_traces(sample_df["Close"]))
    overlays.extend(build_bollinger_traces(sample_df["Close"]))
    vwap_t = build_vwap_trace(sample_df)
    if vwap_t:
        overlays.append(vwap_t)

    fig_main = candlestick_with_volume(sample_df, "TEST", overlay_traces=overlays)
    fig_rsi = rsi_chart(sample_df, ticker="TEST")
    fig_macd = macd_chart(sample_df, ticker="TEST")

    print(f"Candlestick with overlays traces: {len(fig_main.data)}")
    print(f"RSI traces: {len(fig_rsi.data)}")
    print(f"MACD traces: {len(fig_macd.data)}")
    print("All chart builder tests passed!")
