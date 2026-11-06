"""Interactive chart visualization module.

Constructs Plotly figures including candlesticks, OHLC charts, volume bars,
and technical indicator overlays/subplots with consistent styling.
"""

import os
import sys
from typing import Any, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is available on sys.path for direct module execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    CANDLE_DOWN_COLOR,
    CANDLE_UP_COLOR,
    PLOT_TEMPLATE,
    REQUIRED_OHLCV_COLUMNS,
    VOLUME_DOWN_COLOR,
    VOLUME_UP_COLOR,
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
        overlay_traces: Optional list of Plotly trace objects (e.g., Scatter)
                        to overlay directly onto the primary price pane.

    Returns:
        Plotly Figure object configured with dark styling, unified hover mode,
        and shared x-axes.
    """
    clean_ticker = ticker.strip().upper() if ticker else "UNKNOWN"

    # Return clean empty placeholder figure if data is missing or empty
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

    # Initialize 2-row subplot sharing x-axis
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

    # 2. Add any optional overlay traces (e.g., SMA/EMA/Bollinger from Phase 3)
    if overlay_traces:
        for trace in overlay_traces:
            fig.add_trace(trace, row=1, col=1)

    # 3. Volume Bar Colors (Green for close >= open, Red for close < open)
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

    # 4. Rangebreaks to eliminate non-trading gaps
    rangebreaks = [dict(bounds=["sat", "mon"])]

    # Detect if data is intraday to remove non-trading overnight hours
    if len(df.index) > 1:
        time_deltas = df.index[1:] - df.index[:-1]
        median_delta = pd.Series(time_deltas).median()
        if median_delta < pd.Timedelta(days=1):
            if clean_ticker.endswith((".NS", ".BO")):
                # Indian Exchanges (NSE/BSE): 15:30 to 09:15
                rangebreaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
            else:
                # US Exchanges (NYSE/NASDAQ): 16:00 to 09:30
                rangebreaks.append(dict(bounds=[16.0, 9.5], pattern="hour"))

    # Apply x-axis configurations across both subplots
    fig.update_xaxes(
        rangebreaks=rangebreaks,
        rangeslider=dict(visible=False),
        showgrid=True,
        gridcolor="#2A2E39",
    )

    # Configure y-axes
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

    # Overall figure layout
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


if __name__ == "__main__":
    # Self-test block
    dates = pd.date_range("2026-01-01", periods=10, freq="B", tz="UTC")
    sample_df = pd.DataFrame(
        {
            "Open": [150.0, 152.0, 151.0, 154.0, 153.0, 155.0, 157.0, 156.0, 158.0, 160.0],
            "High": [153.0, 154.0, 155.0, 156.0, 155.0, 158.0, 159.0, 158.0, 161.0, 163.0],
            "Low": [149.0, 150.0, 150.0, 152.0, 151.0, 154.0, 155.0, 154.0, 157.0, 159.0],
            "Close": [152.0, 151.0, 154.0, 153.0, 155.0, 157.0, 156.0, 158.0, 160.0, 162.0],
            "Volume": [1200000, 1400000, 1100000, 1600000, 1300000, 1500000, 1700000, 1250000, 1800000, 2000000],
        },
        index=dates,
    )

    fig = candlestick_with_volume(sample_df, "TEST_TICKER")
    print(f"Sample figure generated successfully with {len(fig.data)} traces.")

    empty_fig = candlestick_with_volume(pd.DataFrame(), "EMPTY_TICKER")
    print("Empty figure generated cleanly without exceptions.")
