# Real-Time Stock Market Dashboard

A Streamlit dashboard that tracks and visualizes live stock market data with interactive real-time charts and financial indicators for user-selected stocks.

## Architecture

```
stock-dashboard/
├── app.py                 # Streamlit entry point (UI only, no business logic)
├── data/
│   ├── __init__.py        # Package initializer
│   ├── fetcher.py         # API calls, caching, error handling (DataFetchError)
│   └── processor.py       # Cleaning, resampling, risk/return analytics
├── indicators/
│   ├── __init__.py        # Package initializer
│   └── technical.py       # Hand-coded SMA, EMA, RSI, MACD, Bollinger Bands, VWAP
├── charts/
│   ├── __init__.py        # Package initializer
│   └── plots.py           # Plotly figure builders, dark theme, overlays, heatmaps
├── tests/
│   ├── test_indicators.py # 13 unit tests for technical indicators
│   └── test_processor.py  # 9 unit tests for processing & risk metrics
├── config.py              # Default tickers, intervals, refresh rate, risk-free rate
├── requirements.txt
└── README.md
```

## Features

- **Single Stock Dashboard**:
  - Live metric cards (Current Price, Day Change with % and delta color, Day High/Low, Volume).
  - Candlestick chart with volume subplots, rangebreaks hiding non-trading gaps, and dark theme.
  - Interactive technical indicator overlays (SMA 20/50, EMA 12/26, Bollinger Bands, daily resetting VWAP).
  - Standalone RSI and MACD sub-chart panels.
  - Signal summary table with automated technical interpretations.
  - Live auto-refresh powered by `st.fragment(run_every=...)` with market-status badges.
  - CSV export (with computed indicator columns) and HTML chart download.
- **Multi-Stock Watchlist**:
  - Tabular overview with colored change percentages and dynamic price sparkline trends via `st.column_config.LineChartColumn`.
- **Comparative Analysis**:
  - Normalized performance chart (rebased to 100).
  - International market calendar alignment (US vs. NSE holidays).
  - Returns correlation heatmap (`go.Heatmap` with diverging colors).
  - Risk & return analytics table (Total Return %, Annualized Volatility, Max Drawdown %, Sharpe Ratio).

## Setup & Running

1. **Activate the virtual environment**:
   ```bash
   source ../.venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run unit tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```
