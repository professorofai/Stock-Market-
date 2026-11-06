# Real-Time Stock Market Dashboard

A Streamlit dashboard that tracks and visualizes live stock market data with interactive real-time charts and financial indicators for user-selected stocks.

## Architecture

```
stock-dashboard/
├── app.py                 # Streamlit entry point (UI only, no business logic)
├── data/
│   ├── __init__.py        # Package initializer
│   ├── fetcher.py         # API calls, caching, error handling
│   └── processor.py       # Cleaning and resampling with Pandas
├── indicators/
│   ├── __init__.py        # Package initializer
│   └── technical.py       # SMA, EMA, RSI, MACD, Bollinger Bands, VWAP
├── charts/
│   ├── __init__.py        # Package initializer
│   └── plots.py           # Plotly figure builders
├── config.py              # Default tickers, intervals, refresh rate
├── requirements.txt
└── README.md
```

## Setup & Running

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

