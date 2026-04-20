"""
Configuration for P2-ETF-HAR-RV-FORECASTER engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-har-rv-forecaster-results"

# --- Universe Definitions (mirroring master data exactly) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]

ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- HAR-RV Parameters ---
LOOKBACK_WINDOW = 504                 # 2-year lookback for model training
RV_ESTIMATOR = "parkinson"            # "parkinson", "garman_klass", or "close_to_close"
FORECAST_HORIZONS = [1, 5, 22]        # Forecast horizons (1-day, 1-week, 1-month)
ROLLING_WINDOW = 22                   # Rolling window for realized volatility calculation
MIN_OBSERVATIONS = 252                # Minimum observations required (1 year)

# --- Return Adjustment ---
USE_VOL_ADJUSTED_RETURN = True        # If True, rank by return / forecasted volatility
                                      # If False, rank by raw expected return (from separate model)

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))  # 2008..2024

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
