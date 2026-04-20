"""
Data loading and preprocessing for HAR-RV engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    """
    Downloads master_data.parquet from Hugging Face and loads into DataFrame.
    """
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from master data.")
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def prepare_ohlc_data(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Extract OHLC (Open, High, Low, Close) data for given tickers.
    Returns a long-format DataFrame with Date, ticker, open, high, low, close.
    """
    available_tickers = [t for t in tickers if t in df_wide.columns]
    print(f"Found {len(available_tickers)} ticker columns out of {len(tickers)} expected.")
    
    # The master data contains only price columns (close prices)
    # We'll assume Close = price, and estimate High/Low as Close * (1 ± daily_range_factor)
    # or use the actual OHLC if available in future versions.
    
    df_long = pd.melt(
        df_wide,
        id_vars=['Date'],
        value_vars=available_tickers,
        var_name='ticker',
        value_name='close'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    
    # For simplicity, we'll use close-to-close returns for realized volatility
    # Parkinson requires High/Low which we approximate from daily volatility
    df_long['log_return'] = df_long.groupby('ticker')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    # Estimate High/Low from close and daily volatility (approximation)
    # Using the relationship: High/Low ratio ≈ exp(2 * daily_vol)
    daily_vol = df_long.groupby('ticker')['log_return'].transform(
        lambda x: x.rolling(22, min_periods=5).std()
    )
    df_long['high'] = df_long['close'] * np.exp(daily_vol)
    df_long['low'] = df_long['close'] * np.exp(-daily_vol)
    df_long['open'] = df_long.groupby('ticker')['close'].shift(1)
    
    return df_long.dropna(subset=['open', 'high', 'low', 'close'])

def compute_realized_volatility(df_ohlc: pd.DataFrame, estimator: str = "parkinson") -> pd.DataFrame:
    """
    Compute realized volatility for each ETF using the specified estimator.
    Returns a DataFrame with Date, ticker, and realized_vol.
    """
    if estimator == "parkinson":
        # Parkinson: σ = sqrt( (1 / (4 * ln(2))) * (ln(H/L))^2 )
        df_ohlc['rv'] = (1.0 / (4.0 * np.log(2.0))) * (np.log(df_ohlc['high'] / df_ohlc['low'])) ** 2
    elif estimator == "garman_klass":
        # Garman-Klass: more efficient estimator using OHLC
        df_ohlc['rv'] = 0.5 * (np.log(df_ohlc['high'] / df_ohlc['low'])) ** 2 - \
                        (2 * np.log(2) - 1) * (np.log(df_ohlc['close'] / df_ohlc['open'])) ** 2
    else:
        # Close-to-close squared returns
        df_ohlc['rv'] = df_ohlc['log_return'] ** 2
    
    # Annualize and take square root for volatility
    df_ohlc['realized_vol'] = np.sqrt(df_ohlc['rv'] * 252)
    
    return df_ohlc[['Date', 'ticker', 'realized_vol']].dropna()

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Prepare a wide-format DataFrame of log returns with Date index.
    """
    df_long = pd.melt(
        df_wide,
        id_vars=['Date'],
        value_vars=[t for t in tickers if t in df_wide.columns],
        var_name='ticker',
        value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    pivot_returns = df_long.pivot(index='Date', columns='ticker', values='log_return')
    return pivot_returns.dropna()
