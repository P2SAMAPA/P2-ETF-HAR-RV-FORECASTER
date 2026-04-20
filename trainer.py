"""
Main training script for HAR-RV Forecaster engine.
Computes realized volatility, fits HAR models, forecasts vol, and ranks ETFs.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from har_model import HARModel
import push_results

def compute_expected_return(returns: pd.Series, window: int = 21) -> float:
    """
    Simple expected return: recent average daily return annualized.
    """
    if len(returns) < window:
        return 0.0
    return returns.iloc[-window:].mean() * 252

def run_har_forecast():
    print(f"=== P2-ETF-HAR-RV-FORECASTER Run: {config.TODAY} ===")
    
    df_master = data_manager.load_master_data()
    
    all_results = {}
    top_picks = {}
    
    # 1. Daily Trading (rolling 504d window)
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        
        # Prepare OHLC and compute RV
        df_ohlc = data_manager.prepare_ohlc_data(df_master, tickers)
        df_rv = data_manager.compute_realized_volatility(df_ohlc, estimator=config.RV_ESTIMATOR)
        
        # Prepare returns for expected return calculation
        returns_matrix = data_manager.prepare_returns_matrix(df_master, tickers)
        recent_returns = returns_matrix.iloc[-config.LOOKBACK_WINDOW:]
        
        for ticker in tickers:
            print(f"  Forecasting {ticker}...")
            
            # Get RV series for this ticker
            rv_ticker = df_rv[df_rv['ticker'] == ticker].set_index('Date')['realized_vol']
            if len(rv_ticker) < config.MIN_OBSERVATIONS:
                continue
            
            # Use recent window
            rv_recent = rv_ticker.iloc[-config.LOOKBACK_WINDOW:]
            
            # Fit HAR model
            model = HARModel()
            fit_result = model.fit(rv_recent)
            
            if not fit_result['fitted']:
                continue
            
            # Forecast volatility
            forecasts = model.forecast_all_horizons(rv_recent)
            
            # Compute expected return
            exp_ret = compute_expected_return(recent_returns[ticker])
            
            # Vol-adjusted score: higher return per unit of forecasted vol is better
            vol_forecast = forecasts.get(1, rv_recent.iloc[-1])
            vol_adj_score = exp_ret / (vol_forecast + 1e-6) if config.USE_VOL_ADJUSTED_RETURN else exp_ret
            
            universe_results[ticker] = {
                'ticker': ticker,
                'realized_vol_today': float(rv_recent.iloc[-1]),
                'vol_forecast_1d': forecasts.get(1),
                'vol_forecast_5d': forecasts.get(5),
                'vol_forecast_22d': forecasts.get(22),
                'expected_return': exp_ret,
                'vol_adjusted_score': vol_adj_score,
                'har_r2': fit_result.get('r2', 0),
                'coefficients': fit_result.get('coefficients', {})
            }
        
        if universe_results:
            # Pick top ETF by vol-adjusted score
            top_ticker = max(universe_results, key=lambda t: universe_results[t]['vol_adjusted_score'])
            top_picks[universe_name] = {
                'ticker': top_ticker,
                'vol_adjusted_score': universe_results[top_ticker]['vol_adjusted_score'],
                'vol_forecast_1d': universe_results[top_ticker]['vol_forecast_1d'],
                'expected_return': universe_results[top_ticker]['expected_return']
            }
            print(f"  Top pick: {top_ticker} (score: {top_picks[universe_name]['vol_adjusted_score']:.4f})")
        
        all_results[universe_name] = universe_results
    
    # 2. Shrinking Windows
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        print(f"\n--- Shrinking Window: {window_label} ---")
        
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            df_ohlc_win = data_manager.prepare_ohlc_data(df_window, tickers)
            df_rv_win = data_manager.compute_realized_volatility(df_ohlc_win, estimator=config.RV_ESTIMATOR)
            returns_win = data_manager.prepare_returns_matrix(df_window, tickers)
            
            best_score = -np.inf
            best_ticker = None
            
            for ticker in tickers:
                rv_ticker = df_rv_win[df_rv_win['ticker'] == ticker].set_index('Date')['realized_vol']
                if len(rv_ticker) < config.MIN_OBSERVATIONS:
                    continue
                
                model = HARModel()
                fit_result = model.fit(rv_ticker)
                if not fit_result['fitted']:
                    continue
                
                forecasts = model.forecast_all_horizons(rv_ticker)
                exp_ret = compute_expected_return(returns_win[ticker])
                vol_forecast = forecasts.get(1, rv_ticker.iloc[-1])
                score = exp_ret / (vol_forecast + 1e-6) if config.USE_VOL_ADJUSTED_RETURN else exp_ret
                
                if score > best_score:
                    best_score = score
                    best_ticker = ticker
            
            if best_ticker:
                window_top[universe_name] = {
                    'ticker': best_ticker,
                    'score': best_score
                }
        
        shrinking_results[window_label] = {
            'start_year': start_year,
            'start_date': start_date.isoformat(),
            'top_picks': window_top,
            'n_observations': len(df_window)
        }
    
    # Build output payload
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "rv_estimator": config.RV_ESTIMATOR,
            "use_vol_adjusted_return": config.USE_VOL_ADJUSTED_RETURN
        },
        "daily_trading": {
            "top_picks": top_picks,
            "universes": all_results
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_har_forecast()
