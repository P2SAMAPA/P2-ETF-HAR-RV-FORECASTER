"""
Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Tuple

class HARModel:
    """
    HAR-RV model for forecasting realized volatility.
    RV_t = β0 + β_d * RV_{t-1} + β_w * RV_{t-5} + β_m * RV_{t-22} + ε_t
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.fitted = False
        self.coefficients = None
        
    def _compute_har_features(self, rv_series: pd.Series) -> pd.DataFrame:
        """
        Compute HAR features: daily (lag 1), weekly (avg of lags 1-5), monthly (avg of lags 1-22).
        """
        df = pd.DataFrame(index=rv_series.index)
        df['rv_daily'] = rv_series.shift(1)
        df['rv_weekly'] = rv_series.shift(1).rolling(5, min_periods=1).mean()
        df['rv_monthly'] = rv_series.shift(1).rolling(22, min_periods=1).mean()
        df['target'] = rv_series
        return df.dropna()
    
    def fit(self, rv_series: pd.Series) -> Dict:
        """
        Fit HAR model to the realized volatility series.
        Returns a dictionary with model coefficients and training metrics.
        """
        df = self._compute_har_features(rv_series)
        if len(df) < 22:
            return {'fitted': False, 'error': 'Insufficient data'}
        
        X = df[['rv_daily', 'rv_weekly', 'rv_monthly']].values
        y = df['target'].values
        
        self.model.fit(X, y)
        self.fitted = True
        self.coefficients = {
            'const': self.model.intercept_,
            'daily': self.model.coef_[0],
            'weekly': self.model.coef_[1],
            'monthly': self.model.coef_[2]
        }
        
        # Compute in-sample R²
        y_pred = self.model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-9))
        
        return {
            'fitted': True,
            'coefficients': self.coefficients,
            'r2': r2,
            'n_obs': len(df)
        }
    
    def predict(self, rv_series: pd.Series, horizon: int = 1) -> Optional[float]:
        """
        Predict realized volatility for the given horizon.
        For multi-step forecasts, we recursively predict.
        """
        if not self.fitted:
            return None
        
        # Get the most recent values for features
        if len(rv_series) < 22:
            return None
        
        recent_rv = rv_series.iloc[-22:].values
        
        # Recursive forecasting for horizons > 1
        for _ in range(horizon):
            daily = recent_rv[-1]
            weekly = np.mean(recent_rv[-5:]) if len(recent_rv) >= 5 else daily
            monthly = np.mean(recent_rv[-22:]) if len(recent_rv) >= 22 else daily
            
            X_new = np.array([[daily, weekly, monthly]])
            pred = self.model.predict(X_new)[0]
            recent_rv = np.append(recent_rv[1:], pred)
        
        return pred
    
    def forecast_all_horizons(self, rv_series: pd.Series) -> Dict[int, float]:
        """
        Forecast realized volatility for all configured horizons.
        """
        forecasts = {}
        for h in [1, 5, 22]:
            pred = self.predict(rv_series, horizon=h)
            if pred is not None:
                forecasts[h] = pred
        return forecasts
