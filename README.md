# P2-ETF-HAR-RV-FORECASTER

**Heterogeneous Autoregressive Realized Volatility (HAR-RV) Engine for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-HAR-RV-FORECASTER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HAR-RV-FORECASTER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--har--rv--forecaster--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-har-rv-forecaster-results)

## Overview

`P2-ETF-HAR-RV-FORECASTER` uses the **Heterogeneous Autoregressive (HAR)** model to forecast realized volatility for ETFs. It then ranks ETFs by **volatility-adjusted expected return** (expected return divided by forecasted volatility), identifying those with the best risk-reward profile.

The HAR model captures volatility persistence across daily, weekly, and monthly horizons:
RV_t = β0 + β_d * RV_{t-1} + β_w * RV_{t-5} + β_m * RV_{t-22} + ε_t

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Methodology

1. **Realized Volatility Estimation**: Parkinson estimator (or Garman-Klass) using daily OHLC data.
2. **HAR Model Fitting**: OLS regression on daily, weekly, and monthly volatility components.
3. **Multi-Horizon Forecasting**: Recursive forecasts for 1-day, 5-day, and 22-day horizons.
4. **Volatility-Adjusted Ranking**: Expected return divided by forecasted 1-day volatility.

## File Structure
P2-ETF-HAR-RV-FORECASTER/
├── config.py # Paths, universes, HAR parameters
├── data_manager.py # Data loading and RV computation
├── har_model.py # HAR-RV model implementation
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-HAR-RV-FORECASTER.git
cd P2-ETF-HAR-RV-FORECASTER
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
License
MIT License
