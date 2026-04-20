"""
Streamlit Dashboard for HAR-RV Forecaster.
Displays volatility forecasts and top ETF picks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import numpy as np
import config

st.set_page_config(
    page_title="P2Quant HAR-RV Forecaster",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); 
                 border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-score { font-size: 2rem; font-weight: 600; }
    .footnote { font-size: 0.9rem; color: #666; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_hero_card(ticker: str, score: float, vol_forecast: float, exp_ret: float):
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">📉 TOP PICK (Best Risk-Adjusted Return)</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-score">Score: {score:.4f}</div>
        <div style="margin-top: 1rem;">
            Exp Return (annualized): {exp_ret*100:.2f}%<br>
            Vol Forecast (1‑day, annualized): {vol_forecast*100:.2f}%
        </div>
    </div>
    <div class="footnote">
        * Expected return based on 21‑day average daily return × 252.<br>
        * Volatility forecast is HAR model's 1‑day ahead Parkinson RV (annualized).
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()
st.sidebar.markdown("### 📊 HAR-RV Parameters")
st.sidebar.markdown(f"- Lookback Window: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- RV Estimator: **{config.RV_ESTIMATOR}**")
st.sidebar.markdown(f"- Vol-Adjusted Return: **{config.USE_VOL_ADJUSTED_RETURN}**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

st.sidebar.divider()
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**HAR-RV Forecaster** uses the Heterogeneous Autoregressive model to forecast realized volatility.
- Ranks ETFs by **expected return / forecasted volatility**.
- Higher scores indicate better risk-adjusted return potential.
- All returns and volatilities are **annualized**.
""")

# --- Main Content ---
st.markdown('<div class="main-header">📉 P2Quant HAR-RV Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Heterogeneous Autoregressive Realized Volatility – Risk-Adjusted ETF Ranking</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

daily_data = data['daily_trading']
shrinking_data = data.get('shrinking_windows', {})

tab1, tab2 = st.tabs(["📋 Daily Top Picks", "📆 Shrinking Windows"])

with tab1:
    st.markdown("### Today's Top ETFs by Volatility-Adjusted Score")
    
    top_picks = daily_data['top_picks']
    universes_data = daily_data['universes']
    
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    
    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            if key in universes_data:
                universe_dict = universes_data[key]
                pick = top_picks.get(key, {})
                
                if pick:
                    display_hero_card(
                        pick['ticker'],
                        pick['vol_adjusted_score'],
                        pick.get('vol_forecast_1d', 0),
                        pick.get('expected_return', 0)
                    )
                
                st.markdown("### All ETFs")
                rows = []
                for ticker, vals in universe_dict.items():
                    rows.append({
                        'Ticker': ticker,
                        'Score': f"{vals['vol_adjusted_score']:.4f}",
                        'Exp Return (ann.)': f"{vals['expected_return']*100:.2f}%",
                        'Vol Forecast (1d ann.)': f"{vals['vol_forecast_1d']*100:.2f}%" if vals['vol_forecast_1d'] else 'N/A',
                        'HAR R²': f"{vals.get('har_r2', 0):.3f}"
                    })
                df = pd.DataFrame(rows).sort_values('Score', ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Volatility forecast chart
                st.markdown("### Volatility Forecasts (Top 5 ETFs)")
                top_5 = df.head(5)['Ticker'].tolist()
                fig = go.Figure()
                for t in top_5:
                    if t in universe_dict and universe_dict[t]['vol_forecast_1d']:
                        fig.add_trace(go.Bar(
                            name=t,
                            x=['1-Day', '5-Day', '22-Day'],
                            y=[
                                universe_dict[t]['vol_forecast_1d'] * 100,
                                universe_dict[t].get('vol_forecast_5d', 0) * 100,
                                universe_dict[t].get('vol_forecast_22d', 0) * 100
                            ],
                            text=[f"{universe_dict[t]['vol_forecast_1d']*100:.1f}%",
                                  f"{universe_dict[t].get('vol_forecast_5d', 0)*100:.1f}%",
                                  f"{universe_dict[t].get('vol_forecast_22d', 0)*100:.1f}%"],
                            textposition='auto'
                        ))
                fig.update_layout(
                    title="Volatility Forecasts (Annualized %)",
                    yaxis_title="Annualized Volatility (%)",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key=f"vol_chart_{key}")

with tab2:
    st.markdown("### Top Picks Across Historical Windows")
    
    if not shrinking_data:
        st.warning("No shrinking windows data available.")
        st.stop()
    
    subtabs_sw = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    
    for subtab, key in zip(subtabs_sw, universe_keys):
        with subtab:
            rows = []
            for label, winfo in sorted(shrinking_data.items(), key=lambda x: x[1]['start_year'], reverse=True):
                top = winfo['top_picks'].get(key, {})
                if top:
                    rows.append({
                        'Window': label,
                        'Top Pick': top.get('ticker', 'N/A'),
                        'Score': f"{top.get('score', 0):.4f}",
                        'Observations': winfo.get('n_observations', 0)
                    })
            if rows:
                df_win = pd.DataFrame(rows)
                st.dataframe(df_win, use_container_width=True, hide_index=True)
                
                df_chart = df_win.copy()
                df_chart['Score_val'] = df_chart['Score'].astype(float)
                fig = go.Figure(go.Scatter(
                    x=df_chart['Window'], y=df_chart['Score_val'],
                    mode='lines+markers', text=df_chart['Top Pick'],
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    title=f"{key} – Top Pick Score by Window",
                    xaxis_title="Window Start Year",
                    yaxis_title="Vol-Adjusted Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key=f"sw_chart_{key}")
