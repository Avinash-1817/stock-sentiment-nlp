"""
Simple demo dashboard for your capstone viva.
Run locally with: streamlit run demo_app.py
Install first: pip install streamlit plotly
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Sentiment Analysis", layout="wide")

st.title("Indian Financial News Sentiment vs. Stock Price Movement")
st.caption("Capstone Project — NLP-based sentiment analysis correlated with NSE stock returns")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv("data/merged_sentiment_price_v2.csv")
    merged['date_only'] = pd.to_datetime(merged['date_only'])
    results = pd.read_csv("data/per_ticker_correlation_results_v2.csv")
    return merged, results

merged, results = load_data()

# ---------------------------------------------------------------------------
# Sidebar: ticker selector
# ---------------------------------------------------------------------------
tickers = sorted(results['ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select a company", tickers)

ticker_row = results[results['ticker'] == selected_ticker].iloc[0]

st.sidebar.metric("Pearson r (same-day)", f"{ticker_row['r']:.3f}")
st.sidebar.metric("p-value", f"{ticker_row['p']:.4f}")
st.sidebar.metric("Statistically significant", "Yes" if ticker_row['significant'] else "No")

# ---------------------------------------------------------------------------
# Main chart: sentiment and price overlay
# ---------------------------------------------------------------------------
sub = merged[merged['ticker'] == selected_ticker].sort_values('date_only')

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{selected_ticker}: Sentiment vs. Same-Day Return")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub['avg_sentiment'], y=sub['daily_return'],
        mode='markers', marker=dict(size=8, opacity=0.6),
        name='Ticker-day observation'
    ))
    fig.update_layout(
        xaxis_title="Average daily sentiment score",
        yaxis_title="Same-day return",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"{selected_ticker}: Sentiment & Return Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sub['date_only'], y=sub['avg_sentiment'],
                               name='Sentiment', yaxis='y1', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=sub['date_only'], y=sub['daily_return'],
                               name='Return', yaxis='y2', line=dict(color='orange')))
    fig2.update_layout(
        yaxis=dict(title="Sentiment score"),
        yaxis2=dict(title="Return", overlaying='y', side='right'),
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Overall results table
# ---------------------------------------------------------------------------
st.subheader("All Companies — Correlation Summary")
display_results = results.copy()
display_results['significant'] = display_results['significant'].map({True: "✅ Yes", False: "❌ No"})
display_results = display_results.sort_values('r', ascending=False)
st.dataframe(display_results, use_container_width=True, height=400)

# ---------------------------------------------------------------------------
# Headline stats
# ---------------------------------------------------------------------------
st.subheader("Overall Findings")
c1, c2, c3 = st.columns(3)
c1.metric("Companies studied", len(results))
c2.metric("Statistically significant", f"{results['significant'].sum()}/{len(results)}")
c3.metric("Overall same-day r", "0.220")