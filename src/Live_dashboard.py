"""
Live Stock Sentiment Dashboard
Run with: streamlit run live_dashboard.py

Setup (same as live_sentiment_pipeline.py):
1. pip install streamlit newsapi-python transformers torch yfinance plotly
2. Model downloaded locally at models/finbert_finetuned/final_model
3. NewsAPI key in the project .env file (copy .env.example, fill in your own)
"""

import config
import judge
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Live Stock Sentiment", layout="wide")

# ---------------------------------------------------------------------------
# CONFIG - the NewsAPI key lives in the project .env file (copy .env.example
# and fill in your own key) - never hardcode secrets in source files.
# ---------------------------------------------------------------------------
NEWSAPI_KEY = config.NEWSAPI_KEY
LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH

# Common companies for a dropdown - extend this with any NSE company
COMPANY_TICKERS = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
}

# Historical same-day correlation study (46 tickers, 2017-2020), loaded from
# data/per_ticker_correlation_results_v2.csv with an embedded 10-ticker
# fallback so the app works even without the data directory. Entries are
# {"r", "p", "n", "significant", "source"}.
HISTORICAL_RESULTS = judge.load_study_stats()

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
SCORE_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def render_verdict(verdict, company_name):
    """Render one judgment call (verdict) with its evidence."""
    label = judge.call_label(verdict.call)
    conf_text = f"confidence {verdict.confidence:.0%} ({verdict.confidence_level})"
    if verdict.call == judge.BULLISH:
        st.success(f"**Judge call: {label} (\u25b2 bullish-side tone)** \u2014 {conf_text}")
    elif verdict.call == judge.BEARISH:
        st.warning(f"**Judge call: {label} (\u25bc bearish-side tone)** \u2014 {conf_text}")
    else:
        st.info(f"**Judge call: {label}** \u2014 {conf_text}")
    with st.expander("Why this call \u2014 and what it does NOT mean"):
        for line in judge.describe(verdict, company=company_name):
            st.write(line)


def score_sentiment(texts):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).numpy()
        confidences = torch.max(probs, dim=-1).values.numpy()
    return [LABEL_MAP[p] for p in preds], confidences


@st.cache_data(ttl=3600)  # cache for 1 hour so repeated lookups don't burn API quota
def fetch_recent_news(company_name, days_back=7):
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY is not set - copy .env.example to .env at the project root and add your key")
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    response = newsapi.get_everything(
        q=company_name, from_param=from_date, language="en",
        sort_by="publishedAt", page_size=50,
    )
    articles = response.get("articles", [])
    rows = []
    for a in articles:
        text = (a.get("title") or "") + ". " + (a.get("description") or "")
        rows.append({
            "title": a.get("title"), "text": text,
            "publishedAt": a.get("publishedAt"),
            "source": a.get("source", {}).get("name"), "url": a.get("url"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_recent_prices(ticker, days_back=30):
    data = yf.download(ticker, period=f"{days_back}d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Live Indian Stock News Sentiment Dashboard")
st.caption("Capstone project: NLP-based sentiment analysis using a fine-tuned FinBERT model")

st.info(
    "**This tool provides sentiment and alignment information, not investment advice or forecasts.** "
    "It reads today's news tone and makes a same-day **judgment call** using this company's historical "
    "same-day sentiment-return pattern (2017–2020 study, 46 NSE companies). The study found that "
    "pattern holds on the same day but shows **no reliable edge for the next day's move** — so "
    "nothing here is a prediction, and decisions should not be based solely on this tool."
)

col_select, col_days = st.columns([3, 1])
with col_select:
    company_name = st.selectbox("Select a company", list(COMPANY_TICKERS.keys()))
with col_days:
    days_back = st.number_input("Days of news to analyze", min_value=1, max_value=30, value=7)

ticker = COMPANY_TICKERS[company_name]

if st.button("Analyze", type="primary"):
    with st.spinner("Fetching live news and scoring sentiment..."):
        news_df = fetch_recent_news(company_name, days_back)

    if len(news_df) == 0:
        st.warning("No recent articles found for this company in the selected window.")
    else:
        labels, confidences = score_sentiment(news_df["text"].fillna("").tolist())
        news_df["sentiment"] = labels
        news_df["confidence"] = confidences
        news_df["sentiment_score"] = news_df["sentiment"].map(SCORE_MAP)
        avg_sentiment = news_df["sentiment_score"].mean()

        # Evidence-weighted judgment call on this reading (same-day alignment).
        verdict = judge.judge_call(
            avg_sentiment, len(news_df),
            HISTORICAL_RESULTS.get(ticker), ticker=ticker,
        )

        with st.spinner("Fetching recent price data..."):
            price_df = fetch_recent_prices(ticker, days_back + 20)

        # --- Top metrics row ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles analyzed", len(news_df))
        m2.metric("Average sentiment", f"{avg_sentiment:+.3f}", help="Range: -1 (very negative) to +1 (very positive)")

        if len(price_df) >= 2:
            close_col = price_df["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.squeeze()
            latest = float(close_col.iloc[-1])
            prev = float(close_col.iloc[-2])
            pct_change = (latest - prev) / prev
            m3.metric("Latest close", f"₹{latest:,.2f}", f"{pct_change:+.2%}")
        else:
            m3.metric("Latest close", "N/A")

        hist = HISTORICAL_RESULTS.get(ticker)
        if hist:
            m4.metric("Historical correlation (r)", f"{hist['r']:.3f}",
                      "Significant" if hist["significant"] else "Not significant")

        # --- The judgment call (the tool's conclusion, not a forecast) ---
        render_verdict(verdict, company_name)

        # --- Sentiment breakdown ---
        st.subheader("Sentiment Breakdown")
        c1, c2 = st.columns([1, 2])
        with c1:
            counts = news_df["sentiment"].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=counts.index, values=counts.values,
                marker=dict(colors=["#2ca02c" if l == "Positive" else "#d62728" if l == "Negative" else "#7f7f7f"
                                     for l in counts.index]),
            )])
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], mode='lines', name='Close price'))
            fig2.update_layout(title=f"{ticker} — Recent Price Trend", height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

        # --- Historical context ---
        if hist:
            st.subheader("Historical Study Context")
            if hist["significant"]:
                st.success(
                    f"In our 2017–2020 study, **{company_name}** showed a statistically significant "
                    f"same-day sentiment-return correlation (r = {hist['r']:.3f}, p = {hist['p']:.4f}, "
                    f"n = {hist['n']} ticker-days). Sentiment and price have historically moved together "
                    f"on the same trading day for this company — a correlational finding, not a guarantee."
                )
            else:
                st.warning(
                    f"In our 2017–2020 study, **{company_name}** did NOT show a statistically significant "
                    f"same-day sentiment-return correlation (r = {hist['r']:.3f}, p = {hist['p']:.4f}). "
                    f"For this company, news sentiment has historically been a weaker signal — the judge "
                    f"call above therefore abstains rather than guessing."
                )

        # --- Article table ---
        st.subheader("Recent Articles")
        display_df = news_df[["publishedAt", "source", "title", "sentiment", "confidence"]].copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True, height=400)