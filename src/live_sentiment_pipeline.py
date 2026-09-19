"""
Live sentiment pipeline: fetch recent news for a company via NewsAPI,
score it with your fine-tuned FinBERT model, and pair with current price data.

Run this LOCALLY (no GPU strictly required - inference on a handful of
recent articles is fast even on CPU).

Setup:
1. pip install newsapi-python transformers torch yfinance
2. Get a free API key from https://newsapi.org
3. Download your fine-tuned model folder from Google Drive
   (stock-sentiment-nlp/finbert_finetuned/final_model) to a local folder,
   e.g. models/finbert_finetuned (or set LOCAL_MODEL_PATH)
"""

import os
from datetime import datetime, timedelta

import config
import judge
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# CONFIG - the NewsAPI key lives in the project .env file (copy .env.example
# and fill in your own key) - never hardcode secrets in source files.
# ---------------------------------------------------------------------------
LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH  # fine-tuned FinBERT on disk

# ---------------------------------------------------------------------------
# 1. Load your fine-tuned model (once)
# ---------------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
model.eval()

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
SCORE_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def score_sentiment(texts):
    """Score a list of article texts, return list of (label, score) tuples."""
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).numpy()
    return [LABEL_MAP[p] for p in preds]


# ---------------------------------------------------------------------------
# 2. Fetch recent news for a company
# ---------------------------------------------------------------------------
def fetch_recent_news(company_name, days_back=7, api_key=None):
    api_key = api_key or config.require_newsapi_key()
    newsapi = NewsApiClient(api_key=api_key)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    response = newsapi.get_everything(
        q=company_name,
        from_param=from_date,
        language="en",
        sort_by="publishedAt",
        page_size=50,
    )

    articles = response.get("articles", [])
    rows = []
    for a in articles:
        text = (a.get("title") or "") + ". " + (a.get("description") or "")
        rows.append({
            "title": a.get("title"),
            "text": text,
            "publishedAt": a.get("publishedAt"),
            "source": a.get("source", {}).get("name"),
            "url": a.get("url"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Fetch current price data
# ---------------------------------------------------------------------------
def fetch_recent_prices(ticker, days_back=30):
    data = yf.download(ticker, period=f"{days_back}d")
    # Newer yfinance versions can return multi-level columns (ticker, field)
    # even for a single ticker - flatten them so 'Close' is a plain column.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data


# ---------------------------------------------------------------------------
# 4. Combined pipeline
# ---------------------------------------------------------------------------
def analyze_company(company_name, ticker, days_back=7):
    print(f"\nFetching recent news for '{company_name}'...")
    news_df = fetch_recent_news(company_name, days_back=days_back)
    if len(news_df) == 0:
        print("No recent articles found.")
        return None

    print(f"Found {len(news_df)} articles. Scoring sentiment...")
    news_df["sentiment"] = score_sentiment(news_df["text"].fillna("").tolist())
    news_df["sentiment_score"] = news_df["sentiment"].map(SCORE_MAP)

    avg_sentiment = news_df["sentiment_score"].mean()

    # Evidence-weighted judgment call (same-day alignment, NOT a forecast).
    verdict = judge.judge_call(
        avg_sentiment, len(news_df),
        judge.study_stats_for(ticker), ticker=ticker,
    )

    print(f"\nFetching recent price data for {ticker}...")
    price_df = fetch_recent_prices(ticker, days_back=days_back + 5)

    if len(price_df) >= 2:
        close_col = price_df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.squeeze()
        latest_close = float(close_col.iloc[-1])
        prev_close = float(close_col.iloc[-2])
        recent_return = (latest_close - prev_close) / prev_close
    else:
        recent_return = None

    print("\n" + "=" * 50)
    print(f"SUMMARY: {company_name} ({ticker})")
    print("=" * 50)
    print(f"Articles analyzed (last {days_back} days): {len(news_df)}")
    print(f"Sentiment breakdown: {news_df['sentiment'].value_counts().to_dict()}")
    print(f"Average sentiment score: {avg_sentiment:.3f}  (range: -1 to +1)")
    print(f"Judge call: {judge.call_label(verdict.call)}  "
          f"(confidence {verdict.confidence:.0%}, {verdict.basis})")
    if recent_return is not None:
        print(f"Most recent daily return: {recent_return:.2%}")
    print("\nNOTE: This is a sentiment signal for informational purposes only,")
    print("based on a correlational research finding, NOT a buy/sell recommendation.")

    return {
        "news_df": news_df,
        "price_df": price_df,
        "avg_sentiment": avg_sentiment,
        "recent_return": recent_return,
        "verdict": verdict,
    }


if __name__ == "__main__":
    # Example usage
    result = analyze_company("Reliance Industries", "RELIANCE.NS", days_back=7)