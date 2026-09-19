import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. Load re-scored news data (uses YOUR model's Sentiment_Model column,
#    not the original GPT-labeled Sentiment_GPT column)
# ---------------------------------------------------------------------------
news_df = pd.read_csv("data/news/india_news_rescored.csv")
news_df['date_only'] = pd.to_datetime(news_df['date_only'])

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
news_df['sentiment_score'] = news_df['Sentiment_Model'].map(sentiment_map)

daily_sentiment = news_df.groupby(['ticker', 'date_only']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    num_articles=('sentiment_score', 'count')
).reset_index()

print(f"Total ticker-day sentiment rows (model-scored): {len(daily_sentiment)}")

# ---------------------------------------------------------------------------
# 2. Load price data (same as before, no change needed)
# ---------------------------------------------------------------------------
PRICE_DIR = "data/prices"

def ticker_to_filename(ticker):
    return ticker.replace('.', '_').replace('&', 'and').replace('-', '_') + ".csv"

all_tickers = daily_sentiment['ticker'].unique()

price_frames = []
missing_price_data = []
for ticker in all_tickers:
    fname = ticker_to_filename(ticker)
    path = os.path.join(PRICE_DIR, fname)
    if not os.path.exists(path):
        missing_price_data.append(ticker)
        continue

    p = pd.read_csv(path)
    p['Date'] = pd.to_datetime(p['Date'], errors='coerce')
    p['Close'] = pd.to_numeric(p['Close'], errors='coerce')
    p = p.dropna(subset=['Date', 'Close'])
    if len(p) == 0:
        missing_price_data.append(ticker)
        continue

    p = p.sort_values('Date')
    p['daily_return'] = p['Close'].pct_change()
    p['next_day_return'] = p['daily_return'].shift(-1)
    p['ticker'] = ticker
    price_frames.append(p[['ticker', 'Date', 'Close', 'daily_return', 'next_day_return']])

if missing_price_data:
    print(f"\n{len(missing_price_data)} tickers skipped (no usable price file):")
    for t in missing_price_data:
        print(f"  - {t}")

price_df = pd.concat(price_frames, ignore_index=True)
price_df = price_df.rename(columns={'Date': 'date_only'})

# ---------------------------------------------------------------------------
# 3. Merge sentiment + price
# ---------------------------------------------------------------------------
daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
merged = pd.merge(daily_sentiment, price_df, on=['ticker', 'date_only'], how='inner')
print(f"\nMerged rows (model-scored): {len(merged)}")

merged.to_csv("data/merged_sentiment_price_MODEL.csv", index=False)

merged_valid = merged.dropna(subset=['avg_sentiment', 'daily_return', 'next_day_return'])

# ---------------------------------------------------------------------------
# 4. Correlation — same-day and next-day, overall
# ---------------------------------------------------------------------------
print("\n=== SAME-DAY CORRELATION (model-scored, all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['daily_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

print("\n=== NEXT-DAY CORRELATION (model-scored, all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['next_day_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

# ---------------------------------------------------------------------------
# 5. Per-ticker correlation
# ---------------------------------------------------------------------------
print("\n=== SAME-DAY CORRELATION (model-scored, per ticker) ===")
results = []
for ticker in sorted(merged_valid['ticker'].unique()):
    sub = merged_valid[merged_valid['ticker'] == ticker]
    if len(sub) < 10:
        continue
    c, p = pearsonr(sub['avg_sentiment'], sub['daily_return'])
    significant = "significant" if p < 0.05 else "not significant"
    print(f"{ticker}: n={len(sub)}, Pearson r={c:.4f}, p={p:.4f}  [{significant}]")
    results.append({"ticker": ticker, "n": len(sub), "r": c, "p": p, "significant": p < 0.05})

results_df = pd.DataFrame(results)
results_df.to_csv("data/per_ticker_correlation_results_MODEL.csv", index=False)

n_significant = results_df['significant'].sum()
print(f"\n{n_significant} out of {len(results_df)} tickers significant (model-scored sentiment).")

# ---------------------------------------------------------------------------
# 6. DIRECT COMPARISON: model-scored vs. original GPT-labeled results
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("COMPARISON: Your Fine-Tuned Model vs. Original GPT Labels")
print("="*70)

try:
    original_results = pd.read_csv("data/per_ticker_correlation_results_full.csv")

    comparison = pd.merge(
        original_results[['ticker', 'r', 'p', 'significant']],
        results_df[['ticker', 'r', 'p', 'significant']],
        on='ticker', suffixes=('_GPT', '_Model')
    )
    comparison['r_diff'] = comparison['r_Model'] - comparison['r_GPT']
    comparison.to_csv("data/gpt_vs_model_comparison.csv", index=False)

    print(f"\nGPT labels:   {original_results['significant'].sum()}/{len(original_results)} tickers significant")
    print(f"Model labels: {n_significant}/{len(results_df)} tickers significant")
    print(f"\nMean |r| difference (Model - GPT): {comparison['r_diff'].abs().mean():.4f}")
    print(f"\nFull comparison saved to data/gpt_vs_model_comparison.csv")
except FileNotFoundError:
    print("Original GPT-label results file not found - skipping comparison.")
    print("(Expected at data/per_ticker_correlation_results_full.csv)")