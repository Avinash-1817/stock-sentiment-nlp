import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. Load CLEANED (v2) news sentiment data
# ---------------------------------------------------------------------------
news_df = pd.read_csv("data/news/india_news_final_v2.csv")
news_df['date_only'] = pd.to_datetime(news_df['date_only'])

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
news_df['sentiment_score'] = news_df['Sentiment'].map(sentiment_map)

daily_sentiment = news_df.groupby(['ticker', 'date_only']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    num_articles=('sentiment_score', 'count')
).reset_index()

print(f"Total ticker-day sentiment rows (v2, clean matching): {len(daily_sentiment)}")

# ---------------------------------------------------------------------------
# 2. Load price data (unchanged, reuse existing files)
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
    print(f"\n{len(missing_price_data)} tickers skipped (no usable price file): {missing_price_data}")

price_df = pd.concat(price_frames, ignore_index=True)
price_df = price_df.rename(columns={'Date': 'date_only'})

# ---------------------------------------------------------------------------
# 3. Merge
# ---------------------------------------------------------------------------
daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
merged = pd.merge(daily_sentiment, price_df, on=['ticker', 'date_only'], how='inner')
print(f"\nMerged rows (v2, clean matching): {len(merged)}")

merged.to_csv("data/merged_sentiment_price_v2.csv", index=False)
merged_valid = merged.dropna(subset=['avg_sentiment', 'daily_return', 'next_day_return'])

# ---------------------------------------------------------------------------
# 4. Correlation
# ---------------------------------------------------------------------------
print("\n=== SAME-DAY CORRELATION (v2, all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['daily_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

print("\n=== NEXT-DAY CORRELATION (v2, all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['next_day_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

print("\n=== SAME-DAY CORRELATION (v2, per ticker) ===")
results = []
for ticker in sorted(merged_valid['ticker'].unique()):
    sub = merged_valid[merged_valid['ticker'] == ticker]
    if len(sub) < 10:
        print(f"{ticker}: not enough data points ({len(sub)}) - skipped")
        continue
    c, p = pearsonr(sub['avg_sentiment'], sub['daily_return'])
    significant = "significant" if p < 0.05 else "not significant"
    print(f"{ticker}: n={len(sub)}, Pearson r={c:.4f}, p={p:.4f}  [{significant}]")
    results.append({"ticker": ticker, "n": len(sub), "r": c, "p": p, "significant": p < 0.05})

results_df = pd.DataFrame(results)
results_df.to_csv("data/per_ticker_correlation_results_v2.csv", index=False)

n_significant = results_df['significant'].sum()
print(f"\n{n_significant} out of {len(results_df)} tickers significant (v2, clean matching).")

# ---------------------------------------------------------------------------
# 5. Compare v2 (clean) against v1 (original, noisy matching) results
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("COMPARISON: Clean Matching (v2) vs. Original Loose Matching (v1)")
print("="*70)
try:
    v1_results = pd.read_csv("data/per_ticker_correlation_results_full.csv")
    print(f"\nv1 (loose matching):  {v1_results['significant'].sum()}/{len(v1_results)} tickers significant")
    print(f"v2 (word-boundary):   {n_significant}/{len(results_df)} tickers significant")

    comparison = pd.merge(
        v1_results[['ticker', 'r', 'p', 'significant']],
        results_df[['ticker', 'r', 'p', 'significant']],
        on='ticker', suffixes=('_v1', '_v2')
    )
    comparison.to_csv("data/v1_vs_v2_comparison.csv", index=False)
    print(f"\nFull comparison saved to data/v1_vs_v2_comparison.csv")

    # specifically check the 3 known-noisy tickers
    print(f"\nSpotlight on previously-flagged noisy tickers:")
    for t in ["HAL.NS", "BEL.NS", "RECLTD.NS"]:
        row = comparison[comparison['ticker'] == t]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"  {t}: v1 r={r['r_v1']:.4f} (n from noisy match) -> v2 r={r['r_v2']:.4f} (n from clean match)")
except FileNotFoundError:
    print("v1 results file not found - skipping comparison")