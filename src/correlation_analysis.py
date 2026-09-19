import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. Load news sentiment data (regenerate via clean_dated_news.py first,
#    pointed at the new india_ticker_subset_dated.csv)
# ---------------------------------------------------------------------------
news_df = pd.read_csv("data/news/india_news_final.csv")
news_df['date_only'] = pd.to_datetime(news_df['date_only'])

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
news_df['sentiment_score'] = news_df['Sentiment'].map(sentiment_map)

daily_sentiment = news_df.groupby(['ticker', 'date_only']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    num_articles=('sentiment_score', 'count')
).reset_index()

print(f"Total ticker-day sentiment rows: {len(daily_sentiment)}")

# ---------------------------------------------------------------------------
# 2. Auto-discover which price files actually exist (some tickers like
#    Zomato/Paytm/Nykaa will have no data for this historical period since
#    they IPO'd later, so we don't hardcode a list that might be missing files)
# ---------------------------------------------------------------------------
PRICE_DIR = "data/prices"

# map: ticker name (as used in news data) -> expected price file
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
    print(f"\n{len(missing_price_data)} tickers skipped (no usable price file, likely IPO'd after this period):")
    for t in missing_price_data:
        print(f"  - {t}")

price_df = pd.concat(price_frames, ignore_index=True)
price_df = price_df.rename(columns={'Date': 'date_only'})

# ---------------------------------------------------------------------------
# 3. Merge sentiment + price
# ---------------------------------------------------------------------------
daily_sentiment['date_only'] = pd.to_datetime(daily_sentiment['date_only'])
merged = pd.merge(daily_sentiment, price_df, on=['ticker', 'date_only'], how='inner')
print(f"\nMerged rows (days with both news AND price data): {len(merged)}")

merged.to_csv("data/merged_sentiment_price_full.csv", index=False)

merged_valid = merged.dropna(subset=['avg_sentiment', 'daily_return', 'next_day_return'])

# ---------------------------------------------------------------------------
# 4. Correlation — same-day and next-day, overall and per ticker
# ---------------------------------------------------------------------------
print("\n=== SAME-DAY CORRELATION (all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['daily_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

print("\n=== NEXT-DAY CORRELATION (all tickers combined) ===")
corr, p_value = pearsonr(merged_valid['avg_sentiment'], merged_valid['next_day_return'])
print(f"Pearson correlation: {corr:.4f}  (p-value: {p_value:.4f})")

print("\n=== SAME-DAY CORRELATION (per ticker) ===")
results = []
for ticker in sorted(merged_valid['ticker'].unique()):
    sub = merged_valid[merged_valid['ticker'] == ticker]
    if len(sub) < 10:  # raised threshold slightly for more reliable per-ticker stats
        print(f"{ticker}: not enough data points ({len(sub)}) - skipped")
        continue
    c, p = pearsonr(sub['avg_sentiment'], sub['daily_return'])
    significant = "significant" if p < 0.05 else "not significant"
    print(f"{ticker}: n={len(sub)}, Pearson r={c:.4f}, p={p:.4f}  [{significant}]")
    results.append({"ticker": ticker, "n": len(sub), "r": c, "p": p, "significant": p < 0.05})

results_df = pd.DataFrame(results)
results_df.to_csv("data/per_ticker_correlation_results_full.csv", index=False)
print("\nSaved per-ticker results to data/per_ticker_correlation_results_full.csv")

n_significant = results_df['significant'].sum()
print(f"\n{n_significant} out of {len(results_df)} tickers showed statistically significant same-day correlation (p < 0.05).")

# ---------------------------------------------------------------------------
# 5. Robustness check: exclude tickers with short/ambiguous keyword matches
#    (HAL, BEL, RECLTD had suspiciously high mention counts, likely inflated
#    by false-positive keyword matches in unrelated articles)
# ---------------------------------------------------------------------------
noisy_tickers = ["HAL.NS", "BEL.NS", "RECLTD.NS"]
clean_results = results_df[~results_df['ticker'].isin(noisy_tickers)]
print(f"\nRobustness check (excluding noisy keyword-match tickers: {', '.join(noisy_tickers)}):")
print(f"{clean_results['significant'].sum()} out of {len(clean_results)} tickers significant")