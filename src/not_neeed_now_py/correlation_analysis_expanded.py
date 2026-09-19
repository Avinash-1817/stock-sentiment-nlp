import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 1. Load news sentiment data (regenerate via clean_dated_news.py first,
#    pointed at the new india_ticker_subset_dated.csv)
# ---------------------------------------------------------------------------
news_df = pd.read_csv(
    "data/news/india_news_final.csv",
    usecols=["ticker", "date_only", "Sentiment"],  # skip parsing unused columns
)
news_df['date_only'] = pd.to_datetime(news_df['date_only'])

sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
news_df['sentiment_score'] = news_df['Sentiment'].map(sentiment_map)

daily_sentiment = news_df.groupby(['ticker', 'date_only']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    num_articles=('sentiment_score', 'count')
).reset_index()

print(f"Total ticker-day sentiment rows: {len(daily_sentiment)}")

# ---------------------------------------------------------------------------
# 2. Auto-discover which price files actually exist.
#    NOTE: Established tickers (2017-2020) and late-IPO tickers
#    (Zomato/Eternal, Paytm, Nykaa, PolicyBazaar, LIC — 2021 onward) are
#    fetched over DIFFERENT date windows by the price-fetch script. That's
#    expected — this script doesn't care, it just merges whatever exists
#    for each ticker's actual trading history.
# ---------------------------------------------------------------------------
PRICE_DIR = "data/prices"

def ticker_to_filename(ticker):
    return ticker.replace('.', '_').replace('&', 'and').replace('-', '_') + ".csv"

all_tickers = daily_sentiment['ticker'].unique()

def load_price_file(ticker):
    """Read + clean a single ticker's price CSV. Returns (ticker, df_or_None)."""
    fname = ticker_to_filename(ticker)
    path = os.path.join(PRICE_DIR, fname)
    if not os.path.exists(path):
        return ticker, None

    p = pd.read_csv(
        path,
        usecols=["Date", "Close"],  # skip Open/High/Low/Volume parsing entirely
        parse_dates=["Date"],
    )
    p['Close'] = pd.to_numeric(p['Close'], errors='coerce')
    p = p.dropna(subset=['Date', 'Close'])
    if len(p) == 0:
        return ticker, None

    p = p.sort_values('Date')
    p['daily_return'] = p['Close'].pct_change()
    p['next_day_return'] = p['daily_return'].shift(-1)
    p['ticker'] = ticker
    return ticker, p[['ticker', 'Date', 'Close', 'daily_return', 'next_day_return']]

# Reading 50+ small files is I/O-bound -> parallelize with threads
price_frames = []
missing_price_data = []

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(load_price_file, t): t for t in all_tickers}
    for future in as_completed(futures):
        ticker, df = future.result()
        if df is None:
            missing_price_data.append(ticker)
        else:
            price_frames.append(df)

if missing_price_data:
    print(f"\n{len(missing_price_data)} tickers skipped (no usable price file):")
    for t in sorted(missing_price_data):
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
# 3b. DIAGNOSTIC — trace any ticker of interest through each pipeline stage.
#     Useful for late-IPO tickers (Eternal, Paytm, Nykaa, PolicyBazaar, LICI)
#     where a date-range mismatch between news and price data can silently
#     produce zero merged rows.
# ---------------------------------------------------------------------------
WATCH_TICKERS = ["ETERNAL.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "LICI.NS"]

print("\n=== DIAGNOSTIC: late-IPO ticker trace ===")
for t in WATCH_TICKERS:
    n_sentiment = (daily_sentiment['ticker'] == t).sum()
    n_price = (price_df['ticker'] == t).sum()
    n_merged = (merged['ticker'] == t).sum()
    n_valid = (merged_valid['ticker'] == t).sum()

    sent_range = "n/a"
    if n_sentiment > 0:
        s = daily_sentiment.loc[daily_sentiment['ticker'] == t, 'date_only']
        sent_range = f"{s.min().date()} to {s.max().date()}"

    price_range = "n/a"
    if n_price > 0:
        p = price_df.loc[price_df['ticker'] == t, 'date_only']
        price_range = f"{p.min().date()} to {p.max().date()}"

    print(f"{t}: sentiment_rows={n_sentiment} ({sent_range}), "
          f"price_rows={n_price} ({price_range}), "
          f"merged_rows={n_merged}, valid_rows={n_valid}")

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
for ticker, sub in merged_valid.groupby('ticker'):
    if len(sub) < 10:
        print(f"{ticker}: not enough data points ({len(sub)}) - skipped")
        continue
    c, p = pearsonr(sub['avg_sentiment'], sub['daily_return'])
    significant = "significant" if p < 0.05 else "not significant"
    print(f"{ticker}: n={len(sub)}, Pearson r={c:.4f}, p={p:.4f}  [{significant}]")
    results.append({"ticker": ticker, "n": len(sub), "r": c, "p": p, "significant": p < 0.05})

results_df = pd.DataFrame(results).sort_values("ticker").reset_index(drop=True)
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