"""
Data Sourcing & Cleaning Challenges Summary
Aggregates quality metrics from across the pipeline into one report-ready table.
Run this locally - it just reads your existing saved CSVs, no new scraping/training.
"""

import pandas as pd
import os

print("="*70)
print("DATA SOURCING & CLEANING - SUMMARY REPORT")
print("="*70)

# ---------------------------------------------------------------------------
# 1. Raw dataset size (before any filtering)
# ---------------------------------------------------------------------------
try:
    raw_subset = pd.read_csv("data/news/india_ticker_subset.csv")
    print(f"\n[1] RAW TICKER-FILTERED SUBSET")
    print(f"    Total (ticker, article) rows before date scraping: {len(raw_subset)}")
    print(f"    Unique articles (deduplicated by URL): {raw_subset['URL'].nunique()}")
except FileNotFoundError:
    print("\n[1] Skipped - india_ticker_subset.csv not found")

# ---------------------------------------------------------------------------
# 2. Date-scraping success rate
# ---------------------------------------------------------------------------
try:
    dated = pd.read_csv("data/news/india_ticker_subset_dated.csv")
    total = len(dated)
    success = dated['PublishDate'].notna().sum()
    failed = total - success
    print(f"\n[2] PUBLISH-DATE SCRAPING")
    print(f"    Total rows attempted: {total}")
    print(f"    Successfully dated: {success} ({success/total:.1%})")
    print(f"    Failed to date: {failed} ({failed/total:.1%})")

    if failed > 0:
        failed_domains = dated[dated['PublishDate'].isna()]['URL'].apply(
            lambda u: u.split('/')[2] if isinstance(u, str) else "unknown"
        ).value_counts()
        print(f"    Top domains with missing dates:")
        for domain, count in failed_domains.head(5).items():
            print(f"      - {domain}: {count}")
except FileNotFoundError:
    print("\n[2] Skipped - india_ticker_subset_dated.csv not found")

# ---------------------------------------------------------------------------
# 3. Final cleaned dataset (after dropping undated/unusable rows)
# ---------------------------------------------------------------------------
try:
    final_df = pd.read_csv("data/news/india_news_final.csv")
    print(f"\n[3] FINAL CLEANED DATASET")
    print(f"    Usable articles after cleaning: {len(final_df)}")
    print(f"    Date range: {final_df['date_only'].min()} to {final_df['date_only'].max()}")
    print(f"    Tickers covered: {final_df['ticker'].nunique()}")
    print(f"\n    Articles per ticker (top 5 / bottom 5):")
    counts = final_df['ticker'].value_counts()
    print(f"    Top 5: {dict(counts.head(5))}")
    print(f"    Bottom 5: {dict(counts.tail(5))}")
except FileNotFoundError:
    print("\n[3] Skipped - india_news_final.csv not found")

# ---------------------------------------------------------------------------
# 4. Price data coverage
# ---------------------------------------------------------------------------
try:
    price_dir = "data/prices"
    price_files = [f for f in os.listdir(price_dir) if f.endswith(".csv")]
    print(f"\n[4] PRICE DATA COVERAGE")
    print(f"    Price files successfully downloaded: {len(price_files)}")

    missing_price = []
    if 'final_df' in dir():
        news_tickers = set(final_df['ticker'].unique())
        price_tickers = set(f.replace(".csv", "").replace("_", ".", 1) for f in price_files)
        # not a perfect reverse-mapping given naming quirks (&, -), so this is approximate
except FileNotFoundError:
    print("\n[4] Skipped - data/prices/ not found")

# ---------------------------------------------------------------------------
# 5. Sentiment/price merge yield
# ---------------------------------------------------------------------------
try:
    merged = pd.read_csv("data/merged_sentiment_price_full.csv")
    print(f"\n[5] SENTIMENT-PRICE MERGE")
    print(f"    Ticker-day rows with BOTH news and price data: {len(merged)}")
    if 'final_df' in dir():
        daily_sentiment_rows = final_df.groupby(['ticker', 'date_only']).ngroups
        print(f"    Ticker-days with news data: {daily_sentiment_rows}")
        print(f"    Merge yield: {len(merged)/daily_sentiment_rows:.1%} "
              f"(remainder had news but no matching trading-day price data, "
              f"e.g. weekends/holidays or tickers without full price history)")
except FileNotFoundError:
    print("\n[5] Skipped - merged_sentiment_price_full.csv not found")

# ---------------------------------------------------------------------------
# 6. Keyword-matching noise check (short-acronym tickers)
# ---------------------------------------------------------------------------
try:
    if 'final_df' in dir():
        noisy_tickers = ["HAL.NS", "BEL.NS", "RECLTD.NS"]
        counts = final_df['ticker'].value_counts()
        print(f"\n[6] KEYWORD-MATCHING NOISE CHECK")
        print(f"    Tickers flagged for short/ambiguous keyword matches:")
        for t in noisy_tickers:
            if t in counts.index:
                pct_of_total = counts[t] / len(final_df) * 100
                print(f"      - {t}: {counts[t]} articles ({pct_of_total:.1f}% of total dataset)")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 7. Overall pipeline funnel (the headline numbers for your report)
# ---------------------------------------------------------------------------
print(f"\n" + "="*70)
print("PIPELINE FUNNEL SUMMARY")
print("="*70)
try:
    print(f"  Raw Hugging Face dataset:      26,961 articles (all of India)")
    if 'raw_subset' in dir():
        print(f"  After ticker keyword filter:   {raw_subset['URL'].nunique()} unique articles")
    if 'dated' in dir():
        print(f"  After date scraping:           {dated['PublishDate'].notna().sum()} dated successfully")
    if 'final_df' in dir():
        print(f"  Final usable dataset:          {len(final_df)} articles across {final_df['ticker'].nunique()} tickers")
    if 'merged' in dir():
        print(f"  Ticker-days with price match:  {len(merged)}")
except Exception as e:
    print(f"  (Some pipeline stages unavailable: {e})")