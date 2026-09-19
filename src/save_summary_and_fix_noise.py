"""
Part A: Save the pipeline summary to files (for the report appendix).
Part B: Diagnose and fix the keyword over-matching problem (REC/HAL/BEL).
"""

import pandas as pd
import os
import re

os.makedirs("data/reports", exist_ok=True)

# ---------------------------------------------------------------------------
# PART A: Save summary numbers to a file
# ---------------------------------------------------------------------------
summary_lines = []

raw_subset = pd.read_csv("data/news/india_ticker_subset.csv")
dated = pd.read_csv("data/news/india_ticker_subset_dated.csv")
final_df = pd.read_csv("data/news/india_news_final.csv")
merged = pd.read_csv("data/merged_sentiment_price_full.csv")

summary_lines.append("DATA SOURCING & CLEANING - PIPELINE SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Raw Hugging Face dataset: 26,961 articles")
summary_lines.append(f"After ticker keyword filter (unique articles): {raw_subset['URL'].nunique()}")
summary_lines.append(f"(ticker,article) rows before date scraping: {len(raw_subset)}")
summary_lines.append(f"Successfully dated: {dated['PublishDate'].notna().sum()} "
                      f"({dated['PublishDate'].notna().mean():.1%})")
summary_lines.append(f"Final usable dataset: {len(final_df)} articles, "
                      f"{final_df['ticker'].nunique()} tickers")
summary_lines.append(f"Date range: {final_df['date_only'].min()} to {final_df['date_only'].max()}")
summary_lines.append(f"Ticker-days with matched price data: {len(merged)}")

summary_text = "\n".join(summary_lines)
print(summary_text)

with open("data/reports/pipeline_summary.txt", "w") as f:
    f.write(summary_text)
print("\nSaved: data/reports/pipeline_summary.txt")

# also save per-ticker article counts as a CSV (useful table for report appendix)
ticker_counts = final_df['ticker'].value_counts().reset_index()
ticker_counts.columns = ['ticker', 'article_count']
ticker_counts.to_csv("data/reports/per_ticker_article_counts.csv", index=False)
print("Saved: data/reports/per_ticker_article_counts.csv")

# ---------------------------------------------------------------------------
# PART B: Diagnose the over-matching problem with word-boundary regex
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("KEYWORD OVER-MATCHING DIAGNOSIS")
print("=" * 60)

# Load the raw (pre-filter) dataset to re-test matching with word boundaries
from datasets import load_dataset
raw = load_dataset("kdave/Indian_Financial_News")
raw_df = raw['train'].to_pandas()

# Old (loose) pattern vs new (word-boundary) pattern, for the 3 problem tickers
test_keywords = {
    "RECLTD.NS": ["REC", "Rural Electrification Corporation"],
    "HAL.NS": ["HAL", "Hindustan Aeronautics"],
    "BEL.NS": ["BEL", "Bharat Electronics"],
}

for ticker, keywords in test_keywords.items():
    loose_pattern = "|".join(keywords)
    loose_count = raw_df['Content'].str.contains(loose_pattern, case=False, na=False).sum()

    # word-boundary version: \b ensures the keyword isn't part of a larger word
    tight_pattern = "|".join(rf"\b{re.escape(k)}\b" for k in keywords)
    tight_count = raw_df['Content'].str.contains(tight_pattern, case=False, na=False, regex=True).sum()

    print(f"\n{ticker}:")
    print(f"  Loose match (substring): {loose_count} articles")
    print(f"  Tight match (word boundary): {tight_count} articles")
    print(f"  Reduction: {loose_count - tight_count} false-positive matches removed "
          f"({(1 - tight_count/loose_count):.1%} reduction)")