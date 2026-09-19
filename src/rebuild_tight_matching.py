"""
Rebuild ticker filtering with word-boundary matching (fixes REC/HAL/BEL-style
false positives), reusing the already-scraped publish dates so no
re-scraping is needed. Run this LOCALLY - no GPU, no network scraping needed,
just re-filtering + a merge against your existing date cache.
"""

import re
import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Load raw dataset
# ---------------------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("kdave/Indian_Financial_News")
df = dataset['train'].to_pandas()
print(f"Total articles: {len(df)}")

# ---------------------------------------------------------------------------
# 2. Same ticker list as before, unchanged
# ---------------------------------------------------------------------------
TICKER_KEYWORDS = {
    "RELIANCE.NS": ["Reliance", "RIL"],
    "TCS.NS": ["TCS", "Tata Consultancy Services"],
    "INFY.NS": ["Infosys", "Infy"],
    "HDFCBANK.NS": ["HDFC Bank", "HDFC"],
    "ICICIBANK.NS": ["ICICI Bank", "ICICI"],
    "SBIN.NS": ["State Bank of India", "SBI"],
    "WIPRO.NS": ["Wipro"],
    "BAJFINANCE.NS": ["Bajaj Finance", "Bajaj Fin"],
    "MARUTI.NS": ["Maruti Suzuki", "Maruti"],
    "ITC.NS": ["ITC"],
    "AXISBANK.NS": ["Axis Bank", "Axis"],
    "SUNPHARMA.NS": ["Sun Pharma", "Sun Pharmaceutical"],
    "BHARTIARTL.NS": ["Bharti Airtel", "Airtel"],
    "LT.NS": ["L&T", "Larsen & Toubro"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "TATAMOTORS.NS": ["Tata Motors"],
    "M&M.NS": ["Mahindra and Mahindra", "M&M", "Mahindra & Mahindra"],
    "ULTRACEMCO.NS": ["Ultratech Cement", "UltraTech"],
    "NTPC.NS": ["NTPC"],
    "POWERGRID.NS": ["Power Grid", "PowerGrid", "PGCIL"],
    "ONGC.NS": ["ONGC", "Oil and Natural Gas Corporation"],
    "COALINDIA.NS": ["Coal India", "CIL"],
    "TATASTEEL.NS": ["Tata Steel"],
    "JINDALSTEL.NS": ["Jindal Steel", "JSPL"],
    "JSWSTEEL.NS": ["JSW Steel"],
    "HINDALCO.NS": ["Hindalco"],
    "HAL.NS": ["HAL", "Hindustan Aeronautics"],
    "BEL.NS": ["BEL", "Bharat Electronics"],
    "BHEL.NS": ["BHEL", "Bharat Heavy Electricals"],
    "IRCTC.NS": ["IRCTC", "Indian Railway Catering"],
    "PFC.NS": ["PFC", "Power Finance Corporation"],
    "RECLTD.NS": ["REC", "Rural Electrification Corporation"],
    "BAJAJFINSV.NS": ["Bajaj Finserv"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank", "Kotak"],
    "LICI.NS": ["LIC", "Life Insurance Corporation"],
    "SBILIFE.NS": ["SBI Life"],
    "HDFCLIFE.NS": ["HDFC Life"],
    "CIPLA.NS": ["Cipla"],
    "DRREDDY.NS": ["Dr Reddy", "Dr. Reddy"],
    "APOLLOHOSP.NS": ["Apollo Hospitals", "Apollo Hospital"],
    "TITAN.NS": ["Titan"],
    "ASIANPAINT.NS": ["Asian Paints", "Asian Paint"],
    "NESTLEIND.NS": ["Nestle", "Nestle India"],
    "DMART.NS": ["DMart", "D-Mart", "Avenue Supermarts"],
    "TRENT.NS": ["Trent", "Westside"],
    "BAJAJ-AUTO.NS": ["Bajaj Auto"],
    "EICHERMOT.NS": ["Eicher Motors", "Eicher"],
    "HEROMOTOCO.NS": ["Hero MotoCorp", "Hero Moto"],
    "ZOMATO.NS": ["Zomato", "Blinkit"],
    "PAYTM.NS": ["Paytm", "One97"],
    "NYKAA.NS": ["Nykaa", "FSN E-Commerce"],
    "POLICYBZR.NS": ["Policybazaar", "PB Fintech"],
}

# ---------------------------------------------------------------------------
# 3. WORD-BOUNDARY matching (the fix) instead of loose substring matching
# ---------------------------------------------------------------------------
def build_word_boundary_pattern(keywords):
    # \b ensures keywords like "REC" only match as standalone words,
    # not inside "record", "recent", "recommend", etc.
    # Multi-word keywords (e.g. "Bajaj Finance") are naturally safer already,
    # but we apply \b consistently for all of them.
    return "|".join(rf"\b{re.escape(k)}\b" for k in keywords)

print("\nRe-filtering with word-boundary matching:")
subsets = []
for ticker, keywords in TICKER_KEYWORDS.items():
    pattern = build_word_boundary_pattern(keywords)
    subset = df[df['Content'].str.contains(pattern, case=False, na=False, regex=True)].copy()
    subset['ticker'] = ticker
    subsets.append(subset)
    print(f"  {ticker}: {len(subset)} articles")

combined = pd.concat(subsets, ignore_index=True)
combined = combined.drop_duplicates(subset=['URL', 'ticker'])
print(f"\nTotal (ticker, URL) rows after tight matching: {len(combined)}")
print(f"Unique URLs: {combined['URL'].nunique()}")

# ---------------------------------------------------------------------------
# 4. Reuse your EXISTING date cache - no re-scraping needed, since dates
#    were already scraped per-URL and most URLs will still be in that cache
# ---------------------------------------------------------------------------
existing_dated = pd.read_csv("data/news/india_ticker_subset_dated.csv")

# build a URL -> PublishDate lookup from everything already scraped
url_date_map = existing_dated.drop_duplicates(subset=['URL']).set_index('URL')['PublishDate'].to_dict()

combined['PublishDate'] = combined['URL'].map(url_date_map)

still_missing = combined['PublishDate'].isna().sum()
print(f"\nRows with date recovered from existing cache: {combined['PublishDate'].notna().sum()}")
print(f"Rows still missing a date (would need fresh scraping): {still_missing}")

combined.to_csv("data/news/india_ticker_subset_v2.csv", index=False)
print("\nSaved: data/news/india_ticker_subset_v2.csv")

if still_missing > 0:
    print(f"\nNote: {still_missing} rows have no cached date. These are likely NEW "
          f"article-ticker matches that weren't in your original (looser) subset. "
          f"If this number is small, you can either scrape just these few "
          f"(reuse your threaded scraper on this subset) or drop them.")