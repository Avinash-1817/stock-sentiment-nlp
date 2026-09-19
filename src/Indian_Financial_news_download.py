"""
Indian Financial News -> Ticker-filtered, dated dataset pipeline (expanded ticker list, threaded scraper).
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

pd.set_option('display.max_colwidth', 200)

# ---------------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("kdave/Indian_Financial_News")
df = dataset['train'].to_pandas()
print(df.shape)

# ---------------------------------------------------------------------------
# 2. TICKER LIST (your expanded list, unchanged)
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
    "TMCV.NS": ["Tata Motors", "TaMo","Tata Motors Limited"],
    "TMPV.NS": ["Tata Motors Passenger Vehicles", "TaMo"],
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
    "ETERNAL.NS": ["Zomato", "Blinkit"],
    "PAYTM.NS": ["Paytm", "One97"],
    "NYKAA.NS": ["Nykaa", "FSN E-Commerce"],
    "POLICYBZR.NS": ["Policybazaar", "PB Fintech"],
}

for ticker, keywords in TICKER_KEYWORDS.items():
    pattern = "|".join(keywords)
    count = df['Content'].str.contains(pattern, case=False, na=False).sum()
    print(f"{ticker}: {count} mentions")

# ---------------------------------------------------------------------------
# 3. BUILD TICKER-FILTERED SUBSET
# ---------------------------------------------------------------------------
os.makedirs("data/news", exist_ok=True)

subsets = []
for ticker, keywords in TICKER_KEYWORDS.items():
    pattern = "|".join(keywords)
    subset = df[df['Content'].str.contains(pattern, case=False, na=False)].copy()
    subset['ticker'] = ticker
    subsets.append(subset)

combined = pd.concat(subsets, ignore_index=True)
combined = combined.drop_duplicates(subset=['URL', 'ticker'])
print(f"Total (ticker, URL) rows to date: {len(combined)}")
print(f"Unique URLs actually needing a network request: {combined['URL'].nunique()}")

SUBSET_PATH = "data/news/india_ticker_subset.csv"
combined.to_csv(SUBSET_PATH, index=False)

# ---------------------------------------------------------------------------
# 4. PUBLISH-DATE SCRAPER (same fallback chain as before)
# ---------------------------------------------------------------------------
def get_publish_date(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")

        for meta_name in [
            "article:published_time", "og:article:published_time",
            "publish-date", "date", "pubdate", "og:published_time", "sailthru.date",
        ]:
            tag = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
            if tag and tag.get("content"):
                return tag["content"]

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0]
                for key in ["datePublished", "dateCreated", "dateModified"]:
                    if key in data:
                        return data[key]
            except Exception:
                continue

        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            return time_tag["datetime"]

        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. THREADED SCRAPE WITH CHECKPOINTING (resume-safe, ~15-20x fewer requests
#    than before since we only hit each UNIQUE url once, then broadcast the
#    date to every ticker row that shares that url)
# ---------------------------------------------------------------------------
OUTPUT_PATH = "data/news/india_ticker_subset_dated.csv"
MAX_WORKERS = 20          # number of concurrent requests; 15-25 is safe/polite
CHECKPOINT_EVERY = 100    # save progress every N completed unique URLs

news_df = pd.read_csv(SUBSET_PATH)

if os.path.exists(OUTPUT_PATH):
    done_df = pd.read_csv(OUTPUT_PATH)
    news_df = news_df.merge(done_df[['URL', 'ticker', 'PublishDate']], on=['URL', 'ticker'], how='left')
    print(f"Resuming from checkpoint: {news_df['PublishDate'].notna().sum()} rows already dated.")
else:
    news_df['PublishDate'] = None

# Work at the level of UNIQUE URLs (not ticker-duplicated rows) to avoid
# re-scraping the same article multiple times.
url_to_date = {}
already_done_mask = news_df['PublishDate'].notna()
for u, d in zip(news_df.loc[already_done_mask, 'URL'], news_df.loc[already_done_mask, 'PublishDate']):
    url_to_date[u] = d

remaining_urls = [u for u in news_df['URL'].unique() if u not in url_to_date]
print(f"Unique URLs remaining to scrape: {len(remaining_urls)}")

lock = threading.Lock()
completed_count = 0

def scrape_and_store(url):
    global completed_count
    date = get_publish_date(url)
    with lock:
        url_to_date[url] = date
        global completed_count
        completed_count += 1
        if completed_count % CHECKPOINT_EVERY == 0:
            print(f"{completed_count}/{len(remaining_urls)} unique URLs done...")
            news_df['PublishDate'] = news_df['URL'].map(url_to_date)
            news_df.to_csv(OUTPUT_PATH, index=False)
    return url, date

if remaining_urls:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scrape_and_store, u) for u in remaining_urls]
        for f in as_completed(futures):
            pass  # progress/checkpointing handled inside scrape_and_store

# final write: broadcast url_to_date across all ticker-duplicated rows
news_df['PublishDate'] = news_df['URL'].map(url_to_date)
news_df.to_csv(OUTPUT_PATH, index=False)
print("Finished scraping dates. Saved to", OUTPUT_PATH)

# ---------------------------------------------------------------------------
# 6. REPORT SUCCESS RATE
# ---------------------------------------------------------------------------
print(news_df['PublishDate'].notna().sum(), "out of", len(news_df), "rows dated successfully")
failed = news_df[news_df['PublishDate'].isna()]
if len(failed) > 0:
    print("Domains with missing dates:")
    print(failed['URL'].apply(lambda u: u.split('/')[2]).value_counts())