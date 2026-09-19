"""
Indian Financial News Pipeline
--------------------------------
1. Load historical KDAVE dataset
2. Fetch newer Indian financial news from Google News RSS (date-chunked,
   parallelized, using after:/before: operators for real historical spread)
3. Filter news by ticker
4. Combine historical + latest news
5. Remove duplicate articles
6. Scrape publication dates
7. Save ticker-filtered dated dataset
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
from datetime import date

import pandas as pd
import requests
import feedparser

from bs4 import BeautifulSoup
from datasets import load_dataset


pd.set_option('display.max_colwidth', 200)


# ===========================================================================
# 1. TICKER LIST
#    NOTE: ETERNAL.NS now includes "Eternal" — the company rebranded from
#    Zomato to Eternal in 2023. Recent coverage increasingly uses the new
#    name, and the old keyword list was silently missing it.
# ===========================================================================

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
    "TMCV.NS": ["Tata Motors", "TaMo", "Tata Motors Limited"],
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

    "ETERNAL.NS": ["Zomato", "Blinkit", "Eternal"],  # FIXED: added rebrand name

    "PAYTM.NS": ["Paytm", "One97"],
    "NYKAA.NS": ["Nykaa", "FSN E-Commerce"],
    "POLICYBZR.NS": ["Policybazaar", "PB Fintech"],
}

# Tickers that IPO'd after 2020 — worth extra visibility during fetch since
# their news coverage volume/recency is more fragile than established firms.
LATE_IPO_TICKERS = {"ETERNAL.NS", "PAYTM.NS", "NYKAA.NS", "POLICYBZR.NS", "LICI.NS"}


# ===========================================================================
# 2. SETTINGS
# ===========================================================================

DATA_DIR = "data/news"
os.makedirs(DATA_DIR, exist_ok=True)

# KDAVE is historical.
# We fetch newer news starting from 2021.
NEW_NEWS_START_DATE = "2021-01-01"
NEW_NEWS_END_DATE = date.today().isoformat()

# Google News RSS only surfaces a "recent results" slice per query — to get
# real historical spread across NEW_NEWS_START_DATE..today, we query in
# fixed-width date chunks using after:/before: search operators instead of
# a single unbounded query.
CHUNK_MONTHS = 3

MAX_RESULTS_PER_QUERY = 100

MAX_WORKERS = 15          # for publish-date scraping
RSS_FETCH_WORKERS = 10    # for the RSS fetch itself (parallelized now)

CHECKPOINT_EVERY = 100


# ===========================================================================
# 3. LOAD KDAVE HISTORICAL DATA
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 1 - LOADING HISTORICAL KDAVE DATA")
print("=" * 70)

print("Loading dataset...")
dataset = load_dataset("kdave/Indian_Financial_News")
df_old = dataset["train"].to_pandas()

print(f"KDAVE dataset shape: {df_old.shape}")
print("Columns:", list(df_old.columns))


# ===========================================================================
# 4. FILTER KDAVE DATA BY TICKER
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 2 - FILTERING HISTORICAL NEWS BY TICKER")
print("=" * 70)

content_lower = df_old["Content"].str.lower()  # compute once, not per ticker

old_subsets = []
for ticker, keywords in TICKER_KEYWORDS.items():
    pattern = "|".join(rf"(?<!\w){kw.lower()}(?!\w)" for kw in keywords)
    mask = content_lower.str.contains(pattern, regex=True, na=False)
    subset = df_old[mask].copy()
    if len(subset) > 0:
        subset["ticker"] = ticker
        old_subsets.append(subset)
    print(f"{ticker}: {len(subset)} historical articles")

old_news = pd.concat(old_subsets, ignore_index=True) if old_subsets else pd.DataFrame()
if not old_news.empty:
    old_news = old_news.drop_duplicates(subset=["URL", "ticker"])

print(f"\nHistorical ticker-news rows: {len(old_news)}")


# ===========================================================================
# 5. GOOGLE NEWS RSS COLLECTOR (date-chunked)
# ===========================================================================

def month_chunks(start_str, end_str, months=CHUNK_MONTHS):
    """Yield (chunk_start, chunk_end) date strings covering start..end."""
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=months), end)
        yield cur.date().isoformat(), nxt.date().isoformat()
        cur = nxt


def fetch_google_news(ticker, keyword, after=None, before=None):
    """
    Fetch Google News RSS results for one company keyword, optionally
    bounded to a date window using Google's after:/before: search operators.
    This is what actually gives historical spread instead of just
    whatever's currently trending.
    """
    query = f'"{keyword}" India stock'
    if after:
        query += f" after:{after}"
    if before:
        query += f" before:{before}"

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        response = requests.get(
            rss_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            },
            timeout=20,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        rows = []
        for entry in feed.entries:
            url = entry.get("link", "")
            if not url:
                continue
            summary = entry.get("summary", "")
            rows.append({
                "ticker": ticker,
                "SearchKeyword": keyword,
                "Title": entry.get("title", ""),
                "Content": BeautifulSoup(summary, "html.parser").get_text(" ", strip=True),
                "URL": url,
                "GooglePublishedDate": entry.get("published", ""),
            })
        return rows

    except Exception as e:
        print(f"ERROR fetching {ticker} / {keyword} [{after} to {before}]: {e}")
        return []


# ===========================================================================
# 6. FETCH NEWER NEWS — parallelized, date-chunked
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 3 - FETCHING NEWER NEWS")
print("=" * 70)

date_chunks = list(month_chunks(NEW_NEWS_START_DATE, NEW_NEWS_END_DATE))
print(f"Querying {len(date_chunks)} date windows "
      f"({CHUNK_MONTHS}-month chunks) per keyword, for real historical spread.")

tasks = [
    (ticker, keyword, after, before)
    for ticker, keywords in TICKER_KEYWORDS.items()
    for keyword in keywords
    for after, before in date_chunks
]
print(f"Total RSS queries to run: {len(tasks)}")

new_rows_by_ticker = {t: [] for t in TICKER_KEYWORDS}

def run_task(task):
    ticker, keyword, after, before = task
    rows = fetch_google_news(ticker, keyword, after=after, before=before)
    time.sleep(0.15)  # stay polite to the source even while parallelized
    return ticker, rows

with ThreadPoolExecutor(max_workers=RSS_FETCH_WORKERS) as executor:
    futures = [executor.submit(run_task, t) for t in tasks]
    for future in as_completed(futures):
        ticker, rows = future.result()
        new_rows_by_ticker[ticker].extend(rows)

new_rows = []
for ticker, rows in new_rows_by_ticker.items():
    unique_urls = set()
    unique_rows = []
    for row in rows:
        if row["URL"] in unique_urls:
            continue
        unique_urls.add(row["URL"])
        unique_rows.append(row)
    new_rows.extend(unique_rows)
    flag = "  [late-IPO ticker]" if ticker in LATE_IPO_TICKERS else ""
    print(f"{ticker}: {len(unique_rows)} new/search results{flag}")

new_news = pd.DataFrame(new_rows)
print(f"\nTotal new news rows: {len(new_news)}")


# ===========================================================================
# 7. PARSE GOOGLE'S DATE (but DO NOT drop undated rows here anymore —
#    that's the bug that was silently discarding articles before the
#    more reliable publish-date scraper (Step 5 below) ever saw them)
# ===========================================================================

if not new_news.empty:
    new_news["GooglePublishedDate"] = pd.to_datetime(
        new_news["GooglePublishedDate"], errors="coerce", utc=True
    )

n_undated = new_news["GooglePublishedDate"].isna().sum() if not new_news.empty else 0
print(f"New news rows with missing/unparseable Google date: {n_undated} "
      f"(kept — will attempt real date via Step 5 scraper instead of dropping)")


# ===========================================================================
# 8. COMBINE OLD + NEW NEWS
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 4 - COMBINING HISTORICAL + NEW NEWS")
print("=" * 70)

if not old_news.empty:
    old_news["SourceType"] = "KDAVE"
if not new_news.empty:
    new_news["SourceType"] = "GoogleNews"

old_columns = ["ticker", "URL", "Content", "SourceType"]
new_columns = ["ticker", "URL", "Content", "SourceType", "Title", "GooglePublishedDate"]

old_final = old_news[old_columns].copy() if not old_news.empty else pd.DataFrame()
new_final = new_news[new_columns].copy() if not new_news.empty else pd.DataFrame()

combined_parts = [df for df in (old_final, new_final) if not df.empty]
if not combined_parts:
    raise RuntimeError("No news data was collected.")

combined = pd.concat(combined_parts, ignore_index=True, sort=False)

print(f"Before deduplication: {len(combined)} rows")
combined = combined.drop_duplicates(subset=["URL", "ticker"])
print(f"After deduplication: {len(combined)} rows")

COMBINED_RAW_PATH = "data/news/india_combined_news.csv"
combined.to_csv(COMBINED_RAW_PATH, index=False)
print(f"\nSaved combined news to:\n{COMBINED_RAW_PATH}")


# ===========================================================================
# 9. PUBLICATION DATE SCRAPER (unchanged logic)
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 5 - FINDING PUBLICATION DATES")
print("=" * 70)

def get_publish_date(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.content, "html.parser")

        for meta_name in [
            "article:published_time", "og:article:published_time",
            "publish-date", "date", "pubdate", "og:published_time", "sailthru.date",
        ]:
            tag = (soup.find("meta", attrs={"property": meta_name})
                   or soup.find("meta", attrs={"name": meta_name}))
            if tag and tag.get("content"):
                return tag["content"]

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                if not script.string:
                    continue
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0]
                if not isinstance(data, dict):
                    continue
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


OUTPUT_PATH = "data/news/india_ticker_subset_dated.csv"

news_df = pd.read_csv(COMBINED_RAW_PATH)

if os.path.exists(OUTPUT_PATH):
    try:
        done_df = pd.read_csv(OUTPUT_PATH, usecols=["URL", "ticker", "PublishDate"])
        news_df = news_df.merge(done_df, on=["URL", "ticker"], how="left")
        print("Resuming previous date-scraping checkpoint.")
        print(f"Already dated: {news_df['PublishDate'].notna().sum()}")
    except Exception:
        print("Existing dated file could not be used. Starting date scraping again.")
        news_df["PublishDate"] = None
else:
    news_df["PublishDate"] = None


# ===========================================================================
# 10. SCRAPE ONLY UNIQUE URLS
# ===========================================================================

url_to_date = {}
already_done_mask = news_df["PublishDate"].notna()
for u, d in zip(news_df.loc[already_done_mask, "URL"], news_df.loc[already_done_mask, "PublishDate"]):
    url_to_date[u] = d

remaining_urls = [u for u in news_df["URL"].dropna().unique() if u not in url_to_date]
print(f"Unique URLs remaining to scrape: {len(remaining_urls)}")

lock = threading.Lock()
completed_count = 0

def scrape_and_store(url):
    global completed_count
    date_val = get_publish_date(url)
    with lock:
        url_to_date[url] = date_val
        completed_count += 1
        if completed_count % CHECKPOINT_EVERY == 0:
            print(f"{completed_count}/{len(remaining_urls)} URLs completed...")
            news_df["PublishDate"] = news_df["URL"].map(url_to_date)
            news_df.to_csv(OUTPUT_PATH, index=False)
    return url, date_val

if remaining_urls:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scrape_and_store, url) for url in remaining_urls]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error: {e}")


# ===========================================================================
# 11. FINAL DATE MAPPING + FALLBACK
# ===========================================================================

news_df["PublishDate"] = news_df["URL"].map(url_to_date)

if "GooglePublishedDate" in news_df.columns:
    missing_date = news_df["PublishDate"].isna()
    news_df.loc[missing_date, "PublishDate"] = news_df.loc[missing_date, "GooglePublishedDate"]

news_df["PublishDate"] = pd.to_datetime(news_df["PublishDate"], errors="coerce", utc=True)
news_df["date_only"] = news_df["PublishDate"].dt.date


# ===========================================================================
# 12. FINAL DATE-RANGE FILTER
#    Moved here (was previously applied early, before the reliable
#    scraper ran, which dropped recoverable rows). Now we filter using the
#    best available date — scraped PublishDate, falling back to Google's —
#    only after both sources have had a chance to populate it.
# ===========================================================================

before_filter = len(news_df)
start_ts = pd.Timestamp(NEW_NEWS_START_DATE, tz="UTC")
# Keep all KDAVE rows regardless (they're pre-vetted historical data);
# only apply the start-date floor to GoogleNews-sourced rows.
is_google = news_df.get("SourceType", pd.Series(dtype=str)) == "GoogleNews"
drop_mask = is_google & news_df["PublishDate"].notna() & (news_df["PublishDate"] < start_ts)
news_df = news_df[~drop_mask].copy()
print(f"\nDropped {before_filter - len(news_df)} GoogleNews rows dated before {NEW_NEWS_START_DATE}")


# ===========================================================================
# 13. SAVE FINAL DATASET
# ===========================================================================

news_df.to_csv(OUTPUT_PATH, index=False)

print("\nFinished.")
print(f"Final rows: {len(news_df)}")
print(f"Dated rows: {news_df['PublishDate'].notna().sum()}")
print(f"Missing dates: {news_df['PublishDate'].isna().sum()}")
print(f"\nFinal file:\n{OUTPUT_PATH}")

valid_dates = news_df["PublishDate"].dropna()
if len(valid_dates) > 0:
    print("\nNews date range:")
    print(valid_dates.min())
    print("→")
    print(valid_dates.max())

if "SourceType" in news_df.columns:
    print("\nNews source breakdown:")
    print(news_df["SourceType"].value_counts())

print("\n=== LATE-IPO TICKER COVERAGE CHECK ===")
for t in sorted(LATE_IPO_TICKERS):
    sub = news_df[news_df["ticker"] == t]
    src_counts = sub["SourceType"].value_counts().to_dict() if len(sub) else {}
    date_range = (f"{sub['date_only'].min()} to {sub['date_only'].max()}"
                  if sub["date_only"].notna().any() else "n/a")
    print(f"{t}: {len(sub)} rows, sources={src_counts}, date_range={date_range}")

failed = news_df[news_df["PublishDate"].isna()]
if len(failed) > 0:
    print("\nDomains with missing dates:")
    domains = (
        failed["URL"].dropna()
        .apply(lambda u: u.split("/")[2] if "://" in u else u)
        .value_counts()
    )
    print(domains)