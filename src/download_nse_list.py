"""
Downloads NSE's official equity list (all listed companies, ~2000 rows) and
saves it locally for use in ticker resolution. Run this once, and re-run
occasionally (e.g. monthly) to pick up new listings.

NSE requires a browser-like User-Agent and sometimes a warm-up request to
their homepage first (to set session cookies) before the CSV endpoint works.
"""

import requests
import pandas as pd
import os
import time

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
OUTPUT_PATH = "data/nse_equity_list.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,*/*",
}


def download_nse_equity_list():
    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm-up request - NSE often requires hitting the homepage first to get
    # valid session cookies before the data endpoints will respond.
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
    except requests.RequestException:
        pass  # continue anyway, sometimes not required

    response = session.get(NSE_EQUITY_LIST_URL, timeout=15)
    response.raise_for_status()

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(OUTPUT_PATH)
    print(f"Downloaded {len(df)} listed companies.")
    print(df.columns.tolist())
    print(df.head())
    return df


if __name__ == "__main__":
    try:
        download_nse_equity_list()
    except Exception as e:
        print(f"Automated download failed: {e}")
        print("\nNSE sometimes blocks automated requests. Manual fallback:")
        print("1. Go to https://www.nseindia.com/market-data/securities-available-for-trading")
        print("2. Download the 'Equity List' CSV")
        print(f"3. Save it as {OUTPUT_PATH}")