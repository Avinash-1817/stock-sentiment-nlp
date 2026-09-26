"""
Downloads BSE's equity list to extend company coverage beyond NSE. Some
smaller/mid-cap companies list only on BSE, not NSE.

BSE tickers use a numeric "scrip code" - yfinance needs the format
"<scrip_code>.BO" (not the NSE-style "SYMBOL.NS").

Run locally: python src/download_bse_list.py
"""

import requests
import pandas as pd
import os

# BSE publishes this via their site's "Download" section for equity listings.
# If this URL changes/breaks (BSE updates their site periodically), the
# manual fallback below still works.
BSE_LIST_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripCodes/w"
OUTPUT_PATH = "data/bse_equity_list.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def download_bse_list():
    try:
        response = requests.get(BSE_LIST_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Downloaded {len(df)} BSE-listed companies.")
        print(df.columns.tolist())
        print(df.head())
        return df
    except Exception as e:
        print(f"Automated download failed: {e}")
        print("\nManual fallback:")
        print("1. Go to https://www.bseindia.com/corporates/List_Scrips.aspx")
        print("2. Export the Equity segment list as CSV")
        print(f"3. Save it as {OUTPUT_PATH}, with at minimum columns for")
        print("   company name and scrip code")


if __name__ == "__main__":
    download_bse_list()