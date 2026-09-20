"""
Fetch Nifty 50 index prices for the same date range as your existing study,
needed to compute excess returns (company return minus market return) -
isolates company-specific movement from "the whole market moved that day."

Run locally: python src/fetch_nifty_index.py
"""

import yfinance as yf
import pandas as pd

NIFTY_TICKER = "^NSEI"  # Nifty 50 index
START_DATE = "2017-01-01"   # match/extend slightly beyond your study's earliest date
END_DATE = "2021-01-05"     # match your existing price-fetch window

OUTPUT_PATH = "data/prices/NIFTY50_index.csv"


def main():
    print(f"Fetching Nifty 50 ({NIFTY_TICKER}) from {START_DATE} to {END_DATE}...")
    data = yf.download(NIFTY_TICKER, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"]).sort_values("Date")

    data["nifty_return"] = data["Close"].pct_change()
    data["nifty_next_day_return"] = data["nifty_return"].shift(-1)

    data[["Date", "Close", "nifty_return", "nifty_next_day_return"]].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()