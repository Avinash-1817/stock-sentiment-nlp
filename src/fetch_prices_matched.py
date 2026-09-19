import yfinance as yf
import os

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
START_DATE = "2017-06-01"
END_DATE = "2021-01-05"  # a few days buffer past your news data's end

os.makedirs("data/prices", exist_ok=True)

for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data.reset_index(inplace=True)
    out_path = f"data/prices/{ticker.replace('.', '_')}.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")

print("Done.")