import yfinance as yf
import os
import pandas as pd

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "BAJFINANCE.NS", "MARUTI.NS", "ITC.NS",
    "AXISBANK.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "LT.NS", "HINDUNILVR.NS",
    "TMCV.NS", "TMPV.NS", "M&M.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "COALINDIA.NS", "TATASTEEL.NS", "JINDALSTEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "HAL.NS", "BEL.NS", "BHEL.NS", "IRCTC.NS",
    "PFC.NS", "RECLTD.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS", "LICI.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS",
    "TITAN.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "DMART.NS", "TRENT.NS",
    "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "ETERNAL.NS", "PAYTM.NS",
    "NYKAA.NS", "POLICYBZR.NS",
]

START_DATE = "2017-01-01"
END_DATE = "2026-08-22"

os.makedirs("data/prices", exist_ok=True)

CHUNK_SIZE = 20  # batch size per download() call — keeps requests well-formed
failed_tickers = []
saved_count = 0

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for batch in chunks(TICKERS, CHUNK_SIZE):
    print(f"Fetching batch: {batch}")
    try:
        # threads=True (default) parallelizes the underlying requests for
        # every ticker in this batch instead of looping one at a time
        data = yf.download(
            batch,
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        print(f"  ERROR fetching batch {batch}: {e}")
        failed_tickers.extend(batch)
        continue

    for ticker in batch:
        try:
            # When only one ticker succeeds/exists, yfinance sometimes
            # returns a flat (non-multiindex) frame instead of grouped
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    raise KeyError("ticker not in returned columns")
                ticker_data = data[ticker].dropna(how="all")
            else:
                ticker_data = data.dropna(how="all")

            if ticker_data.empty:
                print(f"  WARNING: no data returned for {ticker} "
                      f"(may not have existed/traded in this date range, "
                      f"e.g. Zomato/Paytm/Nykaa IPO'd after 2020)")
                failed_tickers.append(ticker)
                continue

            ticker_data = ticker_data.reset_index()
            safe_name = ticker.replace('.', '_').replace('&', 'and').replace('-', '_')
            out_path = f"data/prices/{safe_name}.csv"
            ticker_data.to_csv(out_path, index=False)
            saved_count += 1
            print(f"  Saved {len(ticker_data)} rows to {out_path}")

        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            failed_tickers.append(ticker)

print("\nDone.")
print(f"Saved {saved_count}/{len(TICKERS)} tickers.")
if failed_tickers:
    print(f"\n{len(failed_tickers)} tickers failed or had no data in this date range:")
    for t in failed_tickers:
        print(f"  - {t}")
    print("\nNote: Zomato, Paytm, Nykaa, PolicyBazaar all IPO'd in India in late 2021/2022 -"
          " they will have NO price data for your 2017-2020 news window. Expect these to fail;"
          " you'll need to drop them from your correlation analysis for this historical period.")