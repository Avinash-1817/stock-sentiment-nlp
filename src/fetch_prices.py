import yfinance as yf
import os

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "BAJFINANCE.NS", "MARUTI.NS", "ITC.NS",
    "AXISBANK.NS", "SUNPHARMA.NS", "BHARTIARTL.NS", "LT.NS", "HINDUNILVR.NS",
    "TMCV.NS", "TMPV.NS","M&M.NS", "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "COALINDIA.NS", "TATASTEEL.NS", "JINDALSTEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "HAL.NS", "BEL.NS", "BHEL.NS", "IRCTC.NS",
    "PFC.NS", "RECLTD.NS", "BAJAJFINSV.NS", "KOTAKBANK.NS", "LICI.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS",
    "TITAN.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "DMART.NS", "TRENT.NS",
    "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "ETERNAL.NS", "PAYTM.NS",
    "NYKAA.NS", "POLICYBZR.NS",
]

START_DATE = "2017-01-01"
END_DATE = "2021-07-31"

os.makedirs("data/prices", exist_ok=True)

failed_tickers = []
for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        if data.empty:
            print(f"  WARNING: no data returned for {ticker} (may not have existed/traded in this date range, e.g. Zomato/Paytm/Nykaa IPO'd after 2020)")
            failed_tickers.append(ticker)
            continue
        data.reset_index(inplace=True)
        # yfinance sanitize: replace characters not safe for filenames
        safe_name = ticker.replace('.', '_').replace('&', 'and').replace('-', '_')
        out_path = f"data/prices/{safe_name}.csv"
        data.to_csv(out_path, index=False)
        print(f"  Saved {len(data)} rows to {out_path}")
    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        failed_tickers.append(ticker)

print("\nDone.")
if failed_tickers:
    print(f"\n{len(failed_tickers)} tickers failed or had no data in this date range:")
    for t in failed_tickers:
        print(f"  - {t}")
    print("\nNote: Zomato, Paytm, Nykaa, PolicyBazaar all IPO'd in India in late 2021/2022 -"
          " they will have NO price data for your 2017-2020 news window. Expect these to fail;"
          " you'll need to drop them from your correlation analysis for this historical period.")