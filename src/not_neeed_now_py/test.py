import yfinance as yf

for ticker in ["TMCV.NS", "TMPV.NS","ETERNAL.NS"]:
    try:
        df = yf.download(ticker, period="1mo", progress=False)
        print(ticker, "Rows:", len(df))
    except Exception as e:
        print(ticker, e)