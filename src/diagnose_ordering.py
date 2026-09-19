import pandas as pd

df = pd.read_csv("data/news/india_ticker_subset_v2.csv")

print("First 10 rows - ticker and sentiment:")
print(df[['ticker', 'Sentiment']].head(10))

print("\nUnique tickers in the FULL file (not just head):")
print(df['ticker'].nunique(), "tickers total")
print(df['ticker'].value_counts().head(10))

print("\nSentiment distribution for RELIANCE.NS specifically:")
print(df[df['ticker'] == 'RELIANCE.NS']['Sentiment'].value_counts())

print("\nSentiment distribution across the WHOLE file:")
print(df['Sentiment'].value_counts())