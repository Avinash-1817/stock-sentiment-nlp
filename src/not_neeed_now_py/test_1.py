import pandas as pd

news_df = pd.read_csv("data/news/india_news_final.csv")
news_df['date_only'] = pd.to_datetime(news_df['date_only'])
print("NEWS date range:", news_df['date_only'].min(), "to", news_df['date_only'].max())
print("NEWS date dtype:", news_df['date_only'].dtype)
print("NEWS sample tickers:", news_df['ticker'].unique()[:5])

p = pd.read_csv("data/prices/RELIANCE_NS.csv")
print("\nPRICE raw head:")
print(p.head(3))
print("PRICE columns:", p.columns.tolist())

p['Date'] = pd.to_datetime(p['Date'], errors='coerce')
p = p.dropna(subset=['Date'])
print("PRICE date range:", p['Date'].min(), "to", p['Date'].max())
print("PRICE date dtype:", p['Date'].dtype)