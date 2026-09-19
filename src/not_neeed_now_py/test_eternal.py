import pandas as pd

news_df = pd.read_csv(
    "data/news/india_news_final.csv",
    usecols=["ticker", "date_only", "Sentiment"],
)
news_df['date_only'] = pd.to_datetime(news_df['date_only'])

eternal_news = news_df[news_df['ticker'] == 'ETERNAL.NS']
print("Eternal news rows:", len(eternal_news))
if len(eternal_news) > 0:
    print("Sentiment date range:", eternal_news['date_only'].min(), "to", eternal_news['date_only'].max())
    print(eternal_news['Sentiment'].value_counts())
else:
    print("No rows at all for ETERNAL.NS in india_news_final.csv")
    # Check if it's under a different ticker spelling
    print("\nAll tickers containing 'ETERNAL' or 'ZOMATO':")
    print(news_df[news_df['ticker'].str.contains('ETERNAL|ZOMATO', case=False, na=False)]['ticker'].unique())