import pandas as pd

df = pd.read_csv("data/news/india_ticker_subset_dated.csv")
df_clean = df[df['PublishDate'].notna()].copy()
print(f"Final usable dataset: {len(df_clean)} articles")

# convert to proper datetime, strip timezone for easier merging later
df_clean['PublishDate'] = pd.to_datetime(df_clean['PublishDate'], errors='coerce', utc=True)
df_clean = df_clean.dropna(subset=['PublishDate'])  # drop any that failed to parse
df_clean['date_only'] = df_clean['PublishDate'].dt.date

print(df_clean[['ticker', 'date_only', 'Sentiment']].head())
print(df_clean['ticker'].value_counts())

df_clean.to_csv("data/news/india_news_final.csv", index=False)


print("Date range:", df_clean['date_only'].min(), "to", df_clean['date_only'].max())
print(df_clean.groupby('ticker').size())