import pandas as pd

df = pd.read_csv("data/news/india_ticker_subset_v2.csv")
print(f"Before dropping undated rows: {len(df)}")

df_clean = df[df['PublishDate'].notna()].copy()
print(f"After dropping undated rows: {len(df_clean)}")

df_clean['PublishDate'] = pd.to_datetime(df_clean['PublishDate'], errors='coerce', utc=True)
df_clean = df_clean.dropna(subset=['PublishDate'])
df_clean['date_only'] = df_clean['PublishDate'].dt.date

print(f"After parsing dates: {len(df_clean)}")
print(f"\nTickers: {df_clean['ticker'].nunique()}")
print(f"Date range: {df_clean['date_only'].min()} to {df_clean['date_only'].max()}")
print(f"\nArticles per ticker (top 10):")
print(df_clean['ticker'].value_counts().head(10))

df_clean.to_csv("data/news/india_news_final_v2.csv", index=False)
print("\nSaved: data/news/india_news_final_v2.csv")