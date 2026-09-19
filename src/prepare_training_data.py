import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/news/india_news_final.csv")

# Some rows may repeat (same article matched to multiple tickers) - dedupe by
# URL so the model doesn't see the exact same text multiple times across splits.
df = df.drop_duplicates(subset=['URL'])
df = df[['Content', 'Sentiment']].dropna()

print(f"Total unique articles for training: {len(df)}")
print(df['Sentiment'].value_counts())

label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['label'] = df['Sentiment'].map(label_map)

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

os.makedirs("data/model_data", exist_ok=True)
train_df.to_csv("data/model_data/train.csv", index=False)
val_df.to_csv("data/model_data/val.csv", index=False)
test_df.to_csv("data/model_data/test.csv", index=False)
print("\nSaved train/val/test splits to data/model_data/")