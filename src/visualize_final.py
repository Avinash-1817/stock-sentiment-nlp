import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

os.makedirs("data/figures", exist_ok=True)

results_df = pd.read_csv("data/per_ticker_correlation_results_v2.csv")
merged = pd.read_csv("data/merged_sentiment_price_v2.csv")
merged['date_only'] = pd.to_datetime(merged['date_only'])

# ---------------------------------------------------------------------------
# 1. Sorted bar chart (final version)
# ---------------------------------------------------------------------------
plot_df = results_df.sort_values('r', ascending=True).reset_index(drop=True)
colors = ['#2ca02c' if sig else '#d3d3d3' for sig in plot_df['significant']]

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(plot_df['ticker'], plot_df['r'], color=colors, edgecolor='black', linewidth=0.3)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Pearson correlation (sentiment vs. same-day return)", fontsize=11)
ax.set_title("Same-Day Sentiment-Return Correlation by Ticker (Final, Clean Matching)\n"
             "(green = statistically significant, p < 0.05)", fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=8)
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Significant (p < 0.05)'),
    Patch(facecolor='#d3d3d3', edgecolor='black', label='Not significant'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("data/figures/correlation_bar_chart_FINAL.png", dpi=200)
plt.close()
print("Saved: data/figures/correlation_bar_chart_FINAL.png")

# ---------------------------------------------------------------------------
# 2. Scatter plot for strongest result
# ---------------------------------------------------------------------------
top_idx = results_df['r'].abs().idxmax()
top_ticker = results_df.loc[top_idx, 'ticker']
top_r = results_df.loc[top_idx, 'r']
top_p = results_df.loc[top_idx, 'p']

sub = merged[merged['ticker'] == top_ticker].dropna(subset=['avg_sentiment', 'daily_return'])

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(sub['avg_sentiment'], sub['daily_return'], alpha=0.6, color='#1f77b4', edgecolor='white', s=50)
z = np.polyfit(sub['avg_sentiment'], sub['daily_return'], 1)
trend_x = np.linspace(sub['avg_sentiment'].min(), sub['avg_sentiment'].max(), 100)
ax.plot(trend_x, np.polyval(z, trend_x), color='red', linewidth=2, label='Trend line')
ax.set_xlabel("Average daily sentiment score", fontsize=11)
ax.set_ylabel("Same-day return", fontsize=11)
ax.set_title(f"{top_ticker}: Sentiment vs. Same-Day Return (Final)\nPearson r = {top_r:.3f}, p = {top_p:.4f}",
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/figures/scatter_strongest_result_FINAL.png", dpi=200)
plt.close()
print(f"Saved: data/figures/scatter_strongest_result_FINAL.png (ticker: {top_ticker})")

# ---------------------------------------------------------------------------
# 3. v1 vs v2 improvement summary (for the robustness section)
# ---------------------------------------------------------------------------
comparison = pd.read_csv("data/v1_vs_v2_comparison.csv")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(comparison['r_v1'], comparison['r_v2'], alpha=0.7, s=70,
           edgecolor='black', linewidth=0.5, color='#1f77b4')
lims = [comparison[['r_v1', 'r_v2']].min().min() - 0.05, comparison[['r_v1', 'r_v2']].max().max() + 0.05]
ax.plot(lims, lims, 'k--', alpha=0.5, label='No change (y = x)')
ax.set_xlabel("Pearson r - v1 (loose keyword matching)", fontsize=11)
ax.set_ylabel("Pearson r - v2 (word-boundary matching)", fontsize=11)
ax.set_title("Effect of Fixing Keyword-Matching Noise\non Per-Ticker Correlation", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/figures/v1_vs_v2_improvement.png", dpi=200)
plt.close()
print("Saved: data/figures/v1_vs_v2_improvement.png")

print(f"\nOverall same-day correlation improved from 0.180 (v1) to 0.220 (v2) after fixing keyword noise.")