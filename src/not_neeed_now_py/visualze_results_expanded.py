import matplotlib
matplotlib.use('Agg')  # ensure non-interactive backend, avoids GUI init overhead

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

os.makedirs("data/figures", exist_ok=True)

FIG_DPI = 150  # drop from 200 -> faster savefig, still crisp on screen

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results_df = pd.read_csv("data/per_ticker_correlation_results_full.csv")
merged = pd.read_csv("data/merged_sentiment_price_full.csv", parse_dates=['date_only'])

# ---------------------------------------------------------------------------
# 1. SORTED BAR CHART — all tickers, color-coded by significance
# ---------------------------------------------------------------------------
plot_df = results_df.sort_values('r', ascending=True).reset_index(drop=True)
colors = ['#2ca02c' if sig else '#d3d3d3' for sig in plot_df['significant']]

fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(plot_df['ticker'], plot_df['r'], color=colors, edgecolor='black', linewidth=0.3)

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Pearson correlation (sentiment vs. same-day return)", fontsize=11)
ax.set_title("Same-Day Sentiment-Return Correlation by Ticker\n(green = statistically significant, p < 0.05)",
              fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=8)

legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Significant (p < 0.05)'),
    Patch(facecolor='#d3d3d3', edgecolor='black', label='Not significant'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("data/figures/correlation_bar_chart.png", dpi=FIG_DPI)
plt.close()
print("Saved: data/figures/correlation_bar_chart.png")

# ---------------------------------------------------------------------------
# 2. SCATTER PLOT — strongest result (highest |r|)
# ---------------------------------------------------------------------------
top_idx = results_df['r'].abs().idxmax()
top_ticker, top_r, top_p = results_df.loc[top_idx, ['ticker', 'r', 'p']]

sub = merged[merged['ticker'] == top_ticker].dropna(subset=['avg_sentiment', 'daily_return'])

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(sub['avg_sentiment'], sub['daily_return'], alpha=0.6, color='#1f77b4', edgecolor='white', s=50)

z = np.polyfit(sub['avg_sentiment'], sub['daily_return'], 1)
trend_x = np.linspace(sub['avg_sentiment'].min(), sub['avg_sentiment'].max(), 100)
ax.plot(trend_x, np.polyval(z, trend_x), color='red', linewidth=2, label='Trend line')

ax.set_xlabel("Average daily sentiment score", fontsize=11)
ax.set_ylabel("Same-day return", fontsize=11)
ax.set_title(f"{top_ticker}: Sentiment vs. Same-Day Return\nPearson r = {top_r:.3f}, p = {top_p:.4f}",
              fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("data/figures/scatter_strongest_result.png", dpi=FIG_DPI)
plt.close()
print(f"Saved: data/figures/scatter_strongest_result.png (ticker: {top_ticker})")

# ---------------------------------------------------------------------------
# 3. SECTOR-LEVEL GROUPING
#    Updated to include late-IPO tickers (Eternal/Zomato, Paytm, Nykaa,
#    PolicyBazaar, LIC) and to fix TMCV.NS / TMPV.NS, which were missing
#    from the old map (it only had the stale "TATAMOTORS.NS").
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "RELIANCE.NS": "Conglomerate/Energy", "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "AXISBANK.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services",
    "SBILIFE.NS": "Financial Services", "HDFCLIFE.NS": "Financial Services", "PFC.NS": "Financial Services",
    "LICI.NS": "Financial Services", "POLICYBZR.NS": "Financial Services", "PAYTM.NS": "Financial Services",
    "MARUTI.NS": "Auto", "TMCV.NS": "Auto", "TMPV.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma", "APOLLOHOSP.NS": "Pharma",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BHARTIARTL.NS": "Telecom",
    "LT.NS": "Engineering", "ULTRACEMCO.NS": "Cement", "BHEL.NS": "Engineering",
    "NTPC.NS": "Energy/Power", "POWERGRID.NS": "Energy/Power", "ONGC.NS": "Energy/Power",
    "COALINDIA.NS": "Energy/Power",
    "TATASTEEL.NS": "Metals", "JINDALSTEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "HAL.NS": "Defense/PSU", "BEL.NS": "Defense/PSU", "IRCTC.NS": "Defense/PSU", "RECLTD.NS": "Defense/PSU",
    "TITAN.NS": "Consumer/Retail", "ASIANPAINT.NS": "Consumer/Retail", "DMART.NS": "Consumer/Retail",
    "TRENT.NS": "Consumer/Retail", "NYKAA.NS": "Consumer/Retail",
    "ETERNAL.NS": "New-Age Tech/Internet",
}

results_df['sector'] = results_df['ticker'].map(SECTOR_MAP).fillna("Other")

sector_summary = results_df.groupby('sector').agg(
    n_tickers=('ticker', 'count'),
    n_significant=('significant', 'sum'),
    avg_r=('r', 'mean')
).reset_index()
sector_summary['pct_significant'] = (sector_summary['n_significant'] / sector_summary['n_tickers'] * 100).round(1)
sector_summary = sector_summary.sort_values('pct_significant', ascending=False)

print("\n=== SECTOR-LEVEL SUMMARY ===")
print(sector_summary.to_string(index=False))
sector_summary.to_csv("data/sector_summary.csv", index=False)
print("\nSaved: data/sector_summary.csv")

fig, ax = plt.subplots(figsize=(9, 6))
bar_colors = plt.cm.RdYlGn(sector_summary['pct_significant'] / 100)
ax.barh(sector_summary['sector'], sector_summary['pct_significant'], color=bar_colors, edgecolor='black')
ax.set_xlabel("% of tickers with significant same-day correlation", fontsize=11)
ax.set_title("Significance Rate by Sector", fontsize=13, fontweight='bold')
for i, (pct, n) in enumerate(zip(sector_summary['pct_significant'], sector_summary['n_tickers'])):
    ax.text(pct + 1, i, f"n={n}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig("data/figures/sector_significance.png", dpi=FIG_DPI)
plt.close()
print("Saved: data/figures/sector_significance.png")