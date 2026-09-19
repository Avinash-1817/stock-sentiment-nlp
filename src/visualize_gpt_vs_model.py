import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data/figures", exist_ok=True)

comparison = pd.read_csv("data/gpt_vs_model_comparison.csv")

# ---------------------------------------------------------------------------
# Scatter plot: GPT r-value vs Model r-value, per ticker
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

colors = np.where(
    comparison['significant_GPT'] & comparison['significant_Model'], '#2ca02c',   # both significant
    np.where(~comparison['significant_GPT'] & ~comparison['significant_Model'], '#999999',  # neither
             '#ff7f0e')  # disagreement on significance
)

ax.scatter(comparison['r_GPT'], comparison['r_Model'], c=colors, s=80,
           edgecolor='black', linewidth=0.5, alpha=0.85)

# 45-degree reference line (perfect agreement)
lims = [
    min(comparison['r_GPT'].min(), comparison['r_Model'].min()) - 0.05,
    max(comparison['r_GPT'].max(), comparison['r_Model'].max()) + 0.05,
]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect agreement (y = x)')

# correlation between the two sets of r-values themselves
agreement_corr = np.corrcoef(comparison['r_GPT'], comparison['r_Model'])[0, 1]

ax.set_xlabel("Pearson r (GPT-labeled sentiment)", fontsize=11)
ax.set_ylabel("Pearson r (fine-tuned model sentiment)", fontsize=11)
ax.set_title(f"Per-Ticker Correlation: GPT Labels vs. Fine-Tuned Model\n"
             f"Agreement between the two r-value sets: r = {agreement_corr:.3f}",
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(lims)
ax.set_ylim(lims)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Significant in both'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Significant in only one'),
    Patch(facecolor='#999999', edgecolor='black', label='Not significant in either'),
]
ax2 = ax.twinx()
ax2.set_yticks([])
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig("data/figures/gpt_vs_model_comparison.png", dpi=200)
plt.close()
print("Saved: data/figures/gpt_vs_model_comparison.png")
print(f"\nAgreement correlation between GPT r-values and Model r-values: {agreement_corr:.4f}")

# ---------------------------------------------------------------------------
# Summary stats for the report
# ---------------------------------------------------------------------------
both_sig = ((comparison['significant_GPT']) & (comparison['significant_Model'])).sum()
neither_sig = ((~comparison['significant_GPT']) & (~comparison['significant_Model'])).sum()
disagree = len(comparison) - both_sig - neither_sig

print(f"\nOut of {len(comparison)} tickers:")
print(f"  Significant under BOTH GPT and model sentiment: {both_sig}")
print(f"  Not significant under EITHER: {neither_sig}")
print(f"  Disagreement (significant under only one): {disagree}")
print(f"  Overall agreement rate on significance classification: {(both_sig + neither_sig) / len(comparison):.1%}")