"""
Two rigor upgrades to the correlation study, done together since they touch
the same numbers:

1. MARKET-ADJUSTED RETURNS (excess return / alpha):
   Subtract the Nifty 50's same-day return from each stock's return, so we're
   testing sentiment against COMPANY-SPECIFIC price movement, not movement
   that's really just "the whole market went up/down that day."

2. MULTIPLE-COMPARISONS CORRECTION:
   Testing 46 tickers independently at p < 0.05 each means ~2-3 "significant"
   results are expected by pure chance alone, even if nothing were real.
   Apply Bonferroni (strict) and Benjamini-Hochberg (less strict, controls
   false discovery rate) corrections and report how many tickers survive.

Run locally: python src/market_adjusted_and_corrected.py
Requires: data/merged_sentiment_price_v2.csv, data/prices/NIFTY50_index.csv
(run fetch_nifty_index.py first if the latter doesn't exist yet)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

MERGED_PATH = "data/merged_sentiment_price_v2.csv"
NIFTY_PATH = "data/prices/NIFTY50_index.csv"
OUTPUT_PATH = "data/rigor_upgrade_results.csv"

MIN_OBS_PER_TICKER = 10


# ---------------------------------------------------------------------------
# Multiple-comparison correction helpers (no extra dependency needed -
# statsmodels isn't already in requirements.txt, so implemented directly)
# ---------------------------------------------------------------------------
def bonferroni_correct(p_values, alpha=0.05):
    """Strictest correction: reject only if p < alpha / n_tests."""
    n = len(p_values)
    threshold = alpha / n
    return [p < threshold for p in p_values], threshold


def benjamini_hochberg_correct(p_values, alpha=0.05):
    """Controls the expected proportion of false discoveries among the
    rejected hypotheses (less conservative than Bonferroni, and the more
    standard choice for exploratory multi-ticker studies like this one)."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [(i + 1) / n * alpha for i in range(n)]

    # find the largest k such that p_(k) <= threshold_(k)
    significant_up_to = -1
    for rank, (_, p) in enumerate(indexed):
        if p <= thresholds[rank]:
            significant_up_to = rank

    result = [False] * n
    for rank in range(significant_up_to + 1):
        original_idx = indexed[rank][0]
        result[original_idx] = True
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    merged = pd.read_csv(MERGED_PATH)
    merged["date_only"] = pd.to_datetime(merged["date_only"])

    nifty = pd.read_csv(NIFTY_PATH)
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty = nifty.rename(columns={"Date": "date_only"})

    # ---- 1. Compute excess (market-adjusted) returns ----
    merged = pd.merge(merged, nifty[["date_only", "nifty_return", "nifty_next_day_return"]],
                       on="date_only", how="left")
    merged["excess_return"] = merged["daily_return"] - merged["nifty_return"]
    merged["excess_next_day_return"] = merged["next_day_return"] - merged["nifty_next_day_return"]

    valid = merged.dropna(subset=["avg_sentiment", "daily_return", "excess_return"])
    print(f"Rows with market-adjustment available: {len(valid)} / {len(merged)}")

    # ---- 2. Overall correlation: raw vs. excess ----
    r_raw, p_raw = pearsonr(valid["avg_sentiment"], valid["daily_return"])
    r_excess, p_excess = pearsonr(valid["avg_sentiment"], valid["excess_return"])
    print(f"\nOverall same-day correlation:")
    print(f"  Raw return:    r = {r_raw:.4f}, p = {p_raw:.4g}")
    print(f"  Excess return: r = {r_excess:.4f}, p = {p_excess:.4g}")
    print("  (If r_excess is close to r_raw, the effect isn't just market-wide "
          "movement being picked up as 'sentiment correlation'.)")

    # ---- 3. Per-ticker correlation using EXCESS returns ----
    results = []
    for ticker in sorted(valid["ticker"].unique()):
        sub = valid[valid["ticker"] == ticker]
        if len(sub) < MIN_OBS_PER_TICKER:
            continue
        r, p = pearsonr(sub["avg_sentiment"], sub["excess_return"])
        results.append({"ticker": ticker, "n": len(sub), "r_excess": r, "p_excess": p})

    results_df = pd.DataFrame(results)
    n_uncorrected_sig = (results_df["p_excess"] < 0.05).sum()
    print(f"\nUncorrected: {n_uncorrected_sig}/{len(results_df)} tickers significant (p < 0.05) using excess returns")

    # ---- 4. Multiple-comparisons corrections ----
    p_values = results_df["p_excess"].tolist()

    bonf_sig, bonf_threshold = bonferroni_correct(p_values)
    results_df["bonferroni_significant"] = bonf_sig
    n_bonf = sum(bonf_sig)

    bh_sig = benjamini_hochberg_correct(p_values)
    results_df["benjamini_hochberg_significant"] = bh_sig
    n_bh = sum(bh_sig)

    print(f"\nBonferroni correction (threshold p < {bonf_threshold:.6f}): "
          f"{n_bonf}/{len(results_df)} tickers remain significant")
    print(f"Benjamini-Hochberg correction (FDR 5%): "
          f"{n_bh}/{len(results_df)} tickers remain significant")

    results_df = results_df.sort_values("p_excess")
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved full results to {OUTPUT_PATH}")

    print("\nTickers surviving Benjamini-Hochberg correction:")
    print(results_df[results_df["benjamini_hochberg_significant"]][["ticker", "n", "r_excess", "p_excess"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()