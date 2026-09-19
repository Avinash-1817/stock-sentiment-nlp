"""
Per-Ticker Track Record

Breaks down the overall same-day backtest (68.3% hit rate, README) by
individual company, so the dashboard can show "for THIS stock specifically,
here's how the judge's same-day calls have historically aligned."

A "call" is made only when:
  1. This ticker's historical same-day correlation was statistically
     significant (matches the judge's abstention rule - no call on a
     company whose pattern isn't reliable), AND
  2. That day's sentiment was meaningfully non-neutral (|avg_sentiment| > 0.15,
     matching the threshold used elsewhere in the narrative generation).

The call direction follows the sign of sentiment, flipped if the ticker's
historical r is negative. A call is scored "aligned" if the same-day return's
sign matches the call direction.

Run locally: python src/per_ticker_track_record.py
"""

import pandas as pd
import numpy as np

MERGED_PATH = "data/merged_sentiment_price_v2.csv"
CORR_RESULTS_PATH = "data/per_ticker_correlation_results_v2.csv"
OUTPUT_PATH = "data/per_ticker_track_record.csv"

SENTIMENT_THRESHOLD = 0.15  # matches the threshold used in generate_narrative


def main():
    merged = pd.read_csv(MERGED_PATH)
    corr = pd.read_csv(CORR_RESULTS_PATH)

    corr_lookup = corr.set_index("ticker")[["r", "significant"]].to_dict("index")

    rows = []
    for ticker, group in merged.groupby("ticker"):
        stats = corr_lookup.get(ticker)
        if stats is None or not stats["significant"]:
            # Judge would abstain on every day for this ticker - no calls to backtest
            rows.append({
                "ticker": ticker, "n_calls": 0, "n_aligned": 0,
                "hit_rate": None, "mean_return_when_aligned": None,
                "note": "Not significant historically - judge abstains, no track record",
            })
            continue

        r_sign = 1 if stats["r"] > 0 else -1
        group = group.dropna(subset=["avg_sentiment", "daily_return"])

        calls = group[group["avg_sentiment"].abs() > SENTIMENT_THRESHOLD].copy()
        if len(calls) == 0:
            rows.append({
                "ticker": ticker, "n_calls": 0, "n_aligned": 0,
                "hit_rate": None, "mean_return_when_aligned": None,
                "note": "No sufficiently non-neutral sentiment days found",
            })
            continue

        calls["call_direction"] = np.sign(calls["avg_sentiment"]) * r_sign
        calls["actual_direction"] = np.sign(calls["daily_return"])
        calls["aligned"] = calls["call_direction"] == calls["actual_direction"]

        n_calls = len(calls)
        n_aligned = int(calls["aligned"].sum())
        hit_rate = n_aligned / n_calls
        mean_return_when_aligned = calls.loc[calls["aligned"], "daily_return"].abs().mean()

        rows.append({
            "ticker": ticker, "n_calls": n_calls, "n_aligned": n_aligned,
            "hit_rate": round(hit_rate, 4),
            "mean_return_when_aligned": round(mean_return_when_aligned, 4) if n_aligned > 0 else None,
            "note": "",
        })

    track_record = pd.DataFrame(rows).sort_values("hit_rate", ascending=False, na_position="last")
    track_record.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(track_record)} ticker track records to {OUTPUT_PATH}")
    print(f"\nTickers with a usable track record (judge would make calls): "
          f"{(track_record['n_calls'] > 0).sum()} / {len(track_record)}")
    print("\nTop 10 by hit rate:")
    print(track_record[track_record["n_calls"] > 0].head(10).to_string(index=False))


if __name__ == "__main__":
    main()