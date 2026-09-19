"""Unit tests for src/backtest_judgment.py metric helpers.

These tests exercise the metric/statistics functions on small synthetic
frames and the expanding-prior math; they do NOT re-run the full 6,374-row
backtest (run src/backtest_judgment.py for that).

Run from the repo root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest_judgment as bt  # noqa: E402
import judge  # noqa: E402


def make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic ticker-days where sentiment and SAME-day return are strongly
    related (judge should align) and next-day return is independent noise."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    same = 0.6 * s + rng.normal(0, 0.5, n)
    nxt = rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {
            "ticker": ["FAKE.NS"] * n,
            "date_only": pd.date_range("2020-01-01", periods=n, freq="B"),
            "avg_sentiment": s,
            "num_articles": [8] * n,
            "daily_return": same,
            "next_day_return": nxt,
        }
    )
    return df


class ExpandingPriorTests(unittest.TestCase):
    def test_perfect_positive_relationship_converges(self):
        n = 120
        x = np.linspace(-1, 1, n)
        y = x  # perfectly correlated
        df = pd.DataFrame(
            {"avg_sentiment": x, "daily_return": y},
            index=range(n),
        )
        out = bt.expanding_prior(df)
        # No leakage: row i's prior uses rows 0..i-1 only.
        self.assertTrue(np.isnan(out["_r_prior"].iloc[0]))
        self.assertTrue(np.isnan(out["_r_prior"].iloc[1]))
        self.assertAlmostEqual(out["_r_prior"].iloc[2], 1.0, places=6)
        self.assertLess(out["_p_prior"].iloc[-1], 1e-9)
        self.assertEqual(int(out["_n_prior"].iloc[-1]), n - 1)

    def test_flat_series_yields_nan(self):
        df = pd.DataFrame(
            {"avg_sentiment": [0.2] * 5, "daily_return": [0.3] * 5}
        )
        out = bt.expanding_prior(df)
        self.assertTrue(np.isnan(out["_r_prior"]).all())


class MetricTests(unittest.TestCase):
    def test_directional_stats_on_perfect_calls(self):
        hist = {"r": 0.5, "p": 0.001, "n": 300, "significant": True}
        rows = []
        for _, r in make_rows(60, seed=1).iterrows():
            v = judge.judge_call(r["avg_sentiment"], 8, hist)
            rows.append(
                {
                    "call": v.call,
                    "direction": 1 if v.call == judge.BULLISH else (
                        -1 if v.call == judge.BEARISH else 0),
                    "confidence": v.confidence,
                    "daily_return": r["daily_return"],
                    "next_day_return": r["next_day_return"],
                    "alignment_hit": None,
                    "forecast_hit": None,
                }
            )
        df = pd.DataFrame(rows)
        df["alignment_hit"] = np.where(
            (df["direction"] != 0) & (df["daily_return"] * df["direction"] > 0), 1.0,
            np.where((df["direction"] != 0), 0.0, np.nan),
        )
        ds = bt.directional_stats(df, "alignment_hit", "daily_return")
        self.assertGreater(ds["n"], 0)
        # With r=0.5 synthetic data the judge should beat 50% comfortably.
        self.assertGreater(ds["hit_rate"], 0.55)
        self.assertLess(ds["hit_p"], 0.05)

    def test_abstention_stats_only_counts_weak_pattern_rows(self):
        df = pd.DataFrame(
            {
                "reason": ["weak_pattern", "weak_pattern", "directional"],
                "naive_direction": [1, -1, 1],
                "naive_alignment_hit": [1.0, 0.0, 1.0],
                "naive_forecast_hit": [0.0, 1.0, 1.0],
            }
        )
        abst = bt.abstention_stats(df)
        self.assertEqual(abst["n"], 2)
        self.assertEqual(abst["alignment_hit"], 0.5)
        self.assertEqual(abst["forecast_hit"], 0.5)

    def test_abstention_stats_empty_when_no_weak_patterns(self):
        df = pd.DataFrame(
            {"reason": ["directional"], "naive_direction": [1],
             "naive_alignment_hit": [1.0]}
        )
        self.assertEqual(bt.abstention_stats(df)["n"], 0)

    def test_bucket_metrics_counts(self):
        df = pd.DataFrame(
            {
                "call": [judge.BULLISH] * 4 + [judge.NEUTRAL] * 2,
                "alignment_hit": [1.0, 1.0, 0.0, 1.0, np.nan, np.nan],
                "forecast_hit": [0.0, 0.0, 1.0, 0.0, np.nan, np.nan],
                "confidence": [0.8] * 6,
                "daily_return": [0.01, 0.02, -0.01, 0.03, 0.0, 0.0],
                "next_day_return": [-0.01] * 6,
            }
        )
        bm = bt.bucket_metrics(df)
        self.assertEqual(int(bm.loc[judge.BULLISH, "n"]), 4)
        self.assertEqual(bm.loc[judge.BULLISH, "alignment_hit_rate"], 0.75)
        self.assertTrue(np.isnan(bm.loc[judge.NEUTRAL, "alignment_hit_rate"]))


if __name__ == "__main__":
    unittest.main()
