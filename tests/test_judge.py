"""Unit tests for src/judge.py (stdlib unittest - no pytest needed).

Run from the repo root:  python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge  # noqa: E402

SIG_POS = {"r": 0.40, "p": 0.001, "n": 200, "significant": True}
SIG_NEG = {"r": -0.40, "p": 0.001, "n": 200, "significant": True}
NOT_SIG = {"r": 0.03, "p": 0.60, "n": 200, "significant": False}
THIN = {"r": 0.99, "p": 0.3, "n": 5, "significant": False}


class JudgeCallTests(unittest.TestCase):
    def test_significant_positive_hist_positive_tone_is_bullish(self):
        v = judge.judge_call(0.55, 9, SIG_POS, ticker="T.NS")
        self.assertEqual(v.call, judge.BULLISH)
        self.assertEqual(v.reason, "directional")
        self.assertEqual(v.basis, "ticker study")
        self.assertEqual(v.expected_sign, 1)
        self.assertGreaterEqual(v.confidence, 0.8)

    def test_negative_tone_is_bearish(self):
        v = judge.judge_call(-0.5, 9, SIG_POS, ticker="T.NS")
        self.assertEqual(v.call, judge.BEARISH)
        self.assertGreaterEqual(v.confidence, 0.8)

    def test_weak_sentiment_is_neutral(self):
        v = judge.judge_call(0.05, 9, SIG_POS)
        self.assertEqual(v.call, judge.NEUTRAL)
        self.assertEqual(v.reason, "weak_sentiment")
        self.assertFalse(v.is_directional)

    def test_low_coverage_is_neutral(self):
        v = judge.judge_call(0.9, 2, SIG_POS)
        self.assertEqual(v.call, judge.NEUTRAL)
        self.assertEqual(v.reason, "low_coverage")

    def test_weak_pattern_abstains_by_default(self):
        v = judge.judge_call(0.6, 9, NOT_SIG, ticker="HINDUNILVR.NS")
        self.assertEqual(v.call, judge.NEUTRAL)
        self.assertEqual(v.reason, "weak_pattern")

    def test_weak_pattern_can_be_disabled(self):
        v = judge.judge_call(0.6, 9, NOT_SIG, abstain_on_weak_pattern=False)
        self.assertEqual(v.call, judge.BULLISH)
        self.assertLess(v.confidence, 0.65)  # low significance drags confidence

    def test_significant_negative_hist_flips_direction(self):
        v = judge.judge_call(0.6, 9, SIG_NEG)
        self.assertEqual(v.call, judge.BEARISH)
        self.assertEqual(v.expected_sign, -1)

    def test_missing_hist_uses_market_prior(self):
        v = judge.judge_call(0.6, 9, None, ticker="PAYTM.NS")
        self.assertEqual(v.basis, "market prior")
        self.assertEqual(v.call, judge.BULLISH)
        self.assertLess(v.confidence, 0.85)  # down-weighted by the prior penalty

    def test_thin_history_falls_back_to_prior(self):
        v = judge.judge_call(0.6, 9, THIN)
        self.assertEqual(v.basis, "market prior")

    def test_confidence_higher_for_significant_record(self):
        sig = judge.judge_call(0.6, 9, SIG_POS, abstain_on_weak_pattern=False)
        ns = judge.judge_call(0.6, 9, NOT_SIG, abstain_on_weak_pattern=False)
        self.assertGreater(sig.confidence, ns.confidence)

    def test_sentiment_clamped(self):
        self.assertEqual(judge.judge_call(5.0, 9, SIG_POS).sentiment, 1.0)
        self.assertEqual(judge.judge_call(-10.0, 9, SIG_POS).sentiment, -1.0)

    def test_articles_coerced(self):
        v = judge.judge_call(0.6, None, SIG_POS)
        self.assertEqual(v.num_articles, 0)
        self.assertEqual(v.reason, "low_coverage")


class StudyStatsTests(unittest.TestCase):
    def test_missing_csv_returns_embedded_fallback(self):
        stats = judge.load_study_stats("/nonexistent/path.csv")
        self.assertIn("RELIANCE.NS", stats)
        self.assertTrue(stats["RELIANCE.NS"]["significant"])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "results.csv"
            p.write_text(
                "ticker,n,r,p,significant\nFAKE.NS,50,0.33,0.02,True\n"
                "OTH.NS,60,-0.1,0.4,false\n",
                encoding="utf-8",
            )
            stats = judge.load_study_stats(str(p))
        self.assertEqual(stats["FAKE.NS"]["r"], 0.33)
        self.assertTrue(stats["FAKE.NS"]["significant"])
        self.assertFalse(stats["OTH.NS"]["significant"])
        self.assertEqual(stats["OTH.NS"]["r"], -0.1)

    def test_study_stats_for_unknown_ticker_returns_prior(self):
        self.assertEqual(
            judge.study_stats_for("ZZZ.NS", {}), judge.MARKET_PRIOR
        )


class TextTests(unittest.TestCase):
    def test_describe_includes_evidence_and_disclaimer(self):
        v = judge.judge_call(0.6, 8, SIG_POS, ticker="T.NS")
        text = "\n".join(judge.describe(v, company="Test Co"))
        self.assertIn("Positive tilt", text)
        self.assertIn("not investment advice", text)
        self.assertIn("NEXT day", text)

    def test_one_line_for_neutral(self):
        v = judge.judge_call(0.05, 9, SIG_POS, ticker="T.NS")
        self.assertIn("Neutral", judge.one_line(v))


if __name__ == "__main__":
    unittest.main()
