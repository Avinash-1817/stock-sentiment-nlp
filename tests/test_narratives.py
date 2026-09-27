"""Unit tests for src/narratives.py (stdlib unittest).

Run from the repo root:  python -m unittest discover -s tests -v

These pin the two prod bugs the templates used to have:
1. the price move was described with the NEWS window (29d) instead of the
   window the user asked for (e.g. 90d);
2. unstudied companies were called "newer listings" - Wipro (listed 1980)
   triggered that line in prod.

They also pin the Ollama-like tone so prod's prose stays close to local.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from narratives import (  # noqa: E402
    generate_comparison_narrative_fallback,
    generate_narrative_fallback,
)


def wipro_like(**overrides):
    """Mirrors the real prod case: 90-day price window, 29-day news clamp,
    WIPRO studied but not significant (r=0.108, p=0.225, n=128)."""
    base = {
        "company": "Wipro Limited",
        "ticker": "WIPRO.NS",
        "avg_sentiment": 0.05,
        "article_count": 30,
        "news_days": 29,
        "days_back": 90,
        "pct_change_window": -0.1832,
        "pct_change": 0.0023,
        "latest_close": 2082.0,
        "hist_r": 0.1079,
        "hist_sig": False,
    }
    base.update(overrides)
    return base


class NarrativeFallbackTests(unittest.TestCase):
    def test_price_window_is_the_user_requested_window(self):
        text = generate_narrative_fallback(wipro_like())
        self.assertIn("last 90 trading days", text)
        self.assertNotIn("29-day window", text)
        self.assertNotIn("same 29-day", text)

    def test_news_window_labels_the_article_count(self):
        text = generate_narrative_fallback(wipro_like())
        self.assertIn("last 29 days", text)

    def test_no_newer_listing_claim(self):
        text = generate_narrative_fallback(wipro_like(hist_r=None, hist_sig=None))
        self.assertIn("wasn't part of the original historical study", text)
        self.assertNotIn("newer listing", text)

    def test_weak_pattern_line_when_studied_but_not_significant(self):
        text = generate_narrative_fallback(wipro_like())
        self.assertIn("weak and not statistically reliable", text)

    def test_meaningful_pattern_line_when_significant(self):
        text = generate_narrative_fallback(wipro_like(hist_sig=True))
        self.assertIn("statistically meaningful", text)

    def test_keeps_official_company_casing(self):
        text = generate_narrative_fallback(wipro_like())
        self.assertIn("Wipro Limited", text)
        self.assertNotIn("Wipro Limited".title(), text.replace("Wipro Limited", ""))

    def test_direction_words_follow_the_sign(self):
        text = generate_narrative_fallback(wipro_like())
        self.assertIn("down 18.32%", text)
        self.assertIn("up 0.23%", text)

    def test_disclaimer_present(self):
        self.assertIn("not a forecast", generate_narrative_fallback(wipro_like()))

    def test_handles_missing_price_data(self):
        text = generate_narrative_fallback(wipro_like(pct_change_window=None, pct_change=None))
        self.assertIn("last 29 days", text)
        self.assertNotIn("trading days", text.split("Historically")[0].split("most recent")[0])


class ComparisonFallbackTests(unittest.TestCase):
    def test_two_companies_both_described(self):
        data = [
            wipro_like(),
            wipro_like(company="HDFC Bank Limited", ticker="HDFCBANK.NS",
                       avg_sentiment=-0.2, hist_sig=True, pct_change_window=-0.0041),
        ]
        text = generate_comparison_narrative_fallback(data)
        self.assertIn("Wipro Limited", text)
        self.assertIn("HDFC Bank Limited", text)
        self.assertIn("does not indicate which stock is a better investment", text)

    def test_no_winner_declared(self):
        data = [wipro_like(), wipro_like(company="A", ticker="A.NS", hist_sig=True)]
        text = generate_comparison_narrative_fallback(data)
        for banned in ("better investment than", "buy", "outperform"):
            self.assertNotIn(banned, text.lower().replace("better investment.", ""))

    def test_labels_price_window_per_company(self):
        data = [wipro_like(), wipro_like(company="B", ticker="B.NS", days_back=15)]
        text = generate_comparison_narrative_fallback(data)
        self.assertIn("last 90 trading days", text)
        self.assertIn("last 15 trading days", text)


if __name__ == "__main__":
    unittest.main()
