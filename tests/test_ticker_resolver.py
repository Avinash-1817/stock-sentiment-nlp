"""Unit tests for src/ticker_resolver_v2.py (stdlib unittest).

Run from the repo root:  python -m unittest discover -s tests -v

These tests need data/nse_equity_list.csv (the resolver loads it at import).
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticker_resolver_v2 import (  # noqa: E402
    official_name_for_symbol,
    resolve_ticker,
)


class ResolveTickerTests(unittest.TestCase):
    def test_fragment_resolves_to_full_company(self):
        ticker, matched = resolve_ticker("varun")
        self.assertEqual(ticker, "VBL.NS")
        self.assertEqual(matched, "varun beverages limited")

    def test_full_name_resolves(self):
        ticker, _ = resolve_ticker("varun beverages")
        self.assertEqual(ticker, "VBL.NS")

    def test_curated_alias_resolves(self):
        ticker, _ = resolve_ticker("hul")
        self.assertEqual(ticker, "HINDUNILVR.NS")

    def test_unknown_company_returns_none(self):
        ticker, matched = resolve_ticker("zzzz not a real company qqqq")
        self.assertIsNone(ticker)
        self.assertIsNone(matched)

    def test_conversational_fragments_do_not_resolve(self):
        # Regression: difflib at cutoff 0.55 used to map these onto unrelated
        # companies ('show me trends' -> MEESHO.NS).
        for frag in ["show me trends", "last 15 days", "how is it doing"]:
            ticker, matched = resolve_ticker(frag)
            self.assertIsNone(ticker, f"{frag!r} should not resolve")
            self.assertIsNone(matched)


class OfficialNameTests(unittest.TestCase):
    def test_official_name_for_vbl(self):
        self.assertEqual(official_name_for_symbol("VBL.NS"), "Varun Beverages Limited")

    def test_official_name_strips_suffix(self):
        self.assertEqual(official_name_for_symbol("HINDUNILVR.NS"), "Hindustan Unilever Limited")

    def test_falls_back_to_symbol_when_unknown(self):
        self.assertEqual(official_name_for_symbol("ZZZZ.NS"), "ZZZZ")

    def test_handles_none_and_empty(self):
        self.assertEqual(official_name_for_symbol(None), "")
        self.assertEqual(official_name_for_symbol(""), "")


if __name__ == "__main__":
    unittest.main()
