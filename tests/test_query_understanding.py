"""Unit tests for src/query_understanding.py (stdlib unittest).

Run from the repo root:  python -m unittest discover -s tests -v

parse_query_llm() is monkeypatched in these tests so no Ollama call is made.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import query_understanding as qu  # noqa: E402
from ticker_resolver_v2 import find_companies_in_text  # noqa: E402


def _fake_llm(result):
    return mock.patch.object(qu, "parse_query_llm", return_value=result)


class MatchesInTextTests(unittest.TestCase):
    def test_whole_word_match(self):
        self.assertTrue(qu._matches_in_text("paytm", "search for paytm stock"))

    def test_case_and_punctuation_insensitive(self):
        self.assertTrue(qu._matches_in_text("Paytm", "Search for Paytm?"))

    def test_hallucinated_name_not_in_text(self):
        self.assertFalse(
            qu._matches_in_text("indian bank", "show me trends for last 15 days")
        )

    def test_short_names_need_word_boundaries(self):
        # len < 5 must NOT match as a bare substring
        self.assertFalse(qu._matches_in_text("tcs", "tcspro limited"))
        self.assertTrue(qu._matches_in_text("tcs", "compare tcs and infy"))

    def test_longer_names_can_substring_match(self):
        self.assertTrue(qu._matches_in_text("beverages", "varun beverages ltd"))

    def test_empty_company(self):
        self.assertFalse(qu._matches_in_text("", "anything"))


class GroundingTests(unittest.TestCase):
    def test_hallucinated_company_dropped_grounded_kept(self):
        def find_fn(q):
            return ["paytm"]

        out = qu._ground_llm_companies(
            ["indian bank", "paytm"], "show me trends for paytm last 15 days", find_fn
        )
        self.assertEqual(out, ["paytm"])

    def test_all_hallucinated_falls_back_to_text_extraction(self):
        out = qu._ground_llm_companies(
            ["totally fake corp"], "show me trends", find_companies_in_text
        )
        self.assertEqual(out, [])

    def test_alias_survives_when_grounded(self):
        out = qu._ground_llm_companies(
            ["hul"], "how is hul doing", find_companies_in_text
        )
        self.assertEqual(out, ["hul"])


class UnderstandQueryTests(unittest.TestCase):
    def test_hallucinated_llm_company_replaced_by_previous_turn(self):
        with _fake_llm({"companies": ["indian bank"], "days_back": 15, "intent": "trend"}):
            out = qu.understand_query(
                "show me trends for last 15 days",
                find_companies_in_text,
                prev_companies=["paytm"],
            )
        self.assertEqual(out["companies"], ["paytm"])
        self.assertEqual(out["days_back"], 15)

    def test_no_previous_turn_means_no_company(self):
        with _fake_llm({"companies": ["indian bank"], "days_back": 15, "intent": "trend"}):
            out = qu.understand_query(
                "show me trends for last 15 days", find_companies_in_text
            )
        self.assertEqual(out["companies"], [])
        self.assertEqual(out["days_back"], 15)

    def test_llm_days_back_clamped(self):
        with _fake_llm({"companies": ["paytm"], "days_back": 500, "intent": "trend"}):
            out = qu.understand_query("paytm", find_companies_in_text)
        self.assertEqual(out["days_back"], 90)

    def test_llm_days_back_string_coerced(self):
        with _fake_llm({"companies": ["paytm"], "days_back": "15", "intent": "trend"}):
            out = qu.understand_query("paytm", find_companies_in_text)
        self.assertEqual(out["days_back"], 15)

    def test_llm_days_back_none_falls_back_to_regex(self):
        with _fake_llm({"companies": ["paytm"], "days_back": None, "intent": "trend"}):
            out = qu.understand_query("paytm over the last 15 days", find_companies_in_text)
        self.assertEqual(out["days_back"], 15)

    def test_llm_unavailable_falls_back_to_regex(self):
        with _fake_llm(None):
            out = qu.understand_query(
                "show me trends for paytm over the last 15 days", find_companies_in_text
            )
        self.assertEqual(out["companies"], ["paytm"])
        self.assertEqual(out["days_back"], 15)
        self.assertEqual(out["intent"], "trend")

    def test_regex_path_drops_unresolvable_fragments(self):
        def find_fn(q):
            return ["qwertyuiop fakeco", "paytm"]

        with _fake_llm(None):
            out = qu.understand_query("whatever paytm", find_fn)
        self.assertEqual(out["companies"], ["paytm"])

    def test_greeting_does_not_carry_over(self):
        with _fake_llm({"companies": [], "days_back": None, "intent": "other"}):
            out = qu.understand_query(
                "hello there", find_companies_in_text, prev_companies=["paytm"]
            )
        self.assertEqual(out["companies"], [])

    def test_followup_carryover_applies_new_window(self):
        with _fake_llm({"companies": [], "days_back": 30, "intent": "trend"}):
            out = qu.understand_query(
                "now show me the last 30 days",
                find_companies_in_text,
                prev_companies=["paytm"],
            )
        self.assertEqual(out["companies"], ["paytm"])
        self.assertEqual(out["days_back"], 30)

    def test_days_back_still_parsed_when_llm_fails(self):
        with _fake_llm(None):
            out = qu.understand_query("tell me about this month for reliance",
                                      lambda q: ["reliance"])
        self.assertEqual(out["days_back"], 30)


class NoOllamaProdPathTests(unittest.TestCase):
    """Regression for prod (Streamlit Cloud has no Ollama): the regex path
    used to return ['hdfc bank', 'bank', 'hdfc'] for 'HDFC Bank last 3 days',
    rendering a bogus 3-company comparison instead of one company card."""

    def test_hdfc_bank_query_extracts_single_company(self):
        with _fake_llm(None):
            out = qu.understand_query(
                "HDFC Bank last 3 days", find_companies_in_text
            )
        self.assertEqual(out["companies"], ["hdfc bank"])
        self.assertEqual(out["days_back"], 3)

    def test_tcs_query_extracts_single_company(self):
        with _fake_llm(None):
            out = qu.understand_query(
                "how has TCS done in the past 45 days", find_companies_in_text
            )
        self.assertEqual(out["companies"], ["tcs"])

    def test_two_named_companies_both_kept(self):
        with _fake_llm(None):
            out = qu.understand_query(
                "compare paytm and reliance", find_companies_in_text
            )
        self.assertEqual(len(out["companies"]), 2)


class EffectiveNewsDaysTests(unittest.TestCase):
    def test_clamped_to_plan_limit(self):
        self.assertEqual(qu.effective_news_days(45, max_days=29), 29)

    def test_within_limit_unchanged(self):
        self.assertEqual(qu.effective_news_days(15, max_days=29), 15)

    def test_minimum_one(self):
        self.assertEqual(qu.effective_news_days(0, max_days=29), 1)
        self.assertEqual(qu.effective_news_days(-5, max_days=29), 1)

    def test_garbage_defaults_to_7(self):
        self.assertEqual(qu.effective_news_days("abc", max_days=29), 7)

    def test_none_defaults_to_7(self):
        self.assertEqual(qu.effective_news_days(None, max_days=29), 7)

    def test_uses_config_default_without_max_days(self):
        from config import NEWSAPI_MAX_DAYS
        self.assertEqual(qu.effective_news_days(45), NEWSAPI_MAX_DAYS)


class ParseDaysBackTests(unittest.TestCase):
    def test_numeric_days(self):
        self.assertEqual(qu.parse_days_back("trends for last 15 days"), 15)

    def test_word_days(self):
        self.assertEqual(qu.parse_days_back("this week"), 7)
        self.assertEqual(qu.parse_days_back("over the last month"), 30)
        self.assertEqual(qu.parse_days_back("today"), 1)

    def test_clamped(self):
        self.assertEqual(qu.parse_days_back("last 500 days"), 90)

    def test_default(self):
        self.assertEqual(qu.parse_days_back("how is reliance"), 7)


if __name__ == "__main__":
    unittest.main()
