"""Unit tests for src/display.py (stdlib unittest)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from display import format_published_at  # noqa: E402


class FormatPublishedAtTests(unittest.TestCase):
    def test_utc_converted_to_ist(self):
        # 09:14:31Z + 5:30 = 14:44:31 IST
        self.assertEqual(
            format_published_at("2026-09-26T09:14:31Z"), "2026-09-26 14:44:31"
        )

    def test_offset_timestamps_converted_to_ist(self):
        # Already IST via offset -> unchanged wall time.
        self.assertEqual(
            format_published_at("2026-09-26T09:14:31+05:30"), "2026-09-26 09:14:31"
        )
        # 04:00 at UTC-05:30 = 09:30 UTC = 15:00 IST.
        self.assertEqual(
            format_published_at("2026-09-26T04:00:00-05:30"), "2026-09-26 15:00:00"
        )

    def test_naive_timestamp_treated_as_utc(self):
        self.assertEqual(
            format_published_at("2026-09-26T09:14:31"), "2026-09-26 14:44:31"
        )

    def test_day_rolls_over(self):
        # 20:00Z on the 26th + 5:30 = 01:30 IST on the 27th.
        self.assertEqual(
            format_published_at("2026-09-26T20:00:00Z"), "2026-09-27 01:30:00"
        )

    def test_unparseable_string_unchanged(self):
        self.assertEqual(format_published_at("two days ago"), "two days ago")

    def test_none_becomes_empty(self):
        self.assertEqual(format_published_at(None), "")

    def test_non_string_passthrough(self):
        obj = object()
        self.assertIs(format_published_at(obj), obj)


if __name__ == "__main__":
    unittest.main()
