"""Small display-formatting helpers shared by the dashboards.

Kept dependency-free (stdlib only) so tests can import it without pulling in
streamlit/torch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# IST is UTC+05:30 year-round (no DST), so a fixed offset is exact and avoids
# the zoneinfo/tzdata dependency (which needs a pip package on Windows).
IST = timezone(timedelta(hours=5, minutes=30))


def format_published_at(value) -> str:
    """Format a NewsAPI ``publishedAt`` timestamp for display in IST.

    NewsAPI returns ISO-8601 in UTC with a trailing Z, e.g.
    ``2026-09-26T09:14:31Z`` -> ``2026-09-26 14:44:31`` (IST). Timestamps that
    already carry an offset are converted from their own offset; naive
    timestamps are treated as UTC (NewsAPI's convention). Anything that does
    not parse (already formatted, None, empty) is returned unchanged.
    """
    if not value or not isinstance(value, str):
        return value if isinstance(value, str) else ("" if value is None else value)
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
