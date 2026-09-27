"""Plain-English narrative templates for the dashboards.

These are the no-LLM fallbacks used when Ollama isn't available (Streamlit
Community Cloud can't run it). They are pure functions - no streamlit, no
ollama - so they can be unit-tested and they read like the Ollama narratives:
flowing sentences, every number labelled with the window it actually covers.

Window rules (the source of a past prod bug):
- articles were fetched over the NEWS window (``news_days``, clamped to the
  NewsAPI plan limit - about 29 days on the free plan);
- the price change is measured over the window the USER asked for
  (``days_back``, e.g. 90), which can be much longer than the news window.
  Conflating the two made prod claim an 18% move happened "over that same
  29-day window".
"""

from __future__ import annotations


def _tone(avg_sentiment) -> str:
    if avg_sentiment > 0.15:
        return "mostly positive"
    if avg_sentiment < -0.15:
        return "mostly negative"
    return "mixed/neutral"


def _direction(change) -> str:
    return "up" if change > 0 else "down" if change < 0 else "flat"


def _company(d) -> str:
    # Official NSE names already carry proper casing ("HDFC Bank Limited");
    # .title()-ing them produced "Hdfc Bank Limited".
    return str(d.get("company") or "").strip() or "this stock"


def generate_narrative_fallback(d) -> str:
    """Template narrative for ONE company - same content Ollama is prompted
    for: news tone over the news window, price over the user's window, the
    historical same-day pattern (or its absence), and the disclaimer."""
    news_days = d.get("news_days", d.get("days_back", 7))
    price_days = d.get("days_back", 7)

    parts = [
        f"Over the last {news_days} days, {d.get('article_count', 0)} news "
        f"articles about {_company(d)} have carried a "
        f"{_tone(d['avg_sentiment'])} tone."
    ]

    window_change = d.get("pct_change_window")
    pct_change = d.get("pct_change")
    if window_change is not None:
        line = (
            f"The stock has moved {_direction(window_change)} "
            f"{abs(window_change):.2%} over the last {price_days} trading days"
        )
        if pct_change is not None:
            line += (
                f", and {_direction(pct_change)} {abs(pct_change):.2%} on the "
                f"most recent day alone"
            )
        parts.append(line + ".")
    elif pct_change is not None:
        parts.append(
            f"On the most recent trading day, it moved {_direction(pct_change)} "
            f"{abs(pct_change):.2%}."
        )

    hist_r = d.get("hist_r")
    if hist_r is not None:
        if d.get("hist_sig"):
            parts.append(
                "Historically, this stock's price has tended to move in the "
                "same direction as news sentiment on the same day - a pattern "
                "strong enough to be statistically meaningful."
            )
        else:
            parts.append(
                "Historically, its price has tended to move with same-day "
                "news sentiment, though for this stock that pattern has been "
                "weak and not statistically reliable."
            )
    else:
        parts.append(
            "This stock wasn't part of the original historical study, so "
            "there's no historical pattern to reference."
        )

    parts.append(
        "This is a snapshot of current information, not a forecast of what "
        "might happen next."
    )
    return " ".join(parts)


def generate_comparison_narrative_fallback(companies_data) -> str:
    """Template comparison across 2+ companies - describes each company in
    one flowing sentence and never declares a winner."""
    lines = []
    for d in companies_data:
        window_change = d.get("pct_change_window")
        days = d.get("days_back", 7)
        hist_r = d.get("hist_r")
        if hist_r is not None:
            link = (
                "strong and statistically meaningful"
                if d.get("hist_sig")
                else "present but weak and not statistically reliable"
            )
        else:
            link = "not part of the original study"
        if window_change is not None:
            price = (
                f"the stock has moved {_direction(window_change)} "
                f"{abs(window_change):.2%} over the last {days} trading days"
            )
        else:
            price = "recent price data is unavailable"
        lines.append(
            f"{_company(d)}'s news tone has been {_tone(d['avg_sentiment'])}, "
            f"{price}, and its historical sentiment-price link is {link}."
        )
    return " ".join(lines) + (
        " This is a description of the data only - it does not indicate "
        "which stock is a better investment."
    )
