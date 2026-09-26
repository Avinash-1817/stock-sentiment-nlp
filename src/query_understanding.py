import difflib
import re
import json

from ticker_resolver_v2 import resolve_ticker

# ---------------------------------------------------------------------------
# Regex fallback for the day-window (used if Ollama is unavailable/fails,
# and as a fast pre-check even when it is available)
# ---------------------------------------------------------------------------
_DAY_WORDS = {
    "today": 1, "yesterday": 2, "week": 7, "fortnight": 14,
    "month": 30, "quarter": 90,
}

def effective_news_days(days_back, max_days=None):
    """Clamp a requested news-lookback window to the NewsAPI plan limit.

    NewsAPI's free plan only serves articles from the last ~1 month; asking
    for older ones raises parameterInvalid. Prices (yfinance) are NOT subject
    to this limit - clamp only the NEWS window."""
    if max_days is None:
        from config import NEWSAPI_MAX_DAYS
        max_days = NEWSAPI_MAX_DAYS
    try:
        days = int(days_back)
    except (TypeError, ValueError):
        days = 7
    return max(1, min(days, max_days))


def parse_days_back(user_query, default=7):
    """Extract a requested day-window from free text, e.g. 'last 15 days',
    'past 7 days', 'this week', 'last month'. Falls back to `default`."""
    q = user_query.lower()

    m = re.search(r"(\d+)\s*day", q)
    if m:
        return max(1, min(int(m.group(1)), 90))  # clamp to a sane range

    for word, days in _DAY_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return days

    return default


# ---------------------------------------------------------------------------
# Hallucination guard for the LLM extraction path
# ---------------------------------------------------------------------------
def _matches_in_text(company, text):
    """True if the company fragment actually appears in the user's text -
    as a whole word or a >=5-char substring. 'Paytm' matches 'search for
    paytm'; 'Indian Bank' does NOT match 'show me trends for last 15 days'.
    Case-insensitive on both sides."""
    c = re.sub(r"\s+", " ", str(company).strip().lower())
    t = re.sub(r"\s+", " ", str(text).lower())
    if not c:
        return False
    if re.search(rf"\b{re.escape(c)}\b", t):
        return True
    if len(c) >= 5 and c in t:
        return True
    return False


def _confident_match(fragment):
    """True if the fragment resolves to a ticker AND the match is trustworthy:
    the fragment shares a whole token (or a close token spelling) with the
    matched official company name. This blocks fuzzy long-shots like
    'show me trends' -> MEESHO.NS, which resolve_ticker's loose cutoff (0.55)
    accepts but no human would call a company mention."""
    ticker, matched = resolve_ticker(fragment)
    if not ticker:
        return False
    frag_tokens = [t for t in re.split(r"\s+", str(fragment).lower()) if t]
    name_tokens = re.split(r"\s+", str(matched or "").lower())
    for ft in frag_tokens:
        for nt in name_tokens:
            if ft == nt:
                return True
            if len(ft) >= 4 and difflib.SequenceMatcher(None, ft, nt).ratio() >= 0.75:
                return True  # tolerates near-miss token spellings
    return False


def _ground_llm_companies(llm_companies, user_query, find_companies_in_text_fn):
    """Keep only LLM-extracted companies that are grounded in the query text
    AND resolve to a known ticker. Small local models like llama3.2 sometimes
    INVENT companies that appear nowhere in the question (observed: asked
    'show me trends for last 15 days', it returned companies that later
    resolved to an unrelated listed bank). Ungrounded names are dropped; if
    nothing survives, we re-extract deterministically from the raw text."""
    text = re.sub(r"\s+", " ", user_query.lower())
    grounded = []
    for company in llm_companies:
        if _matches_in_text(company, text) and _confident_match(company):
            grounded.append(str(company).strip())
    if not grounded:
        grounded = [c for c in find_companies_in_text_fn(user_query) if _confident_match(c)]
    return grounded


# ---------------------------------------------------------------------------
# Full structured query understanding via Ollama - one call returns
# companies, day-window, and intent together, so it handles phrasing like
# "show me trends for reliance over the last 15 days" contextually instead
# of separate brittle regex passes for each piece.
# ---------------------------------------------------------------------------
def parse_query_llm(user_query, model_name="llama3.2"):
    """Returns {"companies": [...], "days_back": int, "intent": str} or None
    if Ollama isn't available (caller should fall back to regex + direct
    company matching in that case)."""
    import ollama

    prompt = f"""Extract structured information from this question about Indian stocks.
Return ONLY valid JSON, nothing else, in this exact shape:
{{"companies": ["<company name>", ...], "days_back": <integer or null>, "intent": "<trend|compare|explain|other>"}}

Rules:
- "companies": every company name mentioned (use the name as written by the user).
- "days_back": the number of days of data requested, if mentioned (e.g. "last 15 days" -> 15,
  "past week" -> 7, "this month" -> 30). Use null if no time period is mentioned.
- "intent": "compare" if two+ companies are being weighed against each other, "explain" if
  the user is asking WHY something is happening or asking for reasons/considerations,
  "trend" for a general trend/sentiment request, "other" otherwise.

Question: "{user_query}"
JSON:"""

    try:
        # temperature 0 (greedy decoding) -> same question, same extraction.
        # Default sampling temperature ~0.8 made the model invent different
        # "companies" on repeated runs of the identical question.
        response = ollama.generate(
            model=model_name, prompt=prompt, options={"temperature": 0.0}
        )
        raw = response["response"].strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "companies" not in parsed:
            return None
        return {
            "companies": [str(c).strip() for c in parsed.get("companies", []) if c],
            "days_back": parsed.get("days_back"),
            "intent": parsed.get("intent", "other"),
        }
    except Exception:
        return None


def understand_query(user_query, find_companies_in_text_fn, default_days=7,
                     prev_companies=None):
    """Main entry point: tries full LLM understanding first, falls back to
    regex day-parsing + direct company text matching if Ollama isn't
    available. Always returns a dict with the same shape.

    Companies are ALWAYS grounded in the query text - hallucinated names are
    dropped. When the query names no company at all (e.g. a follow-up like
    'show me trends for last 15 days'), `prev_companies` from the previous
    chat turn is reused so the window applies to what the user was just
    looking at. Set prev_companies=[] to force 'no company' behaviour."""
    llm_result = parse_query_llm(user_query)
    if llm_result is not None and llm_result["companies"]:
        companies = _ground_llm_companies(
            llm_result["companies"], user_query, find_companies_in_text_fn
        )
    else:
        # Fallback: direct/fuzzy company matching + regex day parsing
        companies = [c for c in find_companies_in_text_fn(user_query) if _confident_match(c)]

    # No company in THIS query: carry over the previous turn's companies -
    # but only ones that STILL pass the confidence check, so a stale turn
    # from an older session (or pre-fix garbage like 'show me trends') can
    # never resurrect an unrelated company. A bare greeting or thanks means
    # a new topic, not a follow-up.
    if not companies and prev_companies:
        q = user_query.lower()
        if not re.search(r"\b(hi|hello|hey|thanks|thank you|bye)\b", q):
            companies = [c for c in prev_companies if _confident_match(c)]

    if llm_result is not None and llm_result.get("days_back") is not None:
        try:
            days_back = max(1, min(int(llm_result["days_back"]), 90))
        except (TypeError, ValueError):
            days_back = parse_days_back(user_query, default=default_days)
    else:
        days_back = parse_days_back(user_query, default=default_days)

    q_lower = user_query.lower()
    if "why" in q_lower or "should i" in q_lower or "consider" in q_lower:
        intent = "explain"
    elif len(companies) >= 2:
        intent = "compare"
    else:
        intent = "trend"

    return {"companies": companies, "days_back": days_back, "intent": intent}