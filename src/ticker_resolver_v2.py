"""
Resolves a free-text company name to an NSE ticker symbol, using the FULL
official NSE equity list (~2000 companies) instead of a small hand-typed map.

Falls back gracefully to a small curated alias list for common shorthand
names (e.g. "HUL" -> Hindustan Unilever) that don't literally appear in the
official "NAME OF COMPANY" field.
"""

import difflib
import os
import pandas as pd

NSE_LIST_PATH = "data/nse_equity_list.csv"

# Curated aliases for common shorthand/abbreviations that won't fuzzy-match
# well against official long company names (e.g. "HUL" vs "HINDUSTAN UNILEVER LTD")
CURATED_ALIASES = {
    "hul": "HINDUNILVR", "reliance": "RELIANCE", "tcs": "TCS",
    "infy": "INFY", "hdfc bank": "HDFCBANK", "icici": "ICICIBANK",
    "sbi": "SBIN", "l&t": "LT", "m&m": "M&M", "airtel": "BHARTIARTL",
    "maruti": "MARUTI", "hero": "HEROMOTOCO", "rec": "RECLTD",
    "hal": "HAL", "bel": "BEL", "lic": "LICI", "dmart": "DMART",
    "paytm": "PAYTM", "nykaa": "NYKAA", "policybazaar": "POLICYBZR",
    "zomato": "ETERNAL", "eternal": "ETERNAL",
}


def _load_nse_list():
    if not os.path.exists(NSE_LIST_PATH):
        raise FileNotFoundError(
            f"NSE equity list not found at {NSE_LIST_PATH}. "
            f"Run download_nse_list.py first."
        )
    df = pd.read_csv(NSE_LIST_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


# Load once at import time and keep in memory (small, ~2000 rows)
try:
    _NSE_DF = _load_nse_list()
    _NAME_TO_SYMBOL = dict(zip(
        _NSE_DF["NAME OF COMPANY"].str.lower().str.strip(),
        _NSE_DF["SYMBOL"].str.strip()
    ))
    _ALL_NAMES = list(_NAME_TO_SYMBOL.keys())
except FileNotFoundError:
    _NSE_DF = None
    _NAME_TO_SYMBOL = {}
    _ALL_NAMES = []


def _fuzzy_sane(name, official_name):
    """Sanity check for the last-resort fuzzy match: the fragment must share
    a whole token (or a close token spelling) with the official company name.
    Without this, difflib at cutoff 0.55 maps conversational fragments like
    'show me trends' onto unrelated companies (observed: MEESHO.NS)."""
    frag_tokens = [t for t in name.split() if t]
    name_tokens = official_name.split()
    for ft in frag_tokens:
        for nt in name_tokens:
            if ft == nt:
                return True
            if len(ft) >= 4 and difflib.SequenceMatcher(None, ft, nt).ratio() >= 0.75:
                return True
    return False


def resolve_ticker(company_name, cutoff=0.55):
    """
    Resolve a free-text company name to an NSE ticker (e.g. 'RELIANCE.NS').
    Returns (ticker, matched_official_name) or (None, None).
    """
    name = company_name.strip().lower()

    # 1. Curated alias shortcut (handles "HUL", "RIL", etc.)
    if name in CURATED_ALIASES:
        return f"{CURATED_ALIASES[name]}.NS", name

    if not _NAME_TO_SYMBOL:
        return None, None  # NSE list not loaded

    # 2. Exact match against official company name
    if name in _NAME_TO_SYMBOL:
        return f"{_NAME_TO_SYMBOL[name]}.NS", name

    # 3. Substring match - handles "reliance" matching
    #    "RELIANCE INDUSTRIES LIMITED"
    for official_name, symbol in _NAME_TO_SYMBOL.items():
        # match if the query is a meaningful chunk of the official name
        first_word = official_name.split()[0] if official_name.split() else ""
        if name == first_word or (len(name) >= 4 and name in official_name):
            return f"{symbol}.NS", official_name

    # 4. Fuzzy match as last resort (typos, partial names) - but only accept
    #    matches that are sane (share a real token with the official name)
    matches = difflib.get_close_matches(name, _ALL_NAMES, n=1, cutoff=cutoff)
    if matches and _fuzzy_sane(name, matches[0]):
        return f"{_NAME_TO_SYMBOL[matches[0]]}.NS", matches[0]

    return None, None


def official_name_for_symbol(ticker):
    """Return the official NSE company name for a ticker like 'VBL.NS',
    e.g. 'Varun Beverages Limited'. Falls back to the bare symbol when the
    NSE list is unavailable. Used for DISPLAY - the ticker stays the key for
    news search and price fetches."""
    symbol = (ticker or "").split(".")[0].strip().upper()
    if _NSE_DF is not None and symbol:
        row = _NSE_DF.loc[_NSE_DF["SYMBOL"].str.strip() == symbol, "NAME OF COMPANY"]
        if not row.empty:
            return str(row.iloc[0]).strip()
    return symbol or str(ticker or "")


def search_companies(query, max_results=5):
    """Return several candidate matches - useful when a query is ambiguous
    and you want to show the user options rather than guessing one."""
    name = query.strip().lower()
    if not _ALL_NAMES:
        return []
    matches = difflib.get_close_matches(name, _ALL_NAMES, n=max_results, cutoff=0.4)
    return [(m, f"{_NAME_TO_SYMBOL[m]}.NS") for m in matches]


import re

# Words that START many official company names but are ordinary nouns in
# questions ("hdfc bank last 3 days", "which bank is doing well"). A bare
# first-word match on one of these is not a company mention unless the
# company's next name word is also right there in the query ("bank of
# baroda"). Without this, step 2 below treats every generic noun as a
# company and the app analyses unrelated firms (observed in prod:
# 'HDFC Bank last 3 days' -> HDFC Bank + Au Small Finance Bank + HDFC AMC).
_GENERIC_FIRST_WORDS = frozenset({
    "bank", "banks", "banking", "finance", "financial", "capital",
    "industries", "industry", "motors", "power", "steel", "energy",
    "petro", "petroleum", "oil", "gas", "pharma", "pharmaceuticals",
    "life", "insurance", "asset", "assets", "realty", "infra",
    "infrastructure", "technologies", "technology", "tech", "systems",
    "solutions", "services", "global", "india", "indian",
    "international", "holdings", "holding", "company", "corporation",
    "limited", "mart", "media", "retail", "health", "healthcare",
    "hospital", "sugar", "cement", "paper", "textiles", "trade",
    "invest", "investments", "consumer", "foods", "hotels", "travel",
    "logistics",
})


def drop_subsumed_fragments(fragments):
    """Drop fragments already covered by a LONGER found fragment.

    'hdfc' and 'bank' say nothing that 'hdfc bank' doesn't already - but
    each resolves to its own unrelated company (HDFC AMC / Au Small Finance
    Bank), so leaving them in turns one-company questions into bogus
    comparisons. Word-boundary containment only, so 'hdfc bank' and
    'hdfc life' never drop each other.
    """
    frags = [str(f).strip() for f in fragments if str(f).strip()]
    kept = []
    for frag in frags:
        covered = any(
            frag != other and re.search(rf"\b{re.escape(frag)}\b", other)
            for other in frags
        )
        if covered or frag in kept:
            continue
        kept.append(frag)
    return kept


def find_companies_in_text(query, cutoff=0.75):
    """Scan a full SENTENCE for company mentions embedded within it - e.g.
    'search for paytm' -> ['paytm']. Do NOT pass a whole sentence to
    resolve_ticker() directly: fuzzy-matching an entire sentence against
    ~2000 company names can match on incidental character overlap and
    return a wrong, unrelated company (this is what caused Paytm queries
    to resolve to an unrelated logistics company previously)."""
    query_lower = query.lower()
    found = []

    # 1. curated aliases as whole-word matches within the sentence
    for alias in CURATED_ALIASES.keys():
        if re.search(rf"\b{re.escape(alias)}\b", query_lower):
            found.append(alias)

    # 2. longest leading run of each official NSE company name present in
    #    the query. 'bank of baroda' matches its full leading trigram, while
    #    the bare 'bank' in 'which bank is doing well' matches nothing
    #    longer than itself - and a lone generic noun ('bank', 'power', ...)
    #    starts hundreds of official names, so it never counts as a mention
    #    on its own. Without the n-gram walk, 'HDFC Bank last 3 days' also
    #    matched every name that merely STARTS with a word in the query
    #    (observed in prod: + Au Small Finance Bank + HDFC AMC).
    if _NAME_TO_SYMBOL:
        seen = set(found)
        for official_name in _ALL_NAMES:
            words = official_name.split()
            if not words:
                continue
            fragment = None
            for n in range(min(len(words), 4), 0, -1):
                candidate = " ".join(words[:n]).lower()
                if re.search(rf"\b{re.escape(candidate)}\b", query_lower):
                    fragment = candidate
                    break
            if fragment is None:
                continue
            if fragment in _GENERIC_FIRST_WORDS and len(words) > 1:
                continue
            if len(fragment.split()) == 1 and len(fragment) < 4:
                continue  # skip noisy one/two/three-letter first words
            if fragment not in seen:
                seen.add(fragment)
                found.append(fragment)

    # 3. only if nothing matched directly, fuzzy-match individual WORDS
    #    (never the whole sentence) as a last resort
    if not found:
        words = re.split(r"[,\?\s]+", query_lower)
        candidates = [w for w in words if len(w) >= 4]
        candidates += [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1) if words[i] and words[i+1]]
        pool = list(CURATED_ALIASES.keys()) + _ALL_NAMES
        for cand in candidates:
            matches = difflib.get_close_matches(cand, pool, n=1, cutoff=cutoff)
            if matches and matches[0] not in found:
                found.append(matches[0])

    # A longer fragment subsumes shorter ones it contains: next to
    # 'hdfc bank', both 'hdfc' and 'bank' are noise that resolve to wrong
    # companies. Word-boundary containment, so distinct names that merely
    # share a prefix ('hdfc bank' vs 'hdfc life') are all kept.
    return drop_subsumed_fragments(found)


if __name__ == "__main__":
    test_queries = ["Reliance", "paytm", "HDFC bank", "policy bazar", "HUL",
                     "Zomato", "Tata Consultancy", "unknown xyz company"]
    for q in test_queries:
        ticker, matched = resolve_ticker(q)
        print(f"{q!r} -> {ticker}  (matched: {matched})")