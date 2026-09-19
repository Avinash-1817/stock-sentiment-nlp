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

    # 4. Fuzzy match as last resort (typos, partial names)
    matches = difflib.get_close_matches(name, _ALL_NAMES, n=1, cutoff=cutoff)
    if matches:
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

    # 2. first word of official NSE company names, as whole-word matches
    #    (only for names >=4 chars, to avoid noisy one/two-letter matches)
    if _NAME_TO_SYMBOL:
        for official_name in _ALL_NAMES:
            first_word = official_name.split()[0] if official_name.split() else ""
            if len(first_word) >= 4 and re.search(rf"\b{re.escape(first_word)}\b", query_lower):
                if first_word not in found:
                    found.append(first_word)

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

    return found


if __name__ == "__main__":
    test_queries = ["Reliance", "paytm", "HDFC bank", "policy bazar", "HUL",
                     "Zomato", "Tata Consultancy", "unknown xyz company"]
    for q in test_queries:
        ticker, matched = resolve_ticker(q)
        print(f"{q!r} -> {ticker}  (matched: {matched})")