"""
Resolves a free-text company name (as extracted from a user query) to an
NSE ticker symbol. Covers your original 49 studied companies PLUS newer
listings that weren't in the historical dataset (Paytm, Nykaa, etc.) so
live analysis works for them even without historical correlation context.
"""

import difflib

COMPANY_TICKER_MAP = {
    # --- Originally studied companies (historical context available) ---
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS", "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "wipro": "WIPRO.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "itc": "ITC.NS",
    "axis bank": "AXISBANK.NS",
    "sun pharma": "SUNPHARMA.NS",
    "airtel": "BHARTIARTL.NS", "bharti airtel": "BHARTIARTL.NS",
    "l&t": "LT.NS", "larsen and toubro": "LT.NS", "larsen & toubro": "LT.NS",
    "hul": "HINDUNILVR.NS", "hindustan unilever": "HINDUNILVR.NS",
    "tata motors": "TATAMOTORS.NS",
    "m&m": "M&M.NS", "mahindra": "M&M.NS", "mahindra and mahindra": "M&M.NS",
    "ultratech": "ULTRACEMCO.NS", "ultratech cement": "ULTRACEMCO.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS",
    "ongc": "ONGC.NS",
    "coal india": "COALINDIA.NS",
    "tata steel": "TATASTEEL.NS",
    "jindal steel": "JINDALSTEL.NS",
    "jsw steel": "JSWSTEEL.NS",
    "hindalco": "HINDALCO.NS",
    "hal": "HAL.NS", "hindustan aeronautics": "HAL.NS",
    "bel": "BEL.NS", "bharat electronics": "BEL.NS",
    "bhel": "BHEL.NS",
    "irctc": "IRCTC.NS",
    "pfc": "PFC.NS", "power finance corporation": "PFC.NS",
    "rec": "RECLTD.NS", "rural electrification corporation": "RECLTD.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "kotak bank": "KOTAKBANK.NS", "kotak mahindra bank": "KOTAKBANK.NS",
    "lic": "LICI.NS", "life insurance corporation": "LICI.NS",
    "sbi life": "SBILIFE.NS",
    "hdfc life": "HDFCLIFE.NS",
    "cipla": "CIPLA.NS",
    "dr reddy": "DRREDDY.NS", "dr reddys": "DRREDDY.NS",
    "apollo hospitals": "APOLLOHOSP.NS",
    "titan": "TITAN.NS",
    "asian paints": "ASIANPAINT.NS",
    "nestle": "NESTLEIND.NS", "nestle india": "NESTLEIND.NS",
    "dmart": "DMART.NS", "avenue supermarts": "DMART.NS",
    "trent": "TRENT.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "eicher motors": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS",

    # --- Post-2020 IPOs (live analysis works, no historical study context) ---
    "paytm": "PAYTM.NS", "one97": "PAYTM.NS",
    "nykaa": "NYKAA.NS", "fsn e-commerce": "NYKAA.NS",
    "policybazaar": "POLICYBZR.NS", "pb fintech": "POLICYBZR.NS",
    "zomato": "ETERNAL.NS", "eternal": "ETERNAL.NS", "blinkit": "ETERNAL.NS",
}


def resolve_ticker(company_name, cutoff=0.6):
    """
    Resolve a free-text company name to an NSE ticker.
    Returns (ticker, matched_name) or (None, None) if no confident match.
    """
    name = company_name.strip().lower()

    # exact match
    if name in COMPANY_TICKER_MAP:
        return COMPANY_TICKER_MAP[name], name

    # substring match (handles "reliance industries limited" etc.)
    for key, ticker in COMPANY_TICKER_MAP.items():
        if key in name or name in key:
            return ticker, key

    # fuzzy match as last resort (handles minor typos)
    matches = difflib.get_close_matches(name, COMPANY_TICKER_MAP.keys(), n=1, cutoff=cutoff)
    if matches:
        return COMPANY_TICKER_MAP[matches[0]], matches[0]

    return None, None


if __name__ == "__main__":
    # quick test
    for q in ["Reliance", "paytm", "HDFC bank", "policy bazar", "unknown company xyz"]:
        ticker, matched = resolve_ticker(q)
        print(f"{q!r} -> {ticker} (matched: {matched})")