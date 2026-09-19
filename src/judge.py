"""Evidence-weighted judgment layer for the stock-sentiment project.

What it does
------------
Turns a raw live sentiment reading (average score + article count) into a
directional judgment call - ``BULLISH`` / ``BEARISH`` / ``NEUTRAL`` - plus a
0-1 confidence and a plain-English evidence summary.  The direction comes from
fusing today's news tone with the company's historical same-day
sentiment-return relationship measured in the 2017-2020 study
(``data/per_ticker_correlation_results_v2.csv``).

What it deliberately does NOT do
--------------------------------
This module is a *same-day alignment* judgment: "given today's tone and how
this stock's tone and returns moved together historically, today's tone sits
on the bullish/bearish side of that pattern."  The pooled study data shows a
strong *same-day* link (r = 0.220, p ~ 1e-70) but **no** next-day predictive
edge (r ~ 0.000, p ~ 0.96 across 6,373 ticker-days) - so nothing here should
be read as a forecast of tomorrow's price.  Callers that quote a judgment must
keep that distinction visible.

Confidence formula
------------------
    conf = 0.35*significance + 0.25*link_strength + 0.25*sentiment_extremity
           + 0.15*coverage

where *significance* is 1 when the company's historical correlation is
p < 0.05 (0.3 otherwise), *link_strength* = min(1, |r| / 0.25),
*sentiment_extremity* = min(1, |score| / 0.35), and
*coverage* = min(1, article_count / 10).  When a company has no usable
company-specific record the market-wide prior is used and confidence is
down-weighted by 20%.  Thresholds live below so tests can pin them down.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Call types
# ---------------------------------------------------------------------------
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# ---------------------------------------------------------------------------
# Thresholds / weights (tune via the backtest, not ad hoc)
# ---------------------------------------------------------------------------
P_CUT = 0.05            # historical p-value cutoff for "significant"
NEUTRAL_BAND = 0.15     # |avg sentiment| below this -> no directional call
MIN_ARTICLES = 3        # fewer articles than this -> no directional call
MIN_STUDY_N = 20        # min study observations before trusting a ticker's own r
CORR_REF = 0.25         # |r| at/above this counts as a "strong" historical link
SENT_REF = 0.35         # |sentiment| at/above this counts as "extreme" tone
COV_REF = 10            # article count at/above this counts as "good coverage"
W_SIGNIFICANCE = 0.35
W_LINK = 0.25
W_SENTIMENT = 0.25
W_COVERAGE = 0.15
MARKET_PRIOR_PENALTY = 0.8  # down-weight calls that lean only on the pooled prior

# When a company WAS studied but its own same-day link is NOT significant,
# abstain instead of substituting the market-wide pattern for that company.
# (A genuinely unstudied company may still lean on the market prior.)
WEAK_PATTERN_ABSTAIN = True

STUDY_SOURCE = "per-ticker study (2017-2020)"

# Pooled same-day correlation across all 46 tickers / 6,373 ticker-days
# (computed from data/merged_sentiment_price_v2.csv).
MARKET_PRIOR: Dict[str, object] = {
    "r": 0.2198,
    "p": 1.5e-70,
    "n": 6373,
    "significant": True,
    "source": "market-wide prior (pooled across 46 tickers, 2017-2020)",
}

# Embedded subset of the study used only when the CSV is unavailable so the
# dashboards still function without the data directory.  The CSV (when
# present) always takes precedence via load_study_stats().
EMBEDDED_STUDY: Dict[str, Dict[str, object]] = {
    "RELIANCE.NS": {"r": 0.2326, "p": 0.0000, "n": 400, "significant": True},
    "TCS.NS": {"r": 0.1601, "p": 0.0203, "n": 210, "significant": True},
    "INFY.NS": {"r": 0.1560, "p": 0.0135, "n": 250, "significant": True},
    "HDFCBANK.NS": {"r": 0.2915, "p": 0.0000, "n": 393, "significant": True},
    "ICICIBANK.NS": {"r": 0.3303, "p": 0.0000, "n": 347, "significant": True},
    "SBIN.NS": {"r": 0.2224, "p": 0.0000, "n": 392, "significant": True},
    "MARUTI.NS": {"r": 0.3300, "p": 0.0000, "n": 211, "significant": True},
    "TATASTEEL.NS": {"r": 0.3998, "p": 0.0000, "n": 166, "significant": True},
    "BAJFINANCE.NS": {"r": 0.4461, "p": 0.0000, "n": 152, "significant": True},
    "HINDUNILVR.NS": {"r": 0.0345, "p": 0.6637, "n": 161, "significant": False},
}


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Verdict:
    """Output of a single judgment call."""

    ticker: Optional[str]
    call: str                        # BULLISH / BEARISH / NEUTRAL
    confidence: float                # 0..1
    reason: str                      # directional | weak_sentiment | low_coverage
    basis: str                       # "ticker study" | "market prior"
    sentiment: float                 # avg sentiment score used (-1..1)
    num_articles: int                # article count used
    hist_r: Optional[float]          # historical r used (or market prior r)
    hist_p: Optional[float]
    hist_n: Optional[int]
    expected_sign: int               # +1/-1: historical direction of the pattern

    @property
    def is_directional(self) -> bool:
        return self.call in (BULLISH, BEARISH)

    @property
    def confidence_level(self) -> str:
        return confidence_level(self.confidence)

    def as_dict(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "call": self.call,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "reason": self.reason,
            "basis": self.basis,
            "sentiment": self.sentiment,
            "num_articles": self.num_articles,
            "hist_r": self.hist_r,
            "hist_p": self.hist_p,
            "hist_n": self.hist_n,
            "expected_sign": self.expected_sign,
        }


def confidence_level(confidence: float) -> str:
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def _coerce_sentiment(value: object) -> float:
    try:
        s = float(value)
    except (TypeError, ValueError):
        s = 0.0
    if not math.isfinite(s):
        s = 0.0
    return max(-1.0, min(1.0, s))


def _coerce_articles(value: object) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


def _resolve_hist(hist: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Pick the stats a call should lean on.

    A caller-supplied record is trusted only when it carries enough study
    observations; otherwise we fall back to the market-wide prior so early /
    thin historical windows cannot drive noisy calls.
    """
    if hist is not None and int(hist.get("n") or 0) >= MIN_STUDY_N:
        return hist
    return MARKET_PRIOR


def judge_call(
    sentiment: object,
    num_articles: object,
    hist: Optional[Dict[str, object]] = None,
    ticker: Optional[str] = None,
    abstain_on_weak_pattern: bool = WEAK_PATTERN_ABSTAIN,
) -> Verdict:
    """Judge one sentiment reading.

    Parameters
    ----------
    sentiment : avg sentiment score in [-1, 1] (clamped internally).
    num_articles : number of articles the reading is based on.
    hist : optional dict with keys r, p, n, significant, source describing the
        company's historical same-day sentiment-return correlation.  When
        omitted (or too thin) the market-wide prior is used instead.
    ticker : optional symbol used only for labeling.

    Returns
    -------
    Verdict
    """
    s = _coerce_sentiment(sentiment)
    k = _coerce_articles(num_articles)

    use = _resolve_hist(hist)
    r = float(use["r"])
    p = float(use["p"])
    n = int(use.get("n") or 0)
    significant = bool(use.get("significant"))
    basis = "market prior" if use is MARKET_PRIOR else "ticker study"

    # Direction of the historical pattern: positive sentiment accompanied
    # positive same-day returns for most tickers (r > 0).  If a ticker's
    # correlation is significant but negative, the pattern flips.  When a
    # ticker's own correlation is NOT significant we refuse to guess its sign
    # and default to the aggregate direction (+1).
    if significant:
        expected_sign = 1 if r >= 0 else -1
    else:
        expected_sign = 1

    directional_s = s * expected_sign  # > 0 -> tone sits on the bullish side

    sig_comp = 1.0 if significant else 0.30
    link_comp = min(1.0, abs(r) / CORR_REF)
    sent_comp = min(1.0, abs(s) / SENT_REF)
    cov_comp = min(1.0, k / COV_REF)
    confidence = (
        W_SIGNIFICANCE * sig_comp
        + W_LINK * link_comp
        + W_SENTIMENT * sent_comp
        + W_COVERAGE * cov_comp
    )
    if basis == "market prior":
        confidence *= MARKET_PRIOR_PENALTY
    confidence = round(min(1.0, confidence), 3)

    if k < MIN_ARTICLES:
        call, reason = NEUTRAL, "low_coverage"
    elif abs(s) < NEUTRAL_BAND:
        call, reason = NEUTRAL, "weak_sentiment"
    elif basis == "ticker study" and not significant and abstain_on_weak_pattern:
        # The company's own measured pattern is not reliable - do not guess.
        call, reason = NEUTRAL, "weak_pattern"
    else:
        call, reason = (BULLISH if directional_s >= 0 else BEARISH), "directional"

    return Verdict(
        ticker=ticker,
        call=call,
        confidence=confidence,
        reason=reason,
        basis=basis,
        sentiment=round(s, 3),
        num_articles=k,
        hist_r=r,
        hist_p=p,
        hist_n=n,
        expected_sign=expected_sign,
    )


# ---------------------------------------------------------------------------
# Study stats loading
# ---------------------------------------------------------------------------
def _default_csv_path() -> Optional[Path]:
    rel = Path("data/per_ticker_correlation_results_v2.csv")
    if rel.is_file():
        return rel
    try:
        import config  # noqa: F401 - may not be importable in every context

        candidate = config.DATA_DIR / "per_ticker_correlation_results_v2.csv"
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    return None


def load_study_stats(csv_path: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    """Load per-ticker same-day correlation results.

    Prefers the CSV produced by correlation_analysis_v2.py (all 46 tickers).
    Falls back to the embedded 10-ticker subset when the file is missing so
    the dashboards keep working without the data directory.
    """
    stats: Dict[str, Dict[str, object]] = {
        t: dict(v) for t, v in EMBEDDED_STUDY.items()
    }
    path = Path(csv_path) if csv_path else _default_csv_path()
    if path is None or not path.is_file():
        return stats

    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            try:
                stats[ticker] = {
                    "r": float(row["r"]),
                    "p": float(row["p"]),
                    "n": int(float(row["n"])),
                    "significant": str(row.get("significant", "")).strip().lower()
                    in ("true", "1", "yes"),
                    "source": STUDY_SOURCE,
                }
            except (KeyError, ValueError):
                continue  # malformed row - keep whatever we had
    return stats


def study_stats_for(
    ticker: str, stats: Optional[Dict[str, Dict[str, object]]] = None
) -> Dict[str, object]:
    """Return a ticker's study stats, or the market prior if not studied."""
    table = stats if stats is not None else load_study_stats()
    return table.get(ticker, MARKET_PRIOR)


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------
_CALL_LABEL = {
    BULLISH: "Positive tilt",
    BEARISH: "Negative tilt",
    NEUTRAL: "Neutral / no call",
}
_CALL_EMOJI = {BULLISH: "▲", BEARISH: "▼", NEUTRAL: "◆"}
_REASON_TEXT = {
    "directional": "today's tone is strong enough and the historical pattern is clear enough to make a call",
    "weak_sentiment": "sentiment is too close to neutral to take a side",
    "low_coverage": "too few articles to take a side",
    "weak_pattern": "this company's own historical pattern was not statistically reliable, so no per-company call is made",
}


def call_label(call: str) -> str:
    return _CALL_LABEL.get(call, call)


def one_line(verdict: Verdict, company: Optional[str] = None) -> str:
    name = verdict.ticker or company or "this stock"
    label = call_label(verdict.call)
    if not verdict.is_directional:
        return f"{name}: {label} ({_REASON_TEXT[verdict.reason]})."
    return (
        f"{name}: {label} - {verdict.confidence_level} confidence "
        f"({verdict.confidence:.2f}), based on {verdict.num_articles} articles "
        f"and the {verdict.basis}."
    )


def describe(verdict: Verdict, company: Optional[str] = None) -> List[str]:
    """Plain-English evidence summary for a verdict (one item per line)."""
    name = company or verdict.ticker or "this company"
    lines: List[str] = []

    tone = (
        f"positive ({verdict.sentiment:+.2f})"
        if verdict.sentiment > 0
        else f"negative ({verdict.sentiment:+.2f})"
    )
    lines.append(
        f"News tone today is {tone} across {verdict.num_articles} articles "
        f"- a {_CALL_EMOJI[verdict.call]} **{call_label(verdict.call)}** reading."
    )

    if verdict.basis == "ticker study":
        sig = verdict.hist_p is not None and verdict.hist_p < P_CUT
        lines.append(
            f"In the 2017-2020 study, {name}'s news tone and *same-day* returns "
            f"moved together "
            f"({('significantly' if sig else 'but NOT significantly')}: "
            f"r = {verdict.hist_r:.3f}, p = {verdict.hist_p:.3f}, "
            f"n = {verdict.hist_n} ticker-days)."
        )
        if not sig:
            if verdict.reason == "weak_pattern":
                lines.append(
                    "Because that link was not statistically reliable, the rule "
                    "abstains instead of guessing a direction for this company."
                )
            else:
                lines.append(
                    "Because that link was not statistically reliable, direction "
                    "leans on the market-wide pattern rather than this company alone."
                )
    else:
        lines.append(
            "This company has no strong company-specific record in the study, "
            "so the direction leans on the market-wide pattern (pooled r = 0.22 "
            "across 46 tickers, 2017-2020)."
        )

    if verdict.is_directional:
        lines.append(
            "Today's tone therefore sits on the side of the historical pattern "
            "that tended to accompany positive/negative same-day moves for the "
            "market or this stock."
        )
    else:
        lines.append(f"No directional call: {_REASON_TEXT[verdict.reason]}.")

    lines.append(
        "This is a same-day alignment reading - the study found no usable "
        "signal for the NEXT day's move. Information only, not investment advice."
    )
    return lines


if __name__ == "__main__":
    import sys

    try:  # Windows consoles default to cp1252; don't crash on unicode glyphs
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stats = load_study_stats()
    print(f"Study stats loaded for {len(stats)} tickers\n")
    for ticker in ["TATASTEEL.NS", "HINDUNILVR.NS", "RELIANCE.NS", "PAYTM.NS"]:
        h = study_stats_for(ticker, stats)
        for s in (0.55, -0.3, 0.05, 0.6):
            v = judge_call(s, 9, h, ticker=ticker)
            print(f"{ticker:14s} s={s:+.2f} -> {v.call:8s} conf={v.confidence:.2f} "
                  f"basis={v.basis:12s} reason={v.reason}")
        print()
    v = judge_call(0.55, 9, None, ticker="PAYTM.NS")
    print("\n".join(describe(v, company="Paytm")))
