"""Central configuration for the stock-sentiment-nlp project.

Settings are read from environment variables first, then from a ``.env`` file
in the project root (if present).  Secrets must never be hardcoded in source
files: keep them in ``.env`` (git-ignored) and commit only ``.env.example``.
"""

from __future__ import annotations

import os
from pathlib import Path

# config.py lives in src/, so the repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent

_ENV_FILE = REPO_ROOT / ".env"
_DATA_DIR = REPO_ROOT / "data"
_MODEL_DIR = REPO_ROOT / "models" / "finbert_finetuned" / "final_model"


def _load_dotenv(path: Path = _ENV_FILE) -> None:
    """Minimal .env loader (no third-party dependency)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def env(key: str, default: str | None = None) -> str | None:
    """Return an environment variable with an optional default."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NEWSAPI_KEY = env("NEWSAPI_KEY")
# NewsAPI's free plan only serves articles from the last ~1 month. Requesting
# anything older raises parameterInvalid - callers must clamp the NEWS window
# to this (price history from yfinance is NOT affected).
NEWSAPI_MAX_DAYS = int(env("NEWSAPI_MAX_DAYS", "29"))
OLLAMA_MODEL = env("OLLAMA_MODEL", "llama3.2")

# Sieve scrape API (https://scrape.usesieve.com). SIEVE_API_KEY is a
# server-side-only secret with full account access (no scopes): never send it
# to a browser/mobile bundle, a log, an error report, or git. When it is unset
# every Sieve feature is disabled and the rest of the app behaves exactly as
# before.
SIEVE_API_KEY = env("SIEVE_API_KEY")
SIEVE_BASE_URL = env("SIEVE_BASE_URL", "https://scrape.usesieve.com")
#LOCAL_MODEL_PATH = env("LOCAL_MODEL_PATH") or str(_MODEL_DIR)
LOCAL_MODEL_PATH = "avinashgakusei/finbert-stock-sentiment"

# Canonical paths used by the analysis scripts (all relative to the repo root,
# so scripts work no matter which directory they are launched from).
DATA_DIR = _DATA_DIR
MERGED_SENTIMENT_PRICE = _DATA_DIR / "merged_sentiment_price_v2.csv"
PER_TICKER_CORRELATIONS = _DATA_DIR / "per_ticker_correlation_results_v2.csv"


def sieve_enabled() -> bool:
    """True when a Sieve API key is configured. Callers use this to keep all
    Sieve UI/behaviour dormant when the integration is not set up."""
    return bool(SIEVE_API_KEY)


def require_sieve_key() -> str:
    """Return the Sieve API key or raise with setup instructions."""
    if not SIEVE_API_KEY:
        raise RuntimeError(
            "SIEVE_API_KEY is not set. Run `python src/sieve_login.py` to log in "
            "with a device code, or add a key from Settings -> API keys to .env "
            "at the project root (copy .env.example first)."
        )
    return SIEVE_API_KEY


def require_newsapi_key() -> str:
    """Return the NewsAPI key or raise with setup instructions."""
    if not NEWSAPI_KEY:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. Copy .env.example to .env at the project "
            "root and add your key (get one free at https://newsapi.org)."
        )
    return NEWSAPI_KEY
