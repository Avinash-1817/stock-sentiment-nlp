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
OLLAMA_MODEL = env("OLLAMA_MODEL", "llama3.2")
#LOCAL_MODEL_PATH = env("LOCAL_MODEL_PATH") or str(_MODEL_DIR)
LOCAL_MODEL_PATH = "https://huggingface.co/avinashgakusei/finbert-stock-sentiment"

# Canonical paths used by the analysis scripts (all relative to the repo root,
# so scripts work no matter which directory they are launched from).
DATA_DIR = _DATA_DIR
MERGED_SENTIMENT_PRICE = _DATA_DIR / "merged_sentiment_price_v2.csv"
PER_TICKER_CORRELATIONS = _DATA_DIR / "per_ticker_correlation_results_v2.csv"


def require_newsapi_key() -> str:
    """Return the NewsAPI key or raise with setup instructions."""
    if not NEWSAPI_KEY:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. Copy .env.example to .env at the project "
            "root and add your key (get one free at https://newsapi.org)."
        )
    return NEWSAPI_KEY
