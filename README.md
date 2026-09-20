# stock-sentiment-nlp

NLP-based financial sentiment analysis correlated with stock price movement — major project.

Fine-tuned **FinBERT** scores Indian financial news from NewsAPI, and the resulting sentiment is
correlated against NSE price moves (2017–2020 study, 46 companies). A lightweight **judgment
layer** (`src/judge.py`) turns a live reading into a directional call with a confidence score and
plain-English evidence, and a **backtester** (`src/backtest_judgment.py`) verifies the rule honestly.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/judge.py` | Pure judgment layer — sentiment reading + historical stats → `BULLISH`/`BEARISH`/`NEUTRAL` call with 0–1 confidence and evidence text. No heavy dependencies. |
| `src/backtest_judgment.py` | Backtests the judge over `data/merged_sentiment_price_v2.csv` (6,374 ticker-days), separating **same-day alignment** from **next-day forecasting**. |
| `src/config.py` | Central settings. Reads env vars, then `.env` at the repo root. **Never hardcode secrets in source.** |
| `src/Live_dashboard.py` | Streamlit dashboard with a per-company "Judge call" panel (`streamlit run src/Live_dashboard.py`). |
| `src/live_dashboard_chat.py` | Chat version with an Ollama narrative (`streamlit run src/live_dashboard_chat.py`). |
| `src/live_sentiment_pipeline.py` | Library/CLI pipeline — fetch news → score → judge call. |
| `src/correlation_analysis_v2.py` | Recomputes the per-ticker same-day correlation study. |
| `data/reports/judgment_backtest_summary.txt` | Latest backtest output. |
| `tests/` | Stdlib `unittest` tests for the judge and the backtest metrics. |

## Setup

1. Install dependencies: `pip install -r requirements.txt` plus
   `streamlit newsapi-python yfinance plotly` (and `ollama` for the chat app).
2. Copy `.env.example` to `.env` and fill in your **own** NewsAPI key
   (the key once hardcoded in `src/` was exposed — rotate it).
3. The fine-tuned model is loaded from `models/finbert_finetuned/final_model`
   (override with `LOCAL_MODEL_PATH` in `.env`).

## The judgment layer (what "making a call" means here)

`judge_call(sentiment, num_articles, hist)` fuses today's news tone with the company's historical
same-day sentiment–return correlation from the study:

- **Direction** — a significant positive per-ticker `r` means positive tone sits on the bullish
  side of that company's historical pattern (negative `r` flips it).
- **Abstention is a feature** — the judge stays `NEUTRAL` when there are too few articles, the tone
  is near neutral, or the company **was** studied but its own pattern was **not** statistically
  significant. It refuses to guess on a company whose record doesn't support a guess.
- **Market prior fallback** — genuinely unstudied companies (e.g. newer listings) lean on the pooled
  market pattern (r ≈ 0.22), with confidence down-weighted 20%.
- **Confidence** = 0.35·significance + 0.25·link strength + 0.25·sentiment extremity + 0.15·coverage.

**Honest framing:** the study measured a *same-day* relationship. It found **no usable signal for
the next day's move** (pooled next-day r ≈ 0.000, p ≈ 0.96), so the call is a same-day alignment
reading, not a forecast — the UI and `judge.describe()` say so explicitly.

## Backtest results (walk-forward, past data only)

```
Same-day alignment : 68.3% hit on directional calls, mean directional same-day return +1.77%
                     (t = 11.29, p < 0.001)          <- the study's relationship, confirmed
Next-day forecast  : 48.7% hit (p = 0.55)            <- no edge, as expected - NOT a predictor
Judge abstention   : on the 195 days the judge stayed NEUTRAL where a naive sentiment rule would
                     have called, naive would have been right only 49.7% (chance) -> abstaining
                     was the correct call
```

Re-run it any time:

```bash
python src/backtest_judgment.py        # prints + writes data/reports + figures
python -m unittest discover -s tests   # 23 unit tests
python src/judge.py                    # self-check demo of the judge
```

## Disclaimer

Educational/research project. Sentiment and the judgment call are informational — not investment
advice and not price forecasts.
![Tests](https://github.com/Avinash-1817/stock-sentiment-nlp/actions/workflows/tests.yml/badge.svg)