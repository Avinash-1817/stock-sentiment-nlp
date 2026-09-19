"""
Live Stock Sentiment Chat Dashboard
Run with: streamlit run live_dashboard_chat.py

Setup:
1. pip install streamlit newsapi-python transformers torch yfinance plotly ollama
2. Ollama installed and running locally (ollama pull llama3.2) - OPTIONAL: the
   app degrades gracefully to template-based narratives if Ollama isn't
   available (e.g. when deployed to Streamlit Community Cloud).
3. Model downloaded locally at models/finbert_finetuned/final_model
4. NewsAPI key in the project .env file (copy .env.example, fill in your own)
"""

import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
import ollama
import json
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from curl_cffi import requests as cfrequests
from datetime import datetime, timedelta
import plotly.graph_objects as go

import config
import judge

from ticker_resolver_v2 import resolve_ticker, find_companies_in_text, official_name_for_symbol

st.set_page_config(page_title="Stock Sentiment Chat", layout="wide")
_session = cfrequests.Session(impersonate="chrome")

# The NewsAPI key lives in the project .env file (copy .env.example, fill in
# your own key) - never hardcode secrets in source files.
NEWSAPI_KEY = config.NEWSAPI_KEY
LOCAL_MODEL_PATH = config.LOCAL_MODEL_PATH
OLLAMA_MODEL = config.OLLAMA_MODEL

# Historical same-day correlation study (46 tickers, 2017-2020), loaded from
# data/per_ticker_correlation_results_v2.csv with an embedded 10-ticker
# fallback so the app works even without the data directory. Entries are
# {"r", "p", "n", "significant", "source"}.
HISTORICAL_RESULTS = judge.load_study_stats()

# ---------------------------------------------------------------------------
# Model + helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
SCORE_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}


def score_sentiment(texts):
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).numpy()
    return [LABEL_MAP[p] for p in preds]


def extract_companies_direct(user_query):
    """Extract company mentions FROM the sentence (not the whole sentence
    itself). Passing an entire sentence into resolve_ticker() for fuzzy
    matching against ~2000 company names caused wrong matches (e.g. 'search
    for paytm' fuzzy-matching to an unrelated logistics company). This scans
    the sentence for actual company-name substrings first."""
    return find_companies_in_text(user_query)


def extract_companies(user_query, model_name=OLLAMA_MODEL, debug=False):
    direct_matches = extract_companies_direct(user_query)
    if direct_matches:
        if debug:
            st.caption(f" Debug: matched directly -> {direct_matches}")
        return direct_matches

    prompt = f"""You extract company names from a user's question about Indian stocks.
Return ONLY a JSON array of company names mentioned, nothing else.

Question: "{user_query}"
Answer:"""
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        raw = response["response"].strip().replace("```json", "").replace("```", "").strip()
        if debug:
            st.caption(f" Debug: raw Ollama output -> {raw!r}")
        companies = json.loads(raw)
        if isinstance(companies, list):
            return [str(c).strip() for c in companies if c]
    except Exception as e:
        if debug:
            st.caption(f"Debug: Ollama extraction failed -> {e}")
    return []


@st.cache_data(ttl=3600)
def fetch_recent_news(company_name, days_back=7):
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY is not set - copy .env.example to .env at the project root and add your key")
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    response = newsapi.get_everything(q=company_name, from_param=from_date,
                                       language="en", sort_by="publishedAt", page_size=30)
    articles = response.get("articles", [])
    rows = []
    for a in articles:
        text = (a.get("title") or "") + ". " + (a.get("description") or "")
        rows.append({"title": a.get("title"), "text": text, "publishedAt": a.get("publishedAt"),
                     "source": a.get("source", {}).get("name")})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_recent_prices(ticker, days_back=30):
    try:
        data = yf.download(ticker, period=f"{days_back}d", threads=False, progress=False)
    except Exception as e:
        st.error(f"yfinance error for {ticker}: {type(e).__name__}: {e}")
        return pd.DataFrame()
    if data.empty:
        st.warning(f"yfinance returned empty data for {ticker} (no exception raised)")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    return data


@st.cache_data
def load_track_record():
    """Loads the per-ticker backtest track record (see
    src/per_ticker_track_record.py). Returns {} if the file doesn't exist
    yet, so the app degrades gracefully rather than crashing."""
    try:
        df = pd.read_csv("data/per_ticker_track_record.csv")
        return df.set_index("ticker").to_dict("index")
    except FileNotFoundError:
        return {}

TRACK_RECORD = load_track_record()


def generate_narrative_fallback(d):
    """Template-based narrative, no LLM needed - used when Ollama isn't
    available (e.g. on Streamlit Community Cloud, which can't run background
    services like Ollama). Same rules as the Ollama version: no buy/sell
    language, no predictions, clear about which time period each number
    refers to."""
    tone = ("mostly positive" if d['avg_sentiment'] > 0.15
            else "mostly negative" if d['avg_sentiment'] < -0.15
            else "mixed/neutral")
    days = d.get("days_back", 7)
    window_change = d.get("pct_change_window")
    pct_change = d.get("pct_change")

    parts = [
        f"{d['article_count']} news articles about {d['company'].title()} were found in the last {days} days, "
        f"with a tone that's been {tone}."
    ]

    if window_change is not None:
        direction = "up" if window_change > 0 else "down" if window_change < 0 else "flat"
        parts.append(f"Over that same {days}-day window, the stock moved {direction} {abs(window_change):.2%}.")

    if pct_change is not None:
        direction = "up" if pct_change > 0 else "down" if pct_change < 0 else "flat"
        parts.append(f"On the most recent trading day alone, it moved {direction} {abs(pct_change):.2%}.")

    if d["hist_r"] is not None:
        if d["hist_sig"]:
            parts.append(
                "Historically (2017-2020 study), this stock's price tended to move in the same "
                "direction as news sentiment on the same day - a pattern strong enough to be "
                "statistically meaningful."
            )
        else:
            parts.append(
                "Historically (2017-2020 study), this pattern was weak for this stock and not "
                "statistically reliable."
            )
    else:
        parts.append(
            "This stock wasn't part of the original historical study (it's a newer listing), "
            "so there's no historical pattern to reference."
        )

    parts.append("This is current information only, not a forecast.")
    return " ".join(parts)


def generate_narrative(d, model_name=OLLAMA_MODEL):
    """Turn a company's raw metrics into a plain-English, market-commentary
    style explanation - no jargon like 'Pearson r' or 'sentiment score'.
    Tries Ollama first; falls back to a template-based version if Ollama
    isn't available (e.g. when deployed on Streamlit Community Cloud, which
    can't run a background LLM service)."""
    if d.get("error"):
        return None

    hist_line = (
        f"Historically (2017-2020 study), this stock's price has tended to move in the "
        f"same direction as news sentiment on the same day "
        f"({'a pattern strong enough to be statistically meaningful' if d['hist_sig'] else 'though this pattern was weak and not statistically reliable'})."
        if d["hist_r"] is not None else
        "This stock wasn't part of the original historical study (it's a newer listing), so there's no historical pattern to reference."
    )

    window_change = d.get("pct_change_window")
    days = d.get("days_back", 7)

    prompt = f"""You are a financial market commentator explaining stock news sentiment to an
ordinary investor who is NOT technical - no jargon like "sentiment score", "correlation",
"Pearson r", "p-value", or "basis points". Write like a knowledgeable friend explaining
what's going on, in 3-4 plain sentences.

Data for {d['company']} ({d['ticker']}):
- {d['article_count']} news articles in the last {days} days
- News tone has been: {"mostly positive" if d['avg_sentiment'] > 0.15 else "mostly negative" if d['avg_sentiment'] < -0.15 else "mixed/neutral"}
- Price change over the last {days} trading days: {f"{window_change:+.2%}" if window_change is not None else "not available"}
- Most recent single-day move: {f"{d['pct_change']:+.2%}" if d['pct_change'] is not None else "not available"}
- {hist_line}

Rules:
- Do NOT say "buy", "sell", "invest", or predict future price direction.
- Do NOT claim the news caused the price move - just describe both.
- Be clear about WHICH time period each number refers to (last {days} days vs. most recent day).
- Keep it to 3-4 short sentences, plain conversational English.
- End with one honest line noting this is just current info, not a forecast.

Write the summary:"""

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        text = response["response"].strip()
        if not text:
            raise ValueError("Empty response from Ollama")
        return text
    except Exception:
        # Ollama not installed/running (e.g. on Streamlit Cloud) - degrade
        # gracefully instead of crashing the whole app.
        return generate_narrative_fallback(d)


def generate_comparison_narrative_fallback(companies_data):
    """Template-based comparison, no LLM needed - same fallback pattern as
    generate_narrative_fallback above."""
    lines = []
    for d in companies_data:
        tone = "positive" if d['avg_sentiment'] > 0.15 else "negative" if d['avg_sentiment'] < -0.15 else "mixed"
        lines.append(f"{d['company'].title()}: news tone {tone}, historical sentiment-price link "
                      f"{'meaningful' if d['hist_sig'] else 'weak or none' if d['hist_r'] is not None else 'not studied'}.")
    return " ".join(lines) + " This is descriptive only — it does not indicate which stock is a better investment."


def generate_comparison_narrative(companies_data, model_name=OLLAMA_MODEL):
    """Plain-English comparison across 2+ companies - explains differences,
    never declares a winner. Tries Ollama first; falls back to a
    template-based version if Ollama isn't available."""
    summaries = []
    for d in companies_data:
        days = d.get("days_back", 7)
        tone = "mostly positive" if d['avg_sentiment'] > 0.15 else "mostly negative" if d['avg_sentiment'] < -0.15 else "mixed"
        window_chg = d.get("pct_change_window")
        summaries.append(
            f"{d['company']}: news tone {tone}, price change over last {days} trading days: "
            f"{f'{window_chg:+.2%}' if window_chg is not None else 'unknown'}, "
            f"historical sentiment-price link: {'meaningful' if d['hist_sig'] else 'weak/none' if d['hist_r'] is not None else 'not studied'}."
        )

    prompt = f"""You are a financial commentator. Compare these companies for an ordinary
investor in plain English, 4-5 sentences total. Point out genuine differences you see
in the data. Do NOT say one is a better investment, do NOT predict which will "earn more"
or perform better, and do NOT recommend buying either. If the data doesn't support a
clear comparison, say so honestly.

{chr(10).join(summaries)}

Write the comparison:"""

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        text = response["response"].strip()
        if not text:
            raise ValueError("Empty response from Ollama")
        return text
    except Exception:
        return generate_comparison_narrative_fallback(companies_data)


def render_comparison_table(companies_data):
    st.markdown("### How they compare")
    with st.spinner("Writing summary..."):
        narrative = generate_comparison_narrative(companies_data)
    st.write(narrative)
    with st.expander("See the raw numbers behind this"):
        rows = []
        for d in companies_data:
            days = d.get("days_back", 7)
            rows.append({
                "Company": d["company"], "Ticker": d["ticker"],
                f"Articles ({days}d)": d["article_count"],
                "News tone": f"{d['avg_sentiment']:+.3f}",
                "Latest Close": f"₹{d['latest_close']:,.2f}" if d["latest_close"] else "N/A",
                f"Change (last {days}d)": f"{d['pct_change_window']:+.2%}" if d.get("pct_change_window") is not None else "N/A",
                "Change (most recent day)": f"{d['pct_change']:+.2%}" if d["pct_change"] is not None else "N/A",
                "Historical link strength": f"{d['hist_r']:.3f}" if d["hist_r"] is not None else "Not studied",
            })
        st.table(pd.DataFrame(rows))


def analyze_company(company_query, days_back=7):
    """Runs the analysis and returns a data dict WITHOUT rendering anything -
    rendering is separated so we can show single-company detail or feed into
    a comparison table."""
    ticker, matched = resolve_ticker(company_query)
    if not ticker:
        return {"company": company_query, "ticker": None, "error": True}

    # For DISPLAY, always use the official NSE company name (e.g. a query for
    # "varun" becomes "Varun Beverages Limited"). The raw fragment is kept
    # only for news search, since NewsAPI matches better on short names.
    official_name = official_name_for_symbol(ticker)

    news_df = fetch_recent_news(company_query, days_back)
    if len(news_df) == 0:
        return {"company": official_name, "ticker": ticker, "error": True, "no_news": True}

    news_df["sentiment"] = score_sentiment(news_df["text"].fillna("").tolist())
    news_df["sentiment_score"] = news_df["sentiment"].map(SCORE_MAP)
    avg_sentiment = news_df["sentiment_score"].mean()

    price_df = fetch_recent_prices(ticker, days_back + 20)
    latest_close, pct_change_1day, pct_change_window = None, None, None
    if len(price_df) >= 2:
        close_col = price_df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.squeeze()

        # Drop any NaN rows (yfinance sometimes returns an incomplete most-recent
        # row right after market close) so we never silently compute with NaN.
        close_col = pd.to_numeric(close_col, errors="coerce").dropna()

        if len(close_col) >= 2:
            latest_close = float(close_col.iloc[-1])
            prev_close = float(close_col.iloc[-2])
            pct_change_1day = (latest_close - prev_close) / prev_close

            if len(close_col) > days_back:
                window_start_close = float(close_col.iloc[-(days_back + 1)])
                pct_change_window = (latest_close - window_start_close) / window_start_close
            elif len(close_col) >= 2:
                # not enough history for the full window - use whatever we have
                window_start_close = float(close_col.iloc[0])
                pct_change_window = (latest_close - window_start_close) / window_start_close

    hist = HISTORICAL_RESULTS.get(ticker)
    hist_r = float(hist["r"]) if hist else None
    hist_sig = bool(hist["significant"]) if hist else None

    # Evidence-weighted judgment call on this reading (same-day alignment).
    verdict = judge.judge_call(
        avg_sentiment, len(news_df),
        HISTORICAL_RESULTS.get(ticker), ticker=ticker,
    )

    return {
        "company": official_name, "ticker": ticker, "error": False,
        "news_df": news_df, "price_df": price_df,
        "avg_sentiment": avg_sentiment, "article_count": len(news_df),
        "latest_close": latest_close,
        "pct_change": pct_change_1day,          # most recent single day
        "pct_change_window": pct_change_window,  # over the same N-day window as the news
        "days_back": days_back,
        "hist_r": hist_r, "hist_sig": hist_sig,
        "verdict": verdict,
    }


def render_verdict(verdict, company_name, key_suffix=""):
    """Render one judgment call with its evidence (used in company detail)."""
    if verdict is None:
        return
    label = judge.call_label(verdict.call)
    conf_text = f"confidence {verdict.confidence:.0%} ({verdict.confidence_level})"
    if verdict.call == judge.BULLISH:
        st.success(f"**Judge call: {label} (\u25b2 bullish-side tone)** \u2014 {conf_text}")
    elif verdict.call == judge.BEARISH:
        st.warning(f"**Judge call: {label} (\u25bc bearish-side tone)** \u2014 {conf_text}")
    else:
        st.info(f"**Judge call: {label}** \u2014 {conf_text}")
    with st.expander(
        f"Why this call \u2014 {company_name}", key=f"verdict_{key_suffix}_{company_name}"
    ):
        for line in judge.describe(verdict, company=company_name):
            st.write(line)


def render_track_record(ticker, company_name):
    """Shows how often the judge's same-day directional calls have
    historically aligned with the actual price move, FOR THIS SPECIFIC
    company - not just the overall 68.3% figure."""
    if not TRACK_RECORD:
        st.caption("📊 Track record data not loaded (data/per_ticker_track_record.csv not found or empty).")
        return

    record = TRACK_RECORD.get(ticker)
    if record is None:
        st.caption(f"📊 No track record entry found for {ticker}.")
        return

    if not record.get("n_calls") or record["n_calls"] == 0:
        st.caption(
            f"📊 Track record: no historical calls to show for {company_name} — "
            f"{record.get('note', 'insufficient historical data')}."
        )
        return

    n_calls = int(record["n_calls"])
    hit_rate = record["hit_rate"]
    n_aligned = int(record["n_aligned"])

    if hit_rate >= 0.6:
        st.success(
            f"📊 **Track record for {company_name}:** in our 2017–2020 backtest, when the "
            f"same-day signal made a call on this stock, it aligned with the actual move "
            f"**{hit_rate:.1%}** of the time ({n_aligned}/{n_calls} instances)."
        )
    elif hit_rate >= 0.5:
        st.info(
            f"📊 **Track record for {company_name}:** historically aligned **{hit_rate:.1%}** "
            f"of the time ({n_aligned}/{n_calls} instances) — modest, close to a coin flip."
        )
    else:
        st.warning(
            f"📊 **Track record for {company_name}:** historically aligned only **{hit_rate:.1%}** "
            f"of the time ({n_aligned}/{n_calls} instances) — treat any call here with extra caution."
        )

    st.caption(
        "This reflects same-day alignment only (2017–2020 study) — it is a historical "
        "track record, not a guarantee of future accuracy."
    )


def render_company_detail(d, key_suffix=""):
    """Renders the full chart/metric view for ONE company's data dict."""
    if d.get("error"):
        if d.get("no_news"):
            st.warning(f"No recent news found for **{d['company']}**.")
        else:
            st.error(f"Couldn't identify a known NSE company from '{d['company']}'.")
        return

    ticker = d["ticker"]
    st.markdown(f"### {d['company'].title()} ({ticker})")

    # The tool's judgment (same-day alignment reading, not a forecast)
    render_verdict(d.get("verdict"), d["company"], key_suffix=key_suffix)
    render_track_record(ticker, d["company"])

    with st.spinner("Writing summary..."):
        narrative = generate_narrative(d)
    st.write(narrative)

    c1, c2 = st.columns(2)
    with c1:
        counts = d["news_df"]["sentiment"].value_counts()
        fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values,
                        marker=dict(colors=["#2ca02c" if l == "Positive" else "#d62728" if l == "Negative" else "#7f7f7f"
                                            for l in counts.index]))])
        fig.update_layout(height=280, margin=dict(t=20, b=20), title="Recent news tone")
        st.plotly_chart(fig, use_container_width=True, key=f"pie_{ticker}_{key_suffix}")
    with c2:
        price_df = d["price_df"]
        if len(price_df) > 0:
            price_df = price_df.copy()
            price_df["Date"] = pd.to_datetime(price_df["Date"])
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=price_df["Date"], y=price_df["Close"], mode='lines'))
            fig2.update_layout(height=280, margin=dict(t=40, b=20), title=f"{ticker} price, last 30 days")
            st.plotly_chart(fig2, use_container_width=True, key=f"price_{ticker}_{key_suffix}")

    with st.expander("See the numbers behind this summary"):
        days = d.get("days_back", 7)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles analyzed", d["article_count"])
        m2.metric("News tone score", f"{d['avg_sentiment']:+.3f}")
        if d["latest_close"]:
            window_chg = d.get("pct_change_window")
            m3.metric("Latest close", f"₹{d['latest_close']:,.2f}",
                      f"{window_chg:+.2%} over last {days}d" if window_chg is not None else None)
        else:
            m3.metric("Latest close", "Not available")
        if d["pct_change"] is not None:
            m4.metric("Most recent day's move", f"{d['pct_change']:+.2%}")
        else:
            m4.metric("Most recent day's move", "Not available")
        st.dataframe(d["news_df"][["publishedAt", "source", "title", "sentiment"]],
                     use_container_width=True, key=f"table_{ticker}_{key_suffix}")


# ---------------------------------------------------------------------------
# Chat-style UI with PERSISTENT session history
# ---------------------------------------------------------------------------
st.title("Stock Sentiment Chat")
st.caption("Ask about any Indian stock in plain language — powered by a locally fine-tuned FinBERT model")

st.info(
    "**This tool shows sentiment and alignment information, not investment advice or forecasts.** "
    "It reports current news sentiment, recent price movement, and a same-day **judgment call** based "
    "on how strongly sentiment correlated with same-day returns for that company in a 2017–2020 "
    "historical study. That study found no reliable edge for the next day's move, so no prediction is "
    "made — and no 'better' or 'more likely to earn' pick is given when comparing companies."
)

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of {"query": str, "results": [dict, ...]}

debug_mode = st.checkbox("Show debug info (temporary, for troubleshooting)", value=False)

user_query = st.chat_input("e.g. show me trends for reliance and paytm")

if user_query:
    with st.spinner("Understanding your question..."):
        companies = extract_companies(user_query, debug=debug_mode)

    results = []
    if companies:
        for company in companies:
            with st.spinner(f"Analyzing {company}..."):
                results.append(analyze_company(company))

    st.session_state.chat_log.append({"query": user_query, "companies": companies, "results": results})

# --- Render the FULL session history, oldest to newest ---
for turn_idx, turn in enumerate(st.session_state.chat_log):
    st.chat_message("user").write(turn["query"])
    with st.chat_message("assistant"):
        if not turn["companies"]:
            st.write("I couldn't identify any specific companies in your question. "
                     "Try naming a company directly, e.g. 'Reliance' or 'Paytm'.")
        else:
            # Resolve each extracted fragment to its official NSE name so the
            # history line matches what the detail cards show (e.g. a query for
            # "varun" displays "Varun Beverages Limited", not "varun").
            resolved_labels = [
                official_name_for_symbol(resolve_ticker(c)[0] or "") or c.title()
                for c in turn["companies"]
            ]
            st.write(f"Analyzing: {', '.join(resolved_labels)}")

            valid_results = [r for r in turn["results"] if not r.get("error")]

            # show comparison table if 2+ companies successfully resolved
            if len(valid_results) >= 2:
                render_comparison_table(valid_results)
                st.divider()

            for i, d in enumerate(turn["results"]):
                render_company_detail(d, key_suffix=f"{turn_idx}_{i}")
                st.divider()

if st.session_state.chat_log:
    if st.button("Clear chat history"):
        st.session_state.chat_log = []
        st.rerun()