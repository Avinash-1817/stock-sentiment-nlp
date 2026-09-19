"""Backtest the judgment rule from judge.py against the historical dataset.

Run from anywhere:  python src/backtest_judgment.py

Data
----
* data/merged_sentiment_price_v2.csv         6,374 ticker-days (2017-2020),
  each with avg_sentiment, num_articles, daily_return (same day) and
  next_day_return.
* data/per_ticker_correlation_results_v2.csv per-ticker same-day r / p / n.

What is evaluated (honestly separated)
--------------------------------------
1. SAME-DAY ALIGNMENT - on days the judge makes a directional call, did the
   stock's *same-day* return move in the direction the tone suggests?  The
   study measured exactly this relationship, so a usable hit-rate is expected
   here.  This is a real-time-usable check: the judge's inputs (tone + past
   pattern) are all known before the session ends.

2. NEXT-DAY FORECAST - did the call predict the *next* trading day's return?
   The pooled study data shows no such edge (r ~ 0.000), so the expected
   result is no better than chance.  The tool must say so plainly.

Two prior variants are compared:
* full-sample : each ticker's r/p from the whole study period (in-sample
  reference, reproduces the study's own numbers).
* walk-forward: each ticker's r/p recomputed only from days *before* the day
  being judged (expanding window, min 20 observations) - the honest version.

Outputs
-------
* data/reports/judgment_backtest_summary.txt
* data/figures/judgment_backtest_alignment.png
* data/figures/judgment_backtest_forecast.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import judge

try:  # Windows consoles default to cp1252; don't crash on unicode glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MIN_PRIOR_N = 20  # must match judge.MIN_STUDY_N usage in walk-forward prior


def load_data() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    merged = pd.read_csv(config.MERGED_SENTIMENT_PRICE)
    merged["date_only"] = pd.to_datetime(merged["date_only"])
    study = judge.load_study_stats(str(config.PER_TICKER_CORRELATIONS))
    return merged, study


def expanding_prior(group: pd.DataFrame) -> pd.DataFrame:
    """Per-row Pearson stats of sentiment vs same-day return using only rows
    STRICTLY BEFORE that row (expanding window, current day excluded)."""
    x = group["avg_sentiment"].to_numpy(dtype=float)
    y = group["daily_return"].to_numpy(dtype=float)
    n = np.arange(len(group))

    sx = np.concatenate([[0.0], np.cumsum(x)])[:-1]   # sums over prior rows
    sy = np.concatenate([[0.0], np.cumsum(y)])[:-1]
    sxx = np.concatenate([[0.0], np.cumsum(x * x)])[:-1]
    syy = np.concatenate([[0.0], np.cumsum(y * y)])[:-1]
    sxy = np.concatenate([[0.0], np.cumsum(x * y)])[:-1]
    npn = n.astype(float)

    denom = np.sqrt((npn * sxx - sx**2) * (npn * syy - sy**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (npn * sxy - sx * sy) / denom
    r = np.where((npn >= 2) & (denom > 0), r, np.nan)

    # t = r * sqrt((n-2)/(1-r^2)), two-sided p from the t distribution
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = r * np.sqrt((npn - 2) / (1.0 - r * r))
        p = 2.0 * stats.t.sf(np.abs(tstat), np.maximum(npn - 2, 1))

    out = pd.DataFrame(
        {"_n_prior": npn, "_r_prior": r, "_p_prior": p}, index=group.index
    )
    return out


def build_evaluations() -> tuple[pd.DataFrame, pd.DataFrame]:
    merged, study = load_data()

    rows_full, rows_wf = [], []
    for ticker, g in merged.sort_values("date_only").groupby("ticker"):
        # full-sample (in-sample reference) prior per ticker
        full_hist = study.get(ticker, judge.MARKET_PRIOR)
        # walk-forward prior (only the past)
        prior = expanding_prior(g)

        for idx, row in g.iterrows():
            sent = row["avg_sentiment"]
            arts = int(row["num_articles"])
            prior_row = prior.loc[idx]
            if prior_row["_n_prior"] >= MIN_PRIOR_N and not np.isnan(
                prior_row["_r_prior"]
            ):
                wf_hist = {
                    "r": float(prior_row["_r_prior"]),
                    "p": float(prior_row["_p_prior"]),
                    "n": int(prior_row["_n_prior"]),
                    "significant": float(prior_row["_p_prior"]) < judge.P_CUT,
                    "source": "ticker study (expanding prior)",
                }
            else:
                wf_hist = None  # too thin -> judge falls back to market prior

            rows_full.append(
                _eval_row(ticker, row, judge.judge_call(sent, arts, full_hist, ticker))
            )
            rows_wf.append(
                _eval_row(ticker, row, judge.judge_call(sent, arts, wf_hist, ticker))
            )

    return pd.DataFrame(rows_full), pd.DataFrame(rows_wf)


def _eval_row(ticker: str, row: pd.Series, v: judge.Verdict) -> dict:
    dir_call = 1 if v.call == judge.BULLISH else (-1 if v.call == judge.BEARISH else 0)
    dr, nr = row["daily_return"], row["next_day_return"]
    s = float(row["avg_sentiment"])

    def hit(return_val):
        if dir_call == 0 or pd.isna(return_val):
            return np.nan
        return 1.0 if dir_call * float(return_val) > 0 else 0.0

    # sentiment-only baseline (ignores all historical correlation)
    if s >= judge.NEUTRAL_BAND and int(row["num_articles"]) >= judge.MIN_ARTICLES:
        dir_naive = 1
    elif s <= -judge.NEUTRAL_BAND and int(row["num_articles"]) >= judge.MIN_ARTICLES:
        dir_naive = -1
    else:
        dir_naive = 0

    def hit_naive(return_val):
        if dir_naive == 0 or pd.isna(return_val):
            return np.nan
        return 1.0 if dir_naive * float(return_val) > 0 else 0.0

    return {
        "ticker": ticker,
        "date": row["date_only"],
        "call": v.call,
        "direction": dir_call,
        "confidence": v.confidence,
        "basis": v.basis,
        "reason": v.reason,
        "sentiment": s,
        "daily_return": dr,
        "next_day_return": nr,
        "alignment_hit": hit(dr),
        "forecast_hit": hit(nr),
        "naive_direction": dir_naive,
        "naive_alignment_hit": hit_naive(dr),
        "naive_forecast_hit": hit_naive(nr),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bucket_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-call-bucket summary over same-day alignment and next-day forecast."""
    rows = []
    for label, sub in df.groupby("call"):
        d = sub.dropna(subset=["daily_return"])
        rows.append(
            {
                "call": label,
                "n": len(sub),
                "alignment_hit_rate": sub["alignment_hit"].mean(),
                "n_align": int(sub["alignment_hit"].notna().sum()),
                "mean_same_day_return": d["daily_return"].mean(),
                "forecast_hit_rate": sub["forecast_hit"].mean(),
                "n_forecast": int(sub["forecast_hit"].notna().sum()),
                "mean_next_day_return": d["next_day_return"].mean(),
                "mean_confidence": sub["confidence"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("call")


def directional_stats(df: pd.DataFrame, hit_col: str, ret_col: str) -> dict:
    """Hit rate (binomial vs 0.5) and mean directional return (t-test vs 0)."""
    sub = df[df["direction"] != 0].dropna(subset=[hit_col, ret_col])
    out: dict = {"n": len(sub)}
    if len(sub) == 0:
        return out
    out["hit_rate"] = sub[hit_col].mean()
    out["hit_p"] = stats.binomtest(int(sub[hit_col].sum()), len(sub), 0.5).pvalue
    rets = sub[ret_col].to_numpy() * sub["direction"].to_numpy()
    out["mean_dir_return"] = rets.mean()
    out["dir_return_t"] = stats.ttest_1samp(rets, 0.0)
    out["mean_confidence"] = sub["confidence"].mean()
    return out


def abstention_stats(df: pd.DataFrame) -> dict:
    """Days the judge abstained on a weak per-company pattern but the naive
    sentiment-only rule WOULD have called: what would naive have gotten?
    This is where the judge's abstention earns (or loses) its keep."""
    sub = df[(df["reason"] == "weak_pattern") & (df["naive_direction"] != 0)]
    out: dict = {"n": len(sub)}
    if len(sub) == 0:
        return out
    out["alignment_hit"] = sub["naive_alignment_hit"].mean()
    out["forecast_hit"] = sub["naive_forecast_hit"].mean()
    return out


def naive_stats(df: pd.DataFrame, hit_col: str, ret_col: str) -> dict:
    sub = df[df["naive_direction"] != 0].dropna(subset=[hit_col, ret_col])
    out: dict = {"n": len(sub)}
    if len(sub) == 0:
        return out
    out["hit_rate"] = sub[hit_col].mean()
    out["hit_p"] = stats.binomtest(int(sub[hit_col].sum()), len(sub), 0.5).pvalue
    rets = sub[ret_col].to_numpy() * sub["naive_direction"].to_numpy()
    out["mean_dir_return"] = rets.mean()
    out["dir_return_t"] = stats.ttest_1samp(rets, 0.0)
    return out


def fmt_t(t) -> str:
    if t is not None and getattr(t, "statistic", None) is not None:
        return f"t={t.statistic:.2f}, p={t.pvalue:.4f}"
    return "-"


# ---------------------------------------------------------------------------
# Report + figures
# ---------------------------------------------------------------------------
def render_figure(df_a: pd.DataFrame, df_b: pd.DataFrame, target: str, out: Path):
    """One grouped bar chart; target in {'alignment', 'forecast'}."""
    hit_col = "alignment_hit" if target == "alignment" else "forecast_hit"
    calls = [judge.BULLISH, judge.BEARISH, judge.NEUTRAL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (df, label) in zip(
        axes, [(df_a, "Full-sample prior"), (df_b, "Walk-forward prior")]
    ):
        means, ses = [], []
        for c in calls:
            sub = df[df["call"] == c][hit_col].dropna()
            m = sub.mean() if len(sub) else np.nan
            means.append(m)
            ses.append(
                np.sqrt(m * (1 - m) / len(sub)) * 1.96 if len(sub) and not np.isnan(m) else 0
            )
        ax.bar(calls, means, yerr=ses, capsize=4,
               color=["#2ca02c", "#d62728", "#7f7f7f"])
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\n(directional calls: {int((df['call'] != judge.NEUTRAL).sum())} of {len(df)})")
        ax.set_ylabel("Hit rate vs realized move")
        ax.text(2.42, 0.51, "50% chance line", fontsize=8)
        for i, m in enumerate(means):
            if not np.isnan(m):
                ax.text(i, m + 0.03, f"{m:.0%}", ha="center", fontsize=9)
    fig.suptitle(
        "Same-day alignment" if target == "alignment" else "Next-day forecast",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Building walk-forward priors and judging 6,374 ticker-days...")
    df_a, df_b = build_evaluations()

    merged, _ = load_data()
    v = merged.dropna(subset=["avg_sentiment", "daily_return"])
    pooled_r, pooled_p = stats.pearsonr(v["avg_sentiment"], v["daily_return"])
    vn = merged.dropna(subset=["avg_sentiment", "next_day_return"])
    pooled_rn, pooled_pn = stats.pearsonr(vn["avg_sentiment"], vn["next_day_return"])

    lines: list[str] = []
    add = lines.append
    add("=" * 72)
    add("JUDGMENT RULE BACKTEST - data/reports/judgment_backtest_summary.txt")
    add("=" * 72)
    add(f"Dataset      : {len(merged)} ticker-days, {merged['ticker'].nunique()} tickers, "
        f"{merged['date_only'].min().date()} -> {merged['date_only'].max().date()}")
    add(f"Pooled same-day r = {pooled_r:.4f} (p={pooled_p:.2e}) | "
        f"pooled NEXT-day r = {pooled_rn:.4f} (p={pooled_pn:.2e})")
    add(f"Judge rule   : call = f(sentiment, article count, historical r/p), "
        f"thresholds in src/judge.py")
    add("")

    for name, df in [("FULL-SAMPLE PRIOR (in-sample reference)", df_a),
                     ("WALK-FORWARD PRIOR (out-of-sample, past data only)", df_b)]:
        add("-" * 72)
        add(name)
        add("-" * 72)
        add(bucket_metrics(df).round(4).to_string())
        add("")
        for tgt, hit_col, ret_col in [
            ("SAME-DAY ALIGNMENT", "alignment_hit", "daily_return"),
            ("NEXT-DAY FORECAST", "forecast_hit", "next_day_return"),
        ]:
            ds = directional_stats(df, hit_col, ret_col)
            ns = naive_stats(df, hit_col, ret_col)
            add(f"[{tgt}]  directional calls (judge): n={ds.get('n')}, "
                f"hit={ds.get('hit_rate', float('nan')):.1%} "
                f"(vs 50%: p={ds.get('hit_p', float('nan')):.4f}), "
                f"mean dir. return={ds.get('mean_dir_return', float('nan')):+.4f} "
                f"[{fmt_t(ds.get('dir_return_t'))}]")
            add(f"[{tgt}]  sentiment-only baseline : n={ns.get('n')}, "
                f"hit={ns.get('hit_rate', float('nan')):.1%} "
                f"(vs 50%: p={ns.get('hit_p', float('nan')):.4f}), "
                f"mean dir. return={ns.get('mean_dir_return', float('nan')):+.4f} "
                f"[{fmt_t(ns.get('dir_return_t'))}]")
            add("")
        abst = abstention_stats(df)
        if abst.get("n"):
            add(f"[JUDGE ABSTENTION] days the judge said NEUTRAL on a weak per-company "
                f"pattern but naive would have called: n={abst['n']}; naive alignment "
                f"hit on those days would have been {abst['alignment_hit']:.1%} "
                f"(forecast {abst['forecast_hit']:.1%}).")
            add("")

    add("=" * 72)
    add("READ THIS FIRST")
    add("=" * 72)
    add("A directional call only makes sense if hit rate > 50% and/or mean")
    add("directional return > 0 (judge better than the sentiment-only baseline).")
    add("Expect the SAME-DAY rows to show a real effect (that is what the study")
    add("measured) and the NEXT-DAY rows to be statistically indistinguishable")
    add("from chance - the tool is an alignment reading, NOT a forecast.")
    add("")

    report_dir = config.DATA_DIR / "reports"
    fig_dir = config.DATA_DIR / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "judgment_backtest_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig1 = fig_dir / "judgment_backtest_alignment.png"
    fig2 = fig_dir / "judgment_backtest_forecast.png"
    render_figure(df_a, df_b, "alignment", fig1)
    render_figure(df_a, df_b, "forecast", fig2)
    print(f"\nReport : {report_path}")
    print(f"Figures: {fig1}, {fig2}")


if __name__ == "__main__":
    main()
