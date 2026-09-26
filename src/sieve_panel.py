"""Streamlit panel for the Sieve scrape integration.

Rendered by ``live_dashboard_chat.py`` only when ``SIEVE_API_KEY`` is set
(``config.sieve_enabled()``); with no key this module draws nothing and the
rest of the app is byte-for-byte unchanged.

Long scrapes are not blocked on: starting a run and checking its status are
separate actions, and every run is already persisted by ``ScrapeStore`` so the
panel can resume an in-flight session after a page reload.
"""

from __future__ import annotations

import json

import streamlit as st

import config
from sieve_client import (
    SieveAuthError,
    SieveClient,
    SieveCreditsError,
    SieveError,
    SieveNotFoundError,
    SieveRateLimitError,
    conformance_status,
    refusal_code,
)

_COMPLIANCE_HELP = (
    "How strictly Sieve respects each site's access rules. 'conservative' is the "
    "most cautious, 'regular' is the default, and 'yolo' relaxes the site-access "
    "policy - only choose it deliberately."
)


@st.cache_resource
def get_sieve_client() -> SieveClient:
    return SieveClient.from_config()


def _downloads_dir():
    return config.DATA_DIR / "sieve_downloads"


def _human_size(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "-"


def _render_errors(exc: Exception) -> None:
    if isinstance(exc, SieveAuthError):
        st.error(
            "Sieve rejected the API key (401). Check SIEVE_API_KEY in `.env` - "
            "it may be missing or revoked."
        )
    elif isinstance(exc, SieveCreditsError):
        st.error(
            "Sieve says you are out of credits (402). The run was not started. "
            "See GET /api/me/credits for your plan, limit and usage."
        )
    elif isinstance(exc, SieveRateLimitError):
        wait = getattr(exc, "retry_after", None)
        st.warning(
            f"Sieve rate-limited the request (429). Wait {wait or 30:.0f}s and try again."
        )
    elif isinstance(exc, SieveNotFoundError):
        st.error("Sieve could not find that run (404). It may belong to another account.")
    elif isinstance(exc, SieveError):
        st.error(f"Sieve request failed: {exc}")
    else:
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}")


def _render_payload(payload: dict) -> None:
    status = payload.get("status")
    st.write(f"**Status:** `{status}`")

    if status == "refused":
        code = refusal_code(payload)
        st.warning(
            "Sieve refused to run this (the run never started)"
            + (f" - reason code `{code}`." if code else ".")
        )
        if code == "quota":
            st.caption("Refusal code `quota` means credit limits were hit.")
        return

    if status != "done":
        return

    summary = payload.get("summary")
    if summary:
        st.write(summary)

    conformance = conformance_status(payload)
    if conformance:
        if conformance == "pass":
            st.success("Schema conformance: pass.")
        elif conformance == "partial":
            st.info(
                "Schema conformance: partial - no violations, but some declared "
                "columns were missing."
            )
        elif conformance == "fail":
            st.error(
                "Schema conformance: fail - the output does NOT match the requested "
                "schema. Do not treat it as clean data."
            )
        else:
            st.caption(f"Schema conformance: {conformance} (nothing checkable).")

    result = payload.get("result")
    if result is not None:
        with st.expander("Validated result"):
            if isinstance(result, list) and result and all(isinstance(r, dict) for r in result):
                st.dataframe(result, use_container_width=True)
            else:
                st.json(result)

    files = payload.get("files") or []
    if files:
        st.write("**Delivered files**")
        for entry in files:
            size = _human_size(entry.get("size"))
            st.write(f"- `{entry.get('name')}` ({size})")
        if st.button("Download delivered files", key="sieve_dl"):
            client = get_sieve_client()
            try:
                paths = client.download_all(payload, _downloads_dir())
            except SieveError as exc:
                _render_errors(exc)
            else:
                st.success("Saved: " + ", ".join(str(p) for p in paths))


def render_sieve_panel() -> None:
    """Draw the Sieve panel when configured; otherwise draw nothing."""
    if not config.sieve_enabled():
        return

    with st.sidebar:
        st.subheader("Sieve scrape")
        st.caption("Scrape a public page on demand. Runs take minutes.")

        instruction = st.text_area(
            "What to extract", placeholder="e.g. Extract the text and author of each quote",
            key="sieve_instruction",
        )
        target_url = st.text_input(
            "Page URL (public http/https)", placeholder="https://quotes.toscrape.com",
            key="sieve_target_url",
        )
        compliance_mode = st.selectbox(
            "Compliance mode", ["regular", "conservative", "yolo"], index=0,
            help=_COMPLIANCE_HELP, key="sieve_compliance",
        )

        if st.button("Start scrape", key="sieve_start"):
            if not instruction.strip():
                st.warning("Describe what to extract first.")
            elif not target_url.strip():
                st.warning("Enter the page URL to scrape.")
            else:
                try:
                    payload = get_sieve_client().start_scrape(
                        instruction,
                        target_urls=[target_url.strip()],
                        compliance_mode=compliance_mode,
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                    _render_errors(exc)
                else:
                    st.session_state["sieve_active"] = payload.get("session_id")
                    st.success(f"Run queued: `{payload.get('session_id')}`")

        session_id = st.session_state.get("sieve_active")
        store = get_sieve_client().store
        open_runs = store.list_open() if store is not None else []

        if open_runs:
            options = [r["session_id"] for r in open_runs]
            picked = st.selectbox("Resume an in-flight run", options, key="sieve_resume")
            if st.button("Continue this run", key="sieve_resume_btn"):
                st.session_state["sieve_active"] = picked
                session_id = picked

        if session_id:
            st.caption(f"Active run: `{session_id}`")
            if st.button("Check status", key="sieve_check"):
                try:
                    payload = get_sieve_client().get_scrape(session_id)
                except Exception as exc:  # noqa: BLE001
                    _render_errors(exc)
                    return
                st.session_state["sieve_last"] = payload

    payload = st.session_state.get("sieve_last")
    if payload:
        with st.expander("Sieve result", expanded=True):
            _render_payload(payload)
