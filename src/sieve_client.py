"""Client for the Sieve scrape API (https://scrape.usesieve.com).

This module is the single place that talks to Sieve. It reuses the project's
existing conventions instead of introducing new ones:

* HTTP goes through ``requests`` (already a dependency and already used by the
  other download scripts in ``src/``). The ``Session`` is injectable, which is
  how the unit tests record the HTTP boundary without a second HTTP client.
* The API key comes from :mod:`config` (env var, then the git-ignored ``.env``)
  and is only ever attached as an ``Authorization: Bearer ...`` header. It is
  never logged, returned, or persisted.
* Runs are recorded durably as a JSON file under ``data/`` (the project's
  existing file-based persistence pattern), so a crash resumes polling instead
  of starting a duplicate, credit-spending run.

The client is completely inert unless a key is configured - importing it and
constructing it with an empty key is safe, and nothing here runs at import
time.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urljoin

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://scrape.usesieve.com"

# Values of ``status`` on GET /api/scrapes/<id> that we understand. Anything
# else is treated as a protocol error rather than silently assumed done.
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_REFUSED = "refused"
KNOWN_STATUSES = frozenset({STATUS_RUNNING, STATUS_DONE, STATUS_REFUSED})

# Scrapes take minutes (schema-repair turns also read as "running"), so poll
# slowly: start at 5s and back off towards ~30s. There is no short timeout.
POLL_INITIAL_DELAY = 5.0
POLL_MAX_DELAY = 30.0
POLL_BACKOFF = 1.6

# Generous per-request HTTP timeout (connect, read). Never "short" - a status
# call can be slow while a run is being processed.
DEFAULT_TIMEOUT = (10.0, 60.0)

# POST /api/scrapes has no idempotency key and accepted calls spend credits, so
# only these statuses are safe to retry (no run was created by them).
_RETRYABLE_POST_STATUSES = (429, 500, 502, 503, 504)

COMPLIANCE_MODES = frozenset({"conservative", "regular", "yolo"})
TABLE_SHAPES = frozenset({"long", "wide"})

_MAX_OUTPUT_SCHEMA_BYTES = 32 * 1024

# Transport errors we translate into SieveTransportError. Kept explicit so a
# bug in our own code (TypeError, KeyError, ...) is never swallowed as one.
_TRANSPORT_EXCEPTIONS = (requests.exceptions.RequestException, TimeoutError)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class SieveError(Exception):
    """Base class for every Sieve failure."""

    def __init__(self, message, *, status_code=None, code=None, payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.payload = payload


class SieveTransportError(SieveError):
    """Network failure / timeout talking to Sieve.

    For ``POST /api/scrapes`` this is deliberately NOT retried by the client:
    the first call may have been accepted and spent credits, so the caller must
    recover through the persisted ``session_id`` instead.
    """


class SieveBadRequestError(SieveError):
    """400 - the request itself is wrong. Fix it, do not retry."""


class SieveAuthError(SieveError):
    """401 - key missing/revoked. Tell the user to check SIEVE_API_KEY."""


class SieveCreditsError(SieveError):
    """402 - out of credits. Tell the user (see GET /api/me/credits)."""


class SieveNotFoundError(SieveError):
    """404 - not found, or not owned by this account."""


class SieveConflictError(SieveError):
    """409 - a follow-up turn is already in flight; wait and resend."""


class SieveRateLimitError(SieveError):
    """429 - slow down; honour ``retry_after`` when present."""

    retry_after: Optional[float] = None


class SieveServerError(SieveError):
    """5xx - Sieve-side failure. GETs are retried with exponential backoff."""


class SieveProtocolError(SieveError):
    """Sieve returned something outside the documented contract."""


class SieveTimeoutError(SieveError):
    """A bounded poll gave up while the run was still going."""


_ERROR_BY_STATUS = {
    400: SieveBadRequestError,
    401: SieveAuthError,
    402: SieveCreditsError,
    404: SieveNotFoundError,
    409: SieveConflictError,
    429: SieveRateLimitError,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _default_base_url() -> str:
    try:
        import config

        return config.SIEVE_BASE_URL or DEFAULT_BASE_URL
    except Exception:
        return DEFAULT_BASE_URL


def _json_or_none(response) -> Optional[Mapping[str, Any]]:
    try:
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _error_message(payload: Optional[Mapping[str, Any]], text: str = "") -> str:
    if payload:
        for key in ("error", "detail", "message", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        refusal = payload.get("refusal")
        if isinstance(refusal, Mapping) and refusal.get("reason"):
            return str(refusal["reason"])
    return (text or "").strip()[:300] or "Sieve request failed"


def _error_code(payload: Optional[Mapping[str, Any]]) -> Optional[str]:
    if payload:
        code = payload.get("code")
        if isinstance(code, str) and code:
            return code
    return None


def _retry_after(response) -> Optional[float]:
    raw = None
    try:
        raw = response.headers.get("Retry-After")
    except Exception:
        raw = None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _require_json(response, where: str) -> Mapping[str, Any]:
    payload = _json_or_none(response)
    if payload is None:
        raise SieveProtocolError(
            f"{where} did not return JSON", status_code=response.status_code
        )
    return payload


def _as_bytes(document) -> bytes:
    if isinstance(document, bytes):
        return document
    if isinstance(document, (bytearray, memoryview)):
        return bytes(document)
    if isinstance(document, Path):
        return document.read_bytes()
    if hasattr(document, "read"):
        data = document.read()
        return data if isinstance(data, bytes) else str(data).encode("utf-8")
    return str(document).encode("utf-8")


def _validate_output_schema(output_schema) -> None:
    if isinstance(output_schema, (str, bytes, bytearray)):
        size = len(output_schema if isinstance(output_schema, bytes) else output_schema.encode("utf-8"))
    else:
        try:
            size = len(json.dumps(output_schema).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"output_schema is not JSON-serializable: {exc}") from exc
    if size > _MAX_OUTPUT_SCHEMA_BYTES:
        raise ValueError(
            f"output_schema is {size} bytes; the API limit is {_MAX_OUTPUT_SCHEMA_BYTES}"
        )


def status_of(payload: Mapping[str, Any]) -> str:
    """Return a known scrape status or raise :class:`SieveProtocolError`."""
    status = (payload or {}).get("status")
    if status not in KNOWN_STATUSES:
        raise SieveProtocolError(
            f"unexpected scrape status {status!r}", payload=payload
        )
    return status


def conformance_status(payload: Mapping[str, Any]) -> Optional[str]:
    """``schema_conformance.status``, e.g. 'pass'/'partial'/'fail'.

    'fail' output must never be presented as clean data; use :func:`is_clean`.
    """
    conformance = (payload or {}).get("schema_conformance") or {}
    if isinstance(conformance, Mapping):
        return conformance.get("status")
    return None


def is_clean(payload: Mapping[str, Any]) -> bool:
    """True only when a schema-checked run passed with no missing columns."""
    return conformance_status(payload) == "pass"


def refusal_code(payload: Mapping[str, Any]) -> Optional[str]:
    refusal = (payload or {}).get("refusal") or {}
    if isinstance(refusal, Mapping):
        return refusal.get("code")
    return None


def file_url(base_url: str, entry: Mapping[str, Any]) -> str:
    """Absolute URL for a delivered file. ``files[].url`` is relative."""
    url = str((entry or {}).get("url") or "")
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base_url.rstrip("/") + "/", url.lstrip("/"))


def _default_store_path() -> Path:
    try:
        import config

        return Path(config.DATA_DIR) / "sieve_scrapes.json"
    except Exception:
        return Path("data") / "sieve_scrapes.json"


# ---------------------------------------------------------------------------
# Durable run store
# ---------------------------------------------------------------------------
class ScrapeStore:
    """File-backed record of every Sieve run, keyed by ``session_id``.

    The first thing a successful start does is persist the ``session_id`` (see
    :meth:`SieveClient.start_scrape`), so a crash can resume polling instead of
    launching - and paying for - a duplicate run.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else _default_store_path()
        self._records: dict[str, dict] = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".sieve_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._records, fh, indent=2, default=str)
            os.replace(tmp, self.path)  # atomic: readers never see a partial file
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    def record_start(self, session_id, *, instruction, target_urls=None, extra=None) -> dict:
        record = self._records.get(session_id) or {}
        record.update(
            {
                "session_id": session_id,
                "instruction": instruction,
                "target_urls": list(target_urls) if target_urls else [],
                "status": STATUS_RUNNING,
                "created_at": record.get("created_at") or self._now(),
                "updated_at": self._now(),
            }
        )
        if extra:
            record.update(extra)
        self._records[session_id] = record
        self._save()
        return record

    def update(self, session_id, **fields) -> dict:
        record = self._records.get(session_id) or {"session_id": session_id}
        record.update(fields)
        record["updated_at"] = self._now()
        self._records[session_id] = record
        self._save()
        return record

    def get(self, session_id) -> Optional[dict]:
        return self._records.get(session_id)

    def all(self) -> dict:
        return dict(self._records)

    def list_open(self) -> list:
        """Runs that had not reached a terminal state when last seen."""
        return [
            r
            for r in self._records.values()
            if r.get("status") in (None, STATUS_RUNNING)
        ]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SieveClient:
    """Thin, testable wrapper over the documented Sieve HTTP contract."""

    def __init__(
        self,
        api_key: Optional[str],
        *,
        base_url: Optional[str] = None,
        session=None,
        timeout=DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_retries: int = 3,
        store: Optional[ScrapeStore] = None,
    ):
        self.api_key = api_key or None
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self._session = session if session is not None else requests.Session()
        self.timeout = timeout
        self._sleep = sleep
        self._monotonic = monotonic
        self.max_retries = max_retries
        self.store = store

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, **kwargs) -> "SieveClient":
        """Build a client from the project config (key + base URL)."""
        import config

        kwargs.setdefault("base_url", config.SIEVE_BASE_URL)
        if "store" not in kwargs:
            kwargs["store"] = ScrapeStore()
        return cls(config.SIEVE_API_KEY, **kwargs)

    # -- HTTP plumbing ------------------------------------------------------
    def _headers(self, extra: Optional[Mapping[str, str]] = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key or ''}"}
        if extra:
            headers.update(extra)
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", str(path).lstrip("/"))

    def _send(self, method, path, *, json_body=None, data=None, files=None, timeout=None):
        kwargs: dict[str, Any] = {
            "headers": self._headers(),
            "timeout": timeout or self.timeout,
        }
        if json_body is not None:
            kwargs["json"] = json_body
        if data is not None:
            kwargs["data"] = data
        if files is not None:
            kwargs["files"] = files
        try:
            return self._session.request(method, self._url(path), **kwargs)
        except _TRANSPORT_EXCEPTIONS as exc:
            raise SieveTransportError(str(exc)) from exc

    def _raise_for_status(self, response) -> None:
        status = response.status_code
        if status < 400:
            return
        payload = _json_or_none(response)
        message = _error_message(payload, getattr(response, "text", ""))
        cls = _ERROR_BY_STATUS.get(status)
        if cls is None:
            cls = SieveServerError if status >= 500 else SieveError
        exc = cls(message, status_code=status, code=_error_code(payload), payload=payload)
        if cls is SieveRateLimitError:
            exc.retry_after = _retry_after(response)
        raise exc

    def _backoff(self, attempt: int, response=None) -> float:
        delay = min(1.0 * (2 ** attempt), POLL_MAX_DELAY)
        retry_after = _retry_after(response) if response is not None else None
        return max(delay, retry_after or 0.0)

    def _get_json(self, path: str, *, timeout=None, retries: Optional[int] = None):
        retries = self.max_retries if retries is None else retries
        attempt = 0
        while True:
            try:
                response = self._send("GET", path, timeout=timeout)
            except SieveTransportError:
                if attempt >= retries:
                    raise
                self._sleep(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= retries:
                    self._raise_for_status(response)
                self._sleep(self._backoff(attempt, response))
                attempt += 1
                continue
            self._raise_for_status(response)
            return _require_json(response, f"GET {path}")

    @staticmethod
    def _build_body(
        instruction,
        *,
        compliance_mode,
        target_urls=None,
        fields=None,
        schema=None,
        output_schema=None,
        table_shape=None,
    ) -> dict:
        if not instruction or not str(instruction).strip():
            raise ValueError("instruction is required (plain language)")
        if compliance_mode not in COMPLIANCE_MODES:
            raise ValueError(
                f"compliance_mode must be one of {sorted(COMPLIANCE_MODES)}"
            )
        if table_shape is not None and table_shape not in TABLE_SHAPES:
            raise ValueError(f"table_shape must be one of {sorted(TABLE_SHAPES)}")
        if output_schema is not None:
            _validate_output_schema(output_schema)

        body: dict[str, Any] = {
            "instruction": str(instruction).strip(),
            "compliance_mode": compliance_mode,
        }
        if target_urls:
            body["target_urls"] = list(target_urls)
        if fields:
            body["fields"] = list(fields)
        if schema is not None:
            body["schema"] = schema
        if output_schema is not None:
            body["output_schema"] = output_schema
        if table_shape is not None:
            body["table_shape"] = table_shape
        return body

    @staticmethod
    def _request_kwargs(body: Mapping[str, Any], document, document_name) -> dict:
        if document is None:
            return {"json_body": dict(body)}
        data = {
            key: (json.dumps(value) if isinstance(value, (dict, list)) else str(value))
            for key, value in body.items()
        }
        files = {"file": (document_name or "document", _as_bytes(document))}
        return {"data": data, "files": files}

    # -- runs ---------------------------------------------------------------
    def start_scrape(
        self,
        instruction,
        *,
        target_urls=None,
        fields=None,
        schema=None,
        output_schema=None,
        table_shape=None,
        compliance_mode="regular",
        document=None,
        document_name=None,
        timeout=None,
        store: bool = True,
    ) -> dict:
        """Start a run (POST /api/scrapes). Returns the 202 payload.

        Never retried after a timeout or network error - the first call may
        have succeeded. It *is* retried after 429/5xx, which cannot have
        created a run.
        """
        body = self._build_body(
            instruction,
            compliance_mode=compliance_mode,
            target_urls=target_urls,
            fields=fields,
            schema=schema,
            output_schema=output_schema,
            table_shape=table_shape,
        )
        request_kwargs = self._request_kwargs(body, document, document_name)

        attempt = 0
        while True:
            try:
                response = self._send(
                    "POST", "/api/scrapes", timeout=timeout, **request_kwargs
                )
            except SieveTransportError:
                # Deliberately re-raised without a retry: an accepted POST
                # spends credits, so a blind retry could pay twice.
                raise
            if response.status_code == 202:
                break
            if (
                response.status_code in _RETRYABLE_POST_STATUSES
                and attempt < self.max_retries
            ):
                self._sleep(self._backoff(attempt, response))
                attempt += 1
                continue
            self._raise_for_status(response)

        payload = _require_json(response, "POST /api/scrapes")
        session_id = payload.get("session_id")
        if not session_id:
            raise SieveProtocolError(
                "POST /api/scrapes returned 202 without a session_id", payload=payload
            )
        if store and self.store is not None:
            # Durable before anything else so a crash resumes polling rather
            # than starting a duplicate run.
            self.store.record_start(
                session_id,
                instruction=body["instruction"],
                target_urls=body.get("target_urls"),
                extra={"compliance_mode": body.get("compliance_mode")},
            )
        return payload

    def get_scrape(self, session_id, *, timeout=None, retries: Optional[int] = None) -> dict:
        """GET /api/scrapes/<id>. 5xx/network failures are retried."""
        return dict(self._get_json(f"/api/scrapes/{session_id}", timeout=timeout, retries=retries))

    def poll_scrape(
        self,
        session_id,
        *,
        initial_delay: float = POLL_INITIAL_DELAY,
        max_delay: float = POLL_MAX_DELAY,
        max_wait: Optional[float] = None,
        on_update: Optional[Callable[[dict], None]] = None,
        timeout=None,
    ) -> dict:
        """Poll until a terminal status; returns the final payload.

        ``done`` and ``refused`` are terminal (a refusal means the run never
        started - see :func:`refusal_code`). ``running`` keeps polling, which
        also covers schema-repair turns. Any other status raises.
        """
        delay = float(initial_delay)
        started = self._monotonic()
        if self.store is not None:
            self.store.update(session_id, status=STATUS_RUNNING)
        while True:
            self._sleep(delay)
            payload = self.get_scrape(session_id, timeout=timeout)
            status = status_of(payload)
            if self.store is not None:
                self.store.update(session_id, status=status, payload=payload)
            if on_update is not None:
                on_update(payload)
            if status in (STATUS_DONE, STATUS_REFUSED):
                return payload
            delay = min(delay * POLL_BACKOFF, max_delay)
            if max_wait is not None and (self._monotonic() - started) >= max_wait:
                raise SieveTimeoutError(
                    f"scrape {session_id} is still running after {max_wait}s"
                )

    # -- follow-ups ---------------------------------------------------------
    def current_turns(self, session_id, *, timeout=None) -> int:
        """Number of conversation turns completed so far."""
        payload = self.get_scrape(session_id, timeout=timeout)
        try:
            return int(payload.get("turns") or 0)
        except (TypeError, ValueError):
            return 0

    def send_message(
        self,
        session_id,
        instruction,
        *,
        target_urls=None,
        fields=None,
        schema=None,
        output_schema=None,
        table_shape=None,
        compliance_mode="regular",
        document=None,
        document_name=None,
        timeout=None,
        max_retries: Optional[int] = None,
    ) -> dict:
        """POST a follow-up turn (same body fields as start_scrape).

        409 means a turn is already in flight; the client waits and resends.
        """
        body = self._build_body(
            instruction,
            compliance_mode=compliance_mode,
            target_urls=target_urls,
            fields=fields,
            schema=schema,
            output_schema=output_schema,
            table_shape=table_shape,
        )
        request_kwargs = self._request_kwargs(body, document, document_name)
        retries = self.max_retries if max_retries is None else max_retries

        attempt = 0
        while True:
            try:
                response = self._send(
                    "POST",
                    f"/api/scrapes/{session_id}/messages",
                    timeout=timeout,
                    **request_kwargs,
                )
            except SieveTransportError:
                raise  # a retry could double-post the turn
            if response.status_code == 409 and attempt < retries:
                self._sleep(self._backoff(attempt, response))
                attempt += 1
                continue
            if (
                response.status_code in _RETRYABLE_POST_STATUSES
                and attempt < retries
            ):
                self._sleep(self._backoff(attempt, response))
                attempt += 1
                continue
            self._raise_for_status(response)
            return dict(_require_json(response, f"POST /api/scrapes/{session_id}/messages"))

    def wait_for_turn(
        self,
        session_id,
        turns_before: int,
        *,
        initial_delay: float = POLL_INITIAL_DELAY,
        max_delay: float = POLL_MAX_DELAY,
        max_wait: Optional[float] = None,
        timeout=None,
        on_update: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Poll a follow-up until ``done`` AND ``turns`` has advanced.

        The run is ``done`` at *every* turn boundary, so returning on the first
        ``done`` after posting reads the previous answer. The turn counter must
        move past the value recorded before the POST.
        """
        turns_before = int(turns_before or 0)
        delay = float(initial_delay)
        started = self._monotonic()
        while True:
            self._sleep(delay)
            payload = self.get_scrape(session_id, timeout=timeout)
            status = status_of(payload)
            if self.store is not None:
                self.store.update(session_id, status=status, payload=payload)
            if on_update is not None:
                on_update(payload)
            if status == STATUS_REFUSED:
                return payload
            try:
                turns = int(payload.get("turns") or 0)
            except (TypeError, ValueError):
                turns = 0
            if status == STATUS_DONE and turns > turns_before:
                return payload
            delay = min(delay * POLL_BACKOFF, max_delay)
            if max_wait is not None and (self._monotonic() - started) >= max_wait:
                raise SieveTimeoutError(
                    f"follow-up on {session_id} did not complete within {max_wait}s"
                )

    # -- files --------------------------------------------------------------
    def files(self, payload: Mapping[str, Any]) -> list:
        files = (payload or {}).get("files") or []
        return list(files) if isinstance(files, (list, tuple)) else []

    def file_url(self, entry: Mapping[str, Any]) -> str:
        return file_url(self.base_url, entry)

    def download_file(self, entry: Mapping[str, Any], *, timeout=None) -> tuple:
        """Download one delivered file; returns ``(name, bytes)``.

        ``files[].url`` is relative, so it is prefixed with the base URL, and
        the Bearer header is sent with it.
        """
        name = str((entry or {}).get("name") or "download")
        url = self.file_url(entry)
        try:
            response = self._session.get(
                url, headers=self._headers(), timeout=timeout or self.timeout
            )
        except _TRANSPORT_EXCEPTIONS as exc:
            raise SieveTransportError(str(exc)) from exc
        self._raise_for_status(response)
        return name, response.content

    def save_file(self, entry: Mapping[str, Any], dest_dir) -> Path:
        """Download and atomically write one file under ``dest_dir``."""
        name, content = self.download_file(entry)
        dest = Path(dest_dir) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".sieve_dl_", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return dest

    def download_all(self, payload: Mapping[str, Any], dest_dir) -> list:
        return [self.save_file(entry, dest_dir) for entry in self.files(payload)]

    # -- misc ---------------------------------------------------------------
    def credits(self, *, timeout=None) -> dict:
        """GET /api/me/credits -> plan, limit, used, remaining."""
        return dict(self._get_json("/api/me/credits", timeout=timeout))

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()
