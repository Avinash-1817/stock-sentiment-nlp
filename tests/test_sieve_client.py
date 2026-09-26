"""Unit tests for src/sieve_client.py (stdlib unittest).

Run from the repo root:  python -m unittest discover -s tests -v

The HTTP boundary is faked (a recorded response per call); every line of
request building, status handling, polling and error mapping under test is the
real client code. No network access and no API key are needed.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sieve_client as sc  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles at the HTTP boundary
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeSession:
    """Replays queued responses/exceptions and records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._responses:
            raise AssertionError(f"unexpected extra {method} {url}")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def request(self, method, url, **kwargs):
        return self._next(method, url, kwargs)

    def get(self, url, **kwargs):
        return self._next("GET", url, kwargs)

    def close(self):
        pass


class FakeClock:
    """Deterministic sleep + monotonic so poll timing is testable."""

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def sleep(self, delay):
        self.slept.append(delay)
        self.t += delay

    def monotonic(self):
        return self.t


def make_client(responses, **kwargs):
    clock = FakeClock()
    session = FakeSession(responses)
    store = kwargs.pop("store", None)
    client = sc.SieveClient(
        "dc_sk_test",
        base_url="https://scrape.example.com",
        session=session,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )
    client.store = store
    return client, session, clock


def running(turns=0):
    return FakeResponse(200, {"status": "running", "turns": turns})


def done(turns=1, **extra):
    payload = {"status": "done", "turns": turns, "files": []}
    payload.update(extra)
    return FakeResponse(200, payload)


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------
class RequestBuildingTests(unittest.TestCase):
    def test_start_scrape_posts_json_with_bearer_and_defaults(self):
        client, session, _ = make_client(
            [FakeResponse(202, {"status": "queued", "session_id": "s1", "poll": "/api/scrapes/s1"})]
        )
        payload = client.start_scrape("Extract the text and author of each quote",
                                      target_urls=["https://quotes.toscrape.com"])
        self.assertEqual(payload["session_id"], "s1")
        call = session.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://scrape.example.com/api/scrapes")
        self.assertEqual(call["headers"]["Authorization"], "Bearer dc_sk_test")
        self.assertEqual(call["json"]["compliance_mode"], "regular")  # default
        self.assertEqual(call["json"]["target_urls"], ["https://quotes.toscrape.com"])
        self.assertIn("instruction", call["json"])

    def test_optional_fields_are_forwarded(self):
        client, session, _ = make_client([FakeResponse(202, {"session_id": "s2"})])
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        client.start_scrape(
            "go",
            fields=["quote", "author"],
            schema=schema,
            output_schema=schema,
            table_shape="long",
            compliance_mode="conservative",
        )
        body = session.calls[0]["json"]
        self.assertEqual(body["fields"], ["quote", "author"])
        self.assertEqual(body["table_shape"], "long")
        self.assertEqual(body["compliance_mode"], "conservative")
        self.assertEqual(body["output_schema"], schema)

    def test_document_uses_multipart_file_field(self):
        client, session, _ = make_client([FakeResponse(202, {"session_id": "s3"})])
        client.start_scrape("summarise this", document=b"hello", document_name="a.txt")
        call = session.calls[0]
        self.assertNotIn("json", call)
        self.assertIn("files", call)
        self.assertEqual(call["files"]["file"][0], "a.txt")
        self.assertEqual(call["data"]["instruction"], "summarise this")

    def test_invalid_compliance_mode_rejected_before_any_call(self):
        client, session, _ = make_client([])
        with self.assertRaises(ValueError):
            client.start_scrape("go", compliance_mode="reckless")
        self.assertEqual(session.calls, [])

    def test_empty_instruction_rejected(self):
        client, _, _ = make_client([])
        with self.assertRaises(ValueError):
            client.start_scrape("   ")

    def test_oversized_output_schema_rejected(self):
        client, session, _ = make_client([])
        big = {"type": "object", "title": "x" * (33 * 1024)}
        with self.assertRaises(ValueError):
            client.start_scrape("go", output_schema=big)
        self.assertEqual(session.calls, [])


# ---------------------------------------------------------------------------
# Persistence of session_id
# ---------------------------------------------------------------------------
class PersistenceTests(unittest.TestCase):
    def test_session_id_persisted_on_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = sc.ScrapeStore(Path(tmp) / "runs.json")
            client, _, _ = make_client(
                [FakeResponse(202, {"session_id": "abc123"})], store=store
            )
            client.start_scrape("go")
            record = store.get("abc123")
            self.assertIsNotNone(record)
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["instruction"], "go")

    def test_store_reloads_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.json"
            store = sc.ScrapeStore(path)
            store.record_start("s9", instruction="keep me")
            reloaded = sc.ScrapeStore(path)
            self.assertEqual(reloaded.get("s9")["instruction"], "keep me")

    def test_status_updates_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = sc.ScrapeStore(Path(tmp) / "runs.json")
            store.record_start("abc", instruction="go")
            store.update("abc", status="done")
            self.assertEqual(sc.ScrapeStore(store.path).get("abc")["status"], "done")


class NeverRetryPostTests(unittest.TestCase):
    def test_post_scrapes_timeout_is_not_retried(self):
        client, session, clock = make_client([requests.exceptions.Timeout("timed out")])
        with self.assertRaises(sc.SieveTransportError):
            client.start_scrape("go")
        posts = [c for c in session.calls if c["method"] == "POST"]
        self.assertEqual(len(posts), 1, "POST /api/scrapes must never be auto-retried")

    def test_timeout_does_not_write_a_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = sc.ScrapeStore(Path(tmp) / "runs.json")
            client, _, _ = make_client([requests.exceptions.Timeout("boom")], store=store)
            with self.assertRaises(sc.SieveTransportError):
                client.start_scrape("go")
            self.assertEqual(store.all(), {})

    def test_safe_post_statuses_are_retried(self):
        client, session, clock = make_client(
            [FakeResponse(503, {"error": "unavailable"}, headers={"Retry-After": "2"}),
             FakeResponse(202, {"session_id": "s4"})]
        )
        payload = client.start_scrape("go")
        self.assertEqual(payload["session_id"], "s4")
        self.assertEqual(len([c for c in session.calls if c["method"] == "POST"]), 2)
        self.assertEqual(clock.slept, [2.0])  # honoured Retry-After


# ---------------------------------------------------------------------------
# Status handling
# ---------------------------------------------------------------------------
class StatusHandlingTests(unittest.TestCase):
    def test_poll_running_then_done_with_backoff(self):
        client, session, clock = make_client([running(), running(), done()])
        payload = client.poll_scrape("s1")
        self.assertEqual(payload["status"], "done")
        self.assertEqual(len(session.calls), 3)
        self.assertAlmostEqual(clock.slept[0], sc.POLL_INITIAL_DELAY)
        self.assertLess(clock.slept[1], clock.slept[2])
        self.assertLessEqual(max(clock.slept), sc.POLL_MAX_DELAY)

    def test_refused_is_terminal_and_exposes_code(self):
        refused = FakeResponse(200, {"status": "refused", "refusal": {"code": "quota"}})
        client, session, _ = make_client([refused])
        payload = client.poll_scrape("s1")
        self.assertEqual(sc.status_of(payload), sc.STATUS_REFUSED)
        self.assertEqual(sc.refusal_code(payload), "quota")

    def test_unknown_status_raises(self):
        client, _, _ = make_client([FakeResponse(200, {"status": "paused"})])
        with self.assertRaises(sc.SieveProtocolError):
            client.poll_scrape("s1")

    def test_status_of_rejects_missing_status(self):
        with self.assertRaises(sc.SieveProtocolError):
            sc.status_of({})

    def test_max_wait_stops_infinite_polling(self):
        responses = [running() for _ in range(20)]
        client, _, _ = make_client(responses)
        with self.assertRaises(sc.SieveTimeoutError):
            client.poll_scrape("s1", max_wait=10.0)

    def test_get_retries_5xx_then_succeeds(self):
        client, session, _ = make_client([FakeResponse(500, {"error": "boom"}), done()])
        payload = client.get_scrape("s1")
        self.assertEqual(payload["status"], "done")
        self.assertEqual(len(session.calls), 2)


# ---------------------------------------------------------------------------
# Follow-up turns
# ---------------------------------------------------------------------------
class FollowUpTurnTests(unittest.TestCase):
    def test_send_message_resends_on_409(self):
        client, session, _ = make_client(
            [FakeResponse(409, {"error": "turn in flight"}), FakeResponse(202, {"status": "queued"})]
        )
        client.send_message("s1", "now extract the dates")
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(session.calls[0]["url"].endswith("/api/scrapes/s1/messages"))

    def test_wait_for_turn_requires_turns_to_advance(self):
        # First "done" carries the PREVIOUS turn count -> must not be returned.
        client, session, _ = make_client([done(turns=1), running(turns=1), done(turns=2)])
        payload = client.wait_for_turn("s1", turns_before=1)
        self.assertEqual(payload["turns"], 2)
        self.assertEqual(len(session.calls), 3)

    def test_wait_for_turn_returns_early_on_refused(self):
        client, session, _ = make_client(
            [FakeResponse(200, {"status": "refused", "turns": 1, "refusal": {"code": "policy"}})]
        )
        payload = client.wait_for_turn("s1", turns_before=1)
        self.assertEqual(sc.refusal_code(payload), "policy")

    def test_current_turns_reads_counter(self):
        client, _, _ = make_client([done(turns=4)])
        self.assertEqual(client.current_turns("s1"), 4)


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
class ErrorMappingTests(unittest.TestCase):
    def test_status_codes_map_to_typed_errors(self):
        cases = [
            (400, sc.SieveBadRequestError),
            (401, sc.SieveAuthError),
            (402, sc.SieveCreditsError),
            (404, sc.SieveNotFoundError),
            (500, sc.SieveServerError),
        ]
        for status, expected in cases:
            client, _, _ = make_client(
                [FakeResponse(status, {"error": f"e{status}"})], max_retries=0
            )
            with self.subTest(status=status):
                with self.assertRaises(expected):
                    client.start_scrape("go")

    def test_429_maps_to_rate_limit_with_retry_after(self):
        client, _, _ = make_client(
            [FakeResponse(429, {"error": "slow down"}, headers={"Retry-After": "7"})],
            max_retries=0,
        )
        with self.assertRaises(sc.SieveRateLimitError) as ctx:
            client.start_scrape("go")
        self.assertEqual(ctx.exception.retry_after, 7.0)

    def test_get_401_raises_auth_error(self):
        client, _, _ = make_client([FakeResponse(401, {"error": "unauthorized"})])
        with self.assertRaises(sc.SieveAuthError):
            client.get_scrape("s1")

    def test_get_404_raises_not_found(self):
        client, _, _ = make_client([FakeResponse(404, {"error": "not found"})])
        with self.assertRaises(sc.SieveNotFoundError):
            client.get_scrape("s1")


# ---------------------------------------------------------------------------
# Files + schema conformance
# ---------------------------------------------------------------------------
class FileTests(unittest.TestCase):
    def test_relative_file_url_is_prefixed(self):
        client, _, _ = make_client([])
        self.assertEqual(
            client.file_url({"url": "/files/q.csv"}),
            "https://scrape.example.com/files/q.csv",
        )

    def test_absolute_file_url_is_left_alone(self):
        client, _, _ = make_client([])
        self.assertEqual(
            client.file_url({"url": "https://cdn.example.com/q.csv"}),
            "https://cdn.example.com/q.csv",
        )

    def test_download_sends_bearer_header(self):
        client, session, _ = make_client([FakeResponse(200, content=b"a,b\n1,2\n")])
        name, content = client.download_file({"name": "q.csv", "url": "/files/q.csv"})
        self.assertEqual(name, "q.csv")
        self.assertEqual(content, b"a,b\n1,2\n")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "Bearer dc_sk_test")


class SchemaConformanceTests(unittest.TestCase):
    def test_pass_is_clean(self):
        payload = {"schema_conformance": {"status": "pass"}}
        self.assertTrue(sc.is_clean(payload))
        self.assertEqual(sc.conformance_status(payload), "pass")

    def test_partial_is_not_clean(self):
        payload = {"schema_conformance": {"status": "partial"}}
        self.assertFalse(sc.is_clean(payload))

    def test_fail_is_not_clean(self):
        payload = {"schema_conformance": {"status": "fail"}}
        self.assertFalse(sc.is_clean(payload))

    def test_missing_conformance_is_falsey(self):
        self.assertFalse(sc.is_clean({}))
        self.assertIsNone(sc.conformance_status({}))


# ---------------------------------------------------------------------------
# Behaviour when Sieve is not configured
# ---------------------------------------------------------------------------
class ConfigGatingTests(unittest.TestCase):
    def test_sieve_enabled_false_without_key(self):
        import config

        with mock.patch.object(config, "SIEVE_API_KEY", None):
            self.assertFalse(config.sieve_enabled())
            with self.assertRaises(RuntimeError):
                config.require_sieve_key()

    def test_sieve_enabled_true_with_key(self):
        import config

        with mock.patch.object(config, "SIEVE_API_KEY", "dc_sk_x"):
            self.assertTrue(config.sieve_enabled())
            self.assertEqual(config.require_sieve_key(), "dc_sk_x")


if __name__ == "__main__":
    unittest.main()
