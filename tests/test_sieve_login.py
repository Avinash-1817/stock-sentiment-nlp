"""Unit tests for src/sieve_login.py (stdlib unittest).

The device-code endpoint is faked at the HTTP boundary; the polling state
machine and the .env writer under test are the real code.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sieve_login as sl  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def poll_with(responses, **kwargs):
    session = FakeSession(responses)
    clock = {"t": 0.0}
    slept = []

    def sleep(d):
        slept.append(d)
        clock["t"] += d

    status, value = sl.poll_for_token(
        session, "https://scrape.example.com", "dev-code",
        sleep=sleep, monotonic=lambda: clock["t"], **kwargs,
    )
    return status, value, slept, session


class PollForTokenTests(unittest.TestCase):
    def test_authorization_pending_then_success(self):
        status, value, slept, session = poll_with(
            [FakeResponse(400, {"error": "authorization_pending"}),
             FakeResponse(200, {"api_key": "dc_sk_secret"})],
            interval=5,
        )
        self.assertEqual(status, "ok")
        self.assertEqual(value, "dc_sk_secret")
        self.assertEqual(len(session.calls), 2)

    def test_slow_down_adds_five_seconds(self):
        _, _, slept, _ = poll_with(
            [FakeResponse(400, {"error": "slow_down"}),
             FakeResponse(200, {"api_key": "k"})],
            interval=5,
        )
        self.assertEqual(slept, [5, 10])

    def test_access_denied_stops(self):
        status, value, _, _ = poll_with([FakeResponse(400, {"error": "access_denied"})])
        self.assertEqual(status, "denied")
        self.assertIsNone(value)

    def test_expired_token_reports_expired(self):
        status, _, _, _ = poll_with([FakeResponse(400, {"error": "expired_token"})])
        self.assertEqual(status, "expired")

    def test_5xx_is_tolerated_and_polling_continues(self):
        status, value, _, _ = poll_with(
            [FakeResponse(503, {}), FakeResponse(200, {"api_key": "k"})]
        )
        self.assertEqual(status, "ok")
        self.assertEqual(value, "k")

    def test_deadline_returns_expired_without_posting(self):
        session = FakeSession([])
        status, value = sl.poll_for_token(
            session, "https://x", "dev", expires_in=0,
            sleep=lambda d: None, monotonic=lambda: 100.0,
        )
        self.assertEqual(status, "expired")
        self.assertEqual(session.calls, [])


class WriteEnvVarTests(unittest.TestCase):
    def test_appends_without_touching_other_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("NEWSAPI_KEY=abc123\nOLLAMA_MODEL=llama3.2\n", encoding="utf-8")
            sl.write_env_var(path, "SIEVE_API_KEY", "dc_sk_secret")
            text = path.read_text(encoding="utf-8")
            self.assertIn("NEWSAPI_KEY=abc123", text)
            self.assertIn("OLLAMA_MODEL=llama3.2", text)
            self.assertIn("SIEVE_API_KEY=dc_sk_secret", text)

    def test_updates_existing_value_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SIEVE_API_KEY=old\n", encoding="utf-8")
            sl.write_env_var(path, "SIEVE_API_KEY", "new")
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "SIEVE_API_KEY=new")

    def test_creates_file_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            sl.write_env_var(path, "SIEVE_API_KEY", "v")
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "SIEVE_API_KEY=v")


if __name__ == "__main__":
    unittest.main()
