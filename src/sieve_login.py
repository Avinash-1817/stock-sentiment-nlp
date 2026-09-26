"""Device-code login for the Sieve scrape API.

Run from the repo root::

    python src/sieve_login.py

It prints a verification link and a short user code. The user opens the link in
their own browser, signs in / signs up, checks the code matches, and clicks
Approve. This script never opens the link, never signs in, and never approves
on the user's behalf.

On success the returned API key is written straight into the project's secret
store (``.env`` at the repo root, git-ignored) as ``SIEVE_API_KEY``. The key is
never printed, logged, or committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

CLIENT_NAME = "stock-sentiment-nlp"
# Sentinel: the code request expired before the user approved.
EXPIRED = "expired_token"

# A pending device code is cached here so a restart can keep polling the SAME
# code the user is looking at, instead of invalidating it with a new one.
PENDING_PATH = config.DATA_DIR / "sieve_device.json"


def _url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def request_device_code(session, base_url: str, client_name: str = CLIENT_NAME) -> dict:
    """POST /api/auth/device/code -> device_code, user_code, verification_uri..."""
    response = session.post(
        _url(base_url, "/api/auth/device/code"),
        json={"client_name": client_name},
        timeout=(10, 60),
    )
    response.raise_for_status()
    return response.json()


def poll_for_token(
    session,
    base_url: str,
    device_code: str,
    *,
    interval: float = 5,
    expires_in: float = 600,
    sleep=time.sleep,
    monotonic=time.monotonic,
    on_event=None,
):
    """Poll /api/auth/device/token until it succeeds, is denied, or expires.

    Returns ``(status, value)`` where status is ``"ok"`` (value = api_key),
    ``"denied"``, or ``"expired"``.
    """
    interval = float(5 if interval is None else interval)
    deadline = monotonic() + float(600 if expires_in is None else expires_in)
    while True:
        if monotonic() >= deadline:
            return "expired", None
        sleep(interval)
        response = session.post(
            _url(base_url, "/api/auth/device/token"),
            json={"device_code": device_code},
            timeout=(10, 60),
        )
        if response.status_code == 200:
            api_key = (response.json() or {}).get("api_key")
            if not api_key:
                raise RuntimeError("device/token returned 200 without an api_key")
            return "ok", api_key
        if response.status_code == 400:
            try:
                error = (response.json() or {}).get("error")
            except ValueError:
                error = None
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5  # server asked us to back off
                continue
            if error == "access_denied":
                return "denied", None
            if error == "expired_token":
                return "expired", None
            raise RuntimeError(f"device/token error: {error!r}")
        if 500 <= response.status_code < 600:
            continue  # transient server error - keep polling within the window
        response.raise_for_status()


def load_pending(path: Path, now: float | None = None):
    """Return a still-valid pending device-code request, or None."""
    now = time.time() if now is None else now
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("device_code"):
        return None
    if float(data.get("expires_at", 0)) - now <= 5:
        return None  # too close to expiry to be useful
    return data


def save_pending(path: Path, payload: dict, base_url: str, client_name: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "device_code": payload.get("device_code"),
        "user_code": payload.get("user_code"),
        "verification_uri_complete": payload.get("verification_uri_complete")
        or payload.get("verification_uri"),
        "interval": payload.get("interval", 5),
        "base_url": base_url,
        "client_name": client_name,
        "expires_at": time.time() + float(payload.get("expires_in") or 600),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def write_env_var(path: Path, key: str, value: str) -> None:
    """Set ``key=value`` in ``path``, preserving every other line unmodified."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    out = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sieve device-code login")
    parser.add_argument("--client-name", default=CLIENT_NAME)
    parser.add_argument("--base-url", default=config.SIEVE_BASE_URL)
    parser.add_argument("--env-file", default=str(config.REPO_ROOT / ".env"))
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore a pending device code and request a new one",
    )
    args = parser.parse_args(argv)

    session = requests.Session()
    env_path = Path(args.env_file)

    for attempt in range(1, 4):
        pending = None if args.fresh else load_pending(PENDING_PATH)
        if pending:
            response = pending
            print("Reusing the pending device code from a previous start.")
        else:
            response = request_device_code(session, args.base_url, args.client_name)
            save_pending(PENDING_PATH, response, args.base_url, args.client_name)

        user_code = response.get("user_code")
        link = response.get("verification_uri_complete") or response.get("verification_uri")
        interval = response.get("interval", 5)
        if pending:
            expires_in = max(1.0, float(pending["expires_at"]) - time.time())
        else:
            expires_in = response.get("expires_in", 600)

        print()
        print("=" * 70)
        print("Sieve device login - approve in your own browser")
        print("=" * 70)
        print(f"1. Open this link:      {link}")
        print(f"2. Confirm the code:    {user_code}")
        print()
        print("Safety checks before you click Approve:")
        print("  - The approval page shows where this code was requested from, next to")
        print("    your own location - make sure it matches what you expect.")
        print(f"  - The tool name ({args.client_name!r}) is SELF-REPORTED by this script.")
        print("  - Only approve a code YOU started. If you did not run this yourself,")
        print("    do not approve it.")
        print(f"  - The code expires in {int(expires_in)}s and works only once.")
        print("=" * 70)
        print("Waiting for approval...")

        status, api_key = poll_for_token(
            session, args.base_url, response["device_code"],
            interval=interval, expires_in=expires_in,
        )
        if status == "ok":
            write_env_var(env_path, "SIEVE_API_KEY", api_key)
            try:
                PENDING_PATH.unlink()
            except OSError:
                pass
            print(f"SIEVE_API_KEY written to {env_path} (value not shown).")
            return 0
        if status == "denied":
            print("Approval was denied. Nothing was written.")
            return 1
        print(f"The code expired (attempt {attempt} of 3). Starting over...")
        args.fresh = True

    print("Could not complete the device login. Please run this again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
