"""Thin wrapper around the authenticated `gh api` CLI: retries, pacing, rate-limit waits.

This is the only module in the project that touches the network. Waiting and pacing use the wall
clock, but they never change what is fetched or written, so results stay deterministic.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

MIN_INTERVAL_SECONDS = 0.4      # polite pacing between consecutive calls
MAX_ATTEMPTS = 8
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 300.0
RATE_LIMIT_FLOOR = 50           # wait for the reset when fewer GraphQL points than this remain


class GhError(RuntimeError):
    pass


class NotFound(GhError):
    pass


@dataclass
class Completed:
    returncode: int
    stdout: bytes
    stderr: bytes


Runner = Callable[[Sequence[str], bytes | None], Completed]
Sleeper = Callable[[float], None]


def _subprocess_runner(args: Sequence[str], stdin: bytes | None) -> Completed:
    p = subprocess.run(list(args), input=stdin, capture_output=True, check=False)
    return Completed(p.returncode, p.stdout, p.stderr)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _is_rate_limited(text: str) -> bool:
    t = text.lower()
    return "rate limit" in t or "abuse detection" in t or "http 429" in t


def _is_transient(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in ("http 500", "http 502", "http 503", "http 504", "timeout",
                                "timed out", "connection reset", "unexpected eof", "tls handshake",
                                "something went wrong", "i/o timeout", "no such host"))


@dataclass
class Gh:
    """`gh api` client. `runner` and `sleep` are injectable for offline tests."""

    runner: Runner = _subprocess_runner
    sleep: Sleeper = time.sleep
    clock: Callable[[], float] = time.monotonic
    min_interval: float = MIN_INTERVAL_SECONDS
    calls: int = 0
    _last: float = field(default=-1e9, repr=False)

    # -- low level ---------------------------------------------------------

    def _pace(self) -> None:
        wait = self._last + self.min_interval - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last = self.clock()

    def _run(self, args: Sequence[str], stdin: bytes | None = None) -> bytes:
        delay = BACKOFF_BASE_SECONDS
        last_err = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            self.calls += 1
            res = self.runner(["gh", "api", *args], stdin)
            err = res.stderr.decode("utf-8", "replace")
            out_text = res.stdout.decode("utf-8", "replace")
            if res.returncode == 0:
                return res.stdout
            if "HTTP 404" in err:
                raise NotFound(err.strip())
            last_err = err.strip() or out_text[:500]
            if _is_rate_limited(err) or _is_rate_limited(out_text):
                wait = max(delay, 60.0)
                log(f"[gh] rate limited (attempt {attempt}); sleeping {wait:.0f}s")
                self.sleep(wait)
            elif _is_transient(err) or _is_transient(out_text):
                log(f"[gh] transient error (attempt {attempt}): {last_err[:200]}; sleeping {delay:.0f}s")
                self.sleep(delay)
            elif args and args[0] == "graphql" and res.stdout:
                # GraphQL responses with partial errors still carry data; let the caller decide.
                return res.stdout
            else:
                raise GhError(f"gh api {' '.join(args)} failed: {last_err}")
            delay = min(delay * 2, BACKOFF_MAX_SECONDS)
        raise GhError(f"gh api {' '.join(args)} failed after {MAX_ATTEMPTS} attempts: {last_err}")

    # -- REST --------------------------------------------------------------

    def rest(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a REST endpoint and return parsed JSON. Raises NotFound on 404."""
        args = ["-X", "GET", path]
        for k in sorted(params or {}):
            args += ["-f", f"{k}={params[k]}"]
        return json.loads(self._run(args))

    def rest_paginate(self, path: str, per_page: int = 100) -> list[Any]:
        """GET every page of a list endpoint (`--paginate --slurp`)."""
        sep = "&" if "?" in path else "?"
        out = self._run(["--paginate", "--slurp", f"{path}{sep}per_page={per_page}"])
        pages = json.loads(out)
        items: list[Any] = []
        for page in pages:
            items.extend(page if isinstance(page, list) else [page])
        return items

    def rest_raw(self, path: str) -> bytes:
        """Raw file content from the contents API (non-JSON media type, so gh passes bytes through)."""
        return self._run(["-H", "Accept: application/vnd.github.raw", path])

    def rest_blob(self, repo: str, oid: str) -> bytes:
        """Exact bytes of a git blob, via the base64 git blobs API."""
        resp = json.loads(self._run([f"repos/{repo}/git/blobs/{oid}"]))
        if resp.get("encoding") != "base64":
            raise GhError(f"{repo} blob {oid}: unexpected encoding {resp.get('encoding')!r}")
        return base64.b64decode(resp.get("content", ""))

    # -- GraphQL -----------------------------------------------------------

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        """Run a GraphQL query; returns {"data": ..., "errors": [...]} and waits on low budget."""
        body = json.dumps({"query": query, "variables": variables or {}}, sort_keys=True).encode()
        delay = BACKOFF_BASE_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw = self._run(["graphql", "--input", "-"], body)
            try:
                resp = json.loads(raw)
            except json.JSONDecodeError as e:
                raise GhError(f"invalid GraphQL response: {raw[:200]!r}") from e
            errors = resp.get("errors") or []
            if any((e.get("type") == "RATE_LIMITED") for e in errors if isinstance(e, dict)):
                self._wait_for_reset(None)
                continue
            if resp.get("data") is None:
                msg = json.dumps(errors)[:500]
                if _is_transient(msg) and attempt < MAX_ATTEMPTS:
                    log(f"[gh] transient GraphQL error: {msg[:200]}; sleeping {delay:.0f}s")
                    self.sleep(delay)
                    delay = min(delay * 2, BACKOFF_MAX_SECONDS)
                    continue
                raise GhError(f"GraphQL query failed: {msg}")
            rl = (resp.get("data") or {}).get("rateLimit")
            if isinstance(rl, dict) and rl.get("remaining", RATE_LIMIT_FLOOR) < RATE_LIMIT_FLOOR:
                self._wait_for_reset(rl.get("resetAt"))
            return resp
        raise GhError("GraphQL query kept hitting the rate limit")

    def _wait_for_reset(self, reset_at: str | None) -> None:
        wait = 60.0
        if reset_at:
            try:
                reset = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
                wait = max(5.0, (reset - datetime.now(timezone.utc)).total_seconds() + 5.0)
            except ValueError:
                pass
        log(f"[gh] GraphQL budget low; sleeping {wait:.0f}s until reset")
        self.sleep(min(wait, 3700.0))


def gql_str(s: str) -> str:
    """A GraphQL string literal (JSON escaping is valid GraphQL for these inputs)."""
    return json.dumps(s)


def utc_now_iso() -> str:
    """Retrieval timestamp for provenance only; never used to make a decision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
