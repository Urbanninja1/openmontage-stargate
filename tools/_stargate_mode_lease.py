"""Pipeline-scoped Stargate mode lease for OpenMontage shims.

Fork-owned helper module — never synced from upstream.

Replaces per-stage POST /voice/mode with a single lease acquired at
pipeline entry and released on exit. Uses Stargate's RAG Platform
lease API at http://127.0.0.1:8104/v1/runtime/mode (acquire/renew/release).

Usage:
    from tools._stargate_mode_lease import stargate_mode_lease

    with stargate_mode_lease("image_studio", duration_minutes=90):
        # ComfyUI workflows run here — 3090 LLMs evicted for duration
        ...
    # lease released; llama-swap restored via mode_manager

Idempotent: nested `with` in the same mode is a no-op. Handles mode
transitions (image_studio → audio_studio → image_studio) via release +
acquire cycle internally.

Graceful fallback: if RAG Platform lease endpoint unreachable, falls
back to per-stage Agent API POST /voice/mode (original plan baseline).

See docs/specs/openmontage.md §"mode_lease" for config.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("stargate_mode_lease")

LEASE_BASE_DEFAULT = "http://127.0.0.1:8104/v1/runtime/mode"
MODE_SWITCH_URL_DEFAULT = "http://127.0.0.1:8096/voice/mode"
RAG_API_KEY_ENV = "STARGATE_RAG_API_KEY"

# Active lease (contextvar so nested with blocks see it)
_active_lease: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "active_lease", default=None
)
_lease_lock = threading.Lock()


_API_KEY_CACHED: Optional[str] = None
_API_KEY_PATHS = [
    Path("/home/edson/stargate/data/secrets/lane6.env"),
    Path("/etc/stargate/secrets/lane6.env"),
]


def _api_key() -> Optional[str]:
    """Read RAG Platform bearer. Priority: env > lane6.env files.

    Lane 6 fix 2026-04-21: previously the shim only read STARGATE_RAG_API_KEY
    from environment, so every shim invocation from a fresh shell silently
    fell back to 401 + mode_manager direct path. Now falls through to
    simple `KEY=value` env files as a convenience.
    """
    import os
    global _API_KEY_CACHED
    if _API_KEY_CACHED:
        return _API_KEY_CACHED
    env_val = os.environ.get(RAG_API_KEY_ENV)
    if env_val:
        _API_KEY_CACHED = env_val
        return env_val
    for path in _API_KEY_PATHS:
        try:
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().lstrip("export ").strip()
                v = v.strip().strip('"').strip("'")
                if k == RAG_API_KEY_ENV:
                    _API_KEY_CACHED = v
                    return v
        except (OSError, PermissionError):
            continue
    return None


def _acquire_lease_platform(mode: str, duration_minutes: int, base: str) -> Optional[dict]:
    """Try to acquire via RAG Platform. Returns lease dict or None if unreachable."""
    key = _api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = requests.post(
            f"{base}/lease",
            json={"mode": mode, "duration_minutes": duration_minutes},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        log.warning("Platform lease returned %s: %s", r.status_code, r.text[:200])
        return None
    except requests.RequestException as exc:
        log.info("Platform lease unreachable (%s); falling back to mode_manager", exc)
        return None


def _release_lease_platform(lease_id: str, base: str) -> None:
    key = _api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        requests.post(f"{base}/lease/{lease_id}/release", headers=headers, timeout=10)
    except requests.RequestException as exc:
        log.warning("Platform lease release failed (%s); mode_manager idle-timeout will catch it", exc)


def _switch_mode_fallback(mode: str, switch_url: str) -> None:
    """Per-stage fallback: POST /voice/mode. Used when Platform unavailable."""
    try:
        r = requests.post(switch_url, json={"mode": mode}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.error("mode_manager switch to %s failed: %s", mode, exc)
        raise RuntimeError(f"Cannot switch to mode {mode!r}: {exc}") from exc


@contextmanager
def stargate_mode_lease(
    mode: str,
    *,
    duration_minutes: int = 90,
    lease_base: str = LEASE_BASE_DEFAULT,
    switch_url: str = MODE_SWITCH_URL_DEFAULT,
):
    """Pipeline-scoped mode lease.

    On entry: acquire lease via RAG Platform; fall back to per-stage
    POST /voice/mode on Agent API.
    On exit: release lease (finally — always runs).

    Nested with-same-mode is no-op. With-different-mode releases + acquires.
    """
    with _lease_lock:
        current = _active_lease.get()

        # Idempotent nested case
        if current is not None and current.get("mode") == mode:
            log.debug("mode_lease: already in %s (nested no-op)", mode)
            yield current
            return

        # Transition case — release old, acquire new
        if current is not None:
            old_mode = current.get("mode")
            log.info("mode_lease: transition %s → %s", old_mode, mode)
            if current.get("lease_id"):
                _release_lease_platform(current["lease_id"], lease_base)
            else:
                # fallback mode didn't use lease; just switch
                _switch_mode_fallback(mode, switch_url)

        # Try Platform lease first
        lease = _acquire_lease_platform(mode, duration_minutes, lease_base)
        if lease is None:
            # Fallback — set mode directly on mode_manager, no lease
            log.info("mode_lease: fallback switch_mode(%s)", mode)
            _switch_mode_fallback(mode, switch_url)
            lease = {"mode": mode, "lease_id": None, "fallback": True, "t_start": time.time()}
        else:
            lease["mode"] = mode
            lease["t_start"] = time.time()

        token = _active_lease.set(lease)

    try:
        yield lease
    finally:
        with _lease_lock:
            if lease.get("lease_id"):
                _release_lease_platform(lease["lease_id"], lease_base)
            else:
                # Fallback: return to "off" so persistent LLMs restore
                try:
                    _switch_mode_fallback("off", switch_url)
                except Exception:
                    pass  # best-effort; mode_manager idle-timeout backstop
            _active_lease.reset(token)
            log.info("mode_lease: released %s after %.1fs", mode, time.time() - lease["t_start"])


def current_mode() -> Optional[str]:
    """Return the currently leased mode, or None."""
    lease = _active_lease.get()
    return lease["mode"] if lease else None
