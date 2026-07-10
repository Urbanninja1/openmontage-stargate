"""Compatibility context manager for Stargate's authoritative mode manager.

Fork-owned helper module. OpenMontage never starts or stops a GPU service
directly: it asks the Agent API to perform the transactional transition and
restores the baseline mode on exit.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

import requests


log = logging.getLogger("stargate_mode_lease")
MODE_SWITCH_URL_DEFAULT = "http://127.0.0.1:8096/voice/mode"

_active_lease: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "active_lease", default=None
)
_lease_lock = threading.Lock()


def _switch_mode(mode: str, switch_url: str) -> None:
    response = requests.post(switch_url, json={"mode": mode}, timeout=180)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") not in {"ready", "switching"}:
        raise RuntimeError(f"mode transition to {mode!r} was rejected: {payload}")


@contextmanager
def stargate_mode_lease(
    mode: str,
    *,
    duration_minutes: int = 90,
    lease_base: str | None = None,
    switch_url: str = MODE_SWITCH_URL_DEFAULT,
):
    """Enter a mode for one operation and restore ``off`` afterward.

    ``duration_minutes`` and ``lease_base`` remain accepted so older fork
    callers keep working; lifecycle authority belongs to ``mode_manager``.
    """
    del duration_minutes, lease_base
    with _lease_lock:
        current = _active_lease.get()
        if current is not None and current["mode"] == mode:
            yield current
            return
        _switch_mode(mode, switch_url)
        state = {"mode": mode, "t_start": time.time(), "manager": "agent-api"}
        token = _active_lease.set(state)

    try:
        yield state
    finally:
        with _lease_lock:
            try:
                _switch_mode("off", switch_url)
            finally:
                _active_lease.reset(token)
            log.info("mode context released %s after %.1fs", mode, time.time() - state["t_start"])


def current_mode() -> Optional[str]:
    state = _active_lease.get()
    return state["mode"] if state else None
