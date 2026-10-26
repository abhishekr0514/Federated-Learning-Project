"""Short-lived cache for expensive platform capability aggregation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_CACHE: Optional[Tuple[float, Dict[str, Any], Dict[str, str]]] = None
_INFLIGHT: Optional[threading.Event] = None
_INFLIGHT_RESULT: Optional[Tuple[Dict[str, Any], Dict[str, str]]] = None
_TTL_SECONDS = 45.0


def get_cached_capability_bundle(
    builder: Callable[[], Tuple[Dict[str, Any], Dict[str, str]]],
    *,
    force_refresh: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Return (capability_details, capability_summaries).
    Coalesces concurrent callers onto one in-flight build.
    """
    global _CACHE, _INFLIGHT, _INFLIGHT_RESULT
    now = time.monotonic()

    with _CACHE_LOCK:
        if not force_refresh and _CACHE is not None:
            ts, details, caps = _CACHE
            if now - ts < _TTL_SECONDS:
                return details, caps

        if _INFLIGHT is not None and not force_refresh:
            wait_event = _INFLIGHT
        else:
            wait_event = None
            _INFLIGHT = threading.Event()
            _INFLIGHT_RESULT = None

    if wait_event is not None:
        wait_event.wait(timeout=120.0)
        with _CACHE_LOCK:
            if _CACHE is not None:
                _, details, caps = _CACHE
                return details, caps
        # fall through to rebuild if waiter timed out

    started = time.perf_counter()
    details, caps = builder()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "capability_bundle_built elapsed_ms=%.1f keys=%s",
        elapsed_ms,
        list(details.keys()),
    )

    with _CACHE_LOCK:
        _CACHE = (time.monotonic(), details, caps)
        _INFLIGHT_RESULT = (details, caps)
        if _INFLIGHT is not None:
            _INFLIGHT.set()
            _INFLIGHT = None
    return details, caps


def invalidate_capability_cache() -> None:
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
