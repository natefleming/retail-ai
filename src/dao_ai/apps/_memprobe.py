"""TEMPORARY memory-footprint probe for the Apps backend.

Gated behind the ``DAO_AI_MEMPROBE`` env var — a no-op unless it is set to a
truthy value. Logs this process's RSS and PSS (proportional set size, which
splits copy-on-write-shared pages across the processes that share them) at
startup and on a periodic interval. PSS is the metric that reveals how much a
gunicorn ``preload_app`` fork actually shares versus duplicates: summing PSS
across master + workers gives the true resident footprint.

This module exists to answer one empirical question — does fork/COW meaningfully
reduce total memory for the dao-ai agent graph? — and should be removed (or left
inert behind the env gate) once that measurement is done.
"""

from __future__ import annotations

import os
import threading
import time

from loguru import logger

_MEMPROBE_ENV = "DAO_AI_MEMPROBE"


def _enabled() -> bool:
    return os.environ.get(_MEMPROBE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _read_rss_pss_mb() -> tuple[float | None, float | None]:
    """Return (RSS, PSS) for the current process in MiB, or (None, None).

    Reads ``/proc/self/smaps_rollup`` (cheap, kernel-aggregated) for both RSS
    and PSS. Falls back to ``/proc/self/status`` VmRSS when the rollup is
    unavailable (PSS stays None in that case).
    """
    try:
        rss_kb: float | None = None
        pss_kb: float | None = None
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss_kb = float(line.split()[1])
                elif line.startswith("Pss:"):
                    pss_kb = float(line.split()[1])
        rss = rss_kb / 1024.0 if rss_kb is not None else None
        pss = pss_kb / 1024.0 if pss_kb is not None else None
        return rss, pss
    except (OSError, ValueError, IndexError):
        # Fallback: VmRSS only (no PSS without smaps).
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0, None
        except (OSError, ValueError, IndexError):
            pass
        return None, None


def _log_once(tag: str) -> None:
    rss, pss = _read_rss_pss_mb()
    logger.info(
        "dao_ai.memprobe pid={} tag={} rss_mb={} pss_mb={}",
        os.getpid(),
        tag,
        f"{rss:.1f}" if rss is not None else "n/a",
        f"{pss:.1f}" if pss is not None else "n/a",
    )


def start(interval_s: float = 20.0) -> None:
    """Start the probe if ``DAO_AI_MEMPROBE`` is truthy; otherwise a no-op.

    Logs an immediate ``startup`` sample, then a daemon thread logs a
    ``periodic`` sample every ``interval_s`` seconds. Daemon so it never blocks
    interpreter/worker shutdown (fork-safe: no locks held across fork).
    """
    if not _enabled():
        return

    _log_once("startup")

    def _loop() -> None:
        while True:
            time.sleep(interval_s)
            _log_once("periodic")

    threading.Thread(target=_loop, name="dao-ai-memprobe", daemon=True).start()
