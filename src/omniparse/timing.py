from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("omniparse")


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("omniparse")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


@contextmanager
def log_step(name: str, timings: dict[str, float] | None = None, **details: object) -> Iterator[None]:
    extra = _format_details(details)
    logger.info("Starting %s%s", name, extra)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        if timings is not None:
            timings[name] = round(elapsed, 3)
        logger.exception("Failed %s after %.3fs", name, elapsed)
        raise
    elapsed = time.perf_counter() - started
    if timings is not None:
        timings[name] = round(elapsed, 3)
    logger.info("Finished %s in %.3fs%s", name, elapsed, extra)


def _format_details(details: dict[str, object]) -> str:
    if not details:
        return ""
    parts = [f"{key}={value}" for key, value in details.items() if value is not None]
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"
