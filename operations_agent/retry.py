"""Retry-with-backoff for transient tool failures (the 'below the loop' tier).

``call_with_retry`` runs a callable, retrying only on ``TransientError`` with
exponential backoff up to a cap. ``SemanticError`` and anything else propagate
immediately — we don't retry failures that won't fix themselves.

``sleep`` is injectable so tests run instantly (no real waiting) while production
uses ``time.sleep``. The optional ``on_retry`` hook lets the caller record each
retry attempt into the Trace.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from .errors import TransientError

T = TypeVar("T")


def call_with_retry(
    func: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, TransientError, float], None] | None = None,
) -> T:
    """Call ``func``; retry on TransientError with exponential backoff.

    Up to ``max_retries`` *additional* attempts after the first. If the last
    attempt still raises TransientError, that error propagates.
    """
    attempt = 0
    while True:
        try:
            return func()
        except TransientError as err:
            if attempt >= max_retries:
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            if on_retry is not None:
                on_retry(attempt + 1, err, delay)
            sleep(delay)
            attempt += 1
