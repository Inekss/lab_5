from __future__ import annotations

import functools
import logging
from time import perf_counter

log = logging.getLogger("findex")


def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.1f ms", fn.__name__, (perf_counter() - t0) * 1000)
    return wrapper
