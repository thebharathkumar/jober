"""Structured logging setup.

Libraries should not configure the root logger on import — that is the
application's job. So modules call ``get_logger(__name__)`` and stay quiet by
default; the CLI (the application here) calls ``configure_logging`` once to
attach a handler. This keeps Bus Factor well-behaved when imported into a larger
service that owns its own logging.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    """Attach a single stderr handler. Idempotent; safe to call more than once.

    Honors the ``BF_LOG_LEVEL`` env var (e.g. ``DEBUG``) when set, otherwise
    uses INFO, or DEBUG when ``verbose`` is passed.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("BF_LOG_LEVEL", "DEBUG" if verbose else "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("bus_factor")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True
