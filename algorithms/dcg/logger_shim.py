"""Minimal logger satisfying the surface the vendored DCG learner expects.

``q_learner`` / ``dcg_learner`` call ``logger.log_stat(key, value, t)`` and
``logger.console_logger.info(msg)``. PyMARL backs these with Sacred; we back
them with a plain dict + stdout so the learner runs unmodified without pulling
Sacred into the framework.
"""

import logging


class DCGLogger:
    def __init__(self, verbose: bool = False):
        # Latest value seen for each logged key, for optional inspection.
        self.stats: dict[str, tuple[int, float]] = {}
        self.console_logger = logging.getLogger("dcg")
        if not self.console_logger.handlers:
            self.console_logger.addHandler(logging.NullHandler())
        self.verbose = verbose

    def log_stat(self, key: str, value, t: int) -> None:
        self.stats[key] = (t, value)
        if self.verbose:
            print(f"[dcg] t={t} {key}={value:.4f}")
