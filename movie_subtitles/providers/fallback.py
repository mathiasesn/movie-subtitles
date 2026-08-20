import logging
import threading
from pathlib import Path

from movie_subtitles.providers.base import AlignmentProvider

logger = logging.getLogger("fallback")


class FallbackAlign:
    """Compose `AlignmentProvider`s into a degrading chain.

    Tries each provider in list order, skipping any already latched off. On success, logs
    (once per run, INFO) which provider it settled on and returns its result. Any exception
    from a provider latches that provider off for the rest of the run -- a broken aligner
    must not cost one failing call per clip -- logs that once (WARNING), and degrades to the
    next provider. If every provider fails, raises `RuntimeError` chained from the last one.
    """

    def __init__(self, providers: list[AlignmentProvider]) -> None:
        if not providers:
            raise ValueError("FallbackAlign requires at least one provider")

        self.providers = providers
        self._disabled: set[int] = set()
        self._logged: set[int] = set()
        self._lock = threading.Lock()

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        last_exc: Exception | None = None
        for index, provider in enumerate(self.providers):
            with self._lock:
                if index in self._disabled:
                    continue

            try:
                boundaries = provider(clip, text)
            except Exception as exc:
                last_exc = exc
                # Logged inside the lock so the once-per-provider guard and the log call
                # cannot interleave; the lock never spans a provider call, so concurrent
                # measurement is not serialised.
                with self._lock:
                    if index not in self._disabled:
                        self._disabled.add(index)
                        logger.warning(
                            f"{type(provider).__name__} unavailable ({exc}); falling back "
                            "to the next speech-boundary measurement tier."
                        )
                continue

            with self._lock:
                if index not in self._logged:
                    self._logged.add(index)
                    logger.info(
                        f"Measuring dub clip speech boundaries via {type(provider).__name__}."
                    )
            return boundaries

        raise RuntimeError(
            "All speech-boundary measurement tiers are exhausted or latched off; "
            "cannot measure clip speech boundaries."
        ) from last_exc
