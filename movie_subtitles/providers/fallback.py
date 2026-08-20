import logging
import threading
from pathlib import Path

from movie_subtitles.providers.base import AlignmentProvider

logger = logging.getLogger("fallback")


class FallbackAlign:
    """Compose `AlignmentProvider`s into a degrading chain.

    Tries each provider in list order, skipping any already latched off. On success, logs
    (once per run, INFO) which provider it settled on and returns its result. A provider's
    very first call is gated: exactly one thread makes it while any others that arrive
    concurrently block on that outcome, so a broken aligner costs at most one failing vendor
    call for the whole run, not one per waiting thread. That first call raising latches the
    provider off (WARNING, once) and degrades to the next provider; succeeding leaves it
    enabled. Once a tier's first outcome is known, every subsequent call to it runs fully
    unserialised -- the lock is never held across a provider call in steady state. If every
    provider fails, raises `RuntimeError` chained from the last one.
    """

    def __init__(self, providers: list[AlignmentProvider]) -> None:
        if not providers:
            raise ValueError("FallbackAlign requires at least one provider")

        self.providers = providers
        self._disabled: set[int] = set()
        self._settled = [False] * len(providers)
        self._gate_locks = [threading.Lock() for _ in providers]
        self._info_logged = False
        self._lock = threading.Lock()

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        last_exc: Exception | None = None
        for index, provider in enumerate(self.providers):
            # Unlocked reads: a stale `False` on _settled costs at most one redundant
            # gate acquisition below, and a stale "not disabled" costs at most one
            # extra call; the authoritative check happens under the gate lock/self._lock.
            if index in self._disabled:
                continue

            if not self._settled[index]:
                with self._gate_locks[index]:
                    if not self._settled[index]:
                        # First call for this tier: make it while holding the tier's gate
                        # so threads that arrive meanwhile block on the outcome instead of
                        # all dispatching to a dead vendor. The call itself never holds
                        # self._lock, and this gate is only ever taken once per tier.
                        try:
                            boundaries = provider(clip, text)
                        except Exception as exc:
                            last_exc = exc
                            self._latch_off_once(index, provider, exc)
                            self._settled[index] = True
                            continue
                        else:
                            self._settled[index] = True
                            self._log_settled_once(provider)
                            return boundaries
                # Gate released without us being the first caller (another thread
                # settled this tier while we waited) -- fall through to the
                # unserialised path below, using our own outcome.

            if index in self._disabled:
                continue

            try:
                boundaries = provider(clip, text)
            except Exception as exc:
                last_exc = exc
                self._latch_off_once(index, provider, exc)
                continue

            self._log_settled_once(provider)
            return boundaries

        raise RuntimeError(
            "All speech-boundary measurement tiers are exhausted or latched off; "
            "cannot measure clip speech boundaries."
        ) from last_exc

    def _latch_off_once(self, index: int, provider: AlignmentProvider, exc: Exception) -> None:
        with self._lock:
            if index not in self._disabled:
                self._disabled.add(index)
                logger.warning(
                    f"{type(provider).__name__} unavailable ({exc}); falling back to the "
                    "next speech-boundary measurement tier."
                )

    def _log_settled_once(self, provider: AlignmentProvider) -> None:
        if self._info_logged:
            return
        with self._lock:
            if not self._info_logged:
                self._info_logged = True
                logger.info(f"Measuring dub clip speech boundaries via {type(provider).__name__}.")
