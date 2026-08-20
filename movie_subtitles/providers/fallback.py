import logging
import threading
from pathlib import Path

from movie_subtitles.providers.base import AlignmentProvider

logger = logging.getLogger("fallback")


class FallbackAlign:
    """Compose `AlignmentProvider`s into a degrading chain.

    Tries each provider in list order, skipping any already latched off. On success, logs
    (INFO, once per tier actually settled on -- at most one line per tier, not once per
    process) which provider it settled on and returns its result: a run that starts on
    tier 1 and later degrades to tier 2 logs both lines, not just the last one. A tier
    that is broken from the very start is gated: its very first call is made by exactly
    one thread while any others that arrive concurrently block on that outcome, so it
    costs at most one failing vendor call for the whole run, not one per waiting thread.
    That first call raising latches the tier off (WARNING, once) and degrades to the next
    provider; succeeding leaves it enabled and every subsequent call to it runs fully
    unserialised -- the lock is never held across a provider call in steady state. This
    one-failing-call bound holds only for a tier broken from the start: a tier that
    succeeds for a while and then starts failing (e.g. a vendor outage mid-run) is already
    past its gate, so up to as many threads as are calling concurrently can each make one
    failing call to it before `_latch_off_once` disables it -- bounded by the caller's
    worker count, once per tier, and the WARNING still reports it, but it is not the
    single-call bound the initial gate provides. If every provider fails, raises
    `RuntimeError` chained from the last one.
    """

    def __init__(self, providers: list[AlignmentProvider]) -> None:
        if not providers:
            raise ValueError("FallbackAlign requires at least one provider")

        self.providers = providers
        self._disabled: set[int] = set()
        self._settled = [False] * len(providers)
        self._gate_locks = [threading.Lock() for _ in providers]
        self._logged = [False] * len(providers)
        self._lock = threading.Lock()

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        last_exc: Exception | None = None
        for index, provider in enumerate(self.providers):
            # `self._lock` guards every read and write of `_disabled`/`_settled` so
            # neither is ever torn or stale under a memory model that doesn't give
            # ordinary dict/list/set mutations cross-thread visibility for free (a
            # free-threaded build). It is only ever held for the read/write itself,
            # never across a provider call, so an uncontended acquisition per call is
            # the only cost this adds -- the steady-state call path stays unserialised.
            with self._lock:
                disabled = index in self._disabled
                settled = self._settled[index]
            if disabled:
                continue

            if not settled:
                with self._gate_locks[index]:
                    with self._lock:
                        settled = self._settled[index]
                    if not settled:
                        # First call for this tier: make it while holding the tier's gate
                        # so threads that arrive meanwhile block on the outcome instead of
                        # all dispatching to a dead vendor. The call itself never holds
                        # self._lock, and this gate is only ever taken once per tier.
                        try:
                            boundaries = provider(clip, text)
                        except Exception as exc:
                            last_exc = exc
                            self._latch_off_once(index, provider, exc)
                            with self._lock:
                                self._settled[index] = True
                            continue
                        else:
                            with self._lock:
                                self._settled[index] = True
                            self._log_settled_once(index, provider)
                            return boundaries
                # Gate released without us being the first caller (another thread
                # settled this tier while we waited) -- fall through to the
                # unserialised path below, using our own outcome.

            with self._lock:
                disabled = index in self._disabled
            if disabled:
                continue

            try:
                boundaries = provider(clip, text)
            except Exception as exc:
                last_exc = exc
                self._latch_off_once(index, provider, exc)
                continue

            self._log_settled_once(index, provider)
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

    def _log_settled_once(self, index: int, provider: AlignmentProvider) -> None:
        with self._lock:
            logged = self._logged[index]
        if logged:
            return
        with self._lock:
            if not self._logged[index]:
                self._logged[index] = True
                logger.info(f"Measuring dub clip speech boundaries via {type(provider).__name__}.")
