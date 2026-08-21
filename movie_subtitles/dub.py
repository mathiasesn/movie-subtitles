import logging
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from movie_subtitles.ffmpeg import run as run_ffmpeg
from movie_subtitles.providers.base import AlignmentProvider, Segment, TTSProvider

logger = logging.getLogger("dub")

# A new anchor group starts whenever the silence between two consecutive (translated)
# segments exceeds this many seconds. Groups are scenes: segments inside one float
# sequentially from the group's anchor instead of each being pinned to its own ASR start.
_GAP_THRESHOLD = 1.5

# A group is re-synthesised at a shared corrective rate once its accumulated drift
# (placed end vs. source end) exceeds this many seconds.
_DRIFT_TOLERANCE = 0.5

# Spec-mandated clamp for the group-level corrective rate. Intentionally tighter than the
# ElevenLabs API's own supported range (0.7-1.2, see providers/elevenlabs.py) so artefacts
# stay inaudible.
_MIN_RATE = 0.9
_MAX_RATE = 1.15

# A composed rate within this much of the rate already applied is treated as no change:
# below vendor speed granularity, so resynthesising would buy a full group of paid TTS
# (plus one aligner call per segment) for effectively identical audio.
_MIN_RATE_DELTA = 0.02

# Bound on how many correction passes a drifted group may go through. The initial pass
# (1.0x) does not count as a pass; this bounds the correction loop that follows it. Each
# pass is a full round of paid TTS for the groups it touches, so this is a cost control as
# much as a termination guard. `correction_passes=1` bounds correction to a single pass --
# it does not reproduce the old fixed phase-A/phase-B behaviour exactly, since the drift
# metric itself changed (speech time, not wall-clock cue span) and a non-improving pass is
# now discarded rather than always kept.
_MAX_CORRECTION_PASSES = 3

# Default bound on how many TTS/alignment calls may be in flight at once. Threaded through
# from --dub-workers. The default is 1 -- fully serial -- because vendor concurrency caps
# are per-subscription and low: an ElevenLabs plan permitting 3 concurrent requests 429s
# immediately at 8 workers, and the retry cannot help, since every worker backs off in
# lockstep and re-collides. Concurrency is therefore opt-in: raise it to whatever the
# resolved TTS vendor's plan actually allows.
_MAX_WORKERS = 1

# Shared empty default for the `voices` kwarg -- a mutable literal default is a bugbear
# violation (B006), so this module-level constant stands in for it; never mutated.
_NO_VOICES: dict[str | None, str | None] = {}

# Bounded retry around the TTS call: up to this many attempts, with an exponential
# 2**(attempt-1) second backoff between them (1s, 2s, ...).
_TTS_MAX_ATTEMPTS = 3

# Pre/post pad applied to each placed speech span before coalescing, so ducking (in
# mux.py) fades in slightly ahead of the dub's speech and releases slightly after it,
# rather than snapping exactly on the measured boundary.
_SPAN_PAD = 0.15


@dataclass(frozen=True)
class _Clip:
    """A synthesised segment clip and the speech span measured inside it."""

    path: Path
    speech_start: float
    speech_end: float

    @property
    def speech_len(self) -> float:
        return max(self.speech_end - self.speech_start, 0.0)


@dataclass(frozen=True)
class _Placement:
    """A clip's measured speech span placed at `start` on the output timeline."""

    start: float
    clip_path: Path
    trim_start: float
    speech_len: float


def _required_rate(actual_duration: float, slot_duration: float) -> float:
    """The unclamped speaking rate needed to fit `actual_duration` into `slot_duration`."""
    if slot_duration <= 0 or actual_duration <= 0:
        return 1.0
    return actual_duration / slot_duration


def _tts_with_retry(
    tts: TTSProvider, text: str, clip_path: Path, rate: float, voice: str | None
) -> None:
    """Call `tts` with a provider-agnostic bounded retry.

    Up to `_TTS_MAX_ATTEMPTS` attempts, sleeping an exponential `2**(attempt-1)` seconds
    between them; the last exception is re-raised if every attempt fails. Catches plain
    `Exception` rather than a vendor 429 type so no vendor SDK needs importing into this
    module, and transient network blips get covered too. This wraps only the TTS call --
    NOT the aligner call, which is handled separately (see `_synthesise_and_measure`): an
    aligner exception is an intended degrade signal that `FallbackAlign` acts on by moving
    to the next tier and latching the failed one off, so retrying it here would fight that.

    `voice` is passed straight through to every attempt unchanged -- retrying never
    substitutes a different voice for the one the caller resolved. `None` reaches the
    provider as-is, which is what makes it use its own configured default.
    """
    for attempt in range(1, _TTS_MAX_ATTEMPTS + 1):
        try:
            tts(text, clip_path, speed=rate, voice=voice)
            return
        except Exception as exc:
            if attempt == _TTS_MAX_ATTEMPTS:
                raise
            backoff = 2 ** (attempt - 1)
            logger.warning(
                f"TTS call for {clip_path.name} failed ({exc}); retrying in {backoff}s "
                f"(attempt {attempt}/{_TTS_MAX_ATTEMPTS})."
            )
            time.sleep(backoff)


def _speech_bounds(segment: Segment) -> tuple[float, float]:
    """The segment's real speech interval: word bounds when available, else cue bounds.

    `(words[0].start, words[-1].end)` when word timings are available, else
    `(segment.start, segment.end)`. This is the single source of truth for "how long did
    this segment really take to say" -- both drift measurement (`_speech_duration`) and
    anchor grouping (`_inter_segment_gap`) derive from it, so they cannot silently
    disagree about a segment's speech span.
    """
    if segment.words:
        return segment.words[0].start, segment.words[-1].end
    return segment.start, segment.end


def _speech_duration(segment: Segment) -> float:
    """The real speech duration of one segment.

    `words[-1].end - words[0].start` when word timings are available, else the cue span
    `segment.end - segment.start`. This is what drift is measured against (layer 5): the
    wall-clock cue span can include long intra-cue pauses that have nothing to do with how
    long the synthesised speech should take.
    """
    start, end = _speech_bounds(segment)
    return max(end - start, 0.0)


def _group_speech_span(group: list[Segment]) -> float:
    """Summed source speech duration for every segment in `group`."""
    return sum(_speech_duration(segment) for segment in group)


def _lay_out_and_drift(group: list[Segment], clips: list[_Clip]) -> tuple[list[_Placement], float]:
    """Lay `group` out from its own anchor and report its accumulated drift.

    The group's anchor is its first segment's start, derived here so every caller (the
    initial layout pass and each correction pass's re-layout) cannot disagree. Drift is
    measured as the synthesised speech total (summed clip speech lengths) against the
    group's summed *source speech time* (`_group_speech_span`, layer 5) -- not the
    wall-clock cue span the old implementation used, which invented drift out of ordinary
    intra-cue pauses. Positive means the synthesised speech ran long relative to the
    source speech it should match.
    """
    placements, placed_end = _layout_group(group, clips)
    speech_total = sum(clip.speech_len for clip in clips)
    drift = speech_total - _group_speech_span(group)
    return placements, drift


def _synthesise_and_measure(
    segment: Segment,
    text: str,
    tts: TTSProvider,
    aligner: AlignmentProvider,
    work_dir: Path,
    rate: float,
    pass_num: int,
    voice: str | None,
) -> _Clip:
    """Synthesise one segment at `rate` for correction pass `pass_num` and measure its
    speech boundaries.

    This is the unit of work submitted to the thread pool for the initial pass (pass 0)
    and every correction pass thereafter. `pass_num` is baked into the on-disk path
    (`segment_{id:05d}_p{pass_num:02d}.mp3`) rather than the rate: the rate clamp
    saturating means two different passes very often request the identical rate (e.g.
    repeated 0.90x), which would collide on a rate-keyed name. Every `_Clip` this returns
    therefore always points at the exact audio it was measured from, even if a later pass
    is rejected and an earlier pass's `_Clip` is what the caller keeps.

    `voice` is the voice id resolved for this segment's speaker (or `None` for "use the
    provider's configured default"), threaded straight through to `_tts_with_retry` --
    it does not affect the on-disk path, since a segment's speaker (and therefore its
    voice) does not change across correction passes.
    """
    clip_path = work_dir / f"segment_{segment.id:05d}_p{pass_num:02d}.mp3"
    _tts_with_retry(tts, text, clip_path, rate, voice)
    speech_start, speech_end = aligner(clip_path, text)
    return _Clip(clip_path, speech_start, speech_end)


def _inter_segment_gap(prev: Segment, nxt: Segment) -> float:
    """Compute the silence gap between two consecutive segments.

    Computed as `next`'s speech start minus `prev`'s speech end, both via
    `_speech_bounds` -- word-level timings when available (so a backend emitting
    exactly-contiguous cue boundaries, e.g. ElevenLabs Scribe, see
    `specs/fix-dub-out-of-sync-contiguous-cues.md`, still yields real gaps), else the
    cue-boundary fallback (`local`/`openai` ASR, where `words is None`), degrading cleanly
    rather than requiring word timings everywhere.
    """
    _, prev_end = _speech_bounds(prev)
    nxt_start, _ = _speech_bounds(nxt)
    return nxt_start - prev_end


def _group_segments(segments: list[Segment]) -> list[list[Segment]]:
    """Partition segments into anchor groups by inter-segment silence gap.

    A new group starts whenever `_inter_segment_gap(prev, segment) > _GAP_THRESHOLD`,
    computed over the *full* segment list (including untranslated segments) so an
    untranslated segment's span still counts toward the gap and cannot merge two scenes
    that should stay separate. Each group's anchor is its first segment's start.

    If this partitions the whole input into a single group with no internal gap exceeding
    `_GAP_THRESHOLD`, that is logged as a WARNING rather than accepted silently -- it is
    the degenerate case that let a whole-clip group swallow all trailing slack as silence
    (see the spec above).
    """
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    for segment in segments:
        if current and _inter_segment_gap(current[-1], segment) > _GAP_THRESHOLD:
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)

    if len(segments) > 1 and len(groups) == 1:
        logger.warning(
            f"Segmentation produced a single anchor group spanning all {len(segments)} "
            "segment(s) with no inter-segment gap exceeding "
            f"{_GAP_THRESHOLD}s; drift correction will have no inter-scene silence to "
            "absorb into."
        )

    return groups


def _submit_group(
    executor: ThreadPoolExecutor,
    group: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    aligner: AlignmentProvider,
    work_dir: Path,
    rate: float,
    pass_num: int,
    voices: dict[str | None, str | None] = _NO_VOICES,
) -> list[Future[_Clip]]:
    """Submit every segment of `group` to `executor` at `rate` for pass `pass_num`, in
    group order.

    Shared by the initial pass and every correction pass, so the unit-of-work signature is
    named once. `pass_num` (0 for the initial pass, 1.. for each correction pass) is
    threaded through to `_synthesise_and_measure` so each pass writes to its own on-disk
    path instead of overwriting a previous pass's file. `voices` maps speaker label -> voice
    id; each segment looks up its own `voices.get(segment.speaker)` here rather than the
    caller resolving one voice per group, since a group can span more than one speaker.
    `voices=None` (or a segment whose speaker has no entry) resolves to `None`, which is
    "no voice specified" all the way down to the provider.
    """
    return [
        executor.submit(
            _synthesise_and_measure,
            segment,
            translations[segment.id],
            tts,
            aligner,
            work_dir,
            rate,
            pass_num,
            voices.get(segment.speaker),
        )
        for segment in group
    ]


def _resolve(batches: Iterable[list[Future[_Clip]]]) -> list[list[_Clip]]:
    """Wait for every future across every batch in `batches`, failing fast, and return
    their clips per batch.

    Shared by both synthesis phases so their abort behaviour cannot drift, and the only
    way to get at the clips -- collecting results without first waiting is therefore not
    expressible. `batches` is flattened in full before waiting: on a failure, the abort
    cancels across every batch handed in, not just the one containing the failure (phase
    B passes every drifted group's batch at once). Waits with `FIRST_EXCEPTION`; on a
    failure -- including a future in `done` that came back cancelled rather than failed,
    which is treated as an abort just the same, even though nothing upstream cancels a
    future before `_resolve()` sees it today -- the still-queued tail is cancelled first,
    then every future is classified in one pass. The four counts (succeeded / failed /
    cancelled / in-flight) are mutually exclusive and sum to the batch size, and nothing
    that has already completed by the time of that pass can be reported as in-flight --
    but "in-flight" is not a final state: those tasks are still running and may finish
    before the WARNING below is even emitted. Logs that WARNING accounting for each
    category of the batch and re-raises the original exception instance from the
    lowest-submission-index failure among the failures observed during this
    classification pass -- deterministic given that set, but not a total order over every
    failure that could occur, since which failures are visible depends on what else
    completed after `wait()` returned. If the abort was triggered only by a cancellation,
    with no task having actually raised, raises `RuntimeError` instead (there is no
    original exception to re-raise). Already-running tasks are left to finish by the
    caller's pool shutdown, so no worker outlives the exception.
    """
    grouped = list(batches)
    futures = [future for batch in grouped for future in batch]
    done, _ = wait(futures, return_when=FIRST_EXCEPTION)

    # `future.cancelled()` is checked first in both branches below, short-circuiting
    # before `future.exception()` is ever called on a cancelled future -- calling it on
    # one raises `CancelledError` instead of returning `None`. A cancelled future in
    # `done` (nothing produces one today, but a future caller might) is itself treated as
    # an abort, not silently skipped: falling through to `future.result()` for it would
    # raise an unguarded `CancelledError`, which is a `BaseException` outside `main()`'s
    # caught tuple and would surface as a bare traceback instead of a one-line message.
    if any(future.cancelled() or future.exception() is not None for future in done):
        # Cancel the still-queued tail first; a future that already started or finished
        # by the time we get to it below returns False here and is classified from its
        # own (now final) state instead, so it can never land in the in-flight bucket.
        for future in futures:
            if future not in done:
                future.cancel()

        succeeded = failed = cancelled = in_flight = 0
        failures: dict[int, BaseException] = {}
        for index, future in enumerate(futures):
            if future.cancelled():
                cancelled += 1
            elif future.done():
                exc = future.exception()
                if exc is not None:
                    failed += 1
                    failures[index] = exc
                else:
                    succeeded += 1
            else:
                in_flight += 1

        logger.warning(
            f"Dub synthesis task failed: {succeeded} succeeded, {failed} failed, "
            f"{cancelled} cancelled before starting, {in_flight} still in flight and "
            f"will finish before the abort completes, out of {len(futures)} segment(s)."
        )
        if failures:
            raise failures[min(failures)]
        raise RuntimeError(
            "Dub synthesis batch aborted: a task was cancelled with no other task having raised."
        )

    return [[future.result() for future in batch] for batch in grouped]


def _layout_group(group: list[Segment], clips: list[_Clip]) -> tuple[list[_Placement], float]:
    """Lay a group's clips out, pinned to their own ASR start.

    Each clip is placed at `max(segment.start, prev_placed_end)`, seeded from
    `group[0].start`. This is what stops a group whose speech is shorter than its source
    span from piling all the unused slack up as trailing silence at the end: later clips
    still start at/after their own `segment.start` instead of floating from wherever the
    previous clip happened to end. Returns the placements and the placed end of the last
    clip.
    """
    placements: list[_Placement] = []
    prev_placed_end = group[0].start
    for segment, clip in zip(group, clips, strict=True):
        placed_start = max(segment.start, prev_placed_end)
        placements.append(_Placement(placed_start, clip.path, clip.speech_start, clip.speech_len))
        prev_placed_end = placed_start + clip.speech_len

    return placements, prev_placed_end


def _corrective_rate(
    idx: int,
    group: list[Segment],
    current_rate: float,
    drift: float,
    speech_total: float,
) -> float:
    """Compute group `idx`'s next corrective rate by composing onto `current_rate`.

    `speech_total` -- the synthesised speech total the caller already computed while
    measuring `drift` (`_lay_out_and_drift`) -- is the clip-side half of the exact-fit
    ratio; `source_span` (computed here) is the source-side half, so both halves of the
    drift definition are visible together in one place. `speech_total` reflects
    `current_rate`, not 1.0x, so the exact-fit ratio `_required_rate(speech_total,
    source_span)` it implies is relative to `current_rate`, not absolute -- applying it as
    an absolute `voice_settings.speed` would silently discard whatever correction
    `current_rate` already bought (e.g. requesting 1.10x, then on the next pass computing
    1.042x from the result and applying that as-is, which is *slower* than 1.10x and
    undoes the first pass). The next rate is therefore `current_rate *
    _required_rate(...)`, clamped to [_MIN_RATE, _MAX_RATE] same as before. When the
    unclamped composed rate falls outside that range, the clamped rate is used but a
    WARNING names both the unclamped rate and the residual `drift`, so a structural
    shortfall a rate change cannot fix stays visible instead of reading as success.
    """
    source_span = _group_speech_span(group)
    unclamped = current_rate * _required_rate(speech_total, source_span)
    rate = min(max(unclamped, _MIN_RATE), _MAX_RATE)
    if rate != unclamped:
        logger.warning(
            f"Group {idx} ({len(group)} segment(s)) needs an unclamped rate of "
            f"{unclamped:.2f}x (composed onto the already-applied {current_rate:.2f}x) to "
            f"close a {drift:+.2f}s drift; clamping to {rate:.2f}x, which will not fully "
            "correct it -- this is a structural shortfall, not a speaking-rate problem."
        )
    return rate


def _speech_spans(placements: list[_Placement]) -> list[tuple[float, float]]:
    """Coalesced, sorted `(start, end)` speech spans from placed clips.

    Each placement's `[start, start + speech_len]` is padded by `_SPAN_PAD` on both
    sides, then adjacent/overlapping spans are merged so callers (mux.py's ducking
    filter) never see two spans that touch or cross.
    """
    raw = sorted(
        (
            max(p.start - _SPAN_PAD, 0.0),
            p.start + p.speech_len + _SPAN_PAD,
        )
        for p in placements
    )

    merged: list[tuple[float, float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def synthesise_track(
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    out_path: Path,
    work_dir: Path | None = None,
    *,
    aligner: AlignmentProvider,
    max_workers: int = _MAX_WORKERS,
    correction_passes: int = _MAX_CORRECTION_PASSES,
    voices: dict[str | None, str | None] = _NO_VOICES,
) -> tuple[Path, list[tuple[float, float]]]:
    """Synthesise translated segments and assemble them onto a silent timeline.

    `translations` maps segment.id -> translated text (segments with no translation,
    e.g. filtered out upstream, are skipped). Segments are partitioned into scene-anchored
    groups (see `_group_segments`); within a group, clips are pinned to their own ASR
    start and float forward only when the previous clip overran it (`_layout_group`). If a
    group's accumulated drift -- synthesised speech total against summed *source speech
    time*, not wall-clock cue span (`_lay_out_and_drift`) -- exceeds `_DRIFT_TOLERANCE`,
    the whole group is re-synthesised at a shared corrective rate clamped to
    [_MIN_RATE, _MAX_RATE].

    Synthesis and measurement run over one bounded `ThreadPoolExecutor` (size
    `max_workers`). The initial pass submits every translated segment of every group at
    1.0x as one flat batch, each task synthesising with `tts` and then measuring with
    `aligner` (a `(clip: Path, text: str) -> (speech_start, speech_end)` callable, e.g.
    `FallbackAlign`). A purely arithmetic layout pass then walks the groups computing
    drift and, for any group past `_DRIFT_TOLERANCE`, its shared corrective rate.

    What follows is a bounded correction loop, run for up to `correction_passes` passes:
    each pass submits one flat cross-group batch -- only the segments of groups still
    outside tolerance -- through the same `_submit_group`/`_resolve` path, so the
    documented fail-fast and abort-accounting semantics hold per pass exactly as they did
    for the initial pass. After each pass, every corrected group is re-measured; a group
    drops out of the loop as soon as it is back within tolerance, or as soon as a pass
    fails to reduce its drift, or a pass's composed rate has saturated the clamp (cost
    control -- each pass is paid TTS, and a group that cannot improve, or would only
    resynthesise identical audio, is not worth paying for again). Corrective rates compose
    across passes (`_corrective_rate` multiplies onto the rate the group's current clips
    were actually synthesised at, not the 1.0x baseline), since each pass's measurement
    reflects the *previous* pass's rate, not 1.0x. `correction_passes=1` bounds correction
    to a single pass -- it does not reproduce the old fixed phase-A/phase-B behaviour
    exactly, since the drift metric itself changed (speech time, not wall-clock cue span)
    and a non-improving pass is now discarded rather than always kept.

    Each phase/pass fails fast: if any task in its batch raises, no not-yet-started task
    in that batch is started (see `_resolve`), and the original exception propagates to
    the caller after up to `max_workers` already-running tasks finish. Results are
    otherwise collected by submission index, so a successful outcome does not depend on
    completion order. Per-group log lines are emitted from the layout passes, in group
    order, so they never interleave with synthesis. Produces one continuous audio file at
    `out_path`.

    `voices` maps speaker label -> voice id and is threaded down through `_submit_group` to
    every `tts(...)` call, initial pass and every correction pass alike, via
    `voices.get(segment.speaker)`; a segment's voice therefore never changes across passes,
    only its rate does. `voices=None` (the default) or a speaker missing from the mapping
    resolves to `voice=None`, which reaches the provider unchanged and makes it fall back
    to its own configured default -- so callers that never pass `voices` see byte-for-byte
    today's single-voice behaviour. This has no bearing on timing, grouping, drift or the
    correction loop, all of which operate purely on `Segment`/`_Clip` timestamps.

    Returns `(out_path, spans)`, where `spans` is the coalesced, sorted list of
    `(start, end)` placed-speech windows across every group (`_speech_spans`), padded
    slightly on each side and with adjacent/overlapping windows merged. Callers (see
    `cli.py:_dub_and_mux`) thread `spans` into `mux.py:mux_dub()` to duck the original
    audio only while the dub is actually speaking.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="movie-subtitles-dub-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        translated
        for raw_group in _group_segments(segments)
        if (translated := [s for s in raw_group if translations.get(s.id)])
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Initial pass: one flat batch over every segment of every group at 1.0x.
        initial_futures = [
            _submit_group(executor, group, translations, tts, aligner, work_dir, 1.0, 0, voices)
            for group in groups
        ]
        clips_by_group: dict[int, list[_Clip]] = dict(enumerate(_resolve(initial_futures)))

        # Layout pass (pure arithmetic): compute placements/drift and decide which groups
        # need a corrective re-synthesis, keyed by group index. `pending` is the single
        # source of truth for "still being corrected": idx -> (drift, applied_rate), where
        # applied_rate is the rate `clips_by_group[idx]` was actually synthesised at. It
        # replaces two hand-synchronised dicts (a drift map and a separate rate map) whose
        # keysets had to be kept in lockstep by hand; rebuilding it wholesale at the end of
        # each pass (like `still_pending` below) means there is nothing to `del`.
        placements_by_group: list[list[_Placement]] = []
        pending: dict[int, tuple[float, float]] = {}

        for idx, group in enumerate(groups):
            placements, drift = _lay_out_and_drift(group, clips_by_group[idx])
            placements_by_group.append(placements)
            if abs(drift) > _DRIFT_TOLERANCE:
                pending[idx] = (drift, 1.0)
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drifted {drift:+.2f}s against "
                    "its source speech time; entering the correction loop."
                )
            else:
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drift {drift:+.2f}s within "
                    "tolerance; left at 1.0x."
                )

        # Groups the loop gave up on early (saturated/near-identical rate, or a pass that
        # failed to improve), with the residual drift and why, so every abandoned group
        # reaches the single terminal warning loop below alongside groups that simply ran
        # out of passes.
        abandoned: dict[int, tuple[float, str]] = {}

        for pass_num in range(1, correction_passes + 1):
            if not pending:
                break

            rates: dict[int, float] = {}
            for idx, (drift, current_rate) in pending.items():
                group = groups[idx]
                # `speech_total` is the clip-side half of the drift definition that
                # `_lay_out_and_drift` already computed a moment ago (for the initial pass)
                # or a pass ago (for every subsequent one): drift = speech_total -
                # source_span, so it is recovered arithmetically instead of re-summing
                # `clips_by_group[idx]`.
                speech_total = drift + _group_speech_span(group)
                rate = _corrective_rate(idx, group, current_rate, drift, speech_total)
                if abs(rate - current_rate) < _MIN_RATE_DELTA:
                    # The newly-composed rate is within a hair of the one already applied,
                    # so resynthesising would pay for effectively identical audio. Usually
                    # the clamp saturating, but not always -- `_required_rate` also returns
                    # exactly 1.0 for a non-positive span -- so only say "clamp" when the
                    # rate actually sits on a bound. Distinct from "did not reduce drift"
                    # below: this group is not even attempted this pass.
                    reason = (
                        "clamp saturated"
                        if rate in (_MIN_RATE, _MAX_RATE)
                        else "no further adjustment implied"
                    )
                    logger.info(
                        f"Group {idx} pass {pass_num}/{correction_passes}: composed rate "
                        f"{rate:.2f}x is within {_MIN_RATE_DELTA} of the already-applied "
                        f"{current_rate:.2f}x ({reason}); skipping re-synthesis for this "
                        "group."
                    )
                    abandoned[idx] = (drift, "no further rate change could improve it")
                else:
                    rates[idx] = rate

            pending = {idx: pending[idx] for idx in rates}
            if not pending:
                break

            pass_futures = {
                idx: _submit_group(
                    executor,
                    groups[idx],
                    translations,
                    tts,
                    aligner,
                    work_dir,
                    rate,
                    pass_num,
                    voices,
                )
                for idx, rate in rates.items()
            }
            pass_clips = dict(zip(pass_futures, _resolve(pass_futures.values()), strict=True))

            still_pending: dict[int, tuple[float, float]] = {}
            for idx, new_clips in pass_clips.items():
                group = groups[idx]
                old_drift, _ = pending[idx]
                placements, new_drift = _lay_out_and_drift(group, new_clips)

                if abs(new_drift) < abs(old_drift):
                    clips_by_group[idx] = new_clips
                    placements_by_group[idx] = placements
                    logger.info(
                        f"Group {idx} pass {pass_num}/{correction_passes}: drift "
                        f"{old_drift:+.2f}s -> {new_drift:+.2f}s at {rates[idx]:.2f}x."
                    )
                    if abs(new_drift) > _DRIFT_TOLERANCE:
                        still_pending[idx] = (new_drift, rates[idx])
                else:
                    logger.warning(
                        f"Group {idx} pass {pass_num}/{correction_passes} did not reduce "
                        f"drift ({old_drift:+.2f}s -> {new_drift:+.2f}s); stopping "
                        "correction for this group."
                    )
                    abandoned[idx] = (old_drift, "no further rate change could improve it")

            pending = still_pending

        terminal: dict[int, tuple[float, str]] = {
            idx: (drift, f"correction exhausted its {correction_passes}-pass bound")
            for idx, (drift, _rate) in pending.items()
        }
        terminal.update(abandoned)
        for idx, (drift, reason) in terminal.items():
            logger.warning(f"Group {idx} left drifted {drift:+.2f}s; {reason}.")

    inputs: list[_Placement] = []
    for placements in placements_by_group:
        inputs.extend(placements)

    _assemble_timeline(inputs, out_path)
    return out_path, _speech_spans(inputs)


def _assemble_timeline(inputs: list[_Placement], out_path: Path) -> Path:
    """Lay clips onto a silent stereo timeline at their placed offsets via ffmpeg.

    Each input's leading TTS padding is skipped with an input seek offset (`-ss`) and its
    trailing padding is dropped with a matching duration cap (`-t`), so only the measured
    speech span of the clip -- `[trim_start, trim_start + speech_len]` -- is placed on the
    timeline at `placed_start`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        # No segments to place; emit a minimal silent file.
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                "0.1",
                str(out_path),
            ],
            what=f"Writing the empty dub timeline to {out_path.name}",
        )
        return out_path

    cmd = ["ffmpeg", "-y"]
    for placement in inputs:
        cmd += [
            "-ss",
            f"{placement.trim_start:.3f}",
            "-t",
            f"{max(placement.speech_len, 0.01):.3f}",
            "-i",
            str(placement.clip_path),
        ]

    filter_parts = []
    mix_labels = []
    for idx, placement in enumerate(inputs):
        delay_ms = int(round(placement.start * 1000))
        filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")
        mix_labels.append(f"[a{idx}]")

    filter_complex = ";".join(filter_parts)
    filter_complex += f";{''.join(mix_labels)}amix=inputs={len(inputs)}:normalize=0[out]"

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        str(out_path),
    ]

    run_ffmpeg(cmd, what=f"Assembling {len(inputs)} dub clip(s) into {out_path.name}")
    return out_path
