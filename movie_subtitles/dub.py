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

# Bound on how many correction passes a drifted group may go through. Phase A (1.0x) does
# not count as a pass; this bounds the correction loop that follows it. Each pass is a full
# round of paid TTS for the groups it touches, so this is a cost control as much as a
# termination guard. `correction_passes=1` reproduces the old fixed single-phase-B
# behaviour exactly.
_MAX_CORRECTION_PASSES = 3

# Default bound on how many TTS/alignment calls may be in flight at once. Threaded through
# from --dub-workers. The default is 1 -- fully serial -- because vendor concurrency caps
# are per-subscription and low: an ElevenLabs plan permitting 3 concurrent requests 429s
# immediately at 8 workers, and the retry cannot help, since every worker backs off in
# lockstep and re-collides. Concurrency is therefore opt-in: raise it to whatever the
# resolved TTS vendor's plan actually allows.
_MAX_WORKERS = 1

# Bounded retry around the TTS call: up to this many attempts, with an exponential
# 2**(attempt-1) second backoff between them (1s, 2s, ...).
_TTS_MAX_ATTEMPTS = 3


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


def fit_rate(actual_duration: float, slot_duration: float) -> float:
    """Compute the speaking rate needed to fit `actual_duration` into `slot_duration`.

    Returns the rate clamped to [_MIN_RATE, _MAX_RATE].
    """
    return min(max(_required_rate(actual_duration, slot_duration), _MIN_RATE), _MAX_RATE)


def _tts_with_retry(tts: TTSProvider, text: str, clip_path: Path, rate: float) -> None:
    """Call `tts` with a provider-agnostic bounded retry.

    Up to `_TTS_MAX_ATTEMPTS` attempts, sleeping an exponential `2**(attempt-1)` seconds
    between them; the last exception is re-raised if every attempt fails. Catches plain
    `Exception` rather than a vendor 429 type so no vendor SDK needs importing into this
    module, and transient network blips get covered too. This wraps only the TTS call --
    NOT the aligner call, which is handled separately (see `_synthesise_and_measure`): an
    aligner exception is an intended degrade signal that `FallbackAlign` acts on by moving
    to the next tier and latching the failed one off, so retrying it here would fight that.
    """
    for attempt in range(1, _TTS_MAX_ATTEMPTS + 1):
        try:
            tts(text, clip_path, speed=rate)
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


def _speech_duration(segment: Segment) -> float:
    """The real speech duration of one segment.

    `words[-1].end - words[0].start` when word timings are available, else the cue span
    `segment.end - segment.start`. This is what drift is measured against (layer 5): the
    wall-clock cue span can include long intra-cue pauses that have nothing to do with how
    long the synthesised speech should take.
    """
    if segment.words:
        return segment.words[-1].end - segment.words[0].start
    return max(segment.end - segment.start, 0.0)


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
    placements, placed_end = _layout_group(group, clips, group[0].start)
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
    """
    clip_path = work_dir / f"segment_{segment.id:05d}_p{pass_num:02d}.mp3"
    _tts_with_retry(tts, text, clip_path, rate)
    speech_start, speech_end = aligner(clip_path, text)
    return _Clip(clip_path, speech_start, speech_end)


def _inter_segment_gap(prev: Segment, nxt: Segment) -> float:
    """Compute the silence gap between two consecutive segments.

    Uses word-level timings (`next.words[0].start - prev.words[-1].end`) when *both*
    segments carry non-empty `words`, so a backend emitting exactly-contiguous cue
    boundaries (e.g. ElevenLabs Scribe, see `specs/fix-dub-out-of-sync-contiguous-cues.md`)
    still yields real gaps. Falls back to the cue-boundary gap `next.start - prev.end`
    when either side lacks word timings (e.g. `local`/`openai` ASR, where `words is None`),
    degrading cleanly rather than requiring word timings everywhere.
    """
    if prev.words and nxt.words:
        return nxt.words[0].start - prev.words[-1].end
    return nxt.start - prev.end


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
) -> list[Future[_Clip]]:
    """Submit every segment of `group` to `executor` at `rate` for pass `pass_num`, in
    group order.

    Shared by the initial pass and every correction pass, so the unit-of-work signature is
    named once. `pass_num` (0 for the initial pass, 1.. for each correction pass) is
    threaded through to `_synthesise_and_measure` so each pass writes to its own on-disk
    path instead of overwriting a previous pass's file.
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


def _layout_group(
    group: list[Segment],
    clips: list[_Clip],
    anchor: float,
) -> tuple[list[_Placement], float]:
    """Lay a group's clips out, pinned to their own ASR start.

    Each clip is placed at `max(segment.start, prev_placed_end)`: it starts at its own
    source start unless the previous clip is still running past that point, in which case
    it floats forward just enough to avoid overlapping. `anchor` is accepted for interface
    symmetry with the group's declared start but is otherwise unused, since clip 1 already
    pins to `group[0].start`. This is what stops a group whose speech is shorter than its
    source span from piling all the unused slack up as trailing silence at the end: later
    clips still start at/after their own `segment.start` instead of floating from
    wherever the previous clip happened to end. Returns the placements and the placed end
    of the last clip.
    """
    placements: list[_Placement] = []
    prev_placed_end = anchor
    for segment, clip in zip(group, clips, strict=True):
        placed_start = max(segment.start, prev_placed_end)
        placements.append(_Placement(placed_start, clip.path, clip.speech_start, clip.speech_len))
        prev_placed_end = placed_start + clip.speech_len

    return placements, prev_placed_end


def _corrective_rate(idx: int, group: list[Segment], drift: float, clips: list[_Clip]) -> float:
    """Compute group `idx`'s shared corrective rate and warn if the clamp binds.

    The rate is derived from the synthesised speech total against the group's source
    speech span (`_group_speech_span`, layer 5), not the wall-clock cue span. When the
    unclamped rate falls outside [_MIN_RATE, _MAX_RATE], the clamped rate is used but a
    WARNING names both the unclamped rate and the residual `drift`, so a structural
    shortfall a rate change cannot fix stays visible instead of reading as success.
    """
    speech_total = sum(clip.speech_len for clip in clips)
    source_span = _group_speech_span(group)
    unclamped = _required_rate(speech_total, source_span)
    rate = min(max(unclamped, _MIN_RATE), _MAX_RATE)
    if rate != unclamped:
        logger.warning(
            f"Group {idx} ({len(group)} segment(s)) needs an unclamped rate of "
            f"{unclamped:.2f}x to close a {drift:+.2f}s drift; clamping to {rate:.2f}x, "
            "which will not fully correct it -- this is a structural shortfall, not a "
            "speaking-rate problem."
        )
    return rate


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
) -> Path:
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
    fails to reduce its drift (cost control -- each pass is paid TTS, and a group that
    cannot improve is not worth paying for again). `correction_passes=1` runs exactly one
    such pass, reproducing the old fixed phase-A/phase-B behaviour exactly.

    Each phase/pass fails fast: if any task in its batch raises, no not-yet-started task
    in that batch is started (see `_resolve`), and the original exception propagates to
    the caller after up to `max_workers` already-running tasks finish. Results are
    otherwise collected by submission index, so a successful outcome does not depend on
    completion order. Per-group log lines are emitted from the layout passes, in group
    order, so they never interleave with synthesis. Produces one continuous audio file at
    `out_path`. Returns `out_path`.
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
            _submit_group(executor, group, translations, tts, aligner, work_dir, 1.0, 0)
            for group in groups
        ]
        clips_by_group: dict[int, list[_Clip]] = dict(enumerate(_resolve(initial_futures)))

        # Layout pass (pure arithmetic): compute placements/drift and decide which groups
        # need a corrective re-synthesis, keyed by group index.
        placements_by_group: list[list[_Placement]] = []
        drift_by_group: dict[int, float] = {}

        for idx, group in enumerate(groups):
            placements, drift = _lay_out_and_drift(group, clips_by_group[idx])
            placements_by_group.append(placements)
            if abs(drift) > _DRIFT_TOLERANCE:
                drift_by_group[idx] = drift
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drifted {drift:+.2f}s against "
                    "its source speech time; entering the correction loop."
                )
            else:
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drift {drift:+.2f}s within "
                    "tolerance; left at 1.0x."
                )

        # Bounded correction loop: only groups still outside tolerance are resubmitted
        # each pass; a group drops out once it is back in tolerance or a pass fails to
        # improve it.
        for pass_num in range(1, correction_passes + 1):
            if not drift_by_group:
                break

            rates = {
                idx: _corrective_rate(idx, groups[idx], drift, clips_by_group[idx])
                for idx, drift in drift_by_group.items()
            }
            pass_futures = {
                idx: _submit_group(
                    executor, groups[idx], translations, tts, aligner, work_dir, rate, pass_num
                )
                for idx, rate in rates.items()
            }
            pass_clips = dict(zip(pass_futures, _resolve(pass_futures.values()), strict=True))

            still_drifted: dict[int, float] = {}
            for idx, new_clips in pass_clips.items():
                group = groups[idx]
                old_drift = drift_by_group[idx]
                placements, new_drift = _lay_out_and_drift(group, new_clips)

                if abs(new_drift) < abs(old_drift):
                    clips_by_group[idx] = new_clips
                    placements_by_group[idx] = placements
                    logger.info(
                        f"Group {idx} pass {pass_num}/{correction_passes}: drift "
                        f"{old_drift:+.2f}s -> {new_drift:+.2f}s."
                    )
                    if abs(new_drift) > _DRIFT_TOLERANCE:
                        still_drifted[idx] = new_drift
                else:
                    logger.warning(
                        f"Group {idx} pass {pass_num}/{correction_passes} did not reduce "
                        f"drift ({old_drift:+.2f}s -> {new_drift:+.2f}s); stopping "
                        "correction for this group."
                    )

            drift_by_group = still_drifted

        if drift_by_group:
            for idx, drift in drift_by_group.items():
                logger.warning(
                    f"Group {idx} still drifted {drift:+.2f}s after "
                    f"{correction_passes} correction pass(es)."
                )

    inputs: list[_Placement] = []
    for placements in placements_by_group:
        inputs.extend(placements)

    _assemble_timeline(inputs, out_path)
    return out_path


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
