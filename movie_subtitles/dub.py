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

# Default bound on how many TTS/alignment calls may be in flight at once. Threaded through
# from --dub-workers; a value of 1 reproduces today's serial behaviour exactly.
_MAX_WORKERS = 8

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


def fit_rate(actual_duration: float, slot_duration: float) -> float:
    """Compute the speaking rate needed to fit `actual_duration` into `slot_duration`.

    Returns the rate clamped to [_MIN_RATE, _MAX_RATE].
    """
    if slot_duration <= 0 or actual_duration <= 0:
        return 1.0

    return min(max(actual_duration / slot_duration, _MIN_RATE), _MAX_RATE)


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


def _lay_out_and_drift(group: list[Segment], clips: list[_Clip]) -> tuple[list[_Placement], float]:
    """Lay `group` out from its own anchor and report its accumulated drift.

    The group's anchor is its first segment's start and its source end is its last
    segment's end, both derived here so the two callers (the phase-A layout pass and the
    phase-B re-layout) cannot disagree. Drift is the placed end of the last clip measured
    against the group's source end; positive means the group ran long.
    """
    group_source_end = group[-1].end
    placements, placed_end = _layout_group(group, clips, group[0].start, group_source_end)
    return placements, placed_end - group_source_end


def _synthesise_and_measure(
    segment: Segment,
    text: str,
    tts: TTSProvider,
    aligner: AlignmentProvider,
    work_dir: Path,
    rate: float,
) -> _Clip:
    """Synthesise one segment at `rate` and measure its speech boundaries.

    This is the unit of work submitted to the thread pool for both phase A and phase B.
    """
    clip_path = work_dir / f"segment_{segment.id:05d}.mp3"
    _tts_with_retry(tts, text, clip_path, rate)
    speech_start, speech_end = aligner(clip_path, text)
    return _Clip(clip_path, speech_start, speech_end)


def _group_segments(segments: list[Segment]) -> list[list[Segment]]:
    """Partition segments into anchor groups by inter-segment silence gap.

    A new group starts whenever `segment.start - prev.end > _GAP_THRESHOLD`, computed
    over the *full* segment list (including untranslated segments) so an untranslated
    segment's span still counts toward the gap and cannot merge two scenes that should
    stay separate. Each group's anchor is its first segment's start.
    """
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    for segment in segments:
        if current and segment.start - current[-1].end > _GAP_THRESHOLD:
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return groups


def _submit_group(
    executor: ThreadPoolExecutor,
    group: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    aligner: AlignmentProvider,
    work_dir: Path,
    rate: float,
) -> list[Future[_Clip]]:
    """Submit every segment of `group` to `executor` at `rate`, in group order.

    Shared by both synthesis phases, so the unit-of-work signature is named once.
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
    failure, the still-queued tail is cancelled first, then every future is classified in
    one pass over its now-final state, so nothing that has completed can be reported as
    in-flight. Logs a WARNING accounting for each category of the batch (the four counts
    sum to the batch size) and re-raises the original exception instance from the
    lowest-submission-index failure, deterministic even when multiple tasks fail
    concurrently. Already-running tasks are left to finish by the caller's pool shutdown,
    so no worker outlives the exception.
    """
    grouped = list(batches)
    futures = [future for batch in grouped for future in batch]
    done, _ = wait(futures, return_when=FIRST_EXCEPTION)

    if any(future.exception() is not None for future in done):
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
        raise failures[min(failures)]

    return [[future.result() for future in batch] for batch in grouped]


def _layout_group(
    group: list[Segment],
    clips: list[_Clip],
    anchor: float,
    group_source_end: float,
) -> tuple[list[_Placement], float]:
    """Lay a group's clips out sequentially from its anchor.

    Clip 1 is placed at the anchor; each subsequent clip is placed at
    `prev_placed_end + min(source_gap, remaining_slack)`, where `remaining_slack` is the
    room left before the group's natural source end. Clip 1 needs no special case: it
    starts from `prev_source_end = group[0].start`, so its source gap is zero and the
    general formula already puts it exactly on the anchor. Returns the placements and the
    placed end of the last clip.
    """
    placements: list[_Placement] = []
    prev_placed_end = anchor
    prev_source_end = group[0].start
    for segment, clip in zip(group, clips, strict=True):
        source_gap = max(segment.start - prev_source_end, 0.0)
        remaining_slack = max(group_source_end - prev_placed_end, 0.0)
        placed_start = prev_placed_end + min(source_gap, remaining_slack)
        placements.append(_Placement(placed_start, clip.path, clip.speech_start, clip.speech_len))
        prev_placed_end = placed_start + clip.speech_len
        prev_source_end = segment.end

    return placements, prev_placed_end


def synthesise_track(
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    out_path: Path,
    work_dir: Path | None = None,
    *,
    aligner: AlignmentProvider,
    max_workers: int = _MAX_WORKERS,
) -> Path:
    """Synthesise translated segments and assemble them onto a silent timeline.

    `translations` maps segment.id -> translated text (segments with no translation,
    e.g. filtered out upstream, are skipped). Segments are partitioned into scene-anchored
    groups (see `_group_segments`); within a group, clips float sequentially from the
    group's anchor at their natural 1.0x rate. If the group's accumulated drift against its
    source span exceeds `_DRIFT_TOLERANCE`, the whole group is re-synthesised once at a
    single shared corrective rate clamped to [_MIN_RATE, _MAX_RATE].

    Synthesis and measurement run over one bounded `ThreadPoolExecutor` (size
    `max_workers`) in two flat batches, independent of group boundaries: phase A submits
    every translated segment of every group at 1.0x, each task synthesising with `tts` and
    then measuring with `aligner` (a `(clip: Path, text: str) -> (speech_start,
    speech_end)` callable, e.g. `FallbackAlign`). Each phase fails fast: if any task in the
    batch raises, no not-yet-started task in that batch is started (see `_resolve`), and
    the original exception propagates to the caller after up to `max_workers` already-
    running tasks finish. Results are otherwise collected by submission index, so a
    successful outcome does not depend on completion order. A purely arithmetic layout
    pass then walks the groups in order computing drift and, for any group past
    `_DRIFT_TOLERANCE`, its shared corrective rate. Phase B is a second flat batch,
    resynthesising only the segments of drifted groups at their group's rate; those groups
    are then re-laid-out. Per-group log lines are emitted from the layout passes, in group
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
        # Phase A: one flat batch over every segment of every group at 1.0x.
        phase_a_futures = [
            _submit_group(executor, group, translations, tts, aligner, work_dir, 1.0)
            for group in groups
        ]
        phase_a_clips = _resolve(phase_a_futures)

        # Layout pass (pure arithmetic): compute placements/drift and decide which groups
        # need a corrective re-synthesis, keyed by group index.
        placements_by_group: list[list[_Placement]] = []
        correction_rates: dict[int, float] = {}

        for idx, group in enumerate(groups):
            placements, drift = _lay_out_and_drift(group, phase_a_clips[idx])
            placements_by_group.append(placements)

            if abs(drift) > _DRIFT_TOLERANCE:
                group_speech_len = sum(placement.speech_len for placement in placements)
                shared_rate = fit_rate(group_speech_len, max(group[-1].end - group[0].start, 0.0))
                correction_rates[idx] = shared_rate
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drifted {drift:+.2f}s past its "
                    f"source span; re-synthesising once at a shared rate of {shared_rate:.2f}x."
                )
            else:
                logger.info(
                    f"Group {idx} ({len(group)} segment(s)) drift {drift:+.2f}s within "
                    "tolerance; left at 1.0x."
                )

        # Phase B: one flat batch over every segment of every drifted group, at that
        # group's shared rate.
        phase_b_futures = {
            idx: _submit_group(executor, groups[idx], translations, tts, aligner, work_dir, rate)
            for idx, rate in correction_rates.items()
        }
        phase_b_clips = dict(zip(phase_b_futures, _resolve(phase_b_futures.values()), strict=True))

    for idx in sorted(phase_b_clips):
        placements, drift = _lay_out_and_drift(groups[idx], phase_b_clips[idx])
        placements_by_group[idx] = placements
        logger.info(f"Group {idx} corrected; residual drift {drift:+.2f}s.")

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
