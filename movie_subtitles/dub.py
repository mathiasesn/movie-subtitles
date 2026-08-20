import logging
import re
import tempfile
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

# Tolerance, in seconds, for treating a silencedetect interval as touching a clip's edge
# (leading/trailing padding) rather than sitting strictly inside it.
_SILENCE_EPSILON = 0.05

# ffmpeg reports every input's length as "Duration: HH:MM:SS.ss" on stderr, so the
# silencedetect pass already carries the clip duration -- no separate ffprobe needed.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


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


def _probe_duration(fpath: Path) -> float:
    result = run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(fpath),
        ],
        what=f"Probing the duration of {fpath.name}",
    )
    return float(result.stdout.strip())


def _parse_duration(stderr_text: str) -> float | None:
    """Read the input duration out of an ffmpeg run's stderr, if it reported one."""
    match = _DURATION_RE.search(stderr_text)
    if match is None:
        return None

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_silence_intervals(stderr_text: str) -> list[tuple[float, float | None]]:
    """Parse ffmpeg `silencedetect` stderr into a list of (start, end) intervals.

    Each `silence_start: Z` line is paired with the `silence_end: X | silence_duration: Y`
    line that follows it. A silence that runs to end-of-file emits a `silence_start` with
    no matching `silence_end` line at all -- that interval's end is `None`.
    """
    intervals: list[tuple[float, float | None]] = []
    pending_start: float | None = None
    for line in stderr_text.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            if pending_start is not None:
                # A silence_start with no silence_end before the next one started --
                # shouldn't happen in practice, but close it out as open-ended rather
                # than silently dropping it.
                intervals.append((pending_start, None))
            pending_start = float(line.split("silence_start:")[1].strip().split()[0])
        elif "silence_end:" in line and pending_start is not None:
            value = float(line.split("silence_end:")[1].strip().split("|")[0].strip())
            intervals.append((pending_start, value))
            pending_start = None

    if pending_start is not None:
        intervals.append((pending_start, None))

    return intervals


def _boundaries_from_intervals(
    intervals: list[tuple[float, float | None]], duration: float
) -> tuple[float, float]:
    """Derive (speech_start, speech_end) from parsed silence intervals.

    Only an interval that actually touches the clip's edges counts as padding to trim;
    internal pauses (silence surrounded by speech on both sides) are speech content and
    are ignored entirely.
    """
    speech_start = 0.0
    for start, end in intervals:
        if start <= _SILENCE_EPSILON and end is not None:
            speech_start = end
            break

    speech_end = duration
    for start, end in reversed(intervals):
        if end is None or abs(end - duration) <= _SILENCE_EPSILON:
            speech_end = start
            break

    if speech_end <= speech_start:
        return 0.0, duration

    return speech_start, speech_end


def _detect_silence(fpath: Path) -> tuple[float, float]:
    """Find real speech boundaries in a studio-clean TTS clip via ffmpeg `silencedetect`.

    Returns (speech_start, speech_end). TTS output has no background noise, so silence
    detection finds the same leading/trailing padding boundaries a real VAD would.
    Internal (mid-sentence) pauses are detected too but must not be mistaken for padding --
    see `_boundaries_from_intervals`. `-vn` keeps the cost independent of what the clip
    container happens to hold.
    """
    stderr_text = run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(fpath),
            "-vn",
            "-af",
            "silencedetect=noise=-30dB:d=0.1",
            "-f",
            "null",
            "-",
        ],
        what=f"Detecting silence in {fpath.name}",
    ).stderr

    duration = _parse_duration(stderr_text)
    if duration is None:
        duration = _probe_duration(fpath)

    return _boundaries_from_intervals(_parse_silence_intervals(stderr_text), duration)


class _BoundaryMeasurer:
    """Measure the real speech boundaries of synthesised clips via a degrading chain.

    Forced Alignment (when an `aligner` is given) -> ffmpeg `silencedetect` -> ffprobe
    container duration (no trim, logged as a warning). Any exception from the aligner
    degrades to the next tier for that clip *and* latches the aligner off for the rest of
    the run -- a broken aligner would otherwise cost one failing API call per clip. Each
    tier is logged only the first time it is used in this run, not once per clip, so a
    silently-taken fallback still shows up in the log exactly once.
    """

    def __init__(self, aligner: AlignmentProvider | None = None) -> None:
        self.aligner = aligner
        self._logged: set[str] = set()

    def __call__(self, clip_path: Path, text: str) -> tuple[float, float]:
        return self.measure(clip_path, text)

    def _log_once(self, key: str, message: str, *, warn: bool = False) -> None:
        if key in self._logged:
            return

        self._logged.add(key)
        (logger.warning if warn else logger.info)(message)

    def measure(self, clip_path: Path, text: str) -> tuple[float, float]:
        if self.aligner is not None:
            try:
                boundaries = self.aligner(clip_path, text)
                self._log_once(
                    "alignment", "Measuring dub clip speech boundaries via Forced Alignment."
                )
                return boundaries
            except Exception as exc:
                self.aligner = None
                logger.warning(
                    f"Forced Alignment unavailable ({exc}); falling back to ffmpeg "
                    "silencedetect for speech-boundary measurement."
                )

        try:
            boundaries = _detect_silence(clip_path)
        except Exception as exc:
            self._log_once(
                "silencedetect_failed",
                f"ffmpeg silencedetect failed ({exc}); falling back to ffprobe container "
                "duration -- no padding trim, coarser drift tracking.",
                warn=True,
            )
            return 0.0, _probe_duration(clip_path)

        self._log_once(
            "silencedetect", "Measuring dub clip speech boundaries via ffmpeg silencedetect."
        )
        return boundaries


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


def _synthesise_group(
    group: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    work_dir: Path,
    measure: _BoundaryMeasurer,
    rate: float,
) -> list[_Clip]:
    """Synthesise every segment in `group` at `rate` and measure its speech boundaries."""
    clips: list[_Clip] = []
    for segment in group:
        text = translations[segment.id]
        clip_path = work_dir / f"segment_{segment.id:05d}.mp3"
        tts(text, clip_path, speed=rate)
        clips.append(_Clip(clip_path, *measure(clip_path, text)))
    return clips


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


def _synthesise_and_lay_out(
    group: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    work_dir: Path,
    measure: _BoundaryMeasurer,
    anchor: float,
    group_source_end: float,
    rate: float,
) -> tuple[list[_Placement], float]:
    """Synthesise a group at `rate` and lay it out, returning (placements, drift).

    Drift is the placed end of the last clip measured against the group's source end;
    positive means the group ran long. Called once per group at 1.0x and, when that drift
    leaves the tolerance band, a second time at the shared corrective rate.
    """
    clips = _synthesise_group(group, translations, tts, work_dir, measure, rate)
    placements, placed_end = _layout_group(group, clips, anchor, group_source_end)
    return placements, placed_end - group_source_end


def synthesise_track(
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    out_path: Path,
    work_dir: Path | None = None,
    *,
    aligner: AlignmentProvider | None = None,
) -> Path:
    """Synthesise translated segments and assemble them onto a silent timeline.

    `translations` maps segment.id -> translated text (segments with no translation,
    e.g. filtered out upstream, are skipped). Segments are partitioned into scene-anchored
    groups (see `_group_segments`); within a group, clips float sequentially from the
    group's anchor at their natural 1.0x rate. If the group's accumulated drift against its
    source span exceeds `_DRIFT_TOLERANCE`, the whole group is re-synthesised once at a
    single shared corrective rate clamped to [_MIN_RATE, _MAX_RATE]. `aligner`, when given,
    is a `(clip: Path, text: str) -> (speech_start, speech_end)` callable (e.g. ElevenLabs
    Forced Alignment); any exception from it degrades to ffmpeg `silencedetect`, then to
    ffprobe container duration as a last resort (see `_BoundaryMeasurer`). Produces one
    continuous audio file at `out_path`. Returns `out_path`.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="movie-subtitles-dub-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        translated
        for raw_group in _group_segments(segments)
        if (translated := [s for s in raw_group if translations.get(s.id)])
    ]
    measure = _BoundaryMeasurer(aligner)

    inputs: list[_Placement] = []
    for group_idx, group in enumerate(groups):
        anchor = group[0].start
        group_source_end = group[-1].end

        placements, drift = _synthesise_and_lay_out(
            group, translations, tts, work_dir, measure, anchor, group_source_end, rate=1.0
        )

        if abs(drift) > _DRIFT_TOLERANCE:
            group_speech_len = sum(placement.speech_len for placement in placements)
            shared_rate = fit_rate(group_speech_len, max(group_source_end - anchor, 0.0))
            logger.info(
                f"Group {group_idx} ({len(group)} segment(s)) drifted {drift:+.2f}s past its "
                f"source span; re-synthesising once at a shared rate of {shared_rate:.2f}x."
            )
            placements, drift = _synthesise_and_lay_out(
                group,
                translations,
                tts,
                work_dir,
                measure,
                anchor,
                group_source_end,
                rate=shared_rate,
            )
            logger.info(f"Group {group_idx} corrected; residual drift {drift:+.2f}s.")
        else:
            logger.info(
                f"Group {group_idx} ({len(group)} segment(s)) drift {drift:+.2f}s within "
                "tolerance; left at 1.0x."
            )

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
