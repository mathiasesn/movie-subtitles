import re
from pathlib import Path

from movie_subtitles.ffmpeg import run as run_ffmpeg

# Tolerance, in seconds, for treating a silencedetect interval as touching a clip's edge
# (leading/trailing padding) rather than sitting strictly inside it.
_SILENCE_EPSILON = 0.05

# ffmpeg reports every input's length as "Duration: HH:MM:SS.ss" on stderr, so the
# silencedetect pass already carries the clip duration -- no separate ffprobe needed.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


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


class SilenceAlign:
    """Find real speech boundaries in a studio-clean TTS clip via ffmpeg `silencedetect`.

    Returns (speech_start, speech_end). TTS output has no background noise, so silence
    detection finds the same leading/trailing padding boundaries a real VAD would.
    Internal (mid-sentence) pauses are detected too but must not be mistaken for padding --
    see `_boundaries_from_intervals`. `-vn` keeps the cost independent of what the clip
    container happens to hold. `text` is accepted but ignored -- it exists only to satisfy
    the `AlignmentProvider` Protocol.
    """

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        stderr_text = run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(clip),
                "-vn",
                "-af",
                "silencedetect=noise=-30dB:d=0.1",
                "-f",
                "null",
                "-",
            ],
            what=f"Detecting silence in {clip.name}",
        ).stderr

        duration = _parse_duration(stderr_text)
        if duration is None:
            duration = _probe_duration(clip)

        return _boundaries_from_intervals(_parse_silence_intervals(stderr_text), duration)


class DurationAlign:
    """No-trim last resort: the clip's full container duration, via ffprobe.

    Returns (0.0, duration) -- no padding trim, coarser drift tracking. `text` is accepted
    but ignored -- it exists only to satisfy the `AlignmentProvider` Protocol.
    """

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        return 0.0, _probe_duration(clip)
