import logging
import subprocess
import tempfile
from pathlib import Path

from movie_subtitles.providers.base import Segment, TTSProvider

logger = logging.getLogger("dub")

# Spec-mandated clamp for the timing-drift speaking-rate nudge. Intentionally tighter
# than the ElevenLabs API's own supported range (0.7-1.2, see providers/elevenlabs.py)
# so artefacts stay inaudible.
_MIN_RATE = 0.9
_MAX_RATE = 1.15


def fit_rate(actual_duration: float, slot_duration: float) -> float:
    """Compute the speaking rate needed to fit `actual_duration` into `slot_duration`.

    Returns the rate clamped to [_MIN_RATE, _MAX_RATE].
    """
    if slot_duration <= 0 or actual_duration <= 0:
        return 1.0

    ideal_rate = actual_duration / slot_duration
    return min(max(ideal_rate, _MIN_RATE), _MAX_RATE)


def _probe_duration(fpath: Path) -> float:
    result = subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _synthesise_fitted(
    tts: TTSProvider,
    text: str,
    slot_duration: float,
    out_dir: Path,
    segment_id: int,
) -> tuple[Path, bool]:
    """Synthesise `text`, re-synthesising at an adjusted rate if it over- or underruns.

    Returns (clip_path, still_overruns) where `still_overruns` is True if the segment's
    actual, re-measured duration still overruns the slot after the clamped-rate
    re-synthesis (not merely whether the ideal rate fell outside the clamp -- the clamped
    rate is an approximation and re-synthesis can still land long of the slot). An
    underrun never counts here: it just leaves trailing silence in the slot.
    """
    clip_path = out_dir / f"segment_{segment_id:05d}.mp3"
    tts(text, clip_path, speed=1.0)
    duration = _probe_duration(clip_path)

    if duration == slot_duration:
        return clip_path, False

    rate = fit_rate(duration, slot_duration)
    ideal_rate = duration / slot_duration if slot_duration > 0 else 1.0
    if rate != ideal_rate:
        logger.warning(
            f"Segment {segment_id} needs rate {ideal_rate:.2f}x to fit its slot, clamped "
            f"to {rate:.2f}x ({duration:.2f}s audio vs {slot_duration:.2f}s slot)."
        )
    if round(rate, 2) != 1.0:
        if rate > 1.0:
            logger.info(f"Segment {segment_id} overruns its slot; speeding up to {rate:.2f}x.")
        else:
            logger.info(f"Segment {segment_id} underruns its slot; slowing down to {rate:.2f}x.")
        tts(text, clip_path, speed=rate)
        duration = _probe_duration(clip_path)

    # Re-measure the clip that will actually be placed on the timeline: the fit is
    # judged by whether it actually fits its slot now, not by whether the ideal rate was
    # inside the clamp.
    still_overruns = duration > slot_duration

    if still_overruns:
        logger.warning(
            f"Segment {segment_id} does not fit its slot even at the {rate:.2f}x rate "
            f"clamp ({duration:.2f}s audio vs {slot_duration:.2f}s slot); letting it "
            "overrun into the following silence."
        )

    return clip_path, still_overruns


def synthesise_track(
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    out_path: Path,
    work_dir: Path | None = None,
) -> Path:
    """Synthesise translated segments and assemble them onto a silent timeline.

    `translations` maps segment.id -> translated text (segments with no translation,
    e.g. filtered out upstream, are skipped). Produces one continuous audio file at
    `out_path` spanning from time 0 to the end of the last segment, with each clip
    placed at its original segment start offset. Returns `out_path`.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="movie-subtitles-dub-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    overrun_count = 0
    inputs: list[tuple[float, Path]] = []

    for segment in segments:
        text = translations.get(segment.id)
        if not text:
            continue

        slot_duration = max(segment.end - segment.start, 0.0)
        clip_path, still_overruns = _synthesise_fitted(
            tts, text, slot_duration, work_dir, segment.id
        )
        if still_overruns:
            overrun_count += 1
        inputs.append((segment.start, clip_path))

    logger.info(f"{overrun_count} segment(s) still overrun their slot after the rate clamp")

    _assemble_timeline(inputs, out_path)
    return out_path


def _assemble_timeline(inputs: list[tuple[float, Path]], out_path: Path) -> Path:
    """Lay clips onto a silent stereo timeline at their start offsets via ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        # No segments to place; emit a minimal silent file.
        subprocess.run(
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
            check=True,
            capture_output=True,
        )
        return out_path

    cmd = ["ffmpeg", "-y"]
    for _, clip_path in inputs:
        cmd += ["-i", str(clip_path)]

    filter_parts = []
    mix_labels = []
    for idx, (start, _) in enumerate(inputs):
        delay_ms = int(round(start * 1000))
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

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path
