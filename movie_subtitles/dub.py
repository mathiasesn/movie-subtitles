import logging
import subprocess
import tempfile
from pathlib import Path

from movie_subtitles.ffmpeg import run as run_ffmpeg
from movie_subtitles.providers.base import Segment, TTSProvider

logger = logging.getLogger("dub")

# Spec-mandated clamp for the timing-drift speaking-rate nudge. Intentionally tighter
# than the ElevenLabs API's own supported range (0.7-1.2, see providers/elevenlabs.py)
# so artefacts stay inaudible.
_MIN_RATE = 0.9
_MAX_RATE = 1.15

# Product policy, in the same vein as _MIN_RATE/_MAX_RATE above: past this ideal rate the
# clip is more than twice its slot, so a +-15% clamped nudge is noise against the mismatch
# and not worth paying for a second TTS call.
_HOPELESS_RATE = 2.0


def _ideal_rate(actual_duration: float, slot_duration: float) -> float:
    """The unclamped rate that would make `actual_duration` fill `slot_duration` exactly."""
    if slot_duration <= 0 or actual_duration <= 0:
        return 1.0

    return actual_duration / slot_duration


def fit_rate(actual_duration: float, slot_duration: float) -> float:
    """Compute the speaking rate needed to fit `actual_duration` into `slot_duration`.

    Returns the rate clamped to [_MIN_RATE, _MAX_RATE].
    """
    return min(max(_ideal_rate(actual_duration, slot_duration), _MIN_RATE), _MAX_RATE)


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

    Returns (clip_path, still_overruns). For the ordinary case, `still_overruns` is True if
    the segment's actual duration still overruns the slot once fitting is done: fitting
    re-synthesises at a clamped rate only when the ideal rate rounds away from 1.00, so
    `still_overruns` is measured on that re-synthesised clip when a retry happened, or on
    the original 1.0x clip when it did not (an ideal rate close enough to round to 1.00 can
    still leave the clip longer than the slot). An underrun never counts here: it just
    leaves trailing silence in the slot.

    Two cases skip that re-synthesis and re-measurement entirely, returning True without
    ever attempting a clamped-rate fit: a non-positive slot duration, and an ideal rate
    beyond `_HOPELESS_RATE` where a +-15% clamped nudge would be noise against the
    mismatch. In both, the clip is placed unfitted at 1.0x and `still_overruns` is True by
    construction (an overrun into the slot's silence is the whole reason it is hopeless),
    not because an overrun was measured.
    """
    clip_path = out_dir / f"segment_{segment_id:05d}.mp3"
    tts(text, clip_path, speed=1.0)
    duration = _probe_duration(clip_path)

    ideal = _ideal_rate(duration, slot_duration)

    # A slot the clamp cannot meaningfully help -- either hopelessly short relative to
    # the clip, or not a real slot at all -- is not worth a second TTS call. Place the
    # clip unfitted at 1.0x rather than pay for a +-15% nudge that is noise against the
    # mismatch. It still lands on the timeline: dropping the line is worse than an
    # overrun into the following silence.
    if slot_duration <= 0 or ideal > _HOPELESS_RATE:
        logger.warning(
            f"Segment {segment_id} ({duration:.2f}s audio vs {slot_duration:.2f}s slot) is "
            "hopelessly mismatched; leaving it unfitted at 1.0x and letting it overrun into "
            "the following silence by design."
        )
        return clip_path, True

    rate = fit_rate(duration, slot_duration)
    clamped = False
    # A clamped rate is always 0.9 or 1.15, so it can never round to 1.00: every clamped
    # segment goes through this branch and is reported once, at warning level.
    if round(rate, 2) != 1.0:
        clamped = True
        drift = "overruns" if rate > 1.0 else "underruns"
        if rate != ideal:
            logger.warning(
                f"Segment {segment_id} {drift} its slot and would need rate {ideal:.2f}x to "
                f"fit exactly; clamped to {rate:.2f}x ({duration:.2f}s audio vs "
                f"{slot_duration:.2f}s slot)."
            )
        else:
            adjustment = "speeding up" if rate > 1.0 else "slowing down"
            logger.info(f"Segment {segment_id} {drift} its slot; {adjustment} to {rate:.2f}x.")
        tts(text, clip_path, speed=rate)
        duration = _probe_duration(clip_path)

    # Re-measure the clip that will actually be placed on the timeline: the fit is
    # judged by whether it actually fits its slot now, not by whether the ideal rate was
    # inside the clamp.
    still_overruns = duration > slot_duration

    if still_overruns:
        if clamped:
            logger.warning(
                f"Segment {segment_id} does not fit its slot even at the {rate:.2f}x rate "
                f"clamp ({duration:.2f}s audio vs {slot_duration:.2f}s slot); letting it "
                "overrun into the following silence."
            )
        else:
            logger.warning(
                f"Segment {segment_id} does not fit its slot at the natural 1.0x rate; no "
                f"clamp was applied ({duration:.2f}s audio vs {slot_duration:.2f}s slot); "
                "letting it overrun into the following silence."
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

    logger.info(f"{overrun_count} segment(s) still overrun their slot after fitting")

    _assemble_timeline(inputs, out_path)
    return out_path


def _assemble_timeline(inputs: list[tuple[float, Path]], out_path: Path) -> Path:
    """Lay clips onto a silent stereo timeline at their start offsets via ffmpeg."""
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

    run_ffmpeg(cmd, what=f"Assembling {len(inputs)} dub clip(s) into {out_path.name}")
    return out_path
