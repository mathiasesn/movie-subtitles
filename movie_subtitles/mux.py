import logging
import shutil
from pathlib import Path

from movie_subtitles.ffmpeg import audio_codec_for, has_audio_stream, probe_audio_format
from movie_subtitles.ffmpeg import run as run_ffmpeg

logger = logging.getLogger("mux")

# How much the original audio is attenuated while the dub is speaking. 0.25 keeps the
# original's music/effects/ambience audible underneath without competing with the
# translated dialogue.
_DUCK_LEVEL = 0.25

# Per-input gain applied to both streams before `amix=normalize=0`, which -- unlike the
# default `normalize=1` -- does not scale inputs down to avoid clipping on its own. Two
# full-scale inputs summed can clip, so each is pre-attenuated; this is a fixed
# module-level guard, not a tunable.
_MIX_INPUT_GAIN = 0.7


# `volume`'s expression is one ffmpeg argv token; a single filter built from hundreds of
# `between(...)` terms risks exceeding practical expression/argv limits. Spans are
# chunked into filters of at most this many terms each and chained -- each chunk's
# spans are disjoint from every other chunk's (spans are globally coalesced first), so
# multiplying by 1.0 outside a chunk's own spans never double-ducks.
_MAX_SPANS_PER_FILTER = 200


def _duck_expression(spans: list[tuple[float, float]]) -> str:
    """A `volume` filter expression that attenuates only inside `spans`."""
    terms = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in spans)
    return f"volume='if(gt({terms},0),{_DUCK_LEVEL},1)':eval=frame"


def _ducking_filters(spans: list[tuple[float, float]]) -> list[str]:
    """One `volume` filter per chunk of `spans`, chained to bound expression size."""
    chunks = [
        spans[i : i + _MAX_SPANS_PER_FILTER] for i in range(0, len(spans), _MAX_SPANS_PER_FILTER)
    ]
    return [_duck_expression(chunk) for chunk in chunks]


def mux_dub(
    video_path: Path,
    audio_path: Path,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
) -> Path:
    """Overlay `audio_path` onto `video_path`'s video track via ffmpeg.

    Writes `<video>.dubbed<ext>` next to `video_path` and returns that path. When the
    source has an audio track, the output keeps it underneath the dub rather than
    dropping it: the original is mixed with the synthesised track (`amix=normalize=0`),
    ducked to `_DUCK_LEVEL` while `speech_spans` says the dub is speaking so the
    translated dialogue stays dominant. `speech_spans` is a list of `(start, end)`
    windows in seconds (typically `dub.synthesise_track()`'s second return value);
    omitting it (or passing an empty list) mixes the original in unducked throughout.
    A source with no audio stream at all (checked via `ffmpeg.has_audio_stream`) falls
    back to today's dub-only mapping, since there is nothing to mix or duck.

    The video's full duration is authoritative -- no `-shortest`, so a synthesised track
    that ends before the video does (e.g. no speech in the video's tail) does not
    truncate the video; without `-shortest` or `apad`, ffmpeg does not pad the audio at
    all -- the audio stream simply ends early and playback continues on video alone for
    the remainder (verified with synthetic media: a 10s video + 3s audio track muxes to a
    10s output).

    The output keeps the source container, so the audio codec has to be one that container
    accepts -- see `audio_codec_for`. The video track is always stream-copied, so the
    source's video codec is by definition already legal in its own container.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH. Install ffmpeg to use --dub.")

    out_path = video_path.with_name(f"{video_path.stem}.dubbed{video_path.suffix}")
    audio_codec = audio_codec_for(video_path.suffix)

    base_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
    ]

    if not has_audio_stream(video_path):
        logger.info(f"{video_path} has no audio stream; muxing the dub track alone.")
        cmd = base_cmd + [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            str(out_path),
        ]
    else:
        spans = speech_spans or []
        duck_filters = _ducking_filters(spans)

        orig_label = "0:a:0"
        chained_filters = []
        for i, expr in enumerate(duck_filters):
            next_label = f"duck{i}"
            chained_filters.append(f"[{orig_label}]{expr}[{next_label}]")
            orig_label = next_label

        # Target the *source's* own channel layout/sample rate rather than a fixed
        # stereo/48kHz pair -- this whole feature exists to preserve the original
        # audio, so silently downmixing e.g. a 5.1 soundtrack to stereo would be a
        # fidelity regression. `amix` negotiates a common format itself when inputs
        # disagree, but that negotiation is implicit and version-dependent, so both
        # branches are still explicitly formatted to the same target: the original
        # branch is already in that format almost by definition (it's where the
        # target came from), and formatting it too keeps the graph deterministic
        # rather than relying on amix's own negotiation for that branch.
        channel_layout, sample_rate = probe_audio_format(video_path)
        fmt = (
            f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts={channel_layout}"
        )
        gain_a = f"[{orig_label}]volume={_MIX_INPUT_GAIN},{fmt}[origmix]"
        gain_b = f"[1:a:0]volume={_MIX_INPUT_GAIN},{fmt}[dubmix]"
        mix = "[origmix][dubmix]amix=inputs=2:normalize=0[out]"

        filter_complex = ";".join([*chained_filters, gain_a, gain_b, mix])

        cmd = base_cmd + [
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[out]",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            str(out_path),
        ]

    logger.info(f"Muxing dub track over {video_path} -> {out_path} (audio: {audio_codec})")
    run_ffmpeg(cmd, what=f"Muxing the dub track into {out_path.name}")

    return out_path
