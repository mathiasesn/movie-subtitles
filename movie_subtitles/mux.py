import logging
import shutil
from pathlib import Path

from movie_subtitles.ffmpeg import audio_codec_for, probe_audio_format
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


# Caps how many `between(...)` terms a single chained `volume` filter can hold; see
# `_ducking_chain`'s docstring for why.
_MAX_SPANS_PER_FILTER = 200


def _ducking_chain(
    spans: list[tuple[float, float]], in_label: str, duck_level: float
) -> tuple[list[str], str]:
    """Chained `volume` filter statements that duck `in_label` only inside `spans`.

    Returns `(statements, out_label)`: the statements to splice into a `-filter_complex`
    graph, and the label carrying the fully-ducked result -- `in_label` itself when
    `spans` is empty, since then there is nothing to chain.

    `volume`'s expression is one ffmpeg argv token, and `eval=frame` means it's
    re-evaluated once per *audio frame* (~47/s at 48 kHz, not per sample) -- but each
    evaluation still walks every `between()` term in its chunk, so a single filter built
    from hundreds of terms both risks exceeding practical expression/argv limits and
    costs real CPU: a 2-hour film with ~1000 spans is on the order of 1e7-1e8
    expression-node visits, seconds of CPU, non-trivial only because the video track
    is otherwise `-c:v copy`. Spans are chunked into filters of at most
    `_MAX_SPANS_PER_FILTER` terms each and chained -- each chunk's spans are disjoint
    from every other chunk's (spans are globally coalesced before reaching here), so
    multiplying by 1.0 outside a chunk's own spans never double-ducks. If this chunking
    ever isn't enough, the fix is a different mechanism entirely (e.g.
    `sidechaincompress` against the dub track), not more chunks.

    Verified with synthetic media: a 500-span speech_spans list (crossing three chunks
    of 200/200/100 terms) muxed via mux_dub() without an ffmpeg argv/expression
    failure, producing the expected duration/streams, with volumedetect (measured with
    edge-safe windows, clear of the eval=frame transition ramps at span boundaries)
    reading -33.1 dB inside a ducked span versus -21.1 dB outside one -- exactly the
    12.0 dB (20*log10(0.25)) expected at that (default) duck level; a different
    `duck_level` scales this attenuation accordingly.
    """
    chunks = [
        spans[i : i + _MAX_SPANS_PER_FILTER] for i in range(0, len(spans), _MAX_SPANS_PER_FILTER)
    ]
    statements = []
    label = in_label
    for i, chunk in enumerate(chunks):
        terms = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in chunk)
        expr = f"volume='if(gt({terms},0),{duck_level},1)':eval=frame"
        out_label = f"duck{i}"
        statements.append(f"[{label}]{expr}[{out_label}]")
        label = out_label
    return statements, label


def mux_dub(
    video_path: Path,
    audio_path: Path,
    *,
    speech_spans: list[tuple[float, float]] | None = None,
    duck_level: float = _DUCK_LEVEL,
) -> Path:
    """Overlay `audio_path` onto `video_path`'s video track via ffmpeg.

    Writes `<video>.dubbed<ext>` next to `video_path` and returns that path. When the
    source has an audio track, the output keeps it underneath the dub rather than
    dropping it: the original is mixed with the synthesised track (`amix=normalize=0`),
    ducked to `duck_level` (default `_DUCK_LEVEL`) while `speech_spans` says the dub is
    speaking so the translated dialogue stays dominant. `speech_spans` is a list of `(start, end)`
    windows in seconds (typically `dub.synthesise_track()`'s second return value);
    omitting it (or passing an empty list) mixes the original in unducked throughout.
    A source with no audio stream at all (per `ffmpeg.probe_audio_format` returning
    `None`) falls back to today's dub-only mapping, since there is nothing to mix or
    duck.

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

    source_format = probe_audio_format(video_path)

    if source_format is None:
        logger.info(f"{video_path} has no audio stream; muxing the dub track alone.")
        filter_args: list[str] = []
        audio_map = "1:a:0"
    else:
        spans = speech_spans or []
        duck_statements, ducked_label = _ducking_chain(spans, "0:a:0", duck_level)

        # Target the *source's* own channel layout/sample rate rather than a fixed
        # stereo/48kHz pair -- this whole feature exists to preserve the original
        # audio, so silently downmixing e.g. a 5.1 soundtrack to stereo would be a
        # fidelity regression. `amix` negotiates a common format itself when inputs
        # disagree, but that negotiation is implicit and version-dependent, so both
        # branches are still explicitly formatted to the same target: the original
        # branch is already in that format almost by definition (it's where the
        # target came from), and formatting it too keeps the graph deterministic
        # rather than relying on amix's own negotiation for that branch.
        channel_layout, sample_rate = source_format
        fmt = (
            f"aformat=sample_fmts=fltp:sample_rates={sample_rate}:channel_layouts={channel_layout}"
        )
        filter_complex = ";".join(
            [
                *duck_statements,
                f"[{ducked_label}]volume={_MIX_INPUT_GAIN},{fmt}[origmix]",
                f"[1:a:0]volume={_MIX_INPUT_GAIN},{fmt}[dubmix]",
                "[origmix][dubmix]amix=inputs=2:normalize=0[out]",
            ]
        )
        filter_args = ["-filter_complex", filter_complex]
        audio_map = "[out]"

    cmd = base_cmd + [
        *filter_args,
        "-map",
        "0:v:0",
        "-map",
        audio_map,
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        str(out_path),
    ]

    logger.info(f"Muxing dub track over {video_path} -> {out_path} (audio: {audio_codec})")
    run_ffmpeg(cmd, what=f"Muxing the dub track into {out_path.name}")

    return out_path
