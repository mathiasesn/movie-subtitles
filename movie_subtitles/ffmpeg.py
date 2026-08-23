import subprocess
from pathlib import Path

# Matroska/MP4 take AAC, but WebM's spec only admits Vorbis and Opus: handing it
# `-c:a aac` makes ffmpeg fail at header-write time with a bare "Invalid argument"
# and exit 234, which says nothing about the real cause. Keyed by container suffix;
# anything unlisted gets AAC, which every other common container accepts.
_AUDIO_CODEC_BY_SUFFIX = {".webm": "libopus"}
_DEFAULT_AUDIO_CODEC = "aac"

_UNKNOWN_CHANNEL_LAYOUT_FALLBACK = "stereo"

# ffprobe can also report an empty/non-numeric sample_rate; 48 kHz is the safe default.
_UNKNOWN_SAMPLE_RATE_FALLBACK = 48000


def audio_codec_for(suffix: str) -> str:
    """The audio codec to encode to when writing into a `suffix` container."""
    return _AUDIO_CODEC_BY_SUFFIX.get(suffix.lower(), _DEFAULT_AUDIO_CODEC)


def run(cmd: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg/ffprobe command, raising a RuntimeError that quotes its stderr.

    subprocess's CalledProcessError only carries the exit status, and cli.main() logs
    just str(exc) -- so a failing ffmpeg used to report "returned non-zero exit status
    234" and nothing about why. ffmpeg puts the actual diagnosis on stderr, so surface
    its tail in the message.

    Returns the CompletedProcess so callers that need the command's output can read it
    (`stdout` for ffprobe queries, `stderr` for ffmpeg filters such as `silencedetect`);
    callers that only care about success ignore the return value.
    """
    # -hide_banner keeps the library-version block out of the stderr tail below, which
    # would otherwise crowd out the actual error.
    if cmd[0] == "ffmpeg":
        cmd = [cmd[0], "-hide_banner", *cmd[1:]]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(line for line in result.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"{what} failed (ffmpeg exit {result.returncode}):\n{tail}")

    return result


def _probe(fpath: Path, entries: str, *, select: str | None = None, of: str, what: str) -> str:
    """Shared ffprobe invocation: build the argv, run it, return raw stdout.

    `probe_duration` and `probe_audio_format` are both "run ffprobe with these
    `-show_entries`/`-of` options against this file" -- this is that shape, factored
    out once so a future change to how this package invokes ffprobe (a new global flag,
    a different output format) lands here instead of in every call site.
    """
    cmd = ["ffprobe", "-v", "error"]
    if select is not None:
        cmd += ["-select_streams", select]
    cmd += ["-show_entries", entries, "-of", of, str(fpath)]
    return run(cmd, what=what).stdout


def probe_duration(fpath: Path) -> float:
    """Measure a media file's actual duration via ffprobe.

    Used wherever a requested cut/synthesis length must not be trusted at face value
    (an `-ss`/`-t` cut can run past the source's actual end, or a TTS clip's container
    duration is the only thing worth trusting) -- the seconds reported must come from
    the produced file itself.
    """
    stdout = _probe(
        fpath,
        "format=duration",
        of="default=noprint_wrappers=1:nokey=1",
        what=f"Probing the duration of {fpath.name}",
    )
    return float(stdout.strip())


def probe_audio_format(fpath: Path) -> tuple[str, int] | None:
    """The first audio stream's `(channel_layout, sample_rate)`, per ffprobe, or `None`.

    `None` means `fpath` has no audio stream at all -- used before muxing a dub over a
    source video: a source with no audio track (e.g. a silent clip) can't be
    ducked-and-mixed, so callers fall back to a dub-only mapping instead of failing.

    Used (when not `None`) to make a dub mux match the source's own channel
    layout/sample rate instead of forcing a fixed one. ffprobe sometimes reports an
    empty `channel_layout` (e.g. for some containers/codecs it can't derive one from
    just the channel count), in which case that would be an invalid `aformat` argument
    -- callers get `_UNKNOWN_CHANNEL_LAYOUT_FALLBACK` instead.
    """
    stdout = _probe(
        fpath,
        "stream=sample_rate,channel_layout",
        select="a:0",
        of="default=noprint_wrappers=1",
        what=f"Probing the audio format of {fpath.name}",
    )
    lines = stdout.strip().splitlines()
    if not lines:
        # Not reachable via "a stream with unreadable fields": keyed output
        # (`noprint_wrappers=1` without `nokey=1`) makes ffprobe print unavailable
        # fields as `key=unknown`/`key=N/A` rather than omitting the line -- e.g. a
        # container with no derivable channel_layout still yields a `sample_rate=...`
        # line, handled by the `unknown` fallback below. Empty output therefore means
        # there genuinely is no `a:0` stream at all.
        return None
    # Keyed output (no `nokey=1`): ffprobe emits `default` fields in its own internal
    # order, not the order -show_entries asked for, so parsing by position would silently
    # swap the two values if that order ever changed.
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    rate = fields.get("sample_rate", "").strip()
    sample_rate = int(rate) if rate.isdigit() else _UNKNOWN_SAMPLE_RATE_FALLBACK
    channel_layout = fields.get("channel_layout", "").strip()
    if not channel_layout or channel_layout == "unknown":
        channel_layout = _UNKNOWN_CHANNEL_LAYOUT_FALLBACK
    return channel_layout, sample_rate
