import subprocess
from pathlib import Path

# Matroska/MP4 take AAC, but WebM's spec only admits Vorbis and Opus: handing it
# `-c:a aac` makes ffmpeg fail at header-write time with a bare "Invalid argument"
# and exit 234, which says nothing about the real cause. Keyed by container suffix;
# anything unlisted gets AAC, which every other common container accepts.
_AUDIO_CODEC_BY_SUFFIX = {".webm": "libopus"}
_DEFAULT_AUDIO_CODEC = "aac"


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


def probe_duration(fpath: Path) -> float:
    """Measure a media file's actual duration via ffprobe.

    Used wherever a requested cut/synthesis length must not be trusted at face value
    (an `-ss`/`-t` cut can run past the source's actual end, or a TTS clip's container
    duration is the only thing worth trusting) -- the seconds reported must come from
    the produced file itself.
    """
    result = run(
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


def has_audio_stream(fpath: Path) -> bool:
    """Whether `fpath` has at least one audio stream, per ffprobe.

    Used before muxing a dub over a source video: a source with no audio track at all
    (e.g. a silent clip) can't be ducked-and-mixed, so callers fall back to a dub-only
    mapping instead of failing.
    """
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(fpath),
        ],
        what=f"Checking {fpath.name} for an audio stream",
    )
    return bool(result.stdout.strip())
