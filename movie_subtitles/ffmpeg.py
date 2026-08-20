import logging
import subprocess

logger = logging.getLogger("ffmpeg")

# Matroska/MP4 take AAC, but WebM's spec only admits Vorbis and Opus: handing it
# `-c:a aac` makes ffmpeg fail at header-write time with a bare "Invalid argument"
# and exit 234, which says nothing about the real cause. Keyed by container suffix;
# anything unlisted gets AAC, which every other common container accepts.
_AUDIO_CODEC_BY_SUFFIX = {".webm": "libopus"}
_DEFAULT_AUDIO_CODEC = "aac"


def audio_codec_for(suffix: str) -> str:
    """The audio codec to encode to when writing into a `suffix` container."""
    return _AUDIO_CODEC_BY_SUFFIX.get(suffix.lower(), _DEFAULT_AUDIO_CODEC)


def run(cmd: list[str], *, what: str, capture_stderr: bool = False) -> str | None:
    """Run an ffmpeg/ffprobe command, raising a RuntimeError that quotes its stderr.

    subprocess's CalledProcessError only carries the exit status, and cli.main() logs
    just str(exc) -- so a failing ffmpeg used to report "returned non-zero exit status
    234" and nothing about why. ffmpeg puts the actual diagnosis on stderr, so surface
    its tail in the message.

    `capture_stderr`, when True, returns the command's full stderr text on success
    (e.g. for callers that parse ffmpeg filter output such as `silencedetect`) instead
    of the default `None`. Existing callers are unaffected.
    """
    # -hide_banner keeps the library-version block out of the stderr tail below, which
    # would otherwise crowd out the actual error.
    if cmd[0] == "ffmpeg":
        cmd = [cmd[0], "-hide_banner", *cmd[1:]]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(line for line in result.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"{what} failed (ffmpeg exit {result.returncode}):\n{tail}")

    return result.stderr if capture_stderr else None
