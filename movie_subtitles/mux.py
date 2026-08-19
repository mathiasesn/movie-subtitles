import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("mux")


def mux_dub(video_path: Path, audio_path: Path) -> Path:
    """Overlay `audio_path` onto `video_path`'s video track via ffmpeg.

    Writes `<video>.dubbed<ext>` next to `video_path` and returns that path. The
    original video's audio track is dropped; the synthesised track replaces it.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH. Install ffmpeg to use --dub (see AGENTS.md).")

    out_path = video_path.with_name(f"{video_path.stem}.dubbed{video_path.suffix}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]

    logger.info(f"Muxing dub track over {video_path} -> {out_path}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    return out_path
