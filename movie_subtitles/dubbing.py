import logging
import time
from pathlib import Path

from elevenlabs.core.api_error import ApiError

from movie_subtitles.providers.elevenlabs import build_client
from movie_subtitles.providers.errors import vendor_errors

logger = logging.getLogger("dubbing")

# Terminal states reported by GET /v1/dubbing/:dubbing_id (DubbingMetadataResponse.status).
# Confirmed against https://elevenlabs.io/docs/api-reference/dubbing/get.md (2026-08-19):
# status is one of "dubbing" | "dubbed" | "failed".
_STATUS_DONE = "dubbed"
_STATUS_FAILED = "failed"


class ManagedDub:
    """Buy-side dubbing path: create a Dubbing job, poll it, download the render.

    Uses the official `elevenlabs` SDK's `client.dubbing` resource end to end:

    - create: https://elevenlabs.io/docs/api-reference/dubbing/create.md
      POST /v1/dubbing, multipart with `file`, `target_lang`, `source_lang`. Returns
      a `DoDubbingResponse` with `dubbing_id` and `expected_duration_sec`.
    - get: https://elevenlabs.io/docs/api-reference/dubbing/get.md
      GET /v1/dubbing/:dubbing_id. Returns a `DubbingMetadataResponse` with a
      `status` field ("dubbing" while in progress, "dubbed" on success, "failed" on
      error) and an `error` field populated on failure.
    - audio.get: https://elevenlabs.io/docs/api-reference/dubbing/audio/get.md
      GET /v1/dubbing/:dubbing_id/audio/:language_code. Streams the rendered media
      for one target language as bytes.
    """

    def __init__(self, poll_interval: float = 10.0, timeout: float = 1800.0) -> None:
        self.poll_interval = poll_interval
        self.timeout = timeout

        self.client = build_client()

    def __call__(self, fpath: str | Path, source_lang: str, target_lang: str) -> Path:
        return self.dub(fpath, source_lang, target_lang)

    def dub(self, fpath: str | Path, source_lang: str, target_lang: str) -> Path:
        if isinstance(fpath, str):
            fpath = Path(fpath)

        dubbing_id = self._create(fpath, source_lang, target_lang)
        self._poll_until_done(dubbing_id)
        return self._download(fpath, dubbing_id, target_lang)

    def _create(self, fpath: Path, source_lang: str, target_lang: str) -> str:
        logger.info(f"Submitting {fpath} to ElevenLabs Dubbing ({source_lang} -> {target_lang})")
        with (
            vendor_errors(ApiError, "ElevenLabs Dubbing create request"),
            open(fpath, "rb") as media_file,
        ):
            response = self.client.dubbing.create(
                file=media_file,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        logger.info(f"Dubbing job {response.dubbing_id} created")
        return response.dubbing_id

    def _poll_until_done(self, dubbing_id: str) -> None:
        deadline = time.monotonic() + self.timeout

        while True:
            with vendor_errors(ApiError, "ElevenLabs Dubbing status request"):
                metadata = self.client.dubbing.get(dubbing_id)
            status = metadata.status

            if status == _STATUS_DONE:
                logger.info(f"Dubbing job {dubbing_id} completed")
                return
            if status == _STATUS_FAILED:
                raise RuntimeError(f"ElevenLabs dubbing job {dubbing_id} failed: {metadata.error}")

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ElevenLabs dubbing job {dubbing_id} did not complete within "
                    f"{self.timeout:.0f}s (last status: {status!r})"
                )

            logger.info(
                f"Dubbing job {dubbing_id} status: {status}; polling again "
                f"in {self.poll_interval:.0f}s"
            )
            time.sleep(self.poll_interval)

    def _download(self, fpath: Path, dubbing_id: str, target_lang: str) -> Path:
        # dubbing.audio.get's own docstring in the installed SDK
        # (.venv/lib/python3.12/site-packages/elevenlabs/dubbing/audio/client.py) reads:
        # "Returns dub as a streamed MP3 or MP4 file." -- confirmed the same wording against
        # https://elevenlabs.io/docs/api-reference/dubbing/audio/get.md (2026-08-19). For a
        # video source (this CLI's only input type), the endpoint renders an MP4 container,
        # not audio-only bytes, and not necessarily whatever container the source video used
        # (.mov/.mkv/etc). Writing those bytes under `fpath.suffix` mislabels the file
        # whenever the source wasn't already .mp4. Always write `.mp4` so the extension
        # matches the actual container returned.
        out_path = fpath.with_name(f"{fpath.stem}.dubbed.mp4")

        logger.info(f"Downloading dubbed audio for {dubbing_id} ({target_lang}) -> {out_path}")
        with vendor_errors(ApiError, "ElevenLabs Dubbing audio download"):
            chunks = self.client.dubbing.audio.get(dubbing_id, target_lang)
            with open(out_path, "wb") as out_file:
                for chunk in chunks:
                    out_file.write(chunk)

        logger.info(f"Saved managed dub to {out_path}")
        return out_path
