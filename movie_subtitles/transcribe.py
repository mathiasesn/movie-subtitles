import logging
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

logger = logging.getLogger("transcribe")


class Transcribe:
    def __init__(self, model_name: str = "large-v3") -> None:
        self.model_name = model_name

        logger.info(f"Loading model {model_name}")
        self.model = WhisperModel(self.model_name)

    def __call__(self, fpath: str | Path, audio_lang: str = "en") -> Iterable[Segment]:
        return self.transcribe(fpath, audio_lang)

    def transcribe(
        self, fpath: str | Path, audio_lang: str = "en"
    ) -> Iterable[Segment]:
        if isinstance(fpath, Path):
            fpath = fpath.as_posix()

        transcribe = self.model.transcribe(
            fpath,
            task="transcribe",
            language=audio_lang,
            vad_filter=True,
        )

        segments, info = transcribe
        logger.info(
            f"Total duration {info.duration}. Duration with speech {info.duration_after_vad}."
        )

        return segments
