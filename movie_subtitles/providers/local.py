import logging
from collections.abc import Iterable
from pathlib import Path

from faster_whisper import WhisperModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from movie_subtitles.providers.base import Segment

logger = logging.getLogger("local")


class Transcribe:
    def __init__(self, model_name: str = "large-v3") -> None:
        self.model_name = model_name

        logger.info(f"Loading model {model_name}")
        self.model = WhisperModel(self.model_name)

    def __call__(self, fpath: str | Path, audio_lang: str = "en") -> Iterable[Segment]:
        return self.transcribe(fpath, audio_lang)

    def transcribe(self, fpath: str | Path, audio_lang: str = "en") -> Iterable[Segment]:
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

        return (
            Segment(id=segment.id, start=segment.start, end=segment.end, text=segment.text)
            for segment in segments
        )


class Translate:
    def __init__(self, model_name: str = "jbochi/madlad400-3b-mt") -> None:
        self.model_name = model_name

        logger.info(f"Loading model {model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        return self.translate(text, output_lang, budget_chars)

    def translate(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        # budget_chars is ignored: MADLAD400 has no length-control lever to target it.
        text = f"<2{output_lang}> {text}"
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text
