import logging
import os
from collections.abc import Iterable
from pathlib import Path

from elevenlabs.client import ElevenLabs

from movie_subtitles.providers.base import Segment

logger = logging.getLogger("elevenlabs")

_SENTENCE_END_CHARS = (".", "!", "?", "…")


def _build_client() -> ElevenLabs:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY environment variable is not set. "
            "Set it to your ElevenLabs API key to use the 'elevenlabs' engine."
        )

    return ElevenLabs(api_key=api_key)


class ScribeTranscribe:
    def __init__(
        self,
        model_id: str = "scribe_v1",
        max_segment_seconds: float = 7.0,
        max_segment_chars: int = 100,
    ) -> None:
        self.model_id = model_id
        self.max_segment_seconds = max_segment_seconds
        self.max_segment_chars = max_segment_chars

        self.client = _build_client()

    def __call__(self, fpath: str | Path, audio_lang: str) -> Iterable[Segment]:
        return self.transcribe(fpath, audio_lang)

    def transcribe(self, fpath: str | Path, audio_lang: str) -> Iterable[Segment]:
        if isinstance(fpath, str):
            fpath = Path(fpath)

        logger.info(f"Sending {fpath} to Scribe ({self.model_id})")
        with open(fpath, "rb") as audio_file:
            response = self.client.speech_to_text.convert(
                file=audio_file,
                model_id=self.model_id,
                language_code=audio_lang,
            )

        words = getattr(response, "words", None)
        if words is None:
            transcripts = getattr(response, "transcripts", None)
            if not transcripts:
                return iter(())
            words = transcripts[0].words

        return self._group_words(words)

    def _group_words(self, words: Iterable) -> Iterable[Segment]:
        segment_id = 0
        buffer: list = []
        buffer_start: float | None = None
        buffer_end: float | None = None

        def flush() -> Segment | None:
            nonlocal segment_id
            if not buffer:
                return None
            text = "".join(w.text for w in buffer).strip()
            if not text:
                return None
            segment = Segment(id=segment_id, start=buffer_start, end=buffer_end, text=text)
            segment_id += 1
            return segment

        for word in words:
            if word.type == "audio_event":
                continue

            if buffer_start is None:
                buffer_start = word.start

            buffer.append(word)
            buffer_end = word.end

            text_so_far = "".join(w.text for w in buffer).strip()
            duration = buffer_end - buffer_start
            ends_sentence = word.type == "word" and text_so_far.endswith(_SENTENCE_END_CHARS)
            too_long = duration >= self.max_segment_seconds
            too_many_chars = len(text_so_far) >= self.max_segment_chars

            if ends_sentence or too_long or too_many_chars:
                segment = flush()
                if segment is not None:
                    yield segment
                buffer = []
                buffer_start = None
                buffer_end = None

        segment = flush()
        if segment is not None:
            yield segment
