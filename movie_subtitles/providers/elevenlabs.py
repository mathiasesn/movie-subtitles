import logging
import os
from collections.abc import Iterable
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from movie_subtitles.providers.base import Segment

logger = logging.getLogger("elevenlabs")

_SENTENCE_END_CHARS = (".", "!", "?", "…")

# A stock ElevenLabs voice ("Sarah"), multilingual-capable. Any voice_id works with
# eleven_multilingual_v2/eleven_turbo_v2_5; this is just a documented default.
_DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"


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


class Speak:
    """TTSProvider backed by the ElevenLabs text-to-speech convert endpoint.

    Confirmed against https://elevenlabs.io/docs/api-reference/text-to-speech/convert.md
    (2026-08-19): POST to voice_id, with `text`, `model_id` (default
    "eleven_multilingual_v2", the multilingual model doc recommends and the one
    appropriate for Danish output), and an optional `voice_settings` object.

    Note: the spec's timing-drift strategy clamps the requested rate to 0.9-1.15x.
    The API itself supports a wider range for `voice_settings.speed` -- per
    https://elevenlabs.io/docs/best-practices/prompting/controls.md the documented
    valid range is 0.7 (slowest) to 1.2 (fastest), default 1.0, with a warning that
    extreme values can degrade audio quality. The dub.py clamp is intentionally
    tighter than what the API allows.
    """

    def __init__(
        self,
        voice_id: str = _DEFAULT_VOICE_ID,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
    ) -> None:
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format

        self.client = _build_client()

    def __call__(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        return self.speak(text, out_path, speed)

    def speak(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        logger.info(f"Synthesising {len(text)} chars to {out_path} (speed={speed:.3f})")

        audio_chunks = self.client.text_to_speech.convert(
            self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings=VoiceSettings(speed=speed),
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as audio_file:
            for chunk in audio_chunks:
                audio_file.write(chunk)

        return out_path
