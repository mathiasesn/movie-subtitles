import logging
import os
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI

from movie_subtitles.providers.base import Segment

logger = logging.getLogger("openai")

_MIN_SPEED = 0.25
_MAX_SPEED = 4.0

_SYSTEM_PROMPT = (
    "You are a subtitle translator. Translate the given text into the requested "
    "language. Respond with only the translation, no preamble, no explanation, "
    "no quotation marks."
)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it to your OpenAI API key to use the 'openai' engine."
        )

    return OpenAI(api_key=api_key)


class OpenAITranscribe:
    """ASRProvider backed by the OpenAI audio transcriptions endpoint.

    Default model is "gpt-4o-transcribe". Note: gpt-4o-transcribe may not support
    verbose_json/segment timestamps on all accounts -- "whisper-1" is the documented
    fallback model for segment-level timestamps, so the model is an __init__ param the
    caller can override.
    """

    def __init__(self, model: str = "gpt-4o-transcribe") -> None:
        self.model = model
        self.client = build_client()

    def __call__(self, fpath: str | Path, audio_lang: str) -> Iterable[Segment]:
        return self.transcribe(fpath, audio_lang)

    def transcribe(self, fpath: str | Path, audio_lang: str) -> Iterable[Segment]:
        if isinstance(fpath, str):
            fpath = Path(fpath)

        logger.info(f"Sending {fpath} to OpenAI transcriptions ({self.model})")
        with open(fpath, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=audio_lang,
            )

        segments = getattr(response, "segments", None) or ()
        return self._yield_segments(segments)

    def _yield_segments(self, segments: Iterable) -> Iterable[Segment]:
        for segment_id, segment in enumerate(segments):
            text = getattr(segment, "text", "").strip()
            if not text:
                continue
            yield Segment(
                id=segment_id,
                start=segment.start,
                end=segment.end,
                text=text,
            )


class OpenAITranslate:
    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self.client = build_client()

    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        return self.translate(text, output_lang, budget_chars)

    def translate(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        prompt = f"Translate the following text into '{output_lang}':\n\n{text}"
        if budget_chars is not None:
            prompt += (
                f"\n\nThe translation must fit within roughly {budget_chars} characters. "
                "Stay as faithful as possible to the meaning while shortening phrasing, "
                "dropping filler, or rewording as needed to hit that budget."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        translation = response.choices[0].message.content or ""
        return translation.strip()


class OpenAISpeak:
    """TTSProvider backed by the OpenAI audio speech endpoint.

    Note: the spec's timing-drift strategy clamps the requested rate at the dub.py
    layer (0.9-1.15x). The OpenAI `speed` parameter itself supports a wider documented
    range of 0.25 (slowest) to 4.0 (fastest), default 1.0. This class clamps to that
    0.25-4.0 range, mirroring how elevenlabs.Speak clamps to ElevenLabs' 0.7-1.2 range.
    """

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "gpt-4o-mini-tts",
    ) -> None:
        self.voice = voice
        self.model = model
        self.client = build_client()

    def __call__(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        return self.speak(text, out_path, speed)

    def speak(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        speed = max(_MIN_SPEED, min(_MAX_SPEED, speed))
        logger.info(f"Synthesising {len(text)} chars to {out_path} (speed={speed:.3f})")

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            speed=speed,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        response.write_to_file(out_path)

        return out_path
