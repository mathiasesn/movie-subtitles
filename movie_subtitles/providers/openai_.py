import logging
import os
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI

from movie_subtitles.providers.base import Segment
from movie_subtitles.providers.prompt import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger("openai_provider")

_MIN_SPEED = 0.25
_MAX_SPEED = 4.0


def build_client() -> OpenAI:
    """Build a fresh OpenAI client, one per provider instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it to your OpenAI API key to use the 'openai' engine."
        )

    return OpenAI(api_key=api_key)


class OpenAITranscribe:
    """ASRProvider backed by the OpenAI audio transcriptions endpoint.

    Default model is "whisper-1" -- the only OpenAI transcription model that
    supports response_format="verbose_json" with segment-level timestamps.
    "gpt-4o-transcribe" does not support verbose_json or segment/word timestamps
    at all, and this pipeline is built entirely on segment timings (they drive
    every downstream .srt cue and dub timing slot), so only "whisper-1" can
    drive it. The model remains an __init__ param the caller can override for
    other use cases.
    """

    def __init__(self, model: str = "whisper-1") -> None:
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

        segments = getattr(response, "segments", None)
        if not segments:
            raise RuntimeError(
                f"OpenAI transcription response from model '{self.model}' has no "
                "'segments' field. This model may not support segment timestamps "
                "(response_format='verbose_json', timestamp_granularities=['segment']); "
                "try model='whisper-1' instead."
            )
        return self._yield_segments(segments)

    def _yield_segments(self, segments: Iterable) -> Iterable[Segment]:
        # Split out from transcribe() so the missing-segments check above runs when
        # transcribe() is called, not when the first segment is pulled. Do not re-merge.
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
    """TranslationProvider backed by the OpenAI chat completions endpoint.

    Default model is "gpt-5.6-terra", the balanced tier of the current GPT-5.6
    family. GPT-5 series models are reasoning models on this endpoint: they take
    `max_completion_tokens` rather than `max_tokens`, and reject `temperature`.
    `reasoning_effort="none"` keeps a per-segment translation call fast and cheap --
    there is nothing to deliberate over here, and reasoning tokens would otherwise
    eat into the completion budget.
    """

    def __init__(self, model: str = "gpt-5.6-terra") -> None:
        self.model = model
        self.client = build_client()

    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        return self.translate(text, output_lang, budget_chars)

    def translate(self, text: str, output_lang: str, budget_chars: int | None = None) -> str:
        prompt = build_prompt(text, output_lang, budget_chars)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=1024,
            reasoning_effort="none",
        )

        translation = response.choices[0].message.content or ""
        return translation.strip()


class OpenAISpeak:
    """TTSProvider backed by the OpenAI audio speech endpoint.

    Note: the spec's timing-drift strategy clamps the requested rate at the dub.py
    layer (0.9-1.15x). The OpenAI `speed` parameter itself supports a wider documented
    range of 0.25 (slowest) to 4.0 (fastest), default 1.0; `speak()` clamps to that
    0.25-4.0 range as a defensive floor/ceiling on whatever rate it is handed.

    Default model is "tts-1", not "gpt-4o-mini-tts": gpt-4o-mini-tts accepts the
    `speed` parameter but silently ignores it, which would make the dub path's
    rate-fitting lever inert. tts-1 honours `speed`. gpt-4o-mini-tts remains
    reachable via the constructor argument for callers that don't need `speed`.
    """

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "tts-1",
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
