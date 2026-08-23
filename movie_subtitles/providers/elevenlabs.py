import logging
import os
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from movie_subtitles.providers.base import Segment, Word

logger = logging.getLogger("elevenlabs")

_SENTENCE_END_CHARS = (".", "!", "?", "…")

# ElevenLabs' documented voice_settings.speed range; `speak()` clamps to this range,
# see Speak's docstring.
_MIN_SPEED = 0.7
_MAX_SPEED = 1.2

# A stock ElevenLabs voice ("Sarah"), multilingual-capable. Any voice_id works with
# eleven_multilingual_v2/eleven_turbo_v2_5; this is just a documented default.
_DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"


def build_client() -> ElevenLabs:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY environment variable is not set. "
            "Set it to your ElevenLabs API key to use the 'elevenlabs' engine."
        )

    return ElevenLabs(api_key=api_key)


def clone_voice(client: ElevenLabs, name: str, sample_paths: list[Path]) -> str:
    """Create an Instant Voice Clone from one or more sample audio files.

    Wraps `client.voices.ivc.create`. Returns the new voice's id; the caller is
    responsible for deleting it (via `delete_voice`) once it is no longer needed.
    """
    logger.info(f"Cloning voice {name!r} from {len(sample_paths)} sample(s)")
    with ExitStack() as stack:
        files = [stack.enter_context(open(path, "rb")) for path in sample_paths]
        response = client.voices.ivc.create(name=name, files=files)

    return response.voice_id


def delete_voice(client: ElevenLabs, voice_id: str) -> None:
    """Delete a previously cloned voice. Wraps `client.voices.delete`."""
    logger.info(f"Deleting voice {voice_id}")
    client.voices.delete(voice_id)


class ScribeTranscribe:
    def __init__(
        self,
        model_id: str = "scribe_v2",
        max_segment_seconds: float = 7.0,
        max_segment_chars: int = 100,
        diarize: bool = True,
    ) -> None:
        self.model_id = model_id
        self.max_segment_seconds = max_segment_seconds
        self.max_segment_chars = max_segment_chars
        self.diarize = diarize

        self.client = build_client()

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
                diarize=self.diarize,
            )

        words = getattr(response, "words", None)
        if words is None:
            transcripts = getattr(response, "transcripts", None)
            if not transcripts:
                raise RuntimeError(
                    f"Scribe transcription response (model '{self.model_id}') has neither "
                    "a 'words' field nor a 'transcripts' field. The response shape may have "
                    "changed; inspect it before assuming the audio is silent."
                )
            words = transcripts[0].words

        return self._group_words(words)

    # Deliberately separate from movie_subtitles/diarize.py's overlap-based split
    # (used for --asr-engine local/openai, which lack Scribe's native per-word
    # speaker_id) -- do not unify the two.
    def _group_words(self, words: Iterable[Any]) -> Iterable[Segment]:
        segment_id = 0
        buffer: list[Any] = []
        buffer_start: float | None = None
        buffer_end: float | None = None
        buffer_speaker: str | None = None

        def flush() -> Segment | None:
            nonlocal segment_id
            if not buffer or buffer_start is None or buffer_end is None:
                return None
            text = "".join(w.text for w in buffer).strip()
            if not text:
                return None
            word_timings = [
                Word(start=w.start, end=w.end, text=w.text) for w in buffer if w.type == "word"
            ]
            segment = Segment(
                id=segment_id,
                start=buffer_start,
                end=buffer_end,
                text=text,
                words=word_timings or None,
                speaker=buffer_speaker,
            )
            segment_id += 1
            return segment

        for word in words:
            if word.type == "audio_event":
                continue

            speaker_id = getattr(word, "speaker_id", None)
            speaker_changed = (
                buffer
                and speaker_id is not None
                and buffer_speaker is not None
                and speaker_id != buffer_speaker
            )
            if speaker_changed:
                segment = flush()
                if segment is not None:
                    yield segment
                buffer = []
                buffer_start = None
                buffer_end = None
                buffer_speaker = None

            if word.type == "word":
                if buffer_start is None:
                    buffer_start = word.start
                buffer_end = word.end
                if buffer_speaker is None and speaker_id is not None:
                    buffer_speaker = speaker_id

            buffer.append(word)

            text_so_far = "".join(w.text for w in buffer).strip()
            duration = (
                buffer_end - buffer_start
                if buffer_start is not None and buffer_end is not None
                else 0.0
            )
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
                buffer_speaker = None

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
    tighter than what the API allows; `speak()` itself clamps to the API's own
    0.7-1.2 range as a defensive floor/ceiling on whatever rate it is handed.
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

        self.client = build_client()

    def __call__(
        self, text: str, out_path: Path, speed: float = 1.0, voice: str | None = None
    ) -> Path:
        return self.speak(text, out_path, speed, voice)

    def speak(
        self, text: str, out_path: Path, speed: float = 1.0, voice: str | None = None
    ) -> Path:
        speed = max(_MIN_SPEED, min(_MAX_SPEED, speed))
        logger.info(f"Synthesising {len(text)} chars to {out_path} (speed={speed:.3f})")

        audio_chunks = self.client.text_to_speech.convert(
            voice or self.voice_id,
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


class Align:
    """Forced-alignment wrapper backed by ElevenLabs' forced_alignment endpoint.

    Given a synthesised TTS clip and the text it was generated from, this measures
    where speech actually begins and ends inside the clip -- TTS output routinely
    carries leading/trailing padding (silence, breath, etc.) that throws off
    timing-slot fitting, so the caller can trim the clip to its real speech span.
    """

    def __init__(self, client: ElevenLabs | None = None) -> None:
        self.client = client if client is not None else build_client()

    def __call__(self, clip: Path, text: str) -> tuple[float, float]:
        return self.align(clip, text)

    def align(self, clip: Path, text: str) -> tuple[float, float]:
        logger.debug(f"Aligning {clip} against {len(text)} chars of text")
        with open(clip, "rb") as audio_file:
            response = self.client.forced_alignment.create(file=audio_file, text=text)

        words = getattr(response, "words", None)
        if not words:
            raise RuntimeError(f"Forced alignment response for {clip} has no usable word timings.")

        return words[0].start, words[-1].end
