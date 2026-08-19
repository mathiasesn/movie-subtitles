import logging
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm

from movie_subtitles.providers.base import ASRProvider, Segment, TranslationProvider, TTSProvider
from movie_subtitles.srt import format_block

logger = logging.getLogger("cli")


# Assumption, not a measured value: rough average speaking rate used to derive a
# character budget from a segment's duration. Tune this once real dubs are assessed.
_CHARS_PER_SECOND = 15.0


def _budget_chars(start: float, end: float) -> int:
    """Derive a translation length budget (in characters) from a segment's duration."""
    duration = max(end - start, 0.0)
    return max(int(duration * _CHARS_PER_SECOND), 1)


def _build_asr_provider(asr_engine: str, whisper_model_name: str) -> ASRProvider:
    if asr_engine == "local":
        from movie_subtitles.providers.local import Transcribe

        return Transcribe(whisper_model_name)
    elif asr_engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import ScribeTranscribe

        return ScribeTranscribe()
    elif asr_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAITranscribe

        return OpenAITranscribe()
    else:
        raise ValueError(f"Unknown ASR engine: {asr_engine}")


def _build_translation_provider(translation_engine: str, mt_model_name: str) -> TranslationProvider:
    if translation_engine == "local":
        from movie_subtitles.providers.local import Translate

        return Translate(mt_model_name)
    elif translation_engine == "anthropic":
        from movie_subtitles.providers.llm import LLMTranslate

        return LLMTranslate()
    elif translation_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAITranslate

        return OpenAITranslate()
    else:
        raise ValueError(f"Unknown translation engine: {translation_engine}")


_VALID_TTS_ENGINES = {"elevenlabs", "openai"}


def _build_tts_provider(tts_engine: str) -> TTSProvider:
    if tts_engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import Speak

        return Speak()
    elif tts_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAISpeak

        return OpenAISpeak()
    else:
        raise ValueError(f"Unknown TTS engine: {tts_engine}")


def _build_providers(
    asr_engine: str, translation_engine: str, whisper_model_name: str, mt_model_name: str
) -> tuple[ASRProvider, TranslationProvider]:
    """Build the ASR + translation providers, per-stage overridable.

    `asr_engine`/`translation_engine` each independently pick a backend by vendor name
    (`local`, `elevenlabs`, `openai` for ASR; `local`, `anthropic`, `openai` for
    translation) — e.g. Scribe ASR with the local MADLAD400 translator. `--engine` in the
    CLI is only a shorthand that sets `asr_engine`/`translation_engine`/`tts_engine` when
    the per-stage flags are not given; `_build_tts_provider` builds the TTS provider
    separately (see `_dub_and_mux`), since this function only covers the two stages every
    invocation needs.
    """
    return (
        _build_asr_provider(asr_engine, whisper_model_name),
        _build_translation_provider(translation_engine, mt_model_name),
    )


def create_subtitles(
    fpath: str | Path,
    audio_lang: str = "en",
    srt_lang: str = "da",
    whisper_model_name: str = "large-v3",
    mt_model_name: str = "jbochi/madlad400-3b-mt",
    *,
    engine: str = "local",
    asr_engine: str | None = None,
    translation_engine: str | None = None,
    dub: bool = False,
    managed: bool = False,
    tts_engine: str | None = None,
) -> None:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    # --engine is the shorthand that sets all stages; --asr-engine/--translation-engine/
    # --tts-engine override it per stage (Goal 2 of specs/openai-api-key-support.md).
    resolved_asr_engine = asr_engine if asr_engine is not None else engine
    resolved_translation_engine = translation_engine if translation_engine is not None else engine
    resolved_tts_engine = tts_engine if tts_engine is not None else engine

    if managed and dub:
        raise ValueError(
            "--managed and --dub are mutually exclusive: --managed runs the ElevenLabs "
            "Dubbing job end to end instead of the local transcribe/translate/dub pipeline."
        )

    if not managed and translation_engine is None and engine == "elevenlabs":
        raise ValueError(
            "--engine elevenlabs does not set a translation stage: ElevenLabs has no "
            "standalone text-translation endpoint (translation exists only bundled "
            "inside the Dubbing job, see --managed). Pass --translation-engine "
            "{anthropic,openai,local} explicitly."
        )

    if dub and resolved_translation_engine == "local":
        raise ValueError(
            "--dub requires a length-budgeted translator: the local MADLAD400 translator "
            "ignores budget_chars, so timing-drift fitting (step 1 of the drift strategy) "
            "is a silent no-op and the dub would be worse for no stated reason. Use "
            "--translation-engine anthropic (or --engine elevenlabs/openai) with --dub."
        )

    if dub and resolved_tts_engine not in _VALID_TTS_ENGINES:
        raise ValueError(
            f"--dub requires a usable TTS engine, but it resolved to '{resolved_tts_engine}'. "
            "Pass --tts-engine {elevenlabs,openai} explicitly."
        )

    if managed:
        _run_managed(fpath, audio_lang, srt_lang)
        return

    srt_file = fpath.with_suffix(".srt")

    transcriber, translator = _build_providers(
        resolved_asr_engine, resolved_translation_engine, whisper_model_name, mt_model_name
    )
    segments = transcriber(fpath, audio_lang)

    srt_lines = []
    dub_segments: list[Segment] = []
    dub_translations: dict[int, str] = {}
    for segment in tqdm(segments, desc="Writing to srt file"):
        segment_id = segment.id + 1

        budget_chars = _budget_chars(segment.start, segment.end)
        text = translator(segment.text, srt_lang, budget_chars=budget_chars)
        if not text:
            continue

        srt_lines.append(format_block(segment_id, segment.start, segment.end, text))

        if dub:
            dub_segments.append(segment)
            dub_translations[segment.id] = text

    srt_file.write_text("".join(srt_lines), encoding="utf-8")
    logger.info(f"Saved srt file to {srt_file}")

    if dub:
        _dub_and_mux(fpath, dub_segments, dub_translations, resolved_tts_engine)


def _run_managed(fpath: Path, audio_lang: str, srt_lang: str) -> None:
    from movie_subtitles.dubbing import ManagedDub

    managed_dub = ManagedDub()
    out_path = managed_dub(fpath, audio_lang, srt_lang)
    logger.info(f"Saved managed dub to {out_path}")


def _check_ffmpeg_tools() -> None:
    """Raise a clear RuntimeError if ffmpeg or ffprobe is missing, before any work starts.

    dub.py uses ffprobe to measure synthesised clip duration and mux.py uses ffmpeg to
    assemble/mux; a missing ffprobe used to crash with a raw FileNotFoundError from
    subprocess deep inside dub.py. Checking both up front keeps the failure mode
    consistent with mux.py's existing ffmpeg check.
    """
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not on PATH. Install ffmpeg (which provides both "
            "ffmpeg and ffprobe) to use --dub."
        )


def _dub_and_mux(
    fpath: Path,
    segments: list[Segment],
    translations: dict[int, str],
    tts_engine: str,
) -> None:
    from movie_subtitles.dub import synthesise_track
    from movie_subtitles.mux import mux_dub

    _check_ffmpeg_tools()

    audio_track = fpath.with_name(f"{fpath.stem}.dub_audio.mp3")
    tts = _build_tts_provider(tts_engine)

    synthesise_track(segments, translations, tts, audio_track)

    dubbed_path = mux_dub(fpath, audio_track)
    logger.info(f"Saved dubbed video to {dubbed_path}")


def main() -> None:
    parser = ArgumentParser(
        "Command line interface for movie subtitles",
        usage="translation-cli <command> [<args>]",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The input file to transcribe",
    )
    parser.add_argument(
        "--audio-lang",
        type=str,
        default="en",
        help="The language of the audio in the movie",
    )
    parser.add_argument(
        "--srt-lang",
        type=str,
        default="da",
        help="The language of the srt file",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="The whisper model to use",
    )
    parser.add_argument(
        "--mt-model",
        type=str,
        default="jbochi/madlad400-3b-mt",
        help="The machine translation model to use",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["local", "elevenlabs", "openai"],
        default="local",
        help="The provider engine to use for transcription, translation and TTS "
        "(shorthand for --asr-engine, --translation-engine and --tts-engine when those "
        "are not set). --engine elevenlabs requires --translation-engine to be set "
        "explicitly, since ElevenLabs has no standalone translation endpoint",
    )
    parser.add_argument(
        "--asr-engine",
        type=str,
        choices=["local", "elevenlabs", "openai"],
        default=None,
        help="Override --engine for the ASR (transcription) stage only",
    )
    parser.add_argument(
        "--translation-engine",
        type=str,
        choices=["local", "anthropic", "openai"],
        default=None,
        help="Override --engine for the translation stage only. --dub requires this to "
        "resolve to 'anthropic' or 'openai', since the local translator ignores the "
        "length budget",
    )
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=["elevenlabs", "openai"],
        default=None,
        help="Override --engine for the TTS (dubbing) stage only",
    )
    parser.add_argument(
        "--dub",
        action="store_true",
        help=(
            "Synthesise the translated segments with TTS and mux them over the source "
            "video (requires ffmpeg and the API key for the resolved TTS engine)"
        ),
    )
    parser.add_argument(
        "--managed",
        action="store_true",
        help=(
            "Use the managed ElevenLabs Dubbing job (create/poll/download) instead of "
            "the local transcribe/translate/dub pipeline (requires ELEVENLABS_API_KEY). "
            "Mutually exclusive with --dub."
        ),
    )
    args = parser.parse_args()

    try:
        create_subtitles(
            args.input,
            args.audio_lang,
            args.srt_lang,
            args.whisper_model,
            args.mt_model,
            engine=args.engine,
            asr_engine=args.asr_engine,
            translation_engine=args.translation_engine,
            dub=args.dub,
            managed=args.managed,
            tts_engine=args.tts_engine,
        )
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
