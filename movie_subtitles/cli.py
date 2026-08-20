import logging
import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from tqdm.auto import tqdm

from movie_subtitles.providers.base import (
    AlignmentProvider,
    ASRProvider,
    Segment,
    TranslationProvider,
    TTSProvider,
)
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


# Single source of truth: argparse choices, the --dub guard, and its error
# message all derive from this, so adding a backend is one edit.
TTS_ENGINES = ("elevenlabs", "openai")


def _build_tts_provider(tts_engine: str) -> TTSProvider:
    if tts_engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import Speak

        return Speak()
    elif tts_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAISpeak

        return OpenAISpeak()
    else:
        raise ValueError(f"Unknown TTS engine: {tts_engine}")


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
    dub_workers: int = 8,
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
        logger.warning(
            "--dub with the local MADLAD400 translator: it ignores budget_chars, so "
            "timing-drift fitting degrades to TTS-rate-only (step 1 of the drift "
            "strategy is a silent no-op). Use --translation-engine anthropic/openai for "
            "length-budgeted translations if the dub timing is off."
        )

    if dub and resolved_tts_engine not in TTS_ENGINES:
        raise ValueError(
            f"--dub requires a usable TTS engine, but it resolved to '{resolved_tts_engine}'. "
            f"Pass --tts-engine {{{','.join(TTS_ENGINES)}}} explicitly."
        )

    if managed:
        _run_managed(fpath, audio_lang, srt_lang)
        return

    srt_file = fpath.with_suffix(".srt")

    # Build cheap-to-construct providers first so a missing API key or a missing ffmpeg
    # fails in milliseconds, rather than after the ASR model download and a full run of
    # paid per-segment translation calls.
    translator = _build_translation_provider(resolved_translation_engine, mt_model_name)
    tts = None
    if dub:
        _check_ffmpeg_tools()
        tts = _build_tts_provider(resolved_tts_engine)

    transcriber = _build_asr_provider(resolved_asr_engine, whisper_model_name)
    segments = transcriber(fpath, audio_lang)

    srt_lines = []
    dub_segments: list[Segment] = []
    dub_translations: dict[int, str] = {}
    cue_id = 0
    for segment in tqdm(segments, desc="Writing to srt file"):
        budget_chars = _budget_chars(segment.start, segment.end)
        text = translator(segment.text, srt_lang, budget_chars=budget_chars)
        if not text:
            continue

        cue_id += 1
        srt_lines.append(format_block(cue_id, segment.start, segment.end, text))

        if dub:
            dub_segments.append(segment)
            dub_translations[segment.id] = text

    srt_file.write_text("".join(srt_lines), encoding="utf-8")
    logger.info(f"Saved srt file to {srt_file}")

    if tts is not None:
        _dub_and_mux(fpath, dub_segments, dub_translations, tts, dub_workers=dub_workers)


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


def _build_aligner() -> AlignmentProvider:
    """Build the Forced Alignment -> silencedetect -> container-duration degrade chain.

    Forced Alignment is a nice-to-have, not a hard requirement for --dub: a
    --tts-engine openai user may have no ELEVENLABS_API_KEY at all. So any failure
    building it (missing key, client construction error) is caught and logged, and the
    chain is built without it -- `FallbackAlign` degrades through the remaining tiers on
    a per-call basis, ending in the never-failing `DurationAlign`, so this never returns
    `None`.
    """
    from movie_subtitles.providers.base import FallbackAlign
    from movie_subtitles.providers.ffmpeg_align import DurationAlign, SilenceAlign

    providers: list[AlignmentProvider] = []

    if not os.environ.get("ELEVENLABS_API_KEY"):
        logger.info(
            "ELEVENLABS_API_KEY not set: speech-boundary measurement will fall back to "
            "ffmpeg silencedetect instead of Forced Alignment."
        )
    else:
        from movie_subtitles.providers.elevenlabs import Align

        try:
            providers.append(Align())
        except Exception as exc:
            logger.warning(
                f"Could not build the Forced Alignment aligner ({exc}); speech-boundary "
                "measurement will fall back to ffmpeg silencedetect."
            )

    providers.append(SilenceAlign())
    providers.append(DurationAlign())

    return FallbackAlign(providers)


def _dub_and_mux(
    fpath: Path,
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    dub_workers: int = 8,
) -> None:
    from movie_subtitles.dub import synthesise_track
    from movie_subtitles.mux import mux_dub

    audio_track = fpath.with_name(f"{fpath.stem}.dub_audio.mp3")
    aligner = _build_aligner()

    synthesise_track(
        segments, translations, tts, audio_track, aligner=aligner, max_workers=dub_workers
    )

    dubbed_path = mux_dub(fpath, audio_track)
    logger.info(f"Saved dubbed video to {dubbed_path}")

    audio_track.unlink()


def _load_env() -> None:
    """Load API keys from a .env file found by searching upward from the cwd.

    Called from main() only: importing this module stays side-effect free for library
    callers, who set the environment themselves. usecwd=True is deliberate -- the
    default resolves .env relative to *this file*, which finds the repo's own .env even
    when the user is working in some other project. Searching from the cwd instead means
    the .env belonging to the directory you invoked the CLI from is the one that wins.

    Already-exported variables are never overridden, so a real environment variable
    still beats the file. A missing .env is not an error: providers raise their own
    RuntimeError naming the specific key they need.
    """
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment from {dotenv_path}")


def main() -> None:
    _load_env()

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
        help="Override --engine for the translation stage only. --dub with this resolving "
        "to 'local' still works, but timing-drift fitting degrades to TTS-rate-only "
        "since the local translator ignores the length budget",
    )
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=list(TTS_ENGINES),
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
        "--dub-workers",
        type=int,
        default=8,
        help=(
            "Maximum number of TTS/alignment calls to run concurrently during --dub "
            "synthesis. 1 restores serial behaviour, the escape hatch for vendor rate "
            "limits"
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
            dub_workers=args.dub_workers,
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
