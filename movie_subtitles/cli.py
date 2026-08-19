import logging
from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm

from movie_subtitles.srt import format_block

logger = logging.getLogger("cli")


# Assumption, not a measured value: rough average speaking rate used to derive a
# character budget from a segment's duration. Tune this once real dubs are assessed.
_CHARS_PER_SECOND = 15.0


def _budget_chars(start: float, end: float) -> int:
    """Derive a translation length budget (in characters) from a segment's duration."""
    duration = max(end - start, 0.0)
    return max(int(duration * _CHARS_PER_SECOND), 1)


def _build_providers(engine: str, whisper_model_name: str, mt_model_name: str):
    if engine == "local":
        from movie_subtitles.providers.local import Transcribe, Translate

        return Transcribe(whisper_model_name), Translate(mt_model_name)
    elif engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import ScribeTranscribe
        from movie_subtitles.providers.llm import LLMTranslate

        return ScribeTranscribe(), LLMTranslate()
    else:
        raise ValueError(f"Unknown engine: {engine}")


def create_subtitles(
    fpath: str | Path,
    audio_lang: str = "en",
    srt_lang: str = "da",
    whisper_model_name: str = "large-v3",
    mt_model_name: str = "jbochi/madlad400-3b-mt",
    engine: str = "local",
) -> None:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    srt_file = fpath.with_suffix(".srt")

    transcriber, translator = _build_providers(engine, whisper_model_name, mt_model_name)
    segments = transcriber(fpath, audio_lang)

    srt_lines = []
    for segment in tqdm(segments, desc="Writing to srt file"):
        segment_id = segment.id + 1

        budget_chars = _budget_chars(segment.start, segment.end)
        text = translator(segment.text, srt_lang, budget_chars=budget_chars)
        if not text:
            continue

        srt_lines.append(format_block(segment_id, segment.start, segment.end, text))

    srt_file.write_text("".join(srt_lines), encoding="utf-8")
    logger.info(f"Saved srt file to {srt_file}")


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
        choices=["local", "elevenlabs"],
        default="local",
        help="The provider engine to use for transcription and translation",
    )
    args = parser.parse_args()

    create_subtitles(
        args.input,
        args.audio_lang,
        args.srt_lang,
        args.whisper_model,
        args.mt_model,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
