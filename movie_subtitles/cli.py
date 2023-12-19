import logging
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

from tqdm.auto import tqdm

from movie_subtitles.transcribe import Transcribe
from movie_subtitles.translate import Translate

logger = logging.getLogger("cli")


def create_subtitles(
    fpath: str | Path,
    audio_lang: str = "en",
    srt_lang: str = "da",
    whisper_model_name: str = "large-v3",
    mt_model_name: str = "jbochi/madlad400-3b-mt",
) -> None:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    srt_file = fpath.with_suffix(".srt")

    transcriber = Transcribe(whisper_model_name)
    segments = transcriber(fpath, audio_lang)

    translator = Translate(mt_model_name)

    srt_lines = []
    for segment in tqdm(segments, desc="Writing to srt file"):
        start_time = str(0) + str(timedelta(seconds=int(segment.start))) + ",000"
        end_time = str(0) + str(timedelta(seconds=int(segment.end))) + ",000"
        text = segment.text
        segment_id = segment.id + 1

        text = translator(text, srt_lang)
        if not text:
            continue

        segment = f"{segment_id}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"
        srt_lines.append(segment)

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
    args = parser.parse_args()

    create_subtitles(
        args.input,
        args.audio_lang,
        args.srt_lang,
        args.whisper_model,
        args.mt_model,
    )


if __name__ == "__main__":
    main()
