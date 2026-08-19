# Movie subtitles

Command line interface for creating movie subtitles either as .srt files or by adding them to each movie frame.

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

Run without installing (recommended)

```shell
uvx --from git+https://github.com/mathiasesn/movie-subtitles.git movie-subtitles --help
```

Install as a tool

```shell
uv tool install git+https://github.com/mathiasesn/movie-subtitles.git
```

Develop locally

```shell
git clone https://github.com/mathiasesn/movie-subtitles.git
cd movie-subtitles
uv sync
uv run movie-subtitles --help
```

## Usage

```shell
movie-subtitles --help
usage: translation-cli <command> [<args>]

options:
  -h, --help            show this help message and exit
  --input INPUT         The input file to transcribe
  --audio-lang AUDIO_LANG
                        The language of the audio in the movie
  --srt-lang SRT_LANG   The language of the srt file
  --whisper-model WHISPER_MODEL
                        The whisper model to use
  --mt-model MT_MODEL   The machine translation model to use
```
