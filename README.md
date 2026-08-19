<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo-dark.svg">
    <img src="assets/logo/logo.svg" alt="Movie subtitles" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/mathiasesn/movie-subtitles/actions/workflows/ci.yml"><img src="https://github.com/mathiasesn/movie-subtitles/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mathiasesn/movie-subtitles/actions/workflows/publish.yml"><img src="https://github.com/mathiasesn/movie-subtitles/actions/workflows/publish.yml/badge.svg" alt="Publish"></a>
  <a href="https://pypi.org/project/movie-subtitles/"><img src="https://img.shields.io/pypi/v/movie-subtitles.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/movie-subtitles/"><img src="https://img.shields.io/pypi/pyversions/movie-subtitles.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

Command line tool that turns a video or audio file into a translated `.srt` file, and
optionally into a dubbed video with a synthesised translated audio track.

## Installation

Published on PyPI as [`movie-subtitles`](https://pypi.org/project/movie-subtitles/).

```shell
# Run without installing (recommended)
uvx movie-subtitles --input clip.mp4

# Install as a tool
uv tool install movie-subtitles

# Or with pip
pip install movie-subtitles

# Latest from git instead of the release on PyPI
uvx --from git+https://github.com/mathiasesn/movie-subtitles.git movie-subtitles --input clip.mp4

# Develop locally
git clone https://github.com/mathiasesn/movie-subtitles.git
cd movie-subtitles && uv sync && uv run movie-subtitles --help
```

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
`ffmpeg` must be on `PATH` for `--dub`; the plain `.srt` path and `--managed` do not need it.

## Engines

The pipeline has three stages, each backed by a vendor you choose independently:

| Flag | Values |
| --- | --- |
| `--asr-engine` | `local` (faster-whisper `large-v3`), `elevenlabs` (Scribe), `openai` (`whisper-1`) |
| `--translation-engine` | `local` (MADLAD400), `anthropic` (Claude), `openai` (chat completions) |
| `--tts-engine` | `elevenlabs`, `openai` (`tts-1`) |

`--engine` is a shorthand that sets all three at once; per-stage flags always override it.

| Invocation | ASR | Translation | TTS |
| --- | --- | --- | --- |
| `--engine local` (default) | faster-whisper | MADLAD400 | none |
| `--engine openai` | OpenAI | OpenAI | OpenAI |
| `--engine elevenlabs` | Scribe | **error** | ElevenLabs |
| `--engine elevenlabs --translation-engine anthropic` | Scribe | Claude | ElevenLabs |

**`--engine elevenlabs` alone errors on purpose.** ElevenLabs has no standalone
text-translation endpoint — translation exists only bundled inside the Dubbing job
(`--managed`) — so there is nothing honest for the shorthand to resolve the translation
stage to. Pass `--translation-engine {anthropic,openai,local}` explicitly.

Engine values name **vendors, not pipelines**. An earlier version used
`--translation-engine elevenlabs`; that value no longer exists, and its replacement is
`--translation-engine anthropic`, since the call was always Claude.

### Known limitation: `--asr-engine openai`

`whisper-1`'s segment timestamps degrade to uniform 1.000s spans on music-heavy or
dialogue-sparse audio (observed on a movie trailer), which corrupts both `.srt` cue
timings and dub slots. Prefer `--asr-engine elevenlabs` on such material.

## API keys

`--engine local` needs no keys and stays fully offline after the first model download.
Everything else reads one or more of:

```shell
export ELEVENLABS_API_KEY="..."   # --asr-engine/--tts-engine elevenlabs, --managed
export ANTHROPIC_API_KEY="..."    # --translation-engine anthropic
export OPENAI_API_KEY="..."       # any stage set to openai
```

Or copy `.env.example` to `.env` and fill it in — `.env` is gitignored and loaded on
startup, searching upward from the directory you run in. Exported variables take
precedence. A missing key exits with a one-line error naming the variable, no traceback.

## Usage

```shell
# Local, offline, .srt only
movie-subtitles --input clip.mp4

# ElevenLabs ASR + Claude translation, .srt only
movie-subtitles --input clip.mp4 --engine elevenlabs --translation-engine anthropic

# ...and dubbed into clip.dubbed.mp4 (needs ffmpeg)
movie-subtitles --input clip.mp4 --engine elevenlabs --translation-engine anthropic --dub

# OpenAI end to end, .srt only / dubbed
movie-subtitles --input clip.mp4 --engine openai
movie-subtitles --input clip.mp4 --engine openai --dub

# Mixed vendors — Scribe ASR, OpenAI translation and voice
movie-subtitles --input clip.mp4 --asr-engine elevenlabs --translation-engine openai --tts-engine openai --dub

# Managed ElevenLabs Dubbing job — no local ASR/MT/TTS code runs at all
movie-subtitles --input clip.mp4 --managed
```

Other flags: `--audio-lang`, `--srt-lang`, `--whisper-model`, `--mt-model`. Run
`movie-subtitles --help` for the full list.

### Dubbing notes

- **`--dub` replaces the original audio track, it does not mix with it.** Only the
  synthesised track is mapped onto the source video; the original spoken audio is dropped.
- **`--dub` and `--managed` are mutually exclusive.** `--managed` replaces the whole
  pipeline with a single hosted job.
- **`--dub` fails if TTS resolves to something unusable**, e.g. plain `--engine local
  --dub`, since `local` has no TTS backend:

  ```
  [ERROR][cli] --dub requires a usable TTS engine, but it resolved to 'local'. Pass --tts-engine {elevenlabs,openai} explicitly.
  ```

- **`--translation-engine local` with `--dub` is allowed but warns.** MADLAD400 ignores
  the length budget, so timing-drift fitting degrades to TTS-rate-only.
