# movie-subtitles Architecture Documentation

> Generated: 2026-08-19 · Commit: c7557ea · Version: 0.0.1
> Re-read this file at the start of any session touching this codebase. Update it when the architecture changes (new major dependency, restructured layer, changed convention).

## 1. How to Read This Document

Written for AI coding agents working on this repo. It states the stack, the exact commands, the module boundaries, and the conventions to follow — so you do not need to re-explore the tree. The codebase is small (4 source files, ~200 lines); this document is deliberately short. Update it if a module is added, a model backend is swapped, or tests are introduced.

## 2. Overview

`movie-subtitles` is a single-command CLI that turns a video/audio file into a translated `.srt` subtitle file.

Pipeline, end to end, all local and synchronous:

1. `Transcribe` runs faster-whisper (with VAD filtering) over the input file → an iterable of timed `Segment`s in the audio language.
2. For each segment, `Translate` runs a MADLAD400 T5 seq2seq model → target-language text.
3. `create_subtitles` formats each segment as an SRT block and writes `<input>.srt` next to the input file.

There is no server, no persistence, no API layer. Models are pulled from Hugging Face on first use and cached by the underlying libraries.

**Known gap:** the README advertises "adding them to each movie frame" (burned-in subtitles). That is *not implemented* — the only output path is the `.srt` file. Do not assume frame-rendering code exists.

## 3. Technology Stack

- **Python** — `requires-python = ">=3.10"`; the local `.venv` is 3.12. Type hints use `X | Y` union syntax (3.10+).
- **faster-whisper** — speech-to-text (CTranslate2 Whisper). Default model `large-v3`.
- **transformers** (+ **torch**, **sentencepiece** via T5Tokenizer) — machine translation. Default model `jbochi/madlad400-3b-mt`, loaded with `device_map="auto"`.
- **tqdm** — progress bar over segments during SRT writing.
- **argparse** (stdlib) — CLI argument parsing. No Click/Typer, despite `typer` being present in `.venv` as a transitive dep.
- **uv** — dependency resolution, lockfile (`uv.lock`), venv, and tool install. **hatchling** — build backend.
- **ruff** — the only linter/formatter and the only CI gate. Dev dependency group `dev`.

## 4. Project Structure

```
movie_subtitles/          # the entire package — flat, no subpackages
  __init__.py             # side-effecting: configures root logging (stdout, INFO) on import
  cli.py                  # argparse entry point + create_subtitles() orchestration & SRT formatting
  transcribe.py           # Transcribe class — wraps faster_whisper.WhisperModel
  translate.py            # Translate class — wraps T5ForConditionalGeneration + T5Tokenizer
.github/workflows/ci.yml  # lint-only CI (ruff format --check, ruff check)
pyproject.toml            # metadata, deps, console script, ruff config
uv.lock                   # committed lockfile
```

No `tests/`, no `src/` layout, no `docs/` beyond this file.

## 5. Core Architecture Principles

These describe what the code actually does — follow them rather than importing conventions from elsewhere.

1. **Model wrappers are callable classes.** `Transcribe` and `Translate` both load their model in `__init__`, expose a named method (`transcribe` / `translate`), and define `__call__` delegating to it. A new model stage should follow the same shape.
2. **Orchestration lives only in `cli.py`.** `create_subtitles()` is the single place that knows about the pipeline order, timestamps, and SRT format. `transcribe.py` and `translate.py` know nothing about each other or about SRT.
3. **Model names are parameters, never hardcoded at the call site.** Defaults live in *both* the class `__init__` signature and the argparse defaults; keep the two in sync when changing a default.
4. **Streaming-friendly transcription.** `model.transcribe()` returns a lazy generator; the segment loop is what actually drives inference. Do not materialize `segments` into a list without reason — that changes when work happens and breaks the progress bar semantics.
5. **Logging over printing.** Each module takes `logging.getLogger("<module>")`; formatting is configured once in `__init__.py`. Use `logger.info`, not `print`.
6. **Everything is local and offline-capable after first run.** No API keys, no network calls except Hugging Face model downloads.

## 6. Build System & Toolchain

Commands verified against `pyproject.toml` and `.github/workflows/ci.yml`:

```shell
uv sync                                  # create/refresh .venv from uv.lock
uv run movie-subtitles --help            # run the CLI locally
uv run --only-dev ruff format --check .  # CI check #1 — formatting
uv run --only-dev ruff check .           # CI check #2 — lint
uv run ruff format .                     # apply formatting
uv build                                 # hatchling wheel/sdist (not exercised in CI)
```

- Console script: `movie-subtitles` → `movie_subtitles.cli:main` (`[project.scripts]`).
- Ruff config: `line-length = 100`, lint rules `["E", "F", "I", "UP", "B", "SIM"]` (pycodestyle, pyflakes, isort, pyupgrade, bugbear, simplify). Import sorting is enforced by `I` — let ruff order imports.
- CI (`.github/workflows/ci.yml`) runs on push to `main` and on every PR; a single `lint` job on `ubuntu-latest`, with `astral-sh/setup-uv@v5` caching. **Lint is the only gate — there is no test or build job.**
- **No test framework is configured.** There is no pytest dependency, no `tests/` directory, and no test command. If asked to add tests, that means introducing the framework, not discovering it.

## 7. Configuration

There is no config file, no env-var layer, and no dotenv. All configuration is CLI flags:

| Flag | Default | Effect |
|---|---|---|
| `--input` (required) | — | Path to the media file to transcribe |
| `--audio-lang` | `en` | Language passed to Whisper |
| `--srt-lang` | `da` | Target language; becomes the `<2xx>` MADLAD prefix token |
| `--whisper-model` | `large-v3` | faster-whisper model id |
| `--mt-model` | `jbochi/madlad400-3b-mt` | Hugging Face translation model id |

Implicit configuration comes from the environment of the underlying libraries: `HF_HOME`/`HF_HUB_CACHE` control model cache location, and `device_map="auto"` lets accelerate/torch pick CPU vs GPU. Output path is not configurable — it is always `input.with_suffix(".srt")`.

## 8. Command Structure

Single-command CLI; there are no subcommands despite the `usage="translation-cli <command> [<args>]"` string in the parser (that usage line is stale and misleading — it does not reflect the real interface).

- `main()` parses args and forwards them **positionally** to `create_subtitles()`. The argument order there is `fpath, audio_lang, srt_lang, whisper_model_name, mt_model_name` — reordering the signature silently breaks the CLI. Prefer adding new params as keyword arguments at the end.
- `create_subtitles()` is the public, importable API for programmatic use; it accepts `str | Path`.
- Exit codes: none are set explicitly. Success exits 0; argparse errors exit 2; any pipeline failure propagates as an uncaught traceback. There is no error handling layer.

## 9. Subtitle Output Format

SRT blocks are built by hand in `cli.py` (no subtitle library):

- Timestamps: `str(0) + str(timedelta(seconds=int(segment.start))) + ",000"` — yields `0H:MM:SS,000`. Sub-second precision is discarded (`int()`) and milliseconds are always `,000`.
- Segment numbering uses `segment.id + 1` from Whisper, *not* the loop index. Because empty translations are skipped with `continue`, the emitted ids can have gaps.
- A single leading space in the translated text is stripped (`text[1:] if text[0] == ' '`).
- Blocks are accumulated in a list and written once at the end with `srt_file.write_text(..., encoding="utf-8")` — nothing is written incrementally, so an interrupted run produces no file.

## 10. Summary & Key Architectural Decisions

- The whole pipeline is `cli.create_subtitles()`; `Transcribe` and `Translate` are stateless-per-call model wrappers with no knowledge of each other.
- New model stages follow the callable-class pattern: load in `__init__`, named method, `__call__` delegating.
- Defaults are duplicated between class signatures and argparse — change both together.
- `create_subtitles()` is called with positional args; do not reorder its parameters.
- Ruff (format + `E,F,I,UP,B,SIM`, line length 100) is the only enforced standard, and the only CI job. Code must pass both `ruff format --check .` and `ruff check .`.
- uv owns the environment; `uv.lock` is committed and must be updated (`uv sync`/`uv lock`) alongside any dependency change in `pyproject.toml`.
- There are no tests and no error handling — do not assume either exists when reasoning about a change.
- Burned-in subtitles are advertised in the README but not implemented; `.srt` is the only output.
- The parser's `usage=` string is stale; the CLI has no subcommands.
