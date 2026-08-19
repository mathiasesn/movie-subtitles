# AGENTS.md

Guidance for AI coding agents working in this repository.

`ARCHI.md` holds the detailed architecture reference. Read it before non-trivial changes; keep it in sync when the architecture shifts.

## What this project is

`movie-subtitles` is a single-command CLI that turns a video/audio file into a translated `.srt` file. The pipeline is local and synchronous: faster-whisper transcribes → a MADLAD400 T5 model translates each segment → `create_subtitles()` writes `<input>.srt` next to the input.

Note: the README advertises burned-in per-frame subtitles. That is **not implemented** — `.srt` is the only output path.

## Layout

```
movie_subtitles/
  __init__.py      # side-effecting: configures stdout INFO logging on import
  cli.py           # argparse entry point + create_subtitles() orchestration & SRT formatting
  transcribe.py    # Transcribe — wraps faster_whisper.WhisperModel
  translate.py     # Translate — wraps T5ForConditionalGeneration + T5Tokenizer
```

Flat package, no `src/` layout, no `tests/`.

## Commands

```shell
uv sync                                  # create/refresh .venv from uv.lock
uv run movie-subtitles --help            # run the CLI
uv run ruff format .                     # apply formatting
uv run --only-dev ruff format --check .  # CI check #1
uv run --only-dev ruff check .           # CI check #2
```

CI (`.github/workflows/ci.yml`) runs those two ruff checks and nothing else — lint is the only gate. Both must pass before you call a change done.

## Conventions

- **Model wrappers are callable classes.** Load the model in `__init__`, expose a named method (`transcribe` / `translate`), and have `__call__` delegate to it. New model stages follow the same shape.
- **Orchestration lives only in `cli.py`.** `transcribe.py` and `translate.py` know nothing about each other or about SRT.
- **Defaults are duplicated** between class `__init__` signatures and argparse. Change both together.
- **Do not reorder `create_subtitles()` parameters** — `main()` passes them positionally. Add new params as keyword arguments at the end.
- **Do not materialize `segments` into a list.** `model.transcribe()` returns a lazy generator; the loop is what drives inference and the progress bar.
- **Log, don't print.** Each module uses `logging.getLogger("<module>")`; formatting is configured once in `__init__.py`.
- **Ruff owns style**: line length 100, rules `E,F,I,UP,B,SIM`. Let ruff sort imports rather than hand-ordering them.
- Type hints use `X | Y` union syntax (`requires-python >= 3.10`).

## Dependencies

`uv` owns the environment and `uv.lock` is committed. Any change to `pyproject.toml` dependencies must be accompanied by an updated lockfile (`uv lock` / `uv sync`).

## Things that don't exist yet

No tests, no test framework, no error-handling layer, no config file or env-var layer. If asked to add tests, that means introducing pytest and a `tests/` directory from scratch — don't go looking for existing ones.
