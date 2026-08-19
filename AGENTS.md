# AGENTS.md

Guidance for AI coding agents working in this repository.

`ARCHI.md` holds the detailed architecture reference. Read it before non-trivial changes; keep it in sync when the architecture shifts.

## What this project is

`movie-subtitles` turns a video/audio file into a translated `.srt` file, and optionally into a dubbed video with a synthesised translated audio track. Engine values name vendors, not pipelines, and each stage is independently overridable: `--asr-engine {local,elevenlabs,openai}`, `--translation-engine {local,anthropic,openai}`, `--tts-engine {elevenlabs,openai}`. `--engine {local,elevenlabs,openai}` is a shorthand that sets all three when the per-stage flags are not given — `local` is faster-whisper ASR → local MADLAD400 T5 translation, fully offline; `openai` is OpenAI ASR → OpenAI chat-completions translation → OpenAI TTS; `--engine elevenlabs` alone raises `ValueError`, since ElevenLabs has no standalone text-translation endpoint — `--translation-engine {anthropic,openai,local}` must be passed explicitly alongside it. `--dub` synthesises the translated segments with the resolved TTS provider, fits them to their timing slots, and muxes them over the source video with ffmpeg (rejected if the resolved TTS engine is unusable, e.g. `local` — a hard error; a translation stage resolving to `local` under `--dub` is milder, only warning and proceeding, degraded to TTS-rate-only fitting). `--managed` bypasses this repo's pipeline entirely and calls the ElevenLabs Dubbing job API (create/poll/download) instead.

Note: the original README advertised burned-in per-frame subtitles. That was never implemented and has been removed from the README — `.srt`, and with `--dub`/`--managed` a dubbed video, are the only output paths.

## Layout

```
movie_subtitles/
  __init__.py      # side-effecting: configures stdout INFO logging on import
  cli.py           # argparse entry point + create_subtitles() orchestration, engine
                    # selection, translation-budget derivation, top-level error handling
  srt.py           # segment -> SRT block formatting, provider-agnostic
  dub.py           # per-segment TTS synthesis + timing-drift rate-fitting + silent-
                    # timeline assembly via ffmpeg
  mux.py           # mux_dub(): overlays the finished audio track over the source video
                    # with ffmpeg, replacing (not mixing with) the original audio track
  ffmpeg.py         # audio_codec_for() container->codec table + run(): the one place that
                    # invokes ffmpeg and turns a non-zero exit into a RuntimeError quoting
                    # its stderr, shared by dub.py and mux.py
  dubbing.py        # ManagedDub: the --managed path, independent of the rest
  providers/
    base.py          # Segment dataclass + ASRProvider / TranslationProvider /
                      # TTSProvider Protocols
    local.py          # Transcribe (faster-whisper) + Translate (MADLAD400 T5)
    elevenlabs.py      # ScribeTranscribe (ASR) + Speak (TTS); shared build_client()
    llm.py             # LLMTranslate: Claude-backed TranslationProvider
    openai_.py          # OpenAITranscribe (ASR) + OpenAITranslate + OpenAISpeak (TTS);
                         # shared build_client(); trailing underscore avoids shadowing
                         # the `openai` SDK
    prompt.py            # SYSTEM_PROMPT + build_prompt(): translation prompt text
                          # shared by LLMTranslate and OpenAITranslate
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

- **Model/API wrappers are callable classes.** Load the model/client in `__init__`, expose a named method (`transcribe` / `translate` / `speak` / `dub`), and have `__call__` delegate to it. New backends follow the same shape.
- **Orchestration lives only in `cli.py`.** Providers know nothing about each other, about SRT, or about dubbing.
- **Providers implement Protocols (`providers/base.py`), not base classes.** `_build_asr_provider`/`_build_translation_provider`/`_build_tts_provider` in `cli.py` are the only places that pick a concrete class per engine; `_build_providers()` wraps the first two for the stages every invocation needs.
- **Provider modules are imported lazily, inside `cli.py`'s builders.** Each provider module imports its own vendor SDK at module top level, but `cli.py`'s `_build_asr_provider`/`_build_translation_provider`/`_build_tts_provider` import the provider module itself only inside the branch that needs it — this is what lets `--engine local` run without importing the ElevenLabs/Anthropic/OpenAI SDKs, and vice versa.
- **Defaults are duplicated** between class `__init__` signatures and argparse. Change both together.
- **Do not reorder `create_subtitles()` parameters** — `main()` passes the first five positionally. New params are keyword-only, appended at the end.
- **Do not materialize `segments` into a list.** ASR providers return a lazy generator/iterable; the loop is what drives inference and the progress bar.
- **Log, don't print.** Each module uses `logging.getLogger("<module>")`; formatting is configured once in `__init__.py`.
- **Ruff owns style**: line length 100, rules `E,F,I,UP,B,SIM`. Let ruff sort imports rather than hand-ordering them.
- Type hints use `X | Y` union syntax (`requires-python >= 3.10`).

## Dependencies

`uv` owns the environment and `uv.lock` is committed. Any change to `pyproject.toml` dependencies must be accompanied by an updated lockfile (`uv lock` / `uv sync`).

`accelerate` is a runtime dependency: `providers/local.py` loads the MADLAD400 model with `device_map="auto"`, which `transformers` only accepts when `accelerate` is installed — without it, `--engine local` cannot translate.

## Things that don't exist yet

No tests, no test framework, no config file, and no environment variable beyond `ELEVENLABS_API_KEY`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (see `.env.example` at the repo root, which holds placeholders for all three). Those three are loaded from `.env` by `cli._load_env()`, called from `main()` only — `find_dotenv(usecwd=True)` so discovery is anchored to the user's cwd rather than to the installed package, and already-exported variables are never overridden. Importing any module still reads no `.env`; library callers set the environment themselves. There is now a minimal error-handling layer at the `main()` boundary (`RuntimeError | ValueError | FileNotFoundError | TimeoutError | subprocess.CalledProcessError` caught and turned into a one-line message + `SystemExit(1)`); anything else still propagates as a traceback. If asked to add tests, that means introducing pytest and a `tests/` directory from scratch — don't go looking for existing ones.
