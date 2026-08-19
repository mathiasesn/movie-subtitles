# movie-subtitles Architecture Documentation

> Generated: 2026-08-19 · Branch: feat/elevenlabs (Stage 5 of `specs/elevenlabs-port.md`) · Version: 0.0.1
> Re-read this file at the start of any session touching this codebase. Update it when the architecture changes (new major dependency, restructured layer, changed convention).

## 1. How to Read This Document

Written for AI coding agents working on this repo. It states the stack, the exact commands, the module boundaries, and the conventions to follow — so you do not need to re-explore the tree. Update it if a module is added, a model backend is swapped, or tests are introduced. `specs/elevenlabs-port.md` is the design record for how this repo went from a single local pipeline to the two-vendor, provider-pluggable shape described here; read it for the "why", not just the "what".

## 2. Overview

`movie-subtitles` turns a video/audio file into a translated `.srt` file, and optionally into a dubbed video with a synthesised translated audio track. Engine values name vendors, not pipelines, and each of the three stages (ASR, translation, TTS) is independently selectable, plus a fourth path that bypasses this repo's pipeline entirely:

- **`--asr-engine {local,elevenlabs,openai}`**, **`--translation-engine {local,anthropic,openai}`**, **`--tts-engine {elevenlabs,openai}`** each independently pick a backend. `--engine {local,elevenlabs,openai}` is a shorthand that sets all three stages when the per-stage flags are not given.
- **`--engine local`** (default, unchanged from the original tool): faster-whisper ASR → local MADLAD400 T5 translation → `.srt`. Fully local and offline after model download. No TTS stage (`--dub` is rejected: MADLAD400 ignores the length budget).
- **`--engine openai`**: OpenAI ASR (`whisper-1`) → OpenAI chat-completions translation → `.srt`, optionally followed by `--dub` using OpenAI TTS (`gpt-4o-mini-tts`).
- **`--engine elevenlabs` alone raises `ValueError`** before any work starts: ElevenLabs has no standalone text-translation endpoint, so `--translation-engine {anthropic,openai,local}` must be passed explicitly alongside it — e.g. `--engine elevenlabs --translation-engine anthropic` reproduces the old pre-rename behaviour (Scribe ASR → Claude translation → optional `--dub` with ElevenLabs TTS).
- **`--managed`**: calls the ElevenLabs Dubbing job API (create → poll → download) directly. No local ASR/MT/TTS code runs; this is the "buy" side of a build-vs-buy comparison against the hand-rolled `elevenlabs --dub` path.

**Mixed-vendor pipelines are expected, not just tolerated.** Any combination of the three per-stage flags is valid as long as it satisfies the `--dub` guards (translation must not resolve to `local`; TTS must resolve to `elevenlabs` or `openai`) — e.g. Scribe ASR + OpenAI translation + OpenAI TTS. The `elevenlabs`-shorthand translation gap (no standalone text-translation endpoint; translation exists only bundled inside the Dubbing job) was a finding during implementation, not a starting assumption. See README.md "Which stages are ElevenLabs, and which are not" for the full explanation and reasoning.

**Known gap, resolved:** the original README advertised burned-in per-frame subtitles. That was never implemented and has been removed from the README as of this port. `.srt` and (with `--dub`) a muxed dubbed video are the only output paths. Do not assume frame-rendering code exists.

**Known gap, present:** the `--dub` timing-drift strategy only corrects overruns (audio longer than its slot gets a clamped speed-up); underruns are left to sit in silence. See section 5 and README.md "Timing drift".

## 3. Technology Stack

- **Python** — `requires-python = ">=3.10"`. Type hints use `X | Y` union syntax (3.10+).
- **faster-whisper** — local ASR (CTranslate2 Whisper). Default model `large-v3`. `local` engine only.
- **transformers** (+ **torch**, **sentencepiece** via T5Tokenizer) — local machine translation. Default model `jbochi/madlad400-3b-mt`, loaded with `device_map="auto"`. `local` engine only.
- **elevenlabs** (official Python SDK) — Scribe ASR (`speech_to_text.convert`), TTS (`text_to_speech.convert`), and the Dubbing job resource (`dubbing.create` / `.get` / `.audio.get`). `--asr-engine`/`--tts-engine elevenlabs` and `--managed`.
- **anthropic** (official Python SDK) — Claude-backed translation (`messages.create`). `--translation-engine anthropic` only.
- **openai** (official Python SDK) — `audio.transcriptions.create` (ASR, `whisper-1`), `chat.completions.create` (translation, `gpt-4o`), `audio.speech.create` (TTS, `gpt-4o-mini-tts`). `--asr-engine`/`--translation-engine`/`--tts-engine openai`. Module is `providers/openai_.py` (trailing underscore) so it doesn't shadow the `openai` SDK import inside it.
- **ffmpeg / ffprobe** (external binary, not a Python dependency) — required for `--dub` only: `ffprobe` measures synthesised clip duration for the fit loop, `ffmpeg` assembles per-segment clips onto a silent timeline and muxes the result over the source video. Not required for `.srt`-only runs (either engine) or for `--managed` (ElevenLabs renders server-side).
- **tqdm** — progress bar over segments during SRT writing.
- **argparse** (stdlib) — CLI argument parsing. No Click/Typer.
- **uv** — dependency resolution, lockfile (`uv.lock`), venv, and tool install. **hatchling** — build backend.
- **ruff** — the only linter/formatter and the only CI gate. Dev dependency group `dev`.

## 4. Project Structure

```
movie_subtitles/
  __init__.py             # side-effecting: configures root logging (stdout, INFO) on import
  cli.py                  # argparse entry point + create_subtitles() orchestration, engine
                           # selection, translation-budget derivation, top-level error handling
  srt.py                  # segment -> SRT block formatting (format_timestamp, format_block),
                           # lifted out of cli.py; provider-agnostic
  dub.py                  # per-segment TTS synthesis + timing-drift rate-fitting loop
                           # (fit_rate, synthesise_track) and silent-timeline assembly via ffmpeg
  mux.py                  # mux_dub(): overlays a finished audio track over the source video
                           # with ffmpeg, replacing its original audio track
  dubbing.py               # ManagedDub: the --managed path (ElevenLabs Dubbing job:
                           # create/poll/download), independent of the local pipeline
  providers/
    base.py                # Segment dataclass + ASRProvider / TranslationProvider /
                            # TTSProvider Protocols — the contract every backend implements
    local.py                # Transcribe (faster-whisper) + Translate (MADLAD400 T5);
                             # the original two classes, now behind the Protocol shape
    elevenlabs.py            # ScribeTranscribe (ASR) + Speak (TTS); shared _build_client()
                              # reads ELEVENLABS_API_KEY and raises RuntimeError if unset
    llm.py                   # LLMTranslate: Claude-backed TranslationProvider, accepts a
                              # budget_chars hint the prompt asks the model to respect;
                              # reads ANTHROPIC_API_KEY, raises RuntimeError if unset
    openai_.py                # OpenAITranscribe (ASR) + OpenAITranslate + OpenAISpeak
                               # (TTS); shared build_client() reads OPENAI_API_KEY and
                               # raises RuntimeError if unset; trailing underscore in the
                               # module name avoids shadowing the `openai` SDK import
.env.example                # placeholders for ELEVENLABS_API_KEY, ANTHROPIC_API_KEY,
                             # OPENAI_API_KEY
specs/elevenlabs-port.md   # design spec for this port; read for rationale, not just code
specs/openai-api-key-support.md  # design spec for the OpenAI backend + engine-value rename
.github/workflows/ci.yml   # lint-only CI (ruff format --check, ruff check)
pyproject.toml             # metadata, deps, console script, ruff config
uv.lock                    # committed lockfile
```

No `tests/`, no `src/` layout, no `docs/` beyond this file and the spec.

## 5. Core Architecture Principles

These describe what the code actually does — follow them rather than importing conventions from elsewhere.

1. **Model/API wrappers are callable classes.** Every backend (`Transcribe`, `Translate`, `ScribeTranscribe`, `Speak`, `LLMTranslate`, `ManagedDub`) loads its client/model in `__init__`, exposes a named method (`transcribe` / `translate` / `speak` / `dub`), and defines `__call__` delegating to it. A new backend should follow the same shape.
2. **Providers implement Protocols, not base classes.** `providers/base.py` defines `ASRProvider`, `TranslationProvider`, `TTSProvider` as `typing.Protocol`s (structural typing — no inheritance required). `cli.py` has four builders: `_build_asr_provider`, `_build_translation_provider`, and `_build_tts_provider` each pick a concrete class per the resolved per-stage engine value (`local`/`elevenlabs`/`openai`, vocabulary differing per stage — see section 7), and `_build_providers()` wraps the first two for the ASR+translation stages every invocation needs. These are the only places that pick a concrete class per engine; everything downstream (`srt.py`, `dub.py`) is written against the `Segment` shape and the Protocol signatures, not against any vendor concretely.
3. **`Segment` is the interchange type between ASR and everything downstream.** `providers/base.py:Segment(id, start, end, text)`. `ScribeTranscribe` normalises Scribe's word-level response into the same `Segment` shape faster-whisper yields (grouping words into segments on sentence-end punctuation, a max duration, or a max character count — see `providers/elevenlabs.py:_group_words`), so `srt.py` and `dub.py` never branch on which ASR backend produced a segment.
4. **Orchestration lives only in `cli.py`.** `create_subtitles()` is the single place that knows the pipeline order, chooses `--managed` vs. the local/elevenlabs transcribe→translate(→dub) pipeline, and computes the per-segment translation length budget (`_budget_chars`, using an assumed — not measured — 15 chars/second speaking rate). Individual providers know nothing about each other, about SRT, or about dubbing.
5. **Model/backend names and API keys are parameters or environment, never hardcoded at the call site.** Local model-name defaults live in *both* the class `__init__` signature and the argparse defaults; keep the two in sync when changing a default. API keys (`ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are read once per provider via a `build_client()`-style helper and never accepted as CLI flags or written to a file. `.env.example` at the repo root lists all three as placeholders for local `.env` use; the filled-in file is never committed.
6. **Errors that reach `main()` from a known cause are user-facing, not tracebacks.** `build_client()` helpers raise `RuntimeError` naming the missing env var; `ManagedDub._poll_until_done` raises `TimeoutError` on a stuck job and `RuntimeError` on a failed job; `create_subtitles()` raises `ValueError` for: the `--dub`/`--managed` conflict, `--dub` combined with a translation stage resolving to `local`, `--engine elevenlabs` with no explicit `--translation-engine` (ElevenLabs has no standalone text-translation endpoint), and `--dub` when the resolved TTS engine is not in `{elevenlabs, openai}`; `cli.py:_check_ffmpeg_tools()` raises `RuntimeError` up front if either `ffmpeg` or `ffprobe` is missing from `PATH`. `cli.py:main()` catches `RuntimeError | ValueError | FileNotFoundError | TimeoutError | subprocess.CalledProcessError` around the `create_subtitles()` call, logs a one-line message, and exits via `raise SystemExit(1)`. `subprocess.CalledProcessError` is in that tuple specifically so a failing `ffmpeg`/`ffprobe` invocation (bad input file, unsupported codec, etc.) surfaces as a one-line error rather than a raw traceback. Anything outside that set (a genuine bug) still propagates as a traceback — deliberately, so unexpected failures stay loud.
7. **Streaming-friendly transcription.** Both ASR backends return/yield a lazy iterable of `Segment`s; the segment loop in `create_subtitles()` is what actually drives inference (local) or paginates the API response (Scribe). Do not materialize `segments` into a list without reason — that changes when work happens and breaks the progress bar semantics.
8. **Timing-drift fitting is a bounded, single strategy, not a pluggable one.** `dub.py:fit_rate` clamps the TTS speaking-rate adjustment to `[0.9, 1.15]` — intentionally tighter than the ElevenLabs API's own documented `voice_settings.speed` range of `0.7-1.2`. `fit_rate` itself stays pure (rate computation only); "unfittable" is decided by `_synthesise_fitted` re-measuring the actual re-synthesised clip's duration with `ffprobe` after the clamped-rate retry and comparing it to the slot — not by whether the ideal rate merely fell outside the clamp, since the clamped retry can still land short or long of the slot after rounding/API behavior. A segment whose re-measured clip still does not fit is logged and counted (`synthesise_track`'s `unfittable_count`), not force-fit by further distorting speech. Only overruns are corrected; underruns are left to sit in silence. There is no `--drift-strategy` flag — see `specs/elevenlabs-port.md` for the alternatives considered and cut (silence trim/pad, Forced-Alignment re-anchoring).
9. **Logging over printing.** Each module uses `logging.getLogger("<module>")`; formatting is configured once in `__init__.py`. Use `logger.info` / `logger.warning` / `logger.error`, not `print`.
10. **The local engine remains fully offline-capable.** `--engine local` (the default) makes no network calls except Hugging Face model downloads, and needs no API key. Any stage set to `elevenlabs` or `openai`, `--dub`, and `--managed` require network access and the corresponding API key.

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
- Runtime dependencies `elevenlabs`, `anthropic`, and `openai` (all present in `pyproject.toml`) were added with `uv.lock` updated alongside each; any further dependency change must update the lockfile too (`uv lock` / `uv sync`).

## 7. Configuration

No config file, no dotenv loader (an `.env.example` documents the three variables for manual `export`/`.env` use, but nothing in the code reads a `.env` file). Configuration is CLI flags plus three environment variables:

| Flag | Default | Effect |
|---|---|---|
| `--input` (required) | — | Path to the media file to transcribe |
| `--audio-lang` | `en` | Source language passed to the ASR backend |
| `--srt-lang` | `da` | Target language for translation and the SRT output |
| `--whisper-model` | `large-v3` | faster-whisper model id (`local` ASR engine only) |
| `--mt-model` | `jbochi/madlad400-3b-mt` | Hugging Face translation model id (`local` translation engine only) |
| `--engine` | `local` | `local`, `elevenlabs`, or `openai` — shorthand that sets `--asr-engine`, `--translation-engine`, and `--tts-engine` when they are not given individually. `elevenlabs` alone raises `ValueError` unless `--translation-engine` is also passed |
| `--asr-engine` | `None` (falls back to `--engine`) | Per-stage override for the ASR backend only: `local`, `elevenlabs`, or `openai` |
| `--translation-engine` | `None` (falls back to `--engine`) | Per-stage override for the translation backend only: `local`, `anthropic`, or `openai`; `--dub` raises `ValueError` if this resolves to `local` (MADLAD400 ignores `budget_chars`) |
| `--tts-engine` | `None` (falls back to `--engine`) | Per-stage override for the TTS backend only: `elevenlabs` or `openai`; `--dub` raises `ValueError` if this resolves to anything else (e.g. `local`) |
| `--dub` | off | Synthesise translated segments with the resolved TTS provider and mux over the source video (needs ffmpeg + ffprobe + the resolved TTS engine's API key); requires the translation stage to resolve to `anthropic` or `openai`, and the TTS stage to resolve to `elevenlabs` or `openai`, but ASR can still be `local` |
| `--managed` | off | Bypass the local pipeline; run the ElevenLabs Dubbing job end to end (needs `ELEVENLABS_API_KEY`). Mutually exclusive with `--dub` — `create_subtitles()` raises `ValueError` if both are set |

| Environment variable | Required for |
|---|---|
| `ELEVENLABS_API_KEY` | any stage set to `elevenlabs` (ASR or TTS), `--managed` (Dubbing job) |
| `ANTHROPIC_API_KEY` | `--translation-engine anthropic` |
| `OPENAI_API_KEY` | any stage set to `openai` (ASR, translation, or TTS) |

Implicit configuration from the environment of underlying libraries: `HF_HOME`/`HF_HUB_CACHE` control the local model cache location; `device_map="auto"` lets accelerate/torch pick CPU vs GPU for the local translation model. Output paths are not configurable — `.srt` is always `input.with_suffix(".srt")`; `--dub` writes `<input>.dubbed<ext>` (same suffix as the source, muxed locally); `--managed` writes `<input>.dubbed.mp4` (downloaded rendered media, always `.mp4` regardless of the source's suffix — see section 10).

## 8. Command Structure

Single-command CLI; no subcommands, despite the stale `usage="translation-cli <command> [<args>]"` string in the parser (unchanged from before this port — still misleading, not fixed as it's out of this port's scope).

- `main()` parses args, then calls `create_subtitles()` **positionally** for the first five params (`fpath, audio_lang, srt_lang, whisper_model_name, mt_model_name`) and by keyword for everything added since (`engine`, `asr_engine`, `translation_engine`, `dub`, `managed`, `tts_engine`). Do not reorder the positional params — `main()` depends on that order. Add any new param as a keyword-only addition at the end of the signature (`tts_engine` was appended last, after `managed`).
- `create_subtitles()` is the public, importable API for programmatic use; it accepts `str | Path`. When `managed=True` it short-circuits into `_run_managed()` and returns without touching the transcribe/translate/dub path at all. `asr_engine`/`translation_engine`/`tts_engine` default to `None` and fall back to `engine` when unset (`cli.py`, resolved before any provider is built).
- **Exit codes:** 0 on success; 2 on argparse errors (stdlib default); **1** on `RuntimeError | ValueError | FileNotFoundError | TimeoutError | subprocess.CalledProcessError` raised from within `create_subtitles()` — caught in `main()`, logged as one line via `logger.error`, then `raise SystemExit(1)`. `cli.py:_check_ffmpeg_tools()` (called at the start of `_dub_and_mux`) proactively raises `RuntimeError` if `ffmpeg` or `ffprobe` is missing from `PATH`, and any `ffmpeg`/`ffprobe` invocation that exits non-zero raises `subprocess.CalledProcessError` (from `check=True`), which is caught the same way. Any other exception (a bug, not a known failure mode) still propagates as an uncaught traceback — this is intentional, not a gap.

## 9. Subtitle Output Format

SRT formatting lives in `srt.py` now (not inline in `cli.py`), same format as before:

- `format_timestamp`: `str(0) + str(timedelta(seconds=int(seconds))) + ",000"` — yields `0H:MM:SS,000`. Sub-second precision is discarded (`int()`) and milliseconds are always `,000`, for both engines.
- `format_block(index, start, end, text)`: builds one `"{index}\n{start} --> {end}\n{text}\n\n"` block. `index` is passed in by the caller as `segment.id + 1` (ASR-provided id, not the loop index) — because empty translations are skipped with `continue` in `cli.py`, emitted ids can have gaps, for both engines.
- A single leading space in the translated text is stripped.
- `cli.py` accumulates all blocks in a list and writes once at the end (`srt_file.write_text(..., encoding="utf-8")`) — nothing is written incrementally, so an interrupted run produces no `.srt` file. Unchanged from before this port.

## 10. Timing-Drift and Dub Pipeline (new in this port)

Only relevant when `--dub` is passed. `--dub` requires the translation stage to resolve to `anthropic` or `openai` (`create_subtitles()` raises `ValueError` if it resolves to `local`, since MADLAD400 ignores `budget_chars` and would silently skip length-budgeting) and the TTS stage to resolve to `elevenlabs` or `openai` (raises `ValueError` otherwise). ASR can still be `local` under `--dub`.

1. `cli.py:_budget_chars(start, end)` derives a character budget from segment duration using an assumed constant `_CHARS_PER_SECOND = 15.0` (explicitly commented in code as an assumption, not a measured value).
2. The translator is called with that budget (`TranslationProvider.__call__(text, output_lang, budget_chars=...)`). `LLMTranslate` (Claude) uses it in the prompt; `Translate` (MADLAD400) ignores it — no length-control lever exists for that model.
3. `dub.py:synthesise_track` iterates segments, calls `_synthesise_fitted` per segment: synthesise at speed 1.0, measure with `ffprobe`, and if the clip overruns its slot, compute a clamped rate via `fit_rate` and re-synthesise once at that rate. Segments that still overrun at the clamp bound are logged and counted.
4. `_assemble_timeline` builds an ffmpeg `filter_complex` that delays each clip to its segment's original start offset and mixes them onto one continuous track (`adelay` + `amix`).
5. `mux.py:mux_dub` overlays that finished audio track over the source video (`-map 0:v:0 -map 1:a:0 -c:v copy -c:a aac`), dropping the original audio track, writing `<input>.dubbed<ext>`. No `-shortest`: the video's full duration is authoritative, so a synthesised track that ends before the video's tail does not truncate the video (verified with synthetic media: a 10s video + 3s audio track muxes to a 10s output). `cli.py:_check_ffmpeg_tools()` raises `RuntimeError` up front (before any dub work starts) if either `ffmpeg` or `ffprobe` is missing from `PATH`.

`dubbing.py:ManagedDub` (the `--managed` path) is architecturally independent of all of the above — it does not touch `srt.py`, `dub.py`, `mux.py`, or any provider. It only talks to the ElevenLabs Dubbing job resource (`dubbing.create` / `.get` / `.audio.get`), polling on a fixed interval up to a timeout, and streams the rendered download to `<input>.dubbed.mp4`. The `.mp4` extension is fixed regardless of the source file's suffix: the installed SDK's own docstring for `dubbing.audio.get` (`.venv/lib/python3.12/site-packages/elevenlabs/dubbing/audio/client.py`, matching https://elevenlabs.io/docs/api-reference/dubbing/audio/get.md) states it "Returns dub as a streamed MP3 or MP4 file" — for a video source it renders an MP4 container, not necessarily whatever container the source used, so writing the bytes under the source's original suffix (e.g. `.mov`, `.mkv`) would mislabel the file.

## 11. Summary & Key Architectural Decisions

- Three vendors (`local`, `elevenlabs`, `openai`) selectable per stage via `--asr-engine`/`--translation-engine`/`--tts-engine`, `--engine` a shorthand for all three, plus one bypass path (`--managed`), unified only at the `create_subtitles()` orchestration level in `cli.py`.
- `providers/base.py` Protocols (`ASRProvider`, `TranslationProvider`, `TTSProvider`) plus the shared `Segment` dataclass are the contract that keeps `srt.py` and `dub.py` provider-agnostic.
- `--engine elevenlabs` alone spans two vendors and is rejected: ElevenLabs (ASR, TTS, Dubbing) has no standalone translation endpoint, so `--translation-engine {anthropic,openai,local}` must be passed explicitly. This is documented in the README as a finding, not hidden.
- Timing drift is handled by one bounded strategy (length-budgeted translation → clamped rate nudge → log-and-overrun on failure), deliberately not made pluggable, and deliberately only handling overruns.
- Known-cause errors (`RuntimeError`, `ValueError`, `FileNotFoundError`, `TimeoutError`) are caught once in `cli.py:main()` and turned into a one-line message + `SystemExit(1)`; anything else still traces back.
- Defaults are duplicated between class signatures and argparse — change both together.
- `create_subtitles()`'s original five params are positional; new params are keyword-only at the end — do not reorder.
- Ruff (format + `E,F,I,UP,B,SIM`, line length 100) is the only enforced standard, and the only CI job.
- uv owns the environment; `uv.lock` is committed and must be updated alongside any `pyproject.toml` dependency change.
- There are no tests. There is now an error-handling layer at the `main()` boundary, but nowhere else — provider code still lets unexpected exceptions propagate.
- Burned-in subtitles are **not** advertised in the README as of this port; `.srt` and (with `--dub`) a dubbed video are the only outputs.
- The parser's `usage=` string is stale (pre-existing, not touched by this port); the CLI has no subcommands.
- README result numbers (ASR benchmark, TTS quality, unfittable-segment count, build-vs-buy verdict) are explicit `TBD` placeholders as of this port — no API key or media clip was available in the environment this port was built in. Do not treat any number in the README as measured until the repo owner has actually run the commands it documents.
