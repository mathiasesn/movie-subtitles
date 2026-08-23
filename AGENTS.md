# AGENTS.md

Guidance for AI coding agents working in this repository.

`ARCHI.md` holds the detailed architecture reference. Read it before non-trivial changes; keep it in sync when the architecture shifts.

## What this project is

`movie-subtitles` turns a video/audio file into a translated `.srt` file, and optionally into a dubbed video with a synthesised translated audio track. Engine values name vendors, not pipelines, and each stage is independently overridable: `--asr-engine {local,elevenlabs,openai}`, `--translation-engine {local,anthropic,openai}`, `--tts-engine {elevenlabs,openai}`. `--asr-engine openai` has a confirmed vendor limitation: `whisper-1`'s segment timestamps degrade to uniform 1.000s spans on music-heavy or dialogue-sparse audio (observed on a movie trailer; see the `OpenAITranscribe` docstring and `specs/fix-smoke-run-findings.md`), which corrupts both `.srt` cue timings and dub slots. Prefer `--asr-engine elevenlabs` on such material. `--engine {local,elevenlabs,openai}` is a shorthand that sets all three when the per-stage flags are not given — `local` is faster-whisper ASR → local MADLAD400 T5 translation, fully offline; `openai` is OpenAI ASR → OpenAI chat-completions translation → OpenAI TTS; `--engine elevenlabs` alone raises `ValueError`, since ElevenLabs has no standalone text-translation endpoint — `--translation-engine {anthropic,openai,local}` must be passed explicitly alongside it. `--dub` synthesises the translated segments with the resolved TTS provider, lays them out on a scene-anchored timeline, and muxes them over the source video with ffmpeg (rejected if the resolved TTS engine is unusable, e.g. `local` — a hard error; a translation stage resolving to `local` under `--dub` is milder, only warning and proceeding). Synthesis runs over a bounded `ThreadPoolExecutor` (`--dub-workers`, default 1 — i.e. serial by default; concurrency is opt-in because vendor concurrency caps are per-subscription and low, and a cap of 3 429s immediately at 8 workers with the lockstep retry unable to recover; values below 1 are rejected by argparse itself via a custom `_positive_int` type, not swallowed downstream). A group whose accumulated drift (synthesised speech total against summed source *speech* time, not wall-clock cue span) exceeds tolerance goes through a bounded corrective re-synthesis loop, `--dub-correction-passes` (default 3, also a `_positive_int`), rather than a single fixed retry — each pass resynthesises only the groups still outside tolerance, at a per-group clamped rate, and a group drops out once it's back in tolerance or a pass fails to improve it; `--dub-correction-passes 1` bounds correction to a single pass (not an exact reproduction of the old fixed single-retry behaviour, since the drift metric and pass-acceptance rule both changed). `--managed` bypasses this repo's pipeline entirely and calls the ElevenLabs Dubbing job API (create/poll/download) instead (`--managed --separate-background` raises `ValueError` up front, since ElevenLabs already handles background preservation itself). `mux.py:mux_dub` keeps the original audio underneath the synthesised dub, attenuated ("ducked") to `--duck-level` whenever the dub is speaking; the flag is validated at parse time to the inclusive `[0.0, 1.0]` range (`0.0` silences the original under the dub, `1.0` disables ducking altogether). There is no single default: `--duck-level`'s argparse default is `None`, resolved by `mux_dub` itself to `0.25` normally or `0.6` when `--separate-background` succeeded, and an explicit value always wins over either. `--separate-background` (off by default) runs `movie_subtitles/separate.py`'s Demucs (`htdemucs`) model over the source audio before muxing and mixes the dub over the resulting accompaniment (no-vocals) stem instead of the original track, so the original dialogue is removed rather than merely ducked; it is opt-in because Demucs is CPU-minutes-to-tens-of-minutes on a feature-length film and its first run downloads model weights. A runtime separation failure (unreachable weights, a bad input, a decode/inference error) is caught in `cli.py:_dub_and_mux`, logged as one WARNING, and falls back to the ordinary duck-and-mix path over the original audio rather than failing the run; an `ImportError` from a broken install propagates instead.

`--asr-engine elevenlabs` diarizes by default (Scribe's `diarize=True`), and `_group_words` now also flushes a segment on speaker change, so `Segment.speaker` is populated for every cue — this is a visible behaviour change for **every** such run, dub or not: a multi-speaker scene now produces more, shorter `.srt` cues than before, since the boundary is speaker-change, not just punctuation/duration/length. Under `--dub`, `voice_match: str = "auto"` (`--voice-match {off,clone,preset,auto}`) resolves each diarized speaker to a TTS voice id via the `movie_subtitles/voices.py:resolved_voices` context manager, whose yielded mapping is threaded into `dub.synthesise_track(..., voices=...)` as a plain `speaker -> voice_id` dict. `off` keeps today's single-voice behaviour; `clone` instant-clones each speaker via ElevenLabs IVC (ElevenLabs-only — `--tts-engine openai` always gets a preset, in every mode); `preset` classifies each speaker (gender/age band, via a `librosa`+`praat-parselmouth` heuristic) and looks up a curated stock voice, never cloning; `auto` (default) clones when the TTS engine supports it and the speaker has enough clean audio, else falls back to preset. `--clone-min-seconds` (30) / `--clone-target-seconds` (60) bound how much non-overlapping clean audio per speaker is gathered and required; a speaker with too little clean audio, or whose clone call fails, degrades to a preset rather than failing the run. `--voice-preset-table <json>` replaces the built-in preset table wholesale for the engines it names, validated strictly at load time. Cloned voices are deleted after the run (in a `finally`, so this happens even when the dub raises) unless `--keep-cloned-voices` is passed.

Note: the original README advertised burned-in per-frame subtitles. That was never implemented and has been removed from the README — `.srt`, and with `--dub`/`--managed` a dubbed video, are the only output paths.

## Layout

```
movie_subtitles/
  __init__.py      # side-effecting: configures stdout INFO logging on import
  cli.py           # argparse entry point + create_subtitles() orchestration, engine
                    # selection, translation-budget derivation, top-level error handling
  srt.py           # segment -> SRT block formatting, provider-agnostic
  dub.py           # scene-anchored TTS synthesis: groups segments by inter-segment
                    # silence gap (word-level gaps when available, else cue-boundary
                    # gaps), lays each group out pinned to each segment's own ASR start
                    # (floating forward only on overrun), and measures drift against
                    # summed source *speech* time, not wall-clock cue span. An initial
                    # pass at 1.0x is followed by a bounded correction loop
                    # (--dub-correction-passes, default 3): each pass is one flat
                    # cross-group batch over a bounded ThreadPoolExecutor
                    # (--dub-workers), resubmitting only groups still outside
                    # tolerance; a group drops out once in tolerance or a pass fails to
                    # improve it. Every pass's batch is resolved by _resolve(), which
                    # fails fast -- cancels the still-queued tail on the first
                    # exception and re-raises it -- before assembling the silent
                    # timeline via ffmpeg; boundary measurement itself lives in
                    # providers/ffmpeg_align.py, providers/elevenlabs.py and
                    # providers/fallback.py, not here. Also returns the coalesced,
                    # padded (start, end) speech spans (_speech_spans) it placed clips
                    # into, for mux.py to duck against
  mux.py           # mux_dub(): overlays the finished audio track over the source video
                    # with ffmpeg, keeping the original audio (or, with
                    # --separate-background, a separated accompaniment stem via
                    # background_path) underneath -- mixed in (amix=normalize=0) and
                    # ducked to duck_level inside the caller's speech_spans, with a
                    # per-input gain guarding against clipping. duck_level defaults to
                    # None, a sentinel mux_dub resolves itself to _DUCK_LEVEL (0.25)
                    # normally or _SEPARATED_DUCK_LEVEL (0.6) when background_path is
                    # given; an explicit --duck-level always wins. A
                    # source with no audio stream (ffmpeg.probe_audio_format returning
                    # None) falls back to the old dub-only mapping
  separate.py      # Separate: lazy-imports demucs (and torch) only inside __init__/
                    # separate(), builds a demucs.api.Separator (htdemucs, CUDA when
                    # available) once, then separate() hands the file to it -- Separator
                    # decodes and resamples itself, so no intermediate wav is written --
                    # and sums every stem except "vocals" in place (keyed off the
                    # returned stem names) into an accompaniment wav, saved via
                    # demucs.audio.save_audio, for mux.py to mix the dub over. Demucs works internally at 44.1 kHz stereo, so a 5.1/48 kHz
                    # source is downmixed in the stem -- mux_dub still re-targets the
                    # final output to the source's own format. Raises rather than
                    # catching; cli.py:_dub_and_mux is what degrades on failure, and
                    # owns the temp dir the stem is written into
  voices.py        # --voice-match speaker -> TTS voice id resolution: a single sweep
                    # (_clean_segments_by_speaker) computes every speaker's clean
                    # (non-overlapping) segment set up front, extract_speaker_sample()
                    # cuts each speaker's clean clips in one ffmpeg atrim/concat
                    # filtergraph pass, classify_voice() runs the sample (decoded once)
                    # through librosa (F0) + parselmouth (formants) into a gender/age
                    # profile (lazy imports, degrades to "unknown" rather than raising),
                    # and resolved_voices() -- a context manager -- ties cloning
                    # (ElevenLabs IVC only) and preset lookup together per --voice-match
                    # mode, deleting every voice it cloned on exit (success, a raise
                    # during resolution, or a raise from the caller's `with` body) unless
                    # told to keep them
  ffmpeg.py         # audio_codec_for() container->codec table, probe_audio_format()
                    # (the ffprobe query mux.py uses both to decide whether there's an
                    # original track to keep and to read its layout/rate) + run(): the
                    # one place that invokes ffmpeg and turns a non-zero exit into a
                    # RuntimeError quoting its
                    # stderr, shared by dub.py, mux.py and voices.py
  dubbing.py        # ManagedDub: the --managed path, independent of the rest
  providers/
    base.py          # Word (frozen: start/end/text) + Segment dataclass (start/end/
                      # text plus optional `words: list[Word] | None` and `speaker:
                      # str | None`, both populated only by ScribeTranscribe) +
                      # ASRProvider / TranslationProvider / TTSProvider /
                      # AlignmentProvider Protocols -- pure type surface, no behaviour
    fallback.py       # FallbackAlign: composes AlignmentProviders into a
                      # thread-safe degrade chain; no vendor SDK imports
    local.py          # Transcribe (faster-whisper) + Translate (MADLAD400 T5)
    elevenlabs.py      # ScribeTranscribe (ASR) + Speak (TTS) + Align (Forced Alignment,
                       # the first tier of the alignment chain); shared build_client()
    ffmpeg_align.py     # SilenceAlign (ffmpeg silencedetect) + DurationAlign (ffprobe
                        # duration, no-trim last resort); no vendor SDK imports
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

- **Model/API wrappers are callable classes.** Load the model/client in `__init__`, expose a named method (`transcribe` / `translate` / `speak` / `dub` / `align`), and have `__call__` delegate to it. New backends follow the same shape.
- **Orchestration lives only in `cli.py`.** Providers know nothing about each other, about SRT, or about dubbing.
- **Providers implement Protocols (`providers/base.py`), not base classes.** `_build_asr_provider`/`_build_translation_provider`/`_build_tts_provider` in `cli.py` are the only places that pick a concrete class per engine; `_build_providers()` wraps the first two for the stages every invocation needs.
- **Speech-boundary measurement is a degrade chain, not a single call.** `cli.py:_build_aligner()` always returns an `AlignmentProvider` (never `None`): it assembles `providers/fallback.py:FallbackAlign([Align()?, SilenceAlign(), DurationAlign()])`, omitting `Align()` when `ELEVENLABS_API_KEY` is unset or construction fails. `FallbackAlign` tries each tier in order, latches a failing one off for the rest of the run, and raises `RuntimeError` only once every tier is exhausted. `dub.py`'s `synthesise_track(..., *, aligner: AlignmentProvider, max_workers: int = 1)` requires `aligner` — it no longer measures boundaries itself.
- **Dub synthesis batches fail fast, they don't drain.** `dub.py` no longer has a fixed two-phase structure: an initial pass at 1.0x is followed by a bounded correction loop (`--dub-correction-passes`, default 3), each pass submitting one flat cross-group batch -- only the segments of groups still outside tolerance -- to the shared `ThreadPoolExecutor` and getting its clips back from `_resolve()`, which is the only path to them -- collecting results without first waiting is not expressible. `_resolve()` flattens every batch handed to it before waiting, so on the first exception it cancels the still-queued tail across *all* of those batches, not just the one containing the failure -- this holds per pass/phase, not just for the old fixed phase A/phase B. It then classifies every future in one pass: the four counts (succeeded/failed/cancelled/in-flight) are mutually exclusive and sum to the batch size, and nothing already finished can be reported as in-flight, but "in-flight" is not a final state -- those tasks may finish before the WARNING is even emitted. It re-raises the lowest-submission-index failure among the failures observed during that pass, which is deterministic given that set but not a total order over every failure two truly concurrent tasks could raise.
- **Retry is a property of the batch, not of any vendor.** `dub.py` wraps only the TTS call (3 attempts, 1s/2s backoff) in retry, never the aligner call, and this policy lives in `dub.py` rather than inside a provider precisely because it must hold identically for every TTS backend; only `dub.py` knows that one unretried failure now aborts the whole queued batch.
- **Provider modules are imported lazily, inside `cli.py`'s builders.** Each provider module imports its own vendor SDK at module top level, but `cli.py`'s `_build_asr_provider`/`_build_translation_provider`/`_build_tts_provider` import the provider module itself only inside the branch that needs it — this is what lets `--engine local` run without importing the ElevenLabs/Anthropic/OpenAI SDKs, and vice versa.
- **Defaults are duplicated** between argparse and every public callable's signature that shares it, not just class `__init__`s — e.g. the `--dub-workers` default of 1 legitimately exists in argparse, in `create_subtitles()`, and in `dub._MAX_WORKERS`; `--dub-correction-passes` follows the same pattern, its default of 3 duplicated across argparse, `create_subtitles()`, and `dub._MAX_CORRECTION_PASSES`. This is a deliberate choice, not a defect to collapse; change every copy together.
- **Do not reorder `create_subtitles()` parameters** — `main()` passes the first five positionally. New params are keyword-only, appended at the end.
- **Do not materialize `segments` into a list.** ASR providers return a lazy generator/iterable; the loop is what drives inference and the progress bar.
- **Log, don't print.** Each module uses `logging.getLogger("<module>")`; formatting is configured once in `__init__.py`.
- **Ruff owns style**: line length 100, rules `E,F,I,UP,B,SIM`. Let ruff sort imports rather than hand-ordering them.
- Type hints use `X | Y` union syntax (`requires-python >= 3.10`).

## Dependencies

`uv` owns the environment and `uv.lock` is committed. Any change to `pyproject.toml` dependencies must be accompanied by an updated lockfile (`uv lock` / `uv sync`).

`accelerate` is a runtime dependency: `providers/local.py` loads the MADLAD400 model with `device_map="auto"`, which `transformers` only accepts when `accelerate` is installed — without it, `--engine local` cannot translate.

`librosa` and `praat-parselmouth` are runtime dependencies of `voices.py:classify_voice`: `librosa.pyin` gives a noise-robust median F0 (the primary gender cue) and `praat-parselmouth`'s formant tracking (F1/F2) refines gender and splits age bands. Both are imported lazily inside `classify_voice`, not at module top level, so `--voice-match off` (and any code path that never classifies) never pays their import cost.

`demucs` is a runtime dependency of `separate.py`'s `--separate-background` path, providing the `htdemucs` separation model. `separate.py` goes through `demucs.api.Separator`, which owns the decode (resampling to the model's own rate/channels itself) and, with `demucs.audio.save_audio`, the write — so no second audio-I/O library is declared alongside it, and no intermediate extracted wav is written. `demucs.api` and `demucs.audio` are imported lazily, inside `Separate.__init__` and `Separate.separate` respectively, so a run without `--separate-background` never pays Demucs's (or torch's) import cost. `Separate.__init__` selects CUDA when `torch.cuda.is_available()`, matching `providers/local.py`'s `device_map="auto"` posture; htdemucs on CPU is roughly an order of magnitude slower.

## Things that don't exist yet

No tests, no test framework, and no environment variable beyond `ELEVENLABS_API_KEY`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (see `.env.example` at the repo root, which holds placeholders for all three). Those three are loaded from `.env` by `cli._load_env()`, called from `main()` only — `find_dotenv(usecwd=True)` so discovery is anchored to the user's cwd rather than to the installed package, and already-exported variables are never overridden. Importing any module still reads no `.env`; library callers set the environment themselves. There is now a minimal error-handling layer at the `main()` boundary (`RuntimeError | ValueError | FileNotFoundError | TimeoutError | subprocess.CalledProcessError` caught and turned into a one-line message + `SystemExit(1)`); anything else still propagates as a traceback. If asked to add tests, that means introducing pytest and a `tests/` directory from scratch — don't go looking for existing ones.

There **is** now a config file: `--voice-preset-table` takes a path to a JSON file (schema: top-level keys are TTS engine names, each mapping `gender:age_band`/`default` profile keys to a voice id string) that replaces `voices.py`'s built-in preset table wholesale for the engines it names. `voices.py:load_preset_table` validates it strictly at load time — malformed JSON, an unrecognised engine or profile key, a non-string voice id, or a missing `default` entry all raise `ValueError` before any dub work starts, rather than surfacing as a mid-dub `KeyError`.
