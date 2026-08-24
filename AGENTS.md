# AGENTS.md

Guidance for AI coding agents working in this repository.

`ARCHI.md` holds the detailed architecture reference. Read it before non-trivial changes; keep it in sync when the architecture shifts.

## What this project is

`movie-subtitles` turns a video/audio file into a translated `.srt` file, and optionally into a dubbed video with a synthesised translated audio track. Engine values name vendors, not pipelines, and each stage is independently overridable: `--asr-engine {local,elevenlabs,openai}`, `--translation-engine {local,anthropic,openai}`, `--tts-engine {elevenlabs,openai}`. `--asr-engine openai` has a confirmed vendor limitation: `whisper-1`'s segment timestamps degrade to uniform 1.000s spans on music-heavy or dialogue-sparse audio (observed on a movie trailer; see the `OpenAITranscribe` docstring and `specs/fix-smoke-run-findings.md`), which corrupts both `.srt` cue timings and dub slots. Prefer `--asr-engine elevenlabs` on such material. The run now detects this signature itself: when most segments come back at a near-1.000s duration (over both an absolute floor and a proportion threshold, tuned in `cli.py`), it emits a WARNING naming the `--asr-engine elevenlabs` workaround rather than failing silently. `--engine {local,elevenlabs,openai}` is a shorthand that sets all three when the per-stage flags are not given — `local` is faster-whisper ASR → local MADLAD400 T5 translation, offline as long as `--voice-match off` is passed (any other `--voice-match` value runs a pyannote.audio diarisation pass that contacts Hugging Face on first use — see below); `openai` is OpenAI ASR → OpenAI chat-completions translation → OpenAI TTS; `--engine elevenlabs` alone raises `ValueError`, since ElevenLabs has no standalone text-translation endpoint — `--translation-engine {anthropic,openai,local}` must be passed explicitly alongside it. `--dub` synthesises the translated segments with the resolved TTS provider, lays them out on a scene-anchored timeline, and muxes them over the source video with ffmpeg (rejected if the resolved TTS engine is unusable, e.g. `local` — a hard error; a translation stage resolving to `local` under `--dub` is milder, only warning and proceeding). Synthesis runs over a bounded `ThreadPoolExecutor` (`--dub-workers`, default 1 — i.e. serial by default; concurrency is opt-in because vendor concurrency caps are per-subscription and low, and a cap of 3 429s immediately at 8 workers with the lockstep retry unable to recover; values below 1 are rejected by argparse itself via a custom `_positive_int` type, not swallowed downstream). A group whose accumulated drift (synthesised speech total against summed source *speech* time, not wall-clock cue span) exceeds tolerance goes through a bounded corrective re-synthesis loop, `--dub-correction-passes` (default 3, also a `_positive_int`), rather than a single fixed retry — each pass resynthesises only the groups still outside tolerance, at a per-group clamped rate, and a group drops out once it's back in tolerance or a pass fails to improve it; `--dub-correction-passes 1` bounds correction to a single pass (not an exact reproduction of the old fixed single-retry behaviour, since the drift metric and pass-acceptance rule both changed). `--managed` bypasses this repo's pipeline entirely and calls the ElevenLabs Dubbing job API (create/poll/download) instead (`--managed --separate-background` raises `ValueError` up front, since ElevenLabs already handles background preservation itself). `mux.py:mux_dub` keeps the original audio underneath the synthesised dub, attenuated ("ducked") to `--duck-level` whenever the dub is speaking; the flag is validated at parse time to the inclusive `[0.0, 1.0]` range (`0.0` silences the original under the dub, `1.0` disables ducking altogether). There is no single default: `--duck-level`'s argparse default is `None`, resolved by `mux_dub` itself to `0.25` normally or `0.6` when `--separate-background` succeeded, and an explicit value always wins over either. `--separate-background` (off by default) runs `movie_subtitles/separate.py`'s Demucs (`htdemucs`) model over the source audio before muxing and mixes the dub over the resulting accompaniment (no-vocals) stem instead of the original track, so the original dialogue is removed rather than merely ducked; it is opt-in because Demucs is CPU-minutes-to-tens-of-minutes on a feature-length film and its first run downloads model weights. A runtime separation failure (unreachable weights, a bad input, a decode/inference error) is caught in `cli.py:_dub_and_mux`, logged as one WARNING, and falls back to the ordinary duck-and-mix path over the original audio rather than failing the run; an `ImportError` from a broken install propagates instead.

`--asr-engine elevenlabs` diarizes by default (Scribe's `diarize=True`), and `_group_words` now also flushes a segment on speaker change, so `Segment.speaker` is populated for every cue — this is a visible behaviour change for **every** such run, dub or not: a multi-speaker scene now produces more, shorter `.srt` cues than before, since the boundary is speaker-change, not just punctuation/duration/length. `--asr-engine local` and `--asr-engine openai` get the same speaker labelling now too, but via a separate mechanism: neither faster-whisper nor `whisper-1` diarizes natively, so `cli.py` runs a standalone `movie_subtitles/providers/pyannote_.py:Diarize` pass (`pyannote/speaker-diarization-community-1`), up front, whenever `needs_diarization` is true (the resolved ASR engine doesn't diarize and `--voice-match != "off"`) — there is no dedicated flag for this, `--voice-match` is now the single, engine-agnostic "do I care about speakers?" control. `_build_diarizer()` itself takes no arguments and always returns a `DiarizationProvider`; the `needs_diarization` guard in `create_subtitles` is the only place that decides whether diarisation runs at all. Diarize doesn't get the source media directly: pyannote.audio's loader is torchaudio/torchcodec-based and may not decode a video container the way demucs's ffmpeg-backed `Separator` can, so `cli.py` first extracts a mono 16 kHz PCM wav via `movie_subtitles/ffmpeg.py:run()` into a `tempfile.TemporaryDirectory()` and diarizes that wav instead — the extraction call sits inside the same try/except as `diarize()`, so an extraction failure degrades exactly like any other diarisation failure (one WARNING, single-voice fallback), not a separate failure mode. `cli.py:_speaker_for_span`/`_split_segment_by_speaker`/`_diarized_segments` then merge those turns onto each lazily-yielded ASR segment by temporal overlap, splitting a segment into one sub-segment per contiguous speaker run when it carries word timestamps (both `providers/local.py` and `providers/openai_.py` now request and populate `Segment.words` unconditionally, not only when diarising — this also means `dub.py` now prefers word-level gaps over cue-boundary gaps when grouping on these two engines, which shifts dub timing slightly relative to before, even with `--voice-match off`). A word that overlaps no diarisation turn (a silence gap, or a diarizer miss) is smoothed onto a neighbouring speaker label (previous labelled word, else next) in `_split_segment_by_speaker` rather than forming its own one-word run — this stops one such gap from fragmenting a single cue into three. A segment where *no* word got a label at all is left entirely `speaker=None`, distinguishable from a smoothed gap: that's diarisation genuinely finding nothing there. So `local`/`openai` `.srt` output on multi-speaker audio now also splits cues on speaker change, same as ElevenLabs; `--voice-match off` still runs no diarisation, imports no pyannote/torch, and produces the same cue count and cue text on both engines — but on `--asr-engine local` cue *timings* can shift marginally even then, since `providers/local.py` now requests `word_timestamps=True` unconditionally, which switches on faster-whisper's DTW word-alignment pass and that pass can adjust a segment's own `start`/`end`. `pyannote/speaker-diarization-community-1` is a **gated** model: first use needs a Hugging Face token in `HF_TOKEN` and the model's conditions accepted at its model page. A missing token, unaccepted gate, unreachable Hub, or any other runtime diarisation failure is caught in `cli.py`, logged as one WARNING, and the run proceeds with every segment's speaker left `None` (today's single-voice dub); an `ImportError` from a broken install propagates instead, mirroring `separate.py`'s failure split. `--asr-engine openai` additionally logs a WARNING when it diarizes, since `whisper-1`'s segment-timestamp defect (see above) makes overlap-based speaker labels confidently wrong on affected audio, not merely absent — `--asr-engine elevenlabs` is recommended there too. Under `--dub`, `voice_match: str = "auto"` (`--voice-match {off,clone,preset,auto}`) resolves each diarized speaker to a TTS voice id via the `movie_subtitles/voices.py:resolved_voices` context manager, whose yielded mapping is threaded into `dub.synthesise_track(..., voices=...)` as a plain `speaker -> voice_id` dict. `off` keeps today's single-voice behaviour; `clone` instant-clones each speaker via ElevenLabs IVC (ElevenLabs-only — `--tts-engine openai` always gets a preset, in every mode); `preset` classifies each speaker (gender/age band, via a `librosa`+`praat-parselmouth` heuristic) and looks up a curated stock voice, never cloning; `auto` (default) clones when the TTS engine supports it and the speaker has enough clean audio, else falls back to preset. `--clone-min-seconds` (30) / `--clone-target-seconds` (60) bound how much non-overlapping clean audio per speaker is gathered and required; a speaker with too little clean audio, or whose clone call fails, degrades to a preset rather than failing the run. `--voice-preset-table <json>` replaces the built-in preset table wholesale for the engines it names, validated strictly at load time. Cloned voices are deleted after the run (in a `finally`, so this happens even when the dub raises) unless `--keep-cloned-voices` is passed.

Note: the original README advertised burned-in per-frame subtitles. That was never implemented and has been removed from the README — `.srt`, and with `--dub`/`--managed` a dubbed video, are the only output paths.

## Layout

```
movie_subtitles/
  __init__.py      # side-effecting: configures stdout logging (root stays INFO) on
                    # import, then configure_logging() applies MOVIE_SUBTITLES_LOG_LEVEL
                    # to this package's own loggers only, named in _PACKAGE_LOGGERS --
                    # they are flat ("dub", not "movie_subtitles.dub"), so there is no
                    # ancestor to set instead. Root is left at INFO deliberately: raising
                    # it also turns on httpcore/openai/anthropic DEBUG, which buries our
                    # own lines. cli.main() calls configure_logging() again after
                    # _load_env() so the variable works from .env. Unset or unrecognised
                    # falls back to INFO
  cli.py           # argparse entry point + create_subtitles() orchestration, engine
                    # selection, translation-length-budget derivation (larger of source-text
                    # expansion ratio and duration-derived slot capacity -- see
                    # specs/chars-per-second-measurement.md), top-level error handling.
                    # Diarisation is orchestration only here now: the needs_diarization
                    # guard is the only place deciding whether diarisation runs at all,
                    # and _diarize_or_warn(fpath, asr_engine) is the single try/except
                    # that builds pyannote_.py:Diarize, extracts a mono 16 kHz wav via
                    # ffmpeg.py:extract_mono_wav() into a tempfile.TemporaryDirectory()
                    # (pyannote's torchaudio/torchcodec loader may not decode a video
                    # container directly), and calls diarize() on that wav, up front,
                    # before the still-lazy segment loop starts -- one ImportError
                    # re-raise, one Exception -> WARNING + [] degrade, mirroring
                    # _dub_and_mux's handling of separate.py failures. The resulting
                    # Turn list is threaded into diarize.py:label_segments(), which does
                    # the actual overlap-based merging/splitting/smoothing algorithm --
                    # moved out of cli.py entirely, see diarize.py below
  diarize.py       # label_segments(segments, turns): lazily merges pyannote.audio
                    # diarisation turns onto ASR segments by temporal overlap
                    # (word-level, splitting a segment into one sub-segment per
                    # contiguous speaker run, when segment.words is populated; cue-level
                    # majority overlap otherwise), for --asr-engine local/openai, which
                    # don't diarize natively (issue #27). A forward-only _TurnCursor
                    # instance encapsulates the turn-index state that used to be
                    # threaded manually through cli.py's split function. Stays lazy --
                    # never materializes `segments` -- so the caller's tqdm loop still
                    # drives ASR inference and the progress bar. A word overlapping no
                    # turn is smoothed onto a neighbouring labelled word rather than
                    # fragmenting its own run; a segment with no labelled word at all
                    # stays entirely speaker=None. Deliberately separate from
                    # providers/elevenlabs.py:ScribeTranscribe._group_words (Scribe's
                    # native per-word speaker_id split) -- do not unify the two
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
                    # stderr, shared by dub.py, mux.py, voices.py and cli.py.
                    # extract_mono_wav(source, out_path, rate=16000) is the one literal
                    # ffmpeg argv cli.py still needs directly -- a plain -i cut to mono
                    # PCM wav, used to feed the diarizer a container pyannote.audio's
                    # own loader may not decode
  dubbing.py        # ManagedDub: the --managed path, independent of the rest
  providers/
    base.py          # Word (frozen: start/end/text) + Segment dataclass (start/end/
                      # text plus optional `words: list[Word] | None` and `speaker:
                      # str | None`) + Turn (frozen: start/end/speaker) +
                      # ASRProvider / TranslationProvider / TTSProvider /
                      # AlignmentProvider / DiarizationProvider Protocols -- pure type
                      # surface, no behaviour. `words` is now populated by local.py and
                      # openai_.py too, not only ScribeTranscribe; `speaker` is
                      # populated by ScribeTranscribe natively and by cli.py's merge
                      # helpers for local/openai after a pyannote_.py diarisation pass
    fallback.py       # FallbackAlign: composes AlignmentProviders into a
                      # thread-safe degrade chain; no vendor SDK imports
    local.py          # Transcribe (faster-whisper) + Translate (MADLAD400 T5).
                      # Transcribe now passes word_timestamps=True unconditionally and
                      # maps each word into a Word, enabling speaker-change cue
                      # splitting and better dub.py gap grouping on this engine
    elevenlabs.py      # ScribeTranscribe (ASR) + Speak (TTS) + Align (Forced Alignment,
                       # the first tier of the alignment chain); shared build_client()
    ffmpeg_align.py     # SilenceAlign (ffmpeg silencedetect) + DurationAlign (ffprobe
                        # duration, no-trim last resort); no vendor SDK imports
    llm.py             # LLMTranslate: Claude-backed TranslationProvider
    openai_.py          # OpenAITranscribe (ASR) + OpenAITranslate + OpenAISpeak (TTS);
                         # shared build_client(); trailing underscore avoids shadowing
                         # the `openai` SDK. OpenAITranscribe now requests
                         # timestamp_granularities=["segment", "word"] unconditionally
                         # and maps the flat word list back onto segments by midpoint
                         # (two-pointer sweep, both lists already time-ordered)
    pyannote_.py         # Diarize: speaker diarisation for --asr-engine local/openai,
                         # which don't diarize natively; cli.py hands it a wav
                         # extracted from the source media, not the source media
                         # itself. Trailing
                         # underscore avoids shadowing the `pyannote` package.
                         # Pipeline (pyannote/speaker-diarization-community-1) built
                         # once in __init__ (CUDA when available, matching
                         # separate.py's posture), diarize() returns a time-sorted
                         # list[Turn]. Every pyannote.audio/torch import is lazy,
                         # inside __init__ only (diarize() needs no import of its
                         # own), so --voice-match off and --asr-engine elevenlabs
                         # never pay the import cost. Gated model: __init__ reads
                         # HF_TOKEN via os.environ.get() and passes it explicitly
                         # as token= to Pipeline.from_pretrained(), translating a
                         # missing
                         # token/unaccepted-conditions failure (GatedRepoError /
                         # RepositoryNotFoundError / HfHubHTTPError, or
                         # from_pretrained returning None) into one actionable
                         # RuntimeError naming the env var and the model page, rather
                         # than an opaque 401/403. Raises rather than catching, like
                         # separate.py -- cli.py is what decides whether a runtime
                         # failure degrades the run (it does) or an ImportError
                         # propagates (it does)
    prompt.py            # SYSTEM_PROMPT + build_prompt(): translation prompt text
                          # shared by LLMTranslate and OpenAITranslate
```

Flat package, no `src/` layout, no `tests/`.

`data/measurements/` is a top-level tree, not part of the package, and is the only committed subtree under `data/` — `.gitignore` ignores `data/*` then negates `!data/measurements/`, so source media and run outputs elsewhere under `data/` stay ignored. It holds `measure.py`, the analysis script behind `specs/chars-per-second-measurement.md`, plus five committed `*.measure.log` files (`clip-translate-synth`, `trailer-translate-synth`, `verify-clip`, `step2-max-budget`, `step2-max-budget-repeat`) of filtered `[measure]` DEBUG lines captured from real runs. Run with no flags, `measure.py` is free and offline: it parses those logs to reproduce the median source chars/slot-second and median tts-1 Danish chars/second figures backing `cli.py`'s `_EXPANSION_RATIO` and `_TARGET_SPEAKABLE_CPS` constants; `--rerun-paid` instead re-derives the unconstrained en→da expansion ratio via real, paid ElevenLabs ASR + Anthropic translation calls, and is off by default. `measure.py` parses the `[measure]` lines that `cli.py` and `dub.py` emit, rather than being imported by the package — there is no import edge from `movie_subtitles/` into `data/`, and no CI signal, so renaming or reordering a `[measure]` field silently breaks the script; both emitters carry an inline comment naming `measure.py` as their parser, which is the only thing holding that contract today.

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

`pyannote.audio>=4.0` is a runtime dependency of `providers/pyannote_.py`'s speaker diarisation path (`--voice-match != off` on `--asr-engine local`/`openai`), providing `pyannote/speaker-diarization-community-1`. Both `pyannote.audio` and `torch` are imported lazily, inside `Diarize.__init__`/`Diarize.diarize`, so a run that doesn't diarize on these engines (`--voice-match off`, or `--asr-engine elevenlabs`, which diarizes natively) never pays the import cost, even though the dependency itself is unconditional in `pyproject.toml`. **Platform consequence:** pinning `>=4.0` (required — unconstrained resolution lands on 3.4.0, too old for Community-1) collapses `uv.lock`'s previous two-branch torch pin (2.2.2 for `x86_64` macOS, 2.13.0 elsewhere) down to torch 2.13.0 only, which publishes no `x86_64` macOS wheel. **Intel Mac support is dropped as a result** — this is a breaking change, not incidental lock churn.

## Things that don't exist yet

No tests, no test framework. There are now four environment variables, not three: `ELEVENLABS_API_KEY`/`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` plus `HF_TOKEN` (see `.env.example` at the repo root, which holds placeholders for all four). `HF_TOKEN` is the `huggingface_hub` standard name; `providers/pyannote_.py:Diarize.__init__` reads it explicitly via `os.environ.get("HF_TOKEN")` and passes it as `token=` to `Pipeline.from_pretrained()` — `cli._load_env()` needed no per-key change, since it already calls `load_dotenv` over the whole file with no allowlist. All four are loaded from `.env` by `cli._load_env()`, called from `main()` only — `find_dotenv(usecwd=True)` so discovery is anchored to the user's cwd rather than to the installed package, and already-exported variables are never overridden. Importing any module still reads no `.env`; library callers set the environment themselves. `MOVIE_SUBTITLES_LOG_LEVEL` is a fifth environment variable, but not a secret: it raises the logging level of this package's own loggers above their default INFO (the root logger stays at INFO, so vendor SDK DEBUG output stays off) (e.g. `DEBUG`, to surface `cli.py`/`dub.py`'s permanent `[measure]` instrumentation lines — see `specs/chars-per-second-measurement.md`), unset or unrecognised falls back to INFO, and there is deliberately no CLI flag for it. There is now a minimal error-handling layer at the `main()` boundary (`RuntimeError | ValueError | FileNotFoundError | TimeoutError | subprocess.CalledProcessError` caught and turned into a one-line message + `SystemExit(1)`); anything else still propagates as a traceback. If asked to add tests, that means introducing pytest and a `tests/` directory from scratch — don't go looking for existing ones.

There **is** now a config file: `--voice-preset-table` takes a path to a JSON file (schema: top-level keys are TTS engine names, each mapping `gender:age_band`/`default` profile keys to a voice id string) that replaces `voices.py`'s built-in preset table wholesale for the engines it names. `voices.py:load_preset_table` validates it strictly at load time — malformed JSON, an unrecognised engine or profile key, a non-string voice id, or a missing `default` entry all raise `ValueError` before any dub work starts, rather than surfacing as a mid-dub `KeyError`.
