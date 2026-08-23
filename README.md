<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mathiasesn/movie-subtitles/main/assets/logo/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mathiasesn/movie-subtitles/main/assets/logo/logo.svg">
    <img src="https://raw.githubusercontent.com/mathiasesn/movie-subtitles/main/assets/logo/logo.png" alt="Movie subtitles" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/mathiasesn/movie-subtitles/actions/workflows/ci.yml"><img src="https://github.com/mathiasesn/movie-subtitles/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/mathiasesn/movie-subtitles/actions/workflows/publish.yml"><img src="https://github.com/mathiasesn/movie-subtitles/actions/workflows/publish.yml/badge.svg" alt="Publish"></a>
  <a href="https://pypi.org/project/movie-subtitles/"><img src="https://img.shields.io/pypi/v/movie-subtitles.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/movie-subtitles/"><img src="https://img.shields.io/pypi/pyversions/movie-subtitles.svg" alt="Python versions"></a>
  <a href="https://github.com/mathiasesn/movie-subtitles/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
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

**Intel Mac (`x86_64` macOS) is no longer supported.** Speaker diarisation on
`--asr-engine local`/`openai` (see below) is backed by `pyannote.audio>=4.0`, which pulls
in torch 2.13.0 as a hard dependency — that release publishes no `x86_64` macOS wheel.
This collapsed what used to be a two-branch torch pin (a separate, older torch build for
Intel Macs) down to one. If you're on an Intel Mac, `uv sync` will fail to resolve.

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

`--engine local` needs no keys and stays fully offline after the first model download —
**as long as `--voice-match off` is passed.** Any other `--voice-match` value (`auto`,
the default, included) runs a speaker-diarisation pass that contacts Hugging Face on
first use; see "Speaker-matched dub voices" below. Everything else reads one or more of:

```shell
export ELEVENLABS_API_KEY="..."   # --asr-engine/--tts-engine elevenlabs, --managed
export ANTHROPIC_API_KEY="..."    # --translation-engine anthropic
export OPENAI_API_KEY="..."       # any stage set to openai
export HF_TOKEN="..."             # --voice-match != off on --asr-engine local/openai
```

Or copy `.env.example` to `.env` and fill it in — `.env` is gitignored and loaded on
startup, searching upward from the directory you run in. Exported variables take
precedence. A missing key exits with a one-line error naming the variable, no traceback.

**`HF_TOKEN` setup:** speaker diarisation on `--asr-engine local`/`openai` uses
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
a gated model. Before it will load:

1. Log in (or sign up) at [huggingface.co](https://huggingface.co).
2. Visit the [model page](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and accept its conditions.
3. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set it as
   `HF_TOKEN`.

Skipping this (or `--asr-engine elevenlabs`, which diarizes without it) is fine as long as
you also pass `--voice-match off` — no token, no diarisation pass, no network call.

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

Other flags: `--audio-lang`, `--srt-lang`, `--whisper-model`, `--mt-model`,
`--dub-workers`, `--dub-correction-passes`, `--duck-level`, `--separate-background`.
Run `movie-subtitles --help` for the full list.

### Speaker-matched dub voices

`--asr-engine elevenlabs` diarizes the audio (Scribe's `diarize=True`, on by
default) and tags every `Segment` with a `speaker` label. `--asr-engine local` and
`--asr-engine openai` don't diarize natively, but get the same labelling: whenever
`--voice-match` is anything other than `off`, a standalone
[`pyannote.audio`](https://github.com/pyannote/pyannote-audio) diarisation pass
(`pyannote/speaker-diarization-community-1`) runs over the whole source file first, and
its speaker turns are merged onto each ASR segment by temporal overlap. There is no
separate flag for this — `--voice-match` is the single control for "do I care about
speakers?" across all three ASR engines. Under `--dub`, the resulting `speaker` label
drives which TTS voice speaks each line:

| Flag | Default | Effect |
| --- | --- | --- |
| `--voice-match {off,clone,preset,auto}` | `auto` | How to pick a voice per diarized speaker |
| `--keep-cloned-voices` | off | Do not delete ElevenLabs voices cloned this run; the retained voice ids are logged |
| `--clone-min-seconds` | `30` | Minimum seconds of clean (non-overlapping) speech a speaker needs to be eligible for cloning; below this it falls back to a preset |
| `--clone-target-seconds` | `60` | Maximum seconds of clean speech gathered per speaker to build a cloned voice sample |
| `--voice-preset-table <path>` | none | JSON file overriding the built-in gender/age preset voice table |

`--voice-match` modes:

- **`off`** — the single configured voice speaks every line (today's pre-diarization
  behaviour). No diarization work, sample extraction, or voice-matching import happens.
- **`clone`** — instant-clones every eligible speaker's voice via ElevenLabs Instant
  Voice Cloning (IVC). A speaker with fewer than `--clone-min-seconds` of clean audio,
  or whose clone call fails, degrades to a preset voice instead of failing the run
  (a WARNING names the speaker either way).
- **`preset`** — never clones; classifies each speaker's sample and matches it to a
  curated stock voice by (gender, age band).
- **`auto`** (default) — clones when the resolved TTS engine supports it and the
  speaker has enough clean audio, otherwise falls back to preset matching.

**Cloning needs a plan that includes IVC.** ElevenLabs rejects
`voices.ivc.create` with `paid_plan_required` on subscriptions without instant voice
cloning; the run logs a WARNING and every speaker degrades to a preset voice, so
`--voice-match auto` behaves as `preset` on such an account.

**Cloning is ElevenLabs-only.** `--tts-engine openai` always gets preset voices,
regardless of `--voice-match` — OpenAI's TTS has no cloning endpoint. Cloned voices
are deleted automatically once the run finishes (including when the dub raises),
unless `--keep-cloned-voices` is passed.

For each speaker, clean audio is gathered by concatenating that speaker's segments
that don't overlap any other speaker's segment, up to `--clone-target-seconds`. When
classification is needed (preset matching, or a clone fallback), the same sample is
run through a coarse heuristic classifier (median F0 via `librosa.pyin`, formants via
`praat-parselmouth`) into one of six `gender:age_band` profiles, or "unknown" if
either analysis fails or no clean audio was found. These thresholds are hand-picked,
not tuned against a labelled dataset — treat the classification as a rough sort, not
a reliable gender/age read.

`--voice-preset-table` replaces the built-in table wholesale for the engines it
names (no per-key merge — a supplied file must be complete for any engine it
mentions). Schema:

```json
{
  "elevenlabs": {
    "female:young": "voice-id",
    "female:adult": "voice-id",
    "female:elderly": "voice-id",
    "male:young": "voice-id",
    "male:adult": "voice-id",
    "male:elderly": "voice-id",
    "default": "voice-id"
  },
  "openai": {
    "female:young": "nova",
    "...": "...",
    "default": "alloy"
  }
}
```

Top-level keys must be `elevenlabs` or `openai`; each block's keys must be one of
the six `gender:age_band` combinations (`female`/`male` × `young`/`adult`/`elderly`)
or `default`, and every block must include `default`. Loading fails fast with a
specific `ValueError` for malformed JSON, an unknown engine key, an unrecognised
profile key, a non-string voice id, or a missing `default` — not a mid-dub
`KeyError`.

**`.srt` cue boundaries now split at speaker changes.** This is not limited to
`--dub`: any `--asr-engine elevenlabs` run (with diarization on, the default) will
produce more, shorter cues than before whenever a scene contains dialogue between
multiple speakers, because a cue is flushed as soon as the diarized speaker changes.
`--asr-engine local`/`openai` now split the same way whenever `--voice-match != off`
(the diarisation pass above populates `speaker`, and both engines now also request
word-level timestamps, which is what makes splitting on a mid-cue speaker change
possible at all). `--voice-match off` still runs no diarisation and splits no cue on speaker change,
on every engine, so cue count and cue text are unchanged. Cue *timings* aren't
guaranteed byte-for-byte on `--asr-engine local`, though: that engine now requests
word-level timestamps unconditionally (see below), which turns on faster-whisper's
word-alignment pass and can shift a segment's own start/end slightly even with
`--voice-match off`.

**Diarising `local`/`openai` audio also shifts dub timing slightly**, independent of
`--voice-match`'s value: both engines now request word timestamps unconditionally (not
only when diarising), and `dub.py` prefers word-level inter-segment gaps over
cue-boundary gaps when they're available. This is a strict improvement to scene
grouping, not a change to the timing/rate-fitting model itself, but it does mean dub
output on these two engines can differ slightly from earlier runs.

**`--asr-engine openai` caveat:** speaker labels there are assigned by overlapping
diarisation turns against `whisper-1`'s own segment spans, and `whisper-1`'s segment
timestamps are known to degrade to uniform 1.000s spans on music-heavy or
dialogue-sparse audio (see "Known limitation" above). Where that fires, diarisation
produces confidently wrong labels rather than merely absent ones — the run logs a
WARNING recommending `--asr-engine elevenlabs` for multi-speaker material.

**Diarisation degrades rather than fails the run.** A missing/invalid `HF_TOKEN`, an
unaccepted model gate, an unreachable Hugging Face Hub, or any other diarisation
runtime error logs one WARNING and falls back to today's single-voice dub — it does not
abort the run. Only a broken install (an `ImportError` — `pyannote.audio` itself
missing) fails it, the same split `--separate-background` uses for Demucs.

**vs. `--managed`:** the ElevenLabs Dubbing job API used by `--managed` has always
handled multi-speaker audio internally, including its own voice matching — none of
this is needed there. `--voice-match` only applies to this repo's own
transcribe→translate→dub pipeline (`--dub`), which had no notion of "who is
speaking" until this feature.

### Dubbing notes

- **Emitted `.srt` cue ends are padded, not raw ASR timings.** At write time, each cue's end is extended by up to `_CUE_PAD` (0.5s) toward the next cue's start (never past it, and never before the cue's own end) for subtitle readability. This applies to every `.srt` output, not only `--dub` runs; `dub.py` groups, anchors, and measures drift against the unpadded, word-accurate segment timings, so this only affects the written `.srt` file.
- **`--dub` keeps the original audio underneath the synthesised track**, ducked
  (via `--duck-level`, default `0.25`, or `0.6` when `--separate-background`
  succeeded — see below) while the dub is speaking so the translated dialogue stays
  dominant but music, effects and ambience survive rather than being dropped.
  `--duck-level 0.0` silences that bed entirely under the dub; `--duck-level 1.0`
  disables ducking altogether. An explicit `--duck-level` always wins over either
  default. A source with no audio stream at all falls back to a dub-only track.
- **`--separate-background` removes the original dialogue instead of merely
  ducking it.** By default the original audio is only attenuated under the dub, so
  both languages are still audible at once during every cue. Passing
  `--separate-background` runs [Demucs](https://github.com/facebookresearch/demucs)
  (`htdemucs`) over the source audio first, splitting it into a vocals stem and an
  accompaniment (everything else) stem, and mixes the dub over the accompaniment
  stem only — the original dialogue is gone, not ducked. It's opt-in for real
  reasons:
  - **Cost:** Demucs runs on CPU by default and takes CPU-minutes to tens of
    minutes on a feature-length film; the first run also downloads model weights
    (hundreds of MB from a third-party host).
  - **Not perfect:** separation leaves artifacts — some residual original-dialogue
    bleed can remain, and music can smear slightly. The result is better than
    ducking, not clean.
  - **Fidelity cost on surround sources:** Demucs works internally at 44.1 kHz
    stereo, so a 5.1/48 kHz source's accompaniment stem is downmixed to stereo
    internally; `--dub`'s final muxed output still matches the source's own
    layout/rate, only the separated bed loses surround information along the way.
  - **Fails safe:** any separation failure (unreachable model weights, a bad input
    file, an ffmpeg/Demucs error) logs one warning and falls straight back to the
    default duck-and-mix behaviour above, rather than failing the run.
  - **Mutually exclusive with `--managed`** — the managed ElevenLabs Dubbing job
    already handles background preservation itself.
- **`--dub` and `--managed` are mutually exclusive.** `--managed` replaces the whole
  pipeline with a single hosted job.
- **`--dub` fails if TTS resolves to something unusable**, e.g. plain `--engine local
  --dub`, since `local` has no TTS backend:

  ```
  [ERROR][cli] --dub requires a usable TTS engine, but it resolved to 'local'. Pass --tts-engine {elevenlabs,openai} explicitly.
  ```

- **`--translation-engine local` with `--dub` is allowed but warns.** MADLAD400 ignores
  the length budget, so timing-drift fitting degrades to TTS-rate-only.
