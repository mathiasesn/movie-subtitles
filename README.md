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

Other flags: `--audio-lang`, `--srt-lang`, `--whisper-model`, `--mt-model`,
`--dub-workers`, `--dub-correction-passes`. Run `movie-subtitles --help` for the
full list.

### Speaker-matched dub voices

`--asr-engine elevenlabs` diarizes the audio (Scribe's `diarize=True`, on by
default) and tags every `Segment` with a `speaker` label. Under `--dub`, this
label drives which TTS voice speaks each line:

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
Plain `--asr-engine local`/`openai` runs are unaffected — they carry no speaker
labels.

**vs. `--managed`:** the ElevenLabs Dubbing job API used by `--managed` has always
handled multi-speaker audio internally, including its own voice matching — none of
this is needed there. `--voice-match` only applies to this repo's own
transcribe→translate→dub pipeline (`--dub`), which had no notion of "who is
speaking" until this feature.

### Dubbing notes

- **Emitted `.srt` cue ends are padded, not raw ASR timings.** At write time, each cue's end is extended by up to `_CUE_PAD` (0.5s) toward the next cue's start (never past it, and never before the cue's own end) for subtitle readability. This applies to every `.srt` output, not only `--dub` runs; `dub.py` groups, anchors, and measures drift against the unpadded, word-accurate segment timings, so this only affects the written `.srt` file.
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
