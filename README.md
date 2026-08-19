# Movie subtitles

Command line interface that turns a video/audio file into a translated `.srt` file, and
optionally into a dubbed video with a synthesised translated audio track.

Three pipelines live side by side behind `--engine`:

- **`local`** (default): faster-whisper for ASR, a local MADLAD400 T5 model for
  translation. Fully offline after model download, no API keys. No TTS stage.
- **`elevenlabs`**: ElevenLabs Scribe for ASR, ElevenLabs TTS for `--dub`. Translation is
  **not** available under this shorthand — see [Breaking change: engine values name
  vendors, not pipelines](#breaking-change-engine-values-name-vendors-not-pipelines)
  below. See also [Which stages are ElevenLabs](#which-stages-are-elevenlabs-and-which-are-not),
  it is not "all ElevenLabs", and that is itself the interesting finding.
- **`openai`**: OpenAI (`whisper-1`) for ASR, OpenAI chat completions for
  translation, OpenAI (`gpt-4o-mini-tts`) for TTS. A second complete end-to-end pipeline,
  alongside the ElevenLabs one. See
  [OpenAI backend](#openai-backend) below for what is and isn't ElevenLabs here.

There is also `--managed`, which bypasses this repo's pipeline entirely and calls the
ElevenLabs Dubbing job API (create → poll → download) as the "buy" side of a build-vs-buy
comparison. `--managed` is ElevenLabs-only and unaffected by the OpenAI backend.

`--engine` is a shorthand that sets all three stages at once. Each stage can be
overridden independently with `--asr-engine`, `--translation-engine`, and `--tts-engine`
— e.g. Scribe ASR with the local MADLAD400 translator
(`--asr-engine elevenlabs --translation-engine local`), or a fully mixed pipeline
(`--asr-engine elevenlabs --translation-engine openai --tts-engine openai`). Not every
combination makes sense: `--dub` requires a length-budgeted translator, so
`--translation-engine local` (or `--engine local`) combined with `--dub` is rejected —
see [Timing drift](#timing-drift).

## Breaking change: engine values name vendors, not pipelines

**This is a deliberate breaking CLI change from an earlier version of this README/CLI.**
Engine values used to name a pipeline (`elevenlabs` meant "Scribe ASR + Claude
translation + ElevenLabs TTS"). Now that more than one vendor backs each stage, that
one-to-one mapping stops making sense — so engine values were renamed to name the vendor
selected **per stage**:

| Flag | Values |
| --- | --- |
| `--asr-engine` | `local` (faster-whisper), `elevenlabs` (Scribe), `openai` |
| `--translation-engine` | `local` (MADLAD400), `anthropic` (Claude), `openai` |
| `--tts-engine` | `elevenlabs`, `openai` |

The old value `--translation-engine elevenlabs` **no longer exists**. Its replacement is
`--translation-engine anthropic`, since the translation call was always Claude, never an
ElevenLabs product call (see below).

**`--engine elevenlabs` alone now errors.** ElevenLabs has no standalone
text-translation endpoint — translation exists only bundled inside the Dubbing job
(`--managed`) — so there is nothing honest for the `elevenlabs` shorthand to resolve the
translation stage to. Rather than silently routing to Anthropic, the CLI refuses and
names the fix:

```shell
$ uv run movie-subtitles --input clip.mp4 --engine elevenlabs
[19/08/2026-14:09:22][ERROR][cli] --engine elevenlabs does not set a translation stage: ElevenLabs has no standalone text-translation endpoint (translation exists only bundled inside the Dubbing job, see --managed). Pass --translation-engine {anthropic,openai,local} explicitly.
$ echo $?
1
```

The corrected form, reproducing exactly the previous `--engine elevenlabs` behaviour:

```shell
uv run movie-subtitles --input clip.mp4 --engine elevenlabs --translation-engine anthropic
```

### Resolution table

| Invocation | ASR | Translation | TTS |
| --- | --- | --- | --- |
| `--engine local` | faster-whisper | MADLAD400 | none (`--dub` rejected) |
| `--engine openai` | OpenAI | OpenAI | OpenAI |
| `--engine elevenlabs` | Scribe | **error** | ElevenLabs |
| `--engine elevenlabs --translation-engine anthropic` | Scribe | Claude | ElevenLabs |

Per-stage flags always override the shorthand, and supplying `--translation-engine`
always clears the error above.

**`--dub` replaces the original audio track, it does not mix with it.** `mux.py` maps
only the synthesised track (`-map 1:a:0`) onto the source video (`-map 0:v:0`); the
original spoken audio is dropped entirely, not layered under the dub.

## Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- `ffmpeg` on `PATH` — only required for `--dub` (audio synthesis + mux) and `--managed`
  is not affected by ffmpeg since ElevenLabs renders the final file server-side. The
  plain `.srt` path (no `--dub`, no `--managed`) never needs ffmpeg.

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

## Setup: API keys

The `elevenlabs` engine, the `openai` engine, `--dub`, and `--managed` all need one or
more of these environment variables. They are read from the environment only — never
write a real key into a file in this repo. A `.env.example` at the repo root lists all
three as placeholders; copy it to `.env` and fill in real values locally (never commit
the filled-in file):

```shell
export ELEVENLABS_API_KEY="<your-elevenlabs-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
export OPENAI_API_KEY="<your-openai-key>"
```

`ELEVENLABS_API_KEY` is required for `--asr-engine elevenlabs` (Scribe), `--tts-engine
elevenlabs` (`--dub`), and `--managed` (Dubbing job). `ANTHROPIC_API_KEY` is required for
`--translation-engine anthropic` (Claude — see below for why translation on the
ElevenLabs path uses Claude and not ElevenLabs). `OPENAI_API_KEY` is required for
`--asr-engine openai`, `--translation-engine openai`, and `--tts-engine openai` — i.e.
any stage set to `openai`, including the `--engine openai` shorthand. If a required key
is missing, the CLI exits with a one-line error naming the variable — no traceback.
Example, run with no key set:

```shell
$ uv run movie-subtitles --input /tmp/nope.mp4 --engine elevenlabs
[19/08/2026-14:09:22][ERROR][cli] ELEVENLABS_API_KEY environment variable is not set. Set it to your ElevenLabs API key to use the 'elevenlabs' engine.
$ echo $?
1
```

The default `--engine local` path needs neither key and stays fully offline after the
first model download.

## Usage

```shell
$ uv run movie-subtitles --help
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
  --engine {local,elevenlabs,openai}
                        The provider engine to use for transcription,
                        translation and TTS (shorthand for --asr-engine,
                        --translation-engine and --tts-engine when those are
                        not set). --engine elevenlabs requires
                        --translation-engine to be set explicitly, since
                        ElevenLabs has no standalone translation endpoint
  --asr-engine {local,elevenlabs,openai}
                        Override --engine for the ASR (transcription) stage
                        only
  --translation-engine {local,anthropic,openai}
                        Override --engine for the translation stage only.
                        --dub requires this to resolve to 'anthropic' or
                        'openai', since the local translator ignores the
                        length budget
  --tts-engine {elevenlabs,openai}
                        Override --engine for the TTS (dubbing) stage only
  --dub                 Synthesise the translated segments with TTS and mux
                        them over the source video (requires ffmpeg and the
                        API key for the resolved TTS engine)
  --managed             Use the managed ElevenLabs Dubbing job
                        (create/poll/download) instead of the local
                        transcribe/translate/dub pipeline (requires
                        ELEVENLABS_API_KEY). Mutually exclusive with --dub.
```

`--engine local --dub` (or `--translation-engine local --dub`) is rejected up front with a
`ValueError` explaining why, rather than silently shipping a worse dub:

```shell
$ uv run movie-subtitles --input clip.mp4 --engine local --dub
[19/08/2026-14:21:12][ERROR][cli] --dub requires a length-budgeted translator: the local MADLAD400 translator ignores budget_chars, so timing-drift fitting (step 1 of the drift strategy) is a silent no-op and the dub would be worse for no stated reason. Use --translation-engine anthropic (or --engine elevenlabs/openai) with --dub.
$ echo $?
1
```

Local, offline, `.srt` only (unchanged from before this port):

```shell
movie-subtitles --input clip.mp4
```

ElevenLabs ASR + Claude translation, `.srt` only (the old `--engine elevenlabs` shorthand
now requires the translation stage explicitly — see
[Breaking change](#breaking-change-engine-values-name-vendors-not-pipelines) above):

```shell
movie-subtitles --input clip.mp4 --engine elevenlabs --translation-engine anthropic
```

ElevenLabs ASR + Claude translation + ElevenLabs TTS, muxed into a dubbed video
(`clip.dubbed.mp4`), requires `ffmpeg`:

```shell
movie-subtitles --input clip.mp4 --engine elevenlabs --translation-engine anthropic --dub
```

OpenAI ASR + OpenAI translation + OpenAI TTS end to end, `.srt` only:

```shell
movie-subtitles --input clip.mp4 --engine openai
```

OpenAI ASR + OpenAI translation + OpenAI TTS, muxed into a dubbed video, requires
`ffmpeg`:

```shell
movie-subtitles --input clip.mp4 --engine openai --dub
```

Mixed vendors — Scribe ASR + OpenAI translation + OpenAI TTS, isolating voice quality as
a single variable:

```shell
movie-subtitles --input clip.mp4 --asr-engine elevenlabs --translation-engine openai --tts-engine openai --dub
```

Mixed vendors the other way — OpenAI ASR/translation, ElevenLabs voice:

```shell
movie-subtitles --input clip.mp4 --engine openai --tts-engine elevenlabs --dub
```

Fully managed ElevenLabs Dubbing job, no local ASR/MT/TTS code runs at all:

```shell
movie-subtitles --input clip.mp4 --managed
```

`--dub` and `--managed` are mutually exclusive — `--managed` replaces the whole
transcribe/translate/dub pipeline with a single hosted job, so combining it with `--dub`
doesn't mean anything and the CLI rejects it with a `ValueError`.

## Which stages are ElevenLabs, and which are not

This matters because the pipeline now spans three vendors, and it should be stated
plainly rather than discovered by reading code:

| Stage | `--engine elevenlabs` (+ `--translation-engine anthropic`) | `--engine local` | `--engine openai` |
|---|---|---|---|
| ASR (speech-to-text) | **ElevenLabs Scribe** (`speech_to_text.convert`) | faster-whisper `large-v3` | **OpenAI** `whisper-1` — not ElevenLabs |
| Translation | **Claude** (Anthropic API, `anthropic` SDK) — not ElevenLabs | MADLAD400 T5 (local) | **OpenAI** chat completions — not ElevenLabs |
| TTS (`--dub`) | **ElevenLabs** (`text_to_speech.convert`) | not available | **OpenAI** `gpt-4o-mini-tts` — not ElevenLabs |
| `--managed` | **ElevenLabs Dubbing** (create/poll/download, does ASR+translation+TTS internally as one hosted job) | not applicable | not applicable (ElevenLabs-only surface) |

**Plainly stated: on the `openai` engine, no stage is ElevenLabs.** OpenAI TTS in
particular is worth calling out explicitly — it is easy to assume "TTS = ElevenLabs" by
habit from the rest of this README, but `--tts-engine openai` (including as part of
`--engine openai`) routes to `gpt-4o-mini-tts`, a completely different vendor and voice,
with its own speed/timing behaviour (see [OpenAI backend](#openai-backend) below).

**Translation on the `elevenlabs` engine is not an ElevenLabs product call.** This was a
finding during implementation, not a design preference stated up front: ElevenLabs has no
standalone text-translation endpoint. Translation exists only bundled inside the Dubbing
job, which does everything end-to-end and gives you no hook to intercept just the
translated text with a length budget you control. So for the hand-rolled per-segment
`.srt`/`--dub` path, translation is Claude via the `anthropic` SDK
(`movie_subtitles/providers/llm.py`), not ElevenLabs. The local `--engine local` path is
unaffected — it keeps using the local MADLAD400 model, exactly as before this port.

Claude was chosen over the local MADLAD400 model for this path specifically because a
prompted LLM can be asked for **length-matched output** — the translation prompt is given
a target character budget derived from the segment's duration (see
[Timing drift](#timing-drift) below) and asked to translate within it. MADLAD400 is a
plain seq2seq translation model with no length-control lever; asking it for "roughly N
characters" is not something the model can act on. That length control is a direct lever
on timing drift, which is the harder problem in the dub path, so it drove the choice of
translator on this engine.

## OpenAI backend

`--engine openai` (or any per-stage `--asr-engine openai` / `--translation-engine
openai` / `--tts-engine openai`) routes to the OpenAI API instead of ElevenLabs or
Anthropic:

- **ASR**: `whisper-1` (`movie_subtitles/providers/openai_.py:OpenAITranscribe`),
  normalised into the same `Segment` shape faster-whisper and Scribe produce.
- **Translation**: OpenAI chat completions (`gpt-5.6-terra`)
  (`movie_subtitles/providers/openai_.py:OpenAITranslate`), honouring the same
  `budget_chars` length-budget contract as `LLMTranslate` (Claude), so `--dub` is
  permitted with `--translation-engine openai`.
- **TTS**: `gpt-4o-mini-tts` (`movie_subtitles/providers/openai_.py:OpenAISpeak`).

**None of these three are ElevenLabs stages.** OpenAI is the only vendor besides
ElevenLabs able to back every stage of the pipeline (Anthropic only ever covered
translation), which is what makes an all-`openai` end-to-end run and mixed-vendor runs
(e.g. `--asr-engine elevenlabs --translation-engine openai --tts-engine openai`) both
possible.

**Speed/timing control is expected to differ from ElevenLabs, unverified.** ElevenLabs'
`voice_settings.speed` and this repo's rate-clamp (see [Timing drift](#timing-drift)
below) were tuned against ElevenLabs TTS specifically. OpenAI's TTS speed lever works
differently, and its supported range is wider/narrower in ways not yet reconciled against
the 0.9–1.15x clamp `dub.py` applies uniformly. **This is stated as an expectation to
verify, not a measured result:** it is plausible the same clamp strategy fits OpenAI-
synthesised segments noticeably worse (or better) than ElevenLabs ones, and that should
be checked empirically once real clips are run through `--tts-engine openai`, rather than
assumed to carry over unchanged.

## Timing drift

**The problem.** A translated sentence is rarely the same duration as the original when
spoken aloud. If you synthesise each segment's translation at a fixed rate and just place
it at the segment's original start time, sentences drift out of sync with the speaker —
audio for segment N can spill into segment N+1's slot, or leave large silent gaps.

**Implemented strategy: length-budgeted translation, then a clamped rate nudge.**

1. Each segment carries a duration budget: `end - start` from the source timing. The
   translation call (`movie_subtitles/cli.py:_budget_chars`, using an assumed ~15
   chars/second speaking-rate constant — not a measured value, see below) turns that into
   a target character count, passed to the Claude translation prompt as "translate within
   roughly N characters."
2. The segment is synthesised via ElevenLabs TTS and the actual audio duration is
   measured with `ffprobe` (`movie_subtitles/dub.py:_synthesise_fitted`).
3. If it overruns the slot, the TTS speaking rate (`voice_settings.speed`)
   is adjusted and the segment is re-synthesised once, **clamped to 0.9–1.15x**
   (`movie_subtitles/dub.py:_MIN_RATE` / `_MAX_RATE`).
4. If the segment still does not fit at the clamp bound, it is left to overrun into the
   following silence rather than distorting the voice further. Each such segment is
   logged, and the run reports a total count of unfittable segments
   (`movie_subtitles/dub.py:synthesise_track`) — see the placeholder in
   [Results](#results) for the count on the test clip.

**The clamp is a deliberate tightening, not the API's limit.** ElevenLabs' own documented
speed range for `voice_settings.speed` is wider — **0.7 to 1.2**
(https://elevenlabs.io/docs/best-practices/prompting/controls.md, confirmed 2026-08-19) —
with a note that extreme values can degrade audio quality. This implementation clamps to
0.9–1.15x specifically to keep the rate adjustment inaudible, at the cost of fitting fewer
segments purely through rate change.

**Only overruns are corrected.** The current implementation adjusts rate only when audio
is *longer* than its slot. An underrun (translated audio shorter than the slot) is not
stretched to fill the gap — it is simply left to sit in silence for the remainder of the
slot. This is an honest gap, not a design choice with a stated rationale; it just wasn't
built.

**Considered, not implemented:**

- **Trimming/padding inter-segment silence** as the primary fitting mechanism, instead of
  (or as well as) a rate nudge — shrink or grow the gap before/after a segment to absorb
  drift rather than only speeding up speech. Would need silence-detection or an assumed
  minimum gap; cut for scope.
- **Letting segments float and re-anchoring at scene boundaries** via ElevenLabs' Forced
  Alignment endpoint, rather than pinning every segment to its original ASR timestamp.
  This is the more correct long-term fix (drift wouldn't compound across a whole scene)
  but is a materially larger change — periodic re-alignment checkpoints, not per-segment
  math — and out of scope for the two-evening budget this port was built under.

No `--drift-strategy` flag exists; one strategy, documented honestly including its gap
(underruns), was judged better than three half-built ones.

## Build vs. buy

The hand-rolled path (`--engine elevenlabs --dub`: Scribe ASR → Claude translation →
ElevenLabs TTS → ffmpeg mux, all in this repo) and the managed path (`--managed`:
ElevenLabs Dubbing job, create/poll/download, no local pipeline code involved) are two
answers to the same task. The questions this comparison should answer, once real clips
have been run through both:

- **Sync quality.** Does the managed job's internal alignment (whatever ElevenLabs does
  under the hood) hold up better across the clip than the clamped-rate strategy above?
- **Danish output quality.** Since Dubbing bundles its own translation, does its Danish
  read better or worse than the length-budgeted Claude translation feeding local TTS?
- **Control vs. opacity.** The hand-rolled path exposes every knob (budget chars, rate
  clamp, which segments failed to fit) at the cost of being more code to maintain. The
  managed path is a single API call with none of those levers — you get what the job
  produces.
- **Cost and latency** for a clip of this length, hand-rolled (per-character TTS billing
  + per-request translation calls) vs. managed (per-minute job billing).
- **Failure modes** — what happens when a segment's boundary lands mid-word for each
  approach.

**Verdict: pending the first real run of both paths on the test clip below.** This
section intentionally states the comparison framework rather than a conclusion — no clip
has been run through either path in this environment (no `ELEVENLABS_API_KEY` or
`ANTHROPIC_API_KEY`, no media). It is plausible that `--managed` is simply the better
Danish result; if so, that finding will be reported here rather than downplayed once it
exists.

## Results

**Nothing in this section has been measured yet.** No API key was available in the
environment this port was built in, and no media clip was available or committed
(`samples/` is gitignored — see [Setup: test clips](#setup-test-clips)). Every cell below
is `TBD` until the repo owner runs the commands and fills them in.

### ASR: Scribe vs. faster-whisper vs. OpenAI, English clip


```shell
movie-subtitles --input samples/en_clip.mp4 --engine local           # writes en_clip.srt
movie-subtitles --input samples/en_clip.mp4 --asr-engine elevenlabs --translation-engine anthropic  # writes en_clip.srt (overwrite; rename to compare)
movie-subtitles --input samples/en_clip.mp4 --asr-engine openai --translation-engine openai         # writes en_clip.srt (overwrite; rename to compare)
```

| Metric | faster-whisper `large-v3` | ElevenLabs Scribe | OpenAI `whisper-1` |
|---|---|---|---|
| Transcript accuracy (impression) | TBD — not yet run | TBD — not yet run | TBD — not yet run |
| Wall-clock time | TBD — not yet run | TBD — not yet run | TBD — not yet run |
| Rough cost | TBD — not yet run (local, compute only) | TBD — not yet run | TBD — not yet run |

### ASR: Scribe vs. faster-whisper vs. OpenAI, Danish-audio clip (the role-relevant result)

```shell
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --engine local
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --asr-engine elevenlabs --translation-engine anthropic
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --asr-engine openai --translation-engine openai
```

| Metric | faster-whisper `large-v3` | ElevenLabs Scribe | OpenAI `whisper-1` |
|---|---|---|---|
| Danish transcript accuracy (impression) | TBD — not yet run | TBD — not yet run | TBD — not yet run |
| Wall-clock time | TBD — not yet run | TBD — not yet run | TBD — not yet run |
| Rough cost | TBD — not yet run | TBD — not yet run | TBD — not yet run |

### Danish TTS voice quality (en→da dub)

```shell
movie-subtitles --input samples/en_clip.mp4 --engine elevenlabs --translation-engine anthropic --dub
movie-subtitles --input samples/en_clip.mp4 --engine openai --dub
```

TBD — not yet run, for both the ElevenLabs and the OpenAI dub. Listen to the resulting
`en_clip.dubbed.mp4` from each run and note naturalness, pronunciation, and any audible
rate-clamp artefacts once produced. Per the [OpenAI backend](#openai-backend) note above,
also check whether the same 0.9–1.15x clamp fits OpenAI-synthesised segments as well as
ElevenLabs ones — that is an open question, not an assumption.

### Translation budget adherence

How closely each translator actually respects the `budget_chars` target passed in the
prompt (see [Timing drift](#timing-drift)) — e.g. the distribution of `len(actual) /
budget_chars` across the test clip's segments.

- English clip (~60s), en→da, Claude (`--translation-engine anthropic`): **TBD — not yet run**
- English clip (~60s), en→da, OpenAI (`--translation-engine openai`): **TBD — not yet run**

### Unfittable-segment count

Reported directly by the run's log line
(`movie_subtitles/dub.py:synthesise_track`, `"N segment(s) could not be fitted within the
rate clamp"`) for the `--dub` runs above.

- English clip (~60s), en→da, ElevenLabs TTS: **TBD — not yet run**
- English clip (~60s), en→da, OpenAI TTS: **TBD — not yet run**

## Setup: test clips

No media is committed — `samples/` is `.gitignore`d. Cut short local clips with `ffmpeg`
so the runs above are reproducible without shipping video:

```shell
mkdir -p samples

# ~60s English-audio source, for the en->da dub test (matches the --audio-lang en
# --srt-lang da defaults)
ffmpeg -ss 00:05:00 -i /path/to/source.mp4 -t 60 -c copy samples/en_clip.mp4

# ~30s Danish-audio source, for the ASR benchmark only (da->en)
ffmpeg -ss 00:05:00 -i /path/to/danish_source.mp4 -t 30 -c copy samples/da_clip.mp4
```

Adjust `-ss` to a timestamp with clear speech in the source file. `-c copy` avoids
re-encoding; drop it if the cut lands on a keyframe boundary issue.

## Run-book: filling in every TBD

With `ELEVENLABS_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY` set (see
[Setup: API keys](#setup-api-keys)) and both test clips cut (see
[Setup: test clips](#setup-test-clips) above), this is the full set of commands, in
order, to produce every measurement referenced as `TBD — not yet run` in
[Results](#results) above, across both the ElevenLabs and the OpenAI paths in one
sitting:

```shell
# --- ASR benchmark, English clip (local vs. ElevenLabs vs. OpenAI) ---
cp samples/en_clip.srt samples/en_clip.local.srt.bak 2>/dev/null || true
movie-subtitles --input samples/en_clip.mp4 --engine local
mv samples/en_clip.srt samples/en_clip.local.srt
movie-subtitles --input samples/en_clip.mp4 --asr-engine elevenlabs --translation-engine anthropic
mv samples/en_clip.srt samples/en_clip.elevenlabs.srt
movie-subtitles --input samples/en_clip.mp4 --asr-engine openai --translation-engine openai
mv samples/en_clip.srt samples/en_clip.openai.srt

# --- ASR benchmark, Danish-audio clip (local vs. ElevenLabs vs. OpenAI) ---
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --engine local
mv samples/da_clip.srt samples/da_clip.local.srt
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --asr-engine elevenlabs --translation-engine anthropic
mv samples/da_clip.srt samples/da_clip.elevenlabs.srt
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --asr-engine openai --translation-engine openai
mv samples/da_clip.srt samples/da_clip.openai.srt

# --- Danish TTS voice quality + unfittable-segment count, ElevenLabs dub ---
movie-subtitles --input samples/en_clip.mp4 --engine elevenlabs --translation-engine anthropic --dub
mv samples/en_clip.dubbed.mp4 samples/en_clip.elevenlabs.dubbed.mp4

# --- Danish TTS voice quality + unfittable-segment count, OpenAI dub ---
movie-subtitles --input samples/en_clip.mp4 --engine openai --dub
mv samples/en_clip.dubbed.mp4 samples/en_clip.openai.dubbed.mp4
```

Time each `movie-subtitles` invocation above (e.g. wrap with `time`) for the wall-clock
rows, note token/character usage reported by each provider's dashboard for the cost rows,
listen to both `*.dubbed.mp4` files for the TTS-quality rows, and read the
`synthesise_track` log line from each `--dub` run for the unfittable-segment counts.
Translation budget adherence requires instrumenting `budget_chars` vs. `len(actual)` per
segment — not currently logged, so this has to be added ad hoc or computed by comparing
the `.srt` timings against the translated text lengths.

## What's not implemented

- **Burned-in per-frame subtitles.** An earlier version of this README described adding
  subtitles to each movie frame. That was never built — `.srt` (and, with `--dub`, a
  muxed audio track) are the only outputs.
- **Underrun correction** in the timing-drift strategy (see above).
- Speaker diarisation / distinct voices per speaker in the dub.
- Realtime/streaming Scribe or TTS.
- A config-file or env-var layer beyond the three API keys — everything else is a CLI
  flag.

## What I'd do differently / what is unverified

- Every number in [Results](#results) is unverified — this port was built and reviewed
  without API access or media in this environment. The build-vs-buy verdict is explicitly
  not called until those runs happen.
- The `_CHARS_PER_SECOND = 15.0` speaking-rate assumption
  (`movie_subtitles/cli.py`) used to derive the translation length budget is a guess, not
  a measured Danish speaking rate. It should be tuned against the first real dub.
- Given more time, the Forced Alignment re-anchoring approach described under
  [Timing drift](#timing-drift) is the direction I'd actually pursue for a
  production-quality dub — the rate-clamp strategy shipped here is a bounded, honestly
  imperfect first pass, not the final answer to sync.
- I have not verified how `--managed`'s Dubbing job behaves on a source clip that mixes
  English and Danish audio, or on clips shorter than the job's practical minimum; only
  the intended happy path (single source language, ~60s) is exercised by the commands
  above.
