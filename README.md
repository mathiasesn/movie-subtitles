# Movie subtitles

Command line interface that turns a video/audio file into a translated `.srt` file, and
optionally into a dubbed video with a synthesised translated audio track.

Two pipelines live side by side behind `--engine`:

- **`local`** (default): faster-whisper for ASR, a local MADLAD400 T5 model for
  translation. Fully offline after model download, no API keys.
- **`elevenlabs`**: ElevenLabs Scribe for ASR, Claude (Anthropic API) for translation,
  ElevenLabs TTS for `--dub`. See [Which stages are ElevenLabs](#which-stages-are-elevenlabs-and-which-are-not)
  below — it is not "all ElevenLabs", and that is itself the interesting finding.

There is also `--managed`, which bypasses this repo's pipeline entirely and calls the
ElevenLabs Dubbing job API (create → poll → download) as the "buy" side of a build-vs-buy
comparison.

`--engine` is a shorthand that sets both stages at once. Each stage can be overridden
independently with `--asr-engine` and `--translation-engine` — e.g. Scribe ASR with the
local MADLAD400 translator (`--asr-engine elevenlabs --translation-engine local`). Not
every combination makes sense: `--dub` requires a length-budgeted translator, so
`--translation-engine local` (or `--engine local`) combined with `--dub` is rejected —
see [Timing drift](#timing-drift).

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

The `elevenlabs` engine, `--dub`, and `--managed` all need one or both of these
environment variables. They are read from the environment only — never write a real key
into a file in this repo.

```shell
export ELEVENLABS_API_KEY="<your-elevenlabs-key>"
export ANTHROPIC_API_KEY="<your-anthropic-key>"
```

`ELEVENLABS_API_KEY` is required for `--engine elevenlabs` (ASR), `--dub` (TTS), and
`--managed` (Dubbing job). `ANTHROPIC_API_KEY` is required for `--engine elevenlabs`
translation specifically (see below for why translation uses Claude and not ElevenLabs).
If a required key is missing, the CLI exits with a one-line error naming the variable —
no traceback. Example, run with no key set:

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
  --engine {local,elevenlabs}
                        The provider engine to use for transcription and
                        translation (shorthand for --asr-engine and
                        --translation-engine when those are not set)
  --asr-engine {local,elevenlabs}
                        Override --engine for the ASR (transcription) stage
                        only
  --translation-engine {local,elevenlabs}
                        Override --engine for the translation stage only.
                        --dub requires this to resolve to 'elevenlabs', since
                        the local translator ignores the length budget
  --dub                 Synthesise the translated segments with ElevenLabs TTS
                        and mux them over the source video (requires ffmpeg
                        and ELEVENLABS_API_KEY)
  --managed             Use the managed ElevenLabs Dubbing job
                        (create/poll/download) instead of the local
                        transcribe/translate/dub pipeline (requires
                        ELEVENLABS_API_KEY). Mutually exclusive with --dub.
```

`--engine local --dub` (or `--translation-engine local --dub`) is rejected up front with a
`ValueError` explaining why, rather than silently shipping a worse dub:

```shell
$ uv run movie-subtitles --input clip.mp4 --engine local --dub
[19/08/2026-14:21:12][ERROR][cli] --dub requires a length-budgeted translator: the local MADLAD400 translator ignores budget_chars, so timing-drift fitting (step 1 of the drift strategy) is a silent no-op and the dub would be worse for no stated reason. Use --translation-engine elevenlabs (or --engine elevenlabs) with --dub.
$ echo $?
1
```

Local, offline, `.srt` only (unchanged from before this port):

```shell
movie-subtitles --input clip.mp4
```

ElevenLabs ASR + Claude translation, `.srt` only:

```shell
movie-subtitles --input clip.mp4 --engine elevenlabs
```

ElevenLabs ASR + Claude translation + ElevenLabs TTS, muxed into a dubbed video
(`clip.dubbed.mp4`), requires `ffmpeg`:

```shell
movie-subtitles --input clip.mp4 --engine elevenlabs --dub
```

Fully managed ElevenLabs Dubbing job, no local ASR/MT/TTS code runs at all:

```shell
movie-subtitles --input clip.mp4 --managed
```

`--dub` and `--managed` are mutually exclusive — `--managed` replaces the whole
transcribe/translate/dub pipeline with a single hosted job, so combining it with `--dub`
doesn't mean anything and the CLI rejects it with a `ValueError`.

## Which stages are ElevenLabs, and which are not

This matters because the pipeline spans two vendors, and it should be stated plainly
rather than discovered by reading code:

| Stage | `--engine elevenlabs` | `--engine local` |
|---|---|---|
| ASR (speech-to-text) | **ElevenLabs Scribe** (`speech_to_text.convert`) | faster-whisper `large-v3` |
| Translation | **Claude** (Anthropic API, `anthropic` SDK) | MADLAD400 T5 (local) |
| TTS (`--dub`) | **ElevenLabs** (`text_to_speech.convert`) | not available |
| `--managed` | **ElevenLabs Dubbing** (create/poll/download, does ASR+translation+TTS internally as one hosted job) | not applicable |

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

### ASR: Scribe vs. faster-whisper, English clip

```shell
movie-subtitles --input samples/en_clip.mp4 --engine local          # writes en_clip.srt
movie-subtitles --input samples/en_clip.mp4 --engine elevenlabs     # writes en_clip.srt (overwrite; rename to compare)
```

| Metric | faster-whisper `large-v3` | ElevenLabs Scribe |
|---|---|---|
| Transcript accuracy (impression) | TBD — not yet run | TBD — not yet run |
| Wall-clock time | TBD — not yet run | TBD — not yet run |
| Rough cost | TBD — not yet run (local, compute only) | TBD — not yet run |

### ASR: Scribe vs. faster-whisper, Danish-audio clip (the role-relevant result)

```shell
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --engine local
movie-subtitles --input samples/da_clip.mp4 --audio-lang da --srt-lang en --engine elevenlabs
```

| Metric | faster-whisper `large-v3` | ElevenLabs Scribe |
|---|---|---|
| Danish transcript accuracy (impression) | TBD — not yet run | TBD — not yet run |
| Wall-clock time | TBD — not yet run | TBD — not yet run |
| Rough cost | TBD — not yet run | TBD — not yet run |

### Danish TTS voice quality (en→da dub)

```shell
movie-subtitles --input samples/en_clip.mp4 --engine elevenlabs --dub
```

TBD — not yet run. Listen to `en_clip.dubbed.mp4` and note naturalness, pronunciation,
and any audible rate-clamp artefacts once produced.

### Translation budget adherence

How closely `LLMTranslate` (Claude) actually respects the `budget_chars` target passed in
the prompt (see [Timing drift](#timing-drift)) — e.g. the distribution of `len(actual) /
budget_chars` across the test clip's segments.

- English clip (~60s), en→da: **TBD — not yet run**

### Unfittable-segment count

Reported directly by the run's log line
(`movie_subtitles/dub.py:synthesise_track`, `"N segment(s) could not be fitted within the
rate clamp"`) for the `--dub` run above.

- English clip (~60s), en→da: **TBD — not yet run**

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

## What's not implemented

- **Burned-in per-frame subtitles.** An earlier version of this README described adding
  subtitles to each movie frame. That was never built — `.srt` (and, with `--dub`, a
  muxed audio track) are the only outputs.
- **Underrun correction** in the timing-drift strategy (see above).
- Speaker diarisation / distinct voices per speaker in the dub.
- Realtime/streaming Scribe or TTS.
- A config-file or env-var layer beyond the two required API keys — everything else is a
  CLI flag.

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
