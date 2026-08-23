# Warn when ASR segment timings look degenerate

Resolves https://github.com/mathiasesn/movie-subtitles/issues/5.

## Problem / Why

`--asr-engine openai` can produce an `.srt` whose cue timings are wrong, with nothing
in the run indicating anything went amiss. The `.srt` is this tool's primary output,
so the failure is silent and lands directly in the deliverable.

The cause is a confirmed vendor limitation, not a bug in this repo: `whisper-1`'s
segment timestamps degrade to uniform 1.000s spans on music-heavy or dialogue-sparse
audio (diagnostic of 2026-08-19, `specs/fix-smoke-run-findings.md`: 98 of 106 segments
exactly 1.000s, chained contiguously, on a ~137s trailer; the distortion is present in
the raw API response before `_yield_segments` sees it). This is documented in
`OpenAITranscribe`'s docstring, `AGENTS.md`, and `ARCHI.md` — but documentation only
helps someone who already went looking. The run itself should say something.

## Goals

1. A run whose ASR timings are degenerate emits a clear `logger.warning` naming the
   likely cause and the `--asr-engine elevenlabs` workaround.
2. A normal run with genuinely varied timings does not warn — no false positives on
   ordinary dialogue, which does contain occasional 1.000s cues.
3. The lazy-`segments` constraint stays intact: the check accumulates cheap running
   aggregates inside the existing segment loop and never materializes the iterable.

## Non-goals

- Cancelling or short-circuiting a paid run mid-stream (issue approach 2's real
  selling point). The warning is after-the-fact for the current run.
- Changing `dub.py`'s per-segment `_HOPELESS_RATE` guard, which already handles the
  dub-side symptom and stays as-is.
- Tuning `_CHARS_PER_SECOND` — blocked on the same untrustworthy timings, per the
  issue's notes.
- Gathering a second real-world sample to validate the threshold (the issue itself
  flags this as wanted; it requires another degenerate input, which we cannot
  synthesize on demand). The threshold is justified analytically below instead.
- Adding tests / pytest. The repo has none; introducing a framework is its own task.
- Amending the lazy-`segments` constraint (issue approach 3). Not needed: approach 1
  respects it as written, so `AGENTS.md`/`ARCHI.md` need no constraint edit — only a
  mention of the new warning where the constraint and the openai limitation are
  discussed.

## Constraints

- **Do not materialize `segments` into a list** (`AGENTS.md`, `ARCHI.md`). ASR
  providers return a lazy generator; the loop is what drives inference and the
  progress bar. This is what rules out a simple "collect all durations, then analyse"
  check and forces the running-aggregates shape.
- Provider-agnostic check living in `cli.py`: orchestration belongs there, and
  providers know nothing about SRT or about each other. The warning fires for any ASR
  engine whose output matches the signature, not just `openai`.
- Warning severity, not a hard error: degenerate timings still yield usable *text*,
  and a user who only wants a transcript must not be blocked.
- `uv` owns the environment; CI gate is only `ruff format --check .` + `ruff check .`
  (line length 100, rules `E,F,I,UP,B,SIM`).
- Log, don't print: the warning goes through the `cli` module logger at WARNING level,
  consistent with the existing `--dub` + local-translator warning in
  `create_subtitles()`.

## Proposed approach

Issue approach 1: end-of-run warning from running aggregates accumulated inside the
existing segment loop in `cli.py:create_subtitles()`.

**Detection rule.** Count every segment the loop sees (including ones the translator
returns empty text for — their timings are just as much evidence of the vendor
degradation, and skipping them would undercount exactly the runs we want to catch) and
tally how many have a duration within ±0.05s of 1.000s. After the loop, warn iff:

    near-1s tally > 11  AND  near-1s tally / total segments > 0.5

- The absolute floor (>11, i.e. the smallest warning tally is 12) protects short
  genuinely-dialogue content — a 20-cue interview clip with 11 one-second cues is
  plausible dialogue, not a stuck cadence, and must not warn even though 11/20 = 0.55
  clears the proportion bar. (The floor sits just above that case; an earlier draft
  said >10, which it fails — the testable case is authoritative.)
- The proportion (>0.5) protects long dialogue: a feature film legitimately contains
  many 1.000s cues in absolute terms, but they stay a small fraction of thousands of
  varied cues.
- The observed degenerate run (98/106 ≈ 92%) clears both bars with a wide margin;
  ordinary dialogue, where 1.000s cues are occasional, clears neither.
- Both thresholds and the ±0.05s tolerance are module-level named constants in
  `cli.py` with a comment stating they are product policy tuned against the single
  observed sample — same posture as `_CHARS_PER_SECOND` and `dub.py`'s
  `_HOPELESS_RATE`.
- Duration is computed from the raw `segment.start`/`segment.end`, not the padded cue
  times — padding is a write-time `.srt` concern and must not feed detection.
- The tally uses durations rounded/compared with a tolerance (±0.05s) rather than
  exact equality, so a vendor jittering the cadence by a millisecond still matches.

**Warning text.** One `logger.warning` after the loop (before or after the `.srt`
write — the message is about the file just produced either way), naming: the observed
counts (`N of M segments`), the likely cause (`whisper-1`-class ASR timestamps
degrading to a fixed ~1s cadence on music-heavy or dialogue-sparse audio), the
consequence (`.srt` cue timings, and any `--dub` slots derived from them, are likely
wrong), and the workaround (`--asr-engine elevenlabs`). Provider-agnostic wording: it
describes the signature, not the engine name, though mentioning that this has been
observed with `--asr-engine openai`'s `whisper-1` is fine and useful.

**Docs.** `AGENTS.md` gains a sentence where the openai limitation is described noting
the run now self-warns; `ARCHI.md` likewise where the constraint/limitation is
discussed. The constraint text itself is unchanged — approach 1 respects it.

## Acceptance criteria

1. A run whose ASR timings match the degenerate signature (>50% near-1.000s durations
   over >11 segments) emits exactly one WARNING log line naming the likely cause and
   the `--asr-engine elevenlabs` workaround.
2. A normal run with genuinely varied timings emits no such warning — verified by
   construction (both thresholds must be cleared) and by reasoning about the
   false-positive cases above; no paid API call is needed to validate this.
3. `segments` is never materialized: the only new per-segment state is two counters.
   The constraint text in `AGENTS.md`/`ARCHI.md` stands unamended.
4. `uv run --only-dev ruff format --check .` and `uv run --only-dev ruff check .` both
   pass.
5. `--engine local` still runs without importing any vendor SDK (the check is pure
   arithmetic in `cli.py`, no new imports).

## Decisions

- Issue approach 1 (end-of-run warning from running aggregates) chosen over the
  sliding-window and materialize-for-eager-providers alternatives: zero architectural
  change, no false-positive risk on short clips, and the constraint docs stand
  unamended. The after-the-fact timing of the warning is the accepted cost
  (2026-08-23).
- The absolute floor is `_DEGENERATE_MIN_SEGMENTS = 11` with a strict `>` comparison
  (smallest warning tally: 12), not the originally drafted >10: the spec's own
  false-positive case (a 20-cue clip with 11 one-second cues, 11/20 = 0.55) would
  warn under >10, so the testable case won over the draft number (2026-08-23).

## Open questions

- [ ] None.

## Risks

- **The threshold is tuned against a single observed sample**, and that sample is a
  worst-case trailer. A milder real-world degradation (e.g. 40% stuck cadence) would
  pass silently. Accepted: the issue's own notes call for a second sample before
  trusting any threshold; the constants are named and commented so re-tuning is a
  one-line change when that sample appears.
- **A future provider could legitimately emit fixed-cadence cues** (e.g. a
  forced-alignment provider quantizing to 1s grid). The warning is provider-agnostic
  by design, so it would fire — arguably correctly, since such timings are equally
  suspicious downstream. If that ever becomes a false positive in practice, the rule
  can gain an engine carve-out; no such provider exists today.
- **The warning arrives after translation has been paid for.** Inherent to approach
  1; the issue explicitly lists this trade-off and approach 2's mid-run cancellation
  is a possible follow-up, not part of this task.
