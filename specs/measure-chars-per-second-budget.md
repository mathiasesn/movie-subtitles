# Back `_CHARS_PER_SECOND` with a measurement (issue #6)

## Problem / Why

`cli.py:_CHARS_PER_SECOND = 15.0` is an assumption its own comment admits is
unmeasured. `_budget_chars(start, end)` turns it into the character budget that
`providers/prompt.py` writes into the translation prompt, making it step 1 of the
two-step timing-drift strategy (budget the translation, then nudge TTS rate).
If it is wrong, step 1 systematically hands step 2 clips of the wrong length and
the deliberately tight `[0.9, 1.15]` rate clamp in `dub.py` silently absorbs the
error until it runs out of room. Nothing fails loudly; the dub just drifts.

Danish (the repo's default-ish target) runs longer than English for the same
content, so an English-ish speaking rate is expected to under-budget `en -> da`.
Whether `15.0` over- or under-shoots has never been measured.

Issue #5 (degenerate `whisper-1` timings) is now closed — the run warns on the
fixed-cadence signature — so measurement is unblocked provided it is done on an
ASR path with trustworthy segment boundaries (`elevenlabs`, or `local`).

## Goals

- Produce a recorded measurement relating slot duration, translated character
  count, and measured TTS clip duration at speed 1.0, from real runs.
- Replace `15.0` with a value (or derivation) backed by that measurement.
- Remove the "assumption, not a measured value" comment and state the sample the
  value came from: source material, language pair, segment count.
- Keep `ARCHI.md`, `AGENTS.md` and the README consistent if the derivation shape
  changes (e.g. per-target-language table instead of one global constant).

## Non-goals

- Changing the `[0.9, 1.15]` rate clamp or the correction-pass loop.
- Adding a length-control lever to `--translation-engine local` (MADLAD400
  ignores `budget_chars`; out of scope).
- Any change to ASR segmentation itself.
- Deriving the budget for `--translation-engine local`.

## Constraints

- `dub.py` already probes each clip's duration; the measurement should record
  what a run already computes rather than build new synthesis machinery.
- Providers stay orchestration-free; instrumentation belongs where the data
  already lives (`dub.py` producing it, `cli.py` deciding whether to write it).
- Ruff format + check are the only CI gate and must pass.
- Measurement is DEBUG logging only: no new CLI flag, no report file format, no
  new module. The lines are permanent (they land on main), the analysis script is
  not. Logging config lives in `__init__.py`, which sets stdout INFO -- so the
  measurement run must raise the level itself; how it does so is part of the work.
- Defaults duplicated between argparse and callable signatures must move together.
- Measurement runs cost real money (ElevenLabs TTS characters dominate).
- Sample: short animated-trailer material only (see Source file), `en -> da` only,
  `--asr-engine elevenlabs`. Outcome is therefore a value defensible for `en -> da`;
  other pairs stay explicitly untuned.
- Source file: `data/paw-patrol-the-dino-movie-clip.webm` -- **45s**, av1/opus,
  stereo 48kHz. The only other candidate on disk is
  `data/paw-patrol-the-dino-movie.webm`, the 137s trailer issue #6 explicitly
  rejects as a sample.

  This is **not** the feature-length dialogue-dense sample originally planned, and
  it is close to the material issue #6 warns against fitting on. Decision: measure
  both files anyway and land the result as an **explicitly provisional** value,
  named as such in the code comment and the artefact. Issue #6's acceptance
  criteria are satisfied literally (a recorded measurement, sample named, no
  "unmeasured assumption" comment), but the value must be re-fitted when
  feature-length dialogue material exists.

## Proposed approach

Full task: I build the instrumentation, execute the measurement runs myself
against real media using the repo's `.env` keys (real spend on TTS + ASR +
translation), fit the value, and land it with the docs.

1. Emit the per-segment measurement at DEBUG level from where the numbers
   already exist -- `dub.py` (measured clip duration at rate 1.0, applied rate,
   resulting drift, slot duration) and `cli.py` (source and translated char
   counts) -- as one line per segment, keyed by segment id so the two sides can
   be joined. No new CLI flag, no new file format; the run log is the artefact,
   and a small throwaway parse script does the join and the fit.
2. Run it end to end over both `data/paw-patrol-the-dino-movie-clip.webm` (45s)
   and `data/paw-patrol-the-dino-movie.webm` (137s), `en -> da`,
   `--asr-engine elevenlabs`, with a TTS engine that honours `speed` -- pooling
   both gives more segments than either alone, and the 137s file already has a
   known-plausible ElevenLabs-ASR baseline.
3. Fit the **expansion ratio** `translated_chars / source_chars` for `en -> da`
   from the run, and validate it against measured clip durations at rate 1.0
   (does a ratio-budgeted translation actually fit its slot?).
4. Replace duration-derived budgeting:
   `_budget_chars(start, end, text, lang) = min(len(text) * ratio[lang],
   duration * max_speakable_cps)`. The ratio is the measured per-target-language
   expansion factor (with a documented default for unmeasured languages); the
   duration term survives only as an **upper cap**, guarding the case where ASR
   hands back a cue whose source text already overruns its own span. Both the
   ratio and `max_speakable_cps` come from the same run -- `max_speakable_cps` is
   the observed upper end of source chars per slot-second, not a reused 15.0.
5. Land it with the sample named in the comment, and update `ARCHI.md`,
   `AGENTS.md` and the README, all of which currently describe a single global
   duration-derived constant.

## Acceptance criteria

- A measurement artefact exists in the repo (e.g. `specs/`), naming source
  material, language pair, ASR engine, TTS engine and segment count, and stating
  plainly that the sample is short trailer material and the fit is provisional.
- `_CHARS_PER_SECOND` is gone; the expansion ratio and the cap's max-speakable
  chars/second are both derived from that artefact, and their comments name the
  sample instead of claiming the values are assumptions -- and say the fit is
  provisional pending feature-length dialogue material.
- `_budget_chars` takes the segment's source text and the target language, and
  applies the duration cap; its call site in `create_subtitles` is updated.
- If the derivation shape changed, `ARCHI.md` and `AGENTS.md` timing-drift
  sections describe the new shape.
- `uv run --only-dev ruff format --check .` and `ruff check .` pass.

## Open questions

_None._

## Risks

- The duration cap reintroduces a rate constant by the back door; if it is set too
  low it silently becomes the binding term for every segment and the change is a
  retuned `_CHARS_PER_SECOND` wearing a new name. Verify from the data how often
  the cap actually binds, and record it.
- The expansion ratio assumes the source text already fits its own slot. Where ASR
  segmentation is loose (a cue whose text overruns its span), the ratio propagates
  that error instead of correcting it -- the duration-derived budget did not.
- **Sample bias is the headline risk, accepted knowingly.** ~3 minutes of
  fast-cut, music-heavy animated trailer in one language pair. Short exclamatory
  utterances ("Ah!", "Kom her, skonhed.") have a very different chars/second and
  expansion profile from ordinary conversational dialogue, so the fitted ratio may
  not survive contact with a real film. Mitigation is disclosure, not accuracy:
  the value ships labelled provisional, and issue #6 should note that a re-fit is
  owed.
- Translator compliance: the budget is a prompt request, not a hard cap; measured
  chars/second conflates speaking rate with how well the LLM obeys the budget.
- Cost: enough material to conclude from may be expensive to synthesise.
