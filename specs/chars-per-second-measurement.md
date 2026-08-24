# Chars-per-second measurement (issue #6)

Record of the measurement that replaced `_CHARS_PER_SECOND = 15.0` (an unmeasured
assumption) with `cli.py`'s `_EXPANSION_RATIO` table and `_TARGET_SPEAKABLE_CPS`. See
`specs/measure-chars-per-second-budget.md` for the plan this executed.

## Sample

- `data/paw-patrol-the-dino-movie-clip.webm` — 45s, av1/opus, stereo 48kHz.
- `data/paw-patrol-the-dino-movie.webm` — 137s, the trailer the original spec named
  as the wrong sample on its own, included anyway (see Method below) to pool more
  segments.
- Language pair: `en -> da` only.
- Engines: `--asr-engine elevenlabs`, `--translation-engine anthropic`,
  `--tts-engine openai` (resolves to `tts-1`), `--voice-match off`.
- Segment counts: 17 translated segments from the 45s clip, 52 from the 137s
  trailer — 69 translated segments pooled across both files. The two duration/rate
  medians below draw on different subsets of those 69: the tts-1 rate uses all 69
  (every synth pass=0 rate=1.0 measurement had a usable clip); the source-rate
  median drops to n=68 because one segment's `translate` line had a zero-length
  slot or zero source characters and was filtered before computing chars/slot-second
  (see Measured medians).

## Method

Two instrumented `--dub` runs were made over both files with `MOVIE_SUBTITLES_LOG_LEVEL=DEBUG`,
which raises `__init__.py`'s stdout logging so the permanent `[measure]` DEBUG lines
land in the run log: `cli.py`'s translation loop emits slot duration plus source and
translated character counts per segment, and `dub.py`'s `_synthesise_and_measure`
emits the measured TTS clip duration and applied rate per synthesis pass. Both lines
carry the segment id, so a throwaway parse script could join them.

Fitting the *unconstrained* expansion ratio (translated chars / source chars with no
budget applied) needed a second pass: the 69 segments' own translations were produced
under the old `_CHARS_PER_SECOND`-derived budget, so their target lengths were already
compressed toward that budget and could not be used to fit a ratio without circularity.
The fix was to re-translate with `budget_chars=None` (unconstrained) and compare
against real source text — but the 137s trailer's *own* ASR source lines were not
separately recoverable at that point, so only the 45s clip's English source text
(recovered from the ElevenLabs ASR transcript) was re-translated unconstrained. This
gives the ratio measurement a smaller n (16) than the two duration/rate measurements
(68–69), which draw on both files.

## Measured medians

- Source English rate: **15.99** chars per slot-second (n=68 of the 69 pooled
  segments, both files — one segment was excluded by the `slot > 0 and
  src_chars > 0` filter, source text length over each segment's ASR slot
  duration).
- tts-1 speaking Danish: **19.822** chars/second at rate 1.0 (n=69, both files,
  from `dub.py`'s measured clip duration at the initial 1.0x pass).
- Unconstrained (`budget_chars=None`) en->da expansion ratio: **1.036**
  (n=16, translated chars / source chars, 45s clip only, from ElevenLabs-recovered
  English source text — see Method above for why the 137s trailer's translate lines
  could not be used).

## Diagnosis

`15.0` sat close to the measured *source English* rate (15.99), but `_budget_chars`
applied it as a character budget for the *Danish target* — and Danish expands
relative to English, so it was the wrong axis. Every cue was budgeted short:
translations came back at 0.78–0.81x source character count, and synthesised clips
ran 0.56–0.63x source speech duration. The `dub.py` rate clamp (`[0.9, 1.15]`)
saturated on 7 of 10 groups in the 45s clip without closing the gap — a structural
shortfall in the budget, not a speaking-rate problem the clamp could absorb.

## What this does and does not fix

`_budget_chars` no longer treats the expansion-ratio term as primary and the
duration term as a cap on it. Both terms are independently measured lower
bounds on how much room a cue needs — the ratio term from source-text
expansion, the duration term from how many characters the slot can speak at
the measured target rate — and `_budget_chars` now takes `max()` of the two,
not `min()`. Both constants are still measured rather than assumed, which is
what issue #6 asked for, but the fix should not be read as closing the
observed clip-length shortfall, because it does not — it substantially
narrows it (see Verification below) while leaving a structural shortfall in
place for at least one group.

Switching from `min()` to `max()` raised the budget on every one of the 68
logged segments in the offline recompute (see Verification): median budgeted
rate moved from 15.99 to 19.65 chars/slot-second, a **median budget increase
of +26.0%**. That "raised on 100% of segments" figure is true by
construction — `max()` is never below `min()`, and the two terms are (in this
sample) never exactly equal — so it carries no evidential weight on its own;
the median increase is the number that matters.

Even with the larger budget, a translation landing on it does not fill the
slot when spoken by tts-1: the paid verification run below found the median
translated/source character ratio rose from 0.854 to 0.935 and median clip
duration against source speech at 1.0x rose from 0.629 to 0.764 — real,
substantial improvement, but still short of 1.0. Part of the remaining gap is
that budget fill (how much of the larger budget the translator actually used)
*dropped*, from 87% to 74% — the translator is not consuming all the new
headroom. The rest is that tts-1 simply speaks Danish faster than this
sample's English actors deliver their lines, which no budget-derivation change
can fix; the `[0.9, 1.15]` `_MIN_RATE`/`_MAX_RATE` clamp in `dub.py` remains
the last line of defense and still saturates on the worst group.

Per-group drift is **not reproducible enough to carry a claim**, and an earlier
version of this document over-read it. A replicate run under identical code
(see Replication below) put Group 1 at -1.52s and clamp-saturated, where the
first run had it at +0.33s and within tolerance -- so "groups within tolerance
went from 1 of 3 to 2 of 3" held in one run and not the other. Group 1 sits
near the tolerance boundary and lands on either side of it depending on how the
ASR segments that pass.

What *does* hold across both runs is Group 0: -5.24s and -4.54s final drift,
clamp-saturated in both, against -5.81s under the old `min()` derivation. It is
**still a structural shortfall, not a speaking-rate problem** the clamp can
absorb -- the gap narrowed, it did not close. Group 2 stayed within tolerance
in both runs.

The honest follow-up is a further increase to `_EXPANSION_RATIO["da"]` and/or
`_TARGET_SPEAKABLE_CPS` (and/or a translator-side steer toward denser
phrasing, given budget fill fell rather than rose), informed by Group 0's
continued clamp saturation, not another turn of the rate clamp — `_MIN_RATE =
0.9` is already too close to 1.0 to meaningfully slow tts-1's Danish output
into this sample's English pacing.

## Verification

Two steps verified the `min()` -> `max()` change, one free and one paid.

**Step 1 — free offline recompute.** The 68 logged segments' already-captured
source chars, slot durations, and the constants above were recomputed with the
new `max()` formula, entirely offline (no API calls, no spend). Median
budgeted rate moved from 15.99 to 19.65 chars/slot-second (+26.0% median
increase), and segments budgeted at or above the 15.99 chars/slot-second
source speaking rate moved from 34/68 to 68/68. As noted above, that
"100% raised" figure follows automatically from `max()` never being below
`min()`; the median increase is the figure with evidential weight. This
recompute was the spend gate for step 2: only after it confirmed the formula
behaves as intended did the paid run proceed.

**Step 2 — paid instrumented `--dub` run.** A real run over
`data/paw-patrol-the-dino-movie-clip.webm` (en -> da, `--asr-engine
elevenlabs`, `--translation-engine anthropic`, `--tts-engine openai`,
`--voice-match off`) was compared against the recorded post-#30 baseline:

| metric | baseline `min()` | new `max()` |
| --- | --- | --- |
| median budget (chars) | 23 | 32 |
| median translated chars | 18 | 22 |
| median translated/source char ratio | 0.854 | 0.935 |
| median clip duration / source speech at 1.0x | 0.629 | 0.764 |
| median % of budget the translator actually used | 87% | 74% |

**Caveat 1 -- the comparison is not a controlled A/B, and this was measured,
not just suspected.** The ElevenLabs ASR resegments on every run: 18 comparable
translate records versus the baseline's 20, and Group 1 had 6 segments here
versus 7 in the baseline. A replicate run (see Replication below) quantified
what that costs -- the aggregate medians reproduced almost exactly, but the
per-group drift figures moved by more than the improvement attributed to the
change. **Treat the per-group numbers in this table as indicative only; the
aggregate rows are the ones that carry weight.**

**Caveat 2 — budget fill dropped.** Median budget fill fell from 87% to 74%:
the translator does not consume all the new headroom the larger budget makes
available, so the remaining shortfall is not purely a budgeting problem — part
of it is that tts-1 speaks Danish faster than the source actors deliver their
lines, which no budget change can fix.

## Replication

The step 2 run was repeated under identical code to separate the effect of the
`max()` change from ordinary run-to-run variation (ASR resegmentation,
translator and TTS nondeterminism). This matters because the paid runs are not
a controlled A/B -- the ASR resegments on every run.

| metric | baseline `min()` | `max()` run A | `max()` run B |
| --- | --- | --- | --- |
| median translated/source char ratio | 0.854 | 0.935 | 0.935 |
| median clip duration / source speech at 1.0x | 0.629 | 0.764 | 0.736 |
| median % of budget used by the translator | 87% | 74% | 71% |
| Group 0 final drift | -5.81s | -5.24s | -4.54s |
| Group 1 final drift | -2.69s | +0.33s | -1.52s |
| groups within tolerance | 1 of 3 | 2 of 3 | 1 of 3 |

**The aggregate figures replicate; the per-group figures do not.** The
translated/source ratio returned 0.935 to three decimals in both runs, and
clip-duration-against-source-speech (0.736-0.764) sits well clear of the
baseline's 0.629, so the improvement those numbers describe is real. The
per-group drift figures move by more than the improvement attributed to the
change -- Group 1 swings from within tolerance to clamp-saturated between two
runs of the same code -- so no claim should rest on them. Group 0's shortfall
is the exception: it is clamp-saturated in every run measured, baseline and
both `max()` runs.

Aggregates here are medians over 18 segments; the group figures are 3 data
points. That is the whole difference.

## Derived constants

```python
_EXPANSION_RATIO: dict[str, float] = {
    "da": 1.036,
    "default": 1.1,
}
_TARGET_SPEAKABLE_CPS = 19.822
```

`_budget_chars(start, end, text, output_lang)` returns
`max(int(max(len(text) * ratio, duration * _TARGET_SPEAKABLE_CPS)), 1)` — the
expansion-ratio term and the duration-derived term are both independently
measured floors, and the larger one wins; neither is primary and neither caps
the other. Measured across the 68 pooled segments used for the offline
recompute, the duration-derived floor is the larger (binding) term on **55 of
68** segments (81%) — so under `max()` it is the duration term that drives the
budget in the large majority of cases, and the expansion-ratio term binds on
the remaining 13. This is the exact inverse of the old `min()` derivation,
where the duration term bound on only those same 13 segments (19%) as a cap.
That inversion is the change: the term that governs whether a cue can fill its
slot went from being the rarely-active ceiling to being the usually-active
floor.

## Status: provisional

This sample is ~3 minutes of short-utterance, music-heavy animated trailer, not
feature-length dialogue. Short exclamatory lines have a different chars/second and
expansion profile from ordinary conversational dialogue, so both constants above
are expected to need a re-fit once feature-length dialogue material is available.
`_EXPANSION_RATIO["default"]` in particular is an unmeasured, conservative
placeholder for any language pair other than `en -> da`.

## Reproduction

```shell
MOVIE_SUBTITLES_LOG_LEVEL=DEBUG uv run movie-subtitles \
  --input <file> --audio-lang en --srt-lang da \
  --asr-engine elevenlabs --translation-engine anthropic --tts-engine openai \
  --dub --voice-match off
```

## Raw records

- `data/measurements/clip-translate-synth.measure.log` -- 51 `[measure]` lines from
  the 45s clip run (translate + synth measurements).
- `data/measurements/trailer-translate-synth.measure.log` -- 153 `[measure]` lines
  from the ~3-minute trailer run (translate + synth measurements).
- `data/measurements/verify-clip.measure.log` -- 57 `[measure]` lines from the
  verification run over the clip, re-checking the derived constants above.
- `data/measurements/step2-max-budget.measure.log` -- 50 `[measure]` lines from
  the step 2 paid verification run under the new `max()` derivation, the source
  of the Verification table above.
- `data/measurements/step2-max-budget-repeat.measure.log` -- 53 `[measure]` lines
  from the replicate of the step 2 run under identical code, the source of the
  Replication table above.
- `data/measurements/measure.py` -- the script used to parse the first three logs
  above and compute the medians cited in this document (source chars/slot-second,
  tts-1 Danish chars/sec, unconstrained en->da expansion ratio).

These are the filtered `[measure]` DEBUG lines emitted by `cli.py`/`dub.py`
(vendor SDK HTTP chatter stripped), not the complete run logs.
