# Chars-per-second measurement (issue #6)

Record of the measurement that replaced `_CHARS_PER_SECOND = 15.0` (an unmeasured
assumption) with `cli.py`'s `_EXPANSION_RATIO` table and `_MAX_SPEAKABLE_CPS`. See
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

Both constants are now measured rather than assumed, which is what issue #6 asked
for — but the fix should not be read as closing the observed 0.56–0.63x clip-length
shortfall, because it largely does not.

The old budget was `15.0 * duration`. At the measured source rate of 15.99
chars/slot-second, that is `duration ≈ src_chars / 15.99`, so the old budget was
approximately `15.0 * src_chars / 15.99 ≈ 0.94 * src_chars`. The new primary term is
`1.036 * src_chars` — about a **10% budget increase** against a shortfall that was
running **~40%** (translations landing at 0.78–0.81x source character count,
clips at 0.56–0.63x source speech duration).

Worse, even a translation that lands exactly on the new 1.036 budget does not
fill its slot. Spoken by tts-1 at the measured median rate of 19.822 Danish
chars/second, `1.036 * src_chars` characters take `1.036 * src_chars / 19.822`
seconds — against a slot of `src_chars / 15.99` seconds. The ratio of the two is
`(1.036 / 19.822) / (1 / 15.99) ≈ 0.836`: **about 84% of the slot even in the
ideal case**, before any translator under- or over-shoot. The residual gap is not
a budgeting error — it is that tts-1 simply speaks Danish faster than this
trailer's English actors deliver their lines, and no character-budget derivation
can correct a speaking-rate mismatch between two different voices in two
different languages. The `[0.9, 1.15]` `_MIN_RATE`/`_MAX_RATE` clamp in `dub.py`
cannot absorb an 84%-of-slot ideal case either: slowing tts-1 to `0.9x` narrows,
but does not close, that gap, and clamping further would trade one problem
(clips too short) for another (audibly slow speech).

None of this means the change was wasted: a confirmed post-fix verification run
on the 45s clip showed real, if partial, improvement — group 0's drift went from
-7.19s to -5.81s, group 1's from -2.82s to -2.69s, and a third group that had
been out of tolerance came back within it. The gap narrowed; it did not close.

The honest follow-up is a further increase to `_EXPANSION_RATIO["da"]` (and/or a
translator-side steer toward denser phrasing) informed by the 84%-of-slot ceiling
above, not another turn of the rate clamp — `_MIN_RATE = 0.9` is already too
close to 1.0 to meaningfully slow tts-1's Danish output into this trailer's
English pacing.

## Derived constants

```python
_EXPANSION_RATIO: dict[str, float] = {
    "da": 1.036,
    "default": 1.1,
}
_MAX_SPEAKABLE_CPS = 19.822
```

`_budget_chars(start, end, text, output_lang)` returns
`max(int(min(len(text) * ratio, duration * _MAX_SPEAKABLE_CPS)), 1)` — the ratio
against source text length is the primary term; the duration-derived cap only
guards a cue whose source text already overruns its own span. Measured across the
69 pooled segments, the duration cap binds on **13 of 68** segments (19%) — the
ratio, not the cap, drives the budget in the large majority of cases.

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
- `data/measurements/measure.py` -- the script used to parse the three logs above
  and compute the medians (source chars/slot-second, tts-1 Danish chars/sec,
  unconstrained en->da expansion ratio) cited in this document.

These are the filtered `[measure]` DEBUG lines emitted by `cli.py`/`dub.py`
(vendor SDK HTTP chatter stripped), not the complete run logs.
