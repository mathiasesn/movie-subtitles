"""Speaker labelling: merge pyannote.audio diarisation turns onto ASR segments.

Feeds issue #27 (speaker diarisation for `--asr-engine local`/`openai`, neither of
which diarizes natively the way `--asr-engine elevenlabs`'s Scribe does via its own
per-word `speaker_id` -- see `providers/elevenlabs.py:_group_words`, which is a
deliberately separate implementation, not something this module should be unified
with). `cli.py` runs `providers/pyannote_.py:Diarize` once, up front, over the whole
file to get a time-sorted `list[Turn]`; this module's `label_segments()` then merges
that list onto each ASR `Segment` as it streams past, by temporal overlap -- splitting
a segment into one sub-segment per contiguous speaker run when word timestamps are
available, or labelling it whole by majority overlap when they are not.

Non-obvious invariants:

- **Stays lazy.** `label_segments()` is a generator that pulls one segment at a time
  and yields as it goes; it never does `list(segments)` or otherwise materializes the
  underlying ASR generator. The caller's `for segment in tqdm(segments, ...)` loop is
  what drives ASR inference and the progress bar, and that must keep working exactly
  as it does when diarisation is skipped entirely.
- **The turn cursor only advances.** `turns` is sorted by start, and spans (word or
  cue) are visited in ascending time order as segments and their words stream past in
  order -- so a single index into `turns` that only ever moves forward is enough:
  once a turn ends at or before the current span's start it can never overlap a later
  span either, and is skipped for good rather than rescanned. This is what keeps
  labelling O(words + turns) instead of O(words * turns) on a feature-length film's
  worth of words.
- **Unlabelled words are smoothed into neighbours, not left to form their own runs.**
  A word that overlaps no diarisation turn (a silence gap, or a diarizer miss) would,
  left as-is, form a one-word run and fragment a cue that should stay whole (e.g.
  splitting one sentence into three sub-segments around a single unlabelled word in
  the middle). Smoothing it onto an adjacent labelled word instead keeps such gaps
  from fragmenting cues that a human would read as one continuous turn. An entire
  segment with no labelled word at all is a different, genuine case -- "diarisation
  found nothing here" -- and must stay entirely `speaker=None` rather than being
  smoothed into a guess.
"""

from collections.abc import Iterable, Iterator

from movie_subtitles.providers.base import Segment, Turn


class _TurnCursor:
    """A forward-only cursor into a start-sorted `list[Turn]`.

    Holds the "furthest turn that could still matter" index as instance state, so
    repeated `speaker_for()` calls over spans visited in ascending time order stay
    O(words + turns) in total rather than rescanning `turns` from the start on every
    call (see the module docstring).
    """

    def __init__(self, turns: list[Turn]) -> None:
        self._turns = turns
        self._idx = 0

    def speaker_for(self, start: float, end: float) -> str | None:
        """Return the speaker with the greatest overlap with `[start, end)`, or `None`."""
        turns = self._turns
        while self._idx < len(turns) and turns[self._idx].end <= start:
            self._idx += 1

        # Fast path: track the best (speaker, overlap) seen so far in two local
        # variables instead of allocating a dict -- the common case is exactly one
        # overlapping turn. Only fall back to a dict once a second *distinct* speaker
        # overlaps this span, so ties are still resolved exactly the way
        # `max(overlaps, key=overlaps.get)` would: the first speaker encountered with
        # the maximum overlap wins (Python's `max` keeps the first-seen maximum on a
        # tie, and a dict preserves insertion order).
        best_speaker: str | None = None
        best_overlap = 0.0
        overlaps: dict[str, float] | None = None

        idx = self._idx
        while idx < len(turns) and turns[idx].start < end:
            turn = turns[idx]
            overlap = min(turn.end, end) - max(turn.start, start)
            if overlap > 0:
                if overlaps is not None:
                    overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap
                elif best_speaker is None or turn.speaker == best_speaker:
                    best_speaker = turn.speaker
                    best_overlap += overlap
                else:
                    # A second distinct speaker overlaps this span: fall back to a
                    # dict so a speaker's overlap can be split across multiple turns
                    # (e.g. two short turns from the same speaker either side of a
                    # brief interruption) and still accumulate correctly.
                    overlaps = {best_speaker: best_overlap, turn.speaker: overlap}
            idx += 1

        if overlaps is not None:
            return max(overlaps, key=overlaps.get)
        return best_speaker


def _labelled_segment(segment: Segment, speaker: str | None, next_id: int) -> Segment:
    """Build a whole (unsplit) copy of `segment` carrying `speaker`, with a fresh id.

    Shared by the no-words path and the single-run path in `_split_segment_by_speaker`
    -- both keep the original span/text intact rather than reconstructing text from
    word tokens, which would risk drifting spacing/punctuation for no benefit.
    """
    return Segment(
        id=next_id,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        words=segment.words,
        speaker=speaker,
    )


def _split_segment_by_speaker(
    segment: Segment, cursor: _TurnCursor, next_id: int
) -> tuple[list[Segment], int]:
    """Split `segment` into one sub-segment per contiguous speaker run.

    Deliberately separate from providers/elevenlabs.py:ScribeTranscribe._group_words
    (Scribe's native per-word speaker_id split) -- do not unify the two.

    Uses word-level overlap against `cursor`'s turns when `segment.words` is
    populated (true for every local/openai segment now that both request word
    timestamps), which is what makes splitting on speaker change expressible at all.
    A segment with no words is left whole, labelled with its single majority speaker.
    Sub-segments -- and an unsplit segment -- get a fresh id from `next_id`, since a
    split segment no longer maps 1:1 onto the original ASR id.
    """
    if not segment.words:
        speaker = cursor.speaker_for(segment.start, segment.end)
        return [_labelled_segment(segment, speaker, next_id)], next_id + 1

    word_speakers: list[str | None] = [
        cursor.speaker_for(word.start, word.end) for word in segment.words
    ]

    # Smooth gap words onto a neighbouring label (see module docstring) -- but only
    # if at least one word in the segment got a label at all; otherwise every word
    # stays None, which is the genuine "diarisation found nothing here" case.
    if any(speaker is not None for speaker in word_speakers):
        # Forward-fill in place: after this pass, the only remaining `None`s are a
        # leading prefix (nothing before the first labelled word could be filled
        # going forward), since every word from the first labelled one onward is
        # covered by `last_label`.
        last_label: str | None = None
        for i, speaker in enumerate(word_speakers):
            if speaker is None:
                word_speakers[i] = last_label
            else:
                last_label = speaker
        # Fill the leading prefix (if any) with the first labelled value -- no
        # general backward scan needed, since the forward pass already handled
        # everything after the first label.
        first_label = next(s for s in word_speakers if s is not None)
        for i, speaker in enumerate(word_speakers):
            if speaker is None:
                word_speakers[i] = first_label
            else:
                break

    runs: list[tuple[int, int]] = []
    run_start = 0
    for i in range(1, len(segment.words) + 1):
        if i == len(segment.words) or word_speakers[i] != word_speakers[run_start]:
            runs.append((run_start, i))
            run_start = i

    if len(runs) == 1:
        return [_labelled_segment(segment, word_speakers[0], next_id)], next_id + 1

    out: list[Segment] = []
    for run_start_i, run_end_i in runs:
        run_words = segment.words[run_start_i:run_end_i]
        out.append(
            Segment(
                id=next_id,
                start=run_words[0].start,
                end=run_words[-1].end,
                # Concatenate raw tokens rather than joining with a space: both
                # faster-whisper and the OpenAI API already carry their own leading
                # whitespace on each word where the language uses it (e.g. " world"),
                # so plain concatenation preserves original spacing for space-delimited
                # languages while staying correct for CJK/Thai, which a " ".join would
                # corrupt by injecting spurious spaces.
                text="".join(w.text for w in run_words).strip(),
                words=run_words,
                speaker=word_speakers[run_start_i],
            )
        )
        next_id += 1

    return out, next_id


def label_segments(segments: Iterable[Segment], turns: list[Turn]) -> Iterator[Segment]:
    """Lazily label (and, on a speaker change, split) each segment as it is consumed.

    Stays lazy -- pulls from `segments` one at a time and yields as it goes -- so the
    caller's `for segment in tqdm(segments, ...)` loop still drives ASR inference and
    the progress bar; nothing here materializes the underlying generator.
    """
    cursor = _TurnCursor(turns)
    next_id = 0
    for segment in segments:
        subs, next_id = _split_segment_by_speaker(segment, cursor, next_id)
        yield from subs
