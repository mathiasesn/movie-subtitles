import logging
import os
import shutil
import subprocess
import tempfile
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from tqdm.auto import tqdm

from movie_subtitles import configure_logging
from movie_subtitles.diarize import label_segments
from movie_subtitles.ffmpeg import extract_mono_wav
from movie_subtitles.providers.base import (
    AlignmentProvider,
    ASRProvider,
    Segment,
    TranslationProvider,
    TTSProvider,
    Turn,
)
from movie_subtitles.srt import format_block, pad_cue_end
from movie_subtitles.voices import (
    CLONE_MIN_SECONDS,
    CLONE_TARGET_SECONDS,
    VOICE_MATCH_MODES,
    resolved_voices,
)

logger = logging.getLogger("cli")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _positive_float(value: str) -> float:
    n = float(value)
    if n <= 0:
        raise ArgumentTypeError(f"must be > 0, got {n}")
    return n


def _unit_float(value: str) -> float:
    n = float(value)
    if not 0.0 <= n <= 1.0:
        raise ArgumentTypeError(f"must be between 0.0 and 1.0 inclusive, got {n}")
    return n


# Measured, not assumed. Sample:
# data/paw-patrol-the-dino-movie-clip.webm (45s) + data/paw-patrol-the-dino-movie.webm
# (137s), en -> da, --asr-engine elevenlabs, --translation-engine anthropic,
# --tts-engine openai (tts-1), --voice-match off; 17 + 52 = 69 translated segments.
# `_EXPANSION_RATIO["da"]` is the median unconstrained (budget_chars=None)
# tgt_chars/src_chars ratio over 16 segments re-translated from ElevenLabs-recovered
# English source text for the 45s clip (the 137s trailer's own translate lines were
# produced under the old 15.0-derived budget and are therefore compressed toward it,
# so they could not be used to fit an unconstrained ratio). `default` is a
# conservative >1.0 placeholder for any other target language, since en -> da is
# known to expand and no other pair has been measured.
# PROVISIONAL: the sample is ~3 minutes of short-utterance, music-heavy animated
# trailer, not feature-length dialogue -- re-fit when such material is available.
_EXPANSION_RATIO: dict[str, float] = {
    "da": 1.036,
    "default": 1.1,
}

# Measured median chars/second the TTS engine speaks the target language at, rate 1.0.
# A FLOOR in _budget_chars: how many characters of synthesised speech the slot can
# actually hold, so a sparse source cue is not budgeted below what its slot can carry.
# Keyed "<tts engine>:<target language>", like _EXPANSION_RATIO above is keyed by
# language, because speaking rate depends on both -- and because this term now binds
# on 55 of 68 measured segments, so an
# engine or language it was never measured on inherits it as the DOMINANT budget
# term, not a rarely-hit ceiling.
# Only "openai:da" is measured (69 synth pass=0 rate=1.0 measurements, same sample and
# run as _EXPANSION_RATIO). `default` deliberately repeats that same figure rather
# than inventing a second one: it is what every engine/language pair already got
# before this table existed, so the table changes no budget today. It exists to make
# the unmeasured extrapolation visible and to give the next measured fit somewhere to
# go. PROVISIONAL, same caveat as _EXPANSION_RATIO.
_TARGET_SPEAKABLE_CPS: dict[str, float] = {
    "openai:da": 19.822,
    "default": 19.822,
}

# Shared empty default for the `voices` kwarg -- a mutable literal default is a bugbear
# violation (B006), so this module-level constant stands in for it; never mutated.
_NO_VOICES: dict[str | None, str | None] = {}

# Degenerate-ASR-timings detection, all product policy tuned against the single
# observed sample (98 of 106 segments exactly 1.000s on a ~137s trailer): a
# segment whose duration is within
# _DEGENERATE_CUE_TOLERANCE of _DEGENERATE_CUE_SECONDS counts toward the stuck-cadence
# tally (the tolerance, not exact equality, so a vendor jittering the cadence by a
# millisecond still matches), and the run warns only when the tally both clears the
# absolute floor _DEGENERATE_MIN_SEGMENTS (protecting short genuinely-dialogue clips --
# a 20-cue interview clip with 11 one-second cues is plausible dialogue, not a stuck
# cadence, so the floor sits just above it) and exceeds _DEGENERATE_MIN_PROPORTION of
# all segments (protecting long dialogue, where occasional 1.000s cues are normal).
# Re-tune when a second real-world degenerate sample appears.
_DEGENERATE_CUE_SECONDS = 1.0
_DEGENERATE_CUE_TOLERANCE = 0.05
_DEGENERATE_MIN_SEGMENTS = 11
_DEGENERATE_MIN_PROPORTION = 0.5


def _warns_degenerate(segment_count: int, near_fixed_cadence: int) -> bool:
    """The degenerate-timings rule: the tally clears its floor and dominates the run.

    No zero-division guard is needed: `and` short-circuits, and a tally above the
    floor already implies segment_count > 0.
    """
    return (
        near_fixed_cadence > _DEGENERATE_MIN_SEGMENTS
        and near_fixed_cadence / segment_count > _DEGENERATE_MIN_PROPORTION
    )


def _budget_chars(start: float, end: float, text: str, output_lang: str, tts_engine: str) -> int:
    """Derive a translation length budget (in characters) from source text and slot.

    The budget is the LARGER of two independently measured lower bounds, not a
    primary term plus a cap: the expansion-ratio term (`_EXPANSION_RATIO`) estimates
    how long the translation will naturally run given the source text length, and the
    slot-capacity term estimates how many characters the cue's own time span can hold
    at the measured target-language speaking rate (`_TARGET_SPEAKABLE_CPS * duration`).
    Taking the min() of these (the old behaviour) starved cues whose source text was
    sparse relative to a generous slot: measured over 68 real segments, it budgeted
    half of them below the rate the source actors themselves speak at. Taking the
    max() instead ensures a sparse source cue is still budgeted up to what its slot
    can carry, not just what the ratio predicts.

    Known trade-off: the speaking rate is a MEDIAN, not a maximum, so used as a floor
    it will overshoot the slot for roughly half of segments by construction -- if the
    translator actually fills the larger budget, the failure mode can flip from
    underrun (translation too short/rushed) to overrun (translation doesn't fit its
    slot). `tts_engine` selects the rate because how fast a slot can be spoken is a
    property of the engine as well as the language; only "openai:da" is measured
    today, and every other pair falls back to that same figure.
    """
    ratio = _EXPANSION_RATIO.get(output_lang, _EXPANSION_RATIO["default"])
    cps = _TARGET_SPEAKABLE_CPS.get(f"{tts_engine}:{output_lang}", _TARGET_SPEAKABLE_CPS["default"])
    expected = len(text) * ratio
    # No `max(end - start, 0.0)` guard: a degenerate cue makes this term negative,
    # which simply loses the max() below -- the clamp it used to need was a property
    # of the old min().
    slot_capacity = (end - start) * cps
    return max(int(expected), int(slot_capacity), 1)


def _build_asr_provider(
    asr_engine: str, whisper_model_name: str, diarize: bool = True
) -> ASRProvider:
    if asr_engine == "local":
        from movie_subtitles.providers.local import Transcribe

        return Transcribe(whisper_model_name)
    elif asr_engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import ScribeTranscribe

        # diarize is only meaningful (and only wired) here: it's Scribe requesting
        # per-word speaker_id from the API, which _group_words then uses as a fourth
        # cue-flush trigger. Wired from --voice-match rather than --dub, on purpose:
        # the cue-splitting-on-speaker-change improvement to .srt output applies to
        # plain (non-dub) elevenlabs runs too, and the docs advertise it -- so it
        # must not be silently disabled just because --dub was not passed. Only an
        # explicit --voice-match off, which promises byte-for-byte pre-diarization
        # behaviour, turns it off.
        return ScribeTranscribe(diarize=diarize)
    elif asr_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAITranscribe

        return OpenAITranscribe()
    else:
        raise ValueError(f"Unknown ASR engine: {asr_engine}")


def _diarize_or_warn(fpath: Path, asr_engine: str) -> list[Turn]:
    """Build the diarizer, extract a mono wav, and diarize -- or degrade to `[]`.

    Callers only reach this when diarisation is actually needed (see
    `create_subtitles`'s `needs_diarization` guard -- the resolved ASR engine doesn't
    already diarize natively and --voice-match isn't "off"). Build, extraction, and
    `diarize()` all run under one try, mirroring `_dub_and_mux`'s handling of
    separate.py failures: an `ImportError` (a broken install -- pyannote.audio is a
    declared hard dependency) propagates uncaught, while any other failure (a missing
    HF_TOKEN, a decode/inference error, ...) is logged as one WARNING and degrades the
    run to a single voice (every segment's speaker stays None) rather than aborting it.

    The pyannote.audio (and therefore torch) import happens only inside this call, so
    a run that doesn't need diarisation never pays that import cost -- mirroring the
    lazy-import convention the other builders and separate.py already follow.
    """
    try:
        from movie_subtitles.providers.pyannote_ import Diarize

        diarizer = Diarize()

        if asr_engine == "openai":
            logger.warning(
                "Diarizing --asr-engine openai output: speaker labels are assigned "
                "by overlapping diarisation turns against whisper-1's ASR segment "
                "spans. whisper-1's segment timestamps are known to collapse to "
                "uniform 1.000s spans on music-heavy or dialogue-sparse audio, which "
                "would make those labels confidently wrong rather than merely "
                "absent. Prefer --asr-engine elevenlabs for multi-speaker material."
            )

        # pyannote.audio's loader is torchaudio/torchcodec-based and may not be able
        # to decode a video container the way demucs's ffmpeg-backed Separator can --
        # so extract a small mono 16 kHz wav with ffmpeg first and hand pyannote that
        # instead of the source media directly.
        with tempfile.TemporaryDirectory(prefix="movie-subtitles-diarize-") as tmp_dir:
            audio_path = Path(tmp_dir) / f"{fpath.stem}.diarize.wav"
            extract_mono_wav(fpath, audio_path)
            return diarizer.diarize(audio_path)
    except ImportError:
        # pyannote.audio is a declared hard dependency, so an ImportError here (or
        # from Diarize.__init__'s own lazy torch/pyannote imports) means a broken
        # install, not a runtime failure -- fail the run rather than silently
        # producing a worse dub. Mirrors separate.py's ImportError/Exception split.
        raise
    except Exception as exc:
        logger.warning(f"Diarization failed ({exc}); dubbing will use a single voice.")
        return []


def _build_translation_provider(translation_engine: str, mt_model_name: str) -> TranslationProvider:
    if translation_engine == "local":
        from movie_subtitles.providers.local import Translate

        return Translate(mt_model_name)
    elif translation_engine == "anthropic":
        from movie_subtitles.providers.llm import LLMTranslate

        return LLMTranslate()
    elif translation_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAITranslate

        return OpenAITranslate()
    else:
        raise ValueError(f"Unknown translation engine: {translation_engine}")


# Single source of truth: argparse choices, the --dub guard, and its error
# message all derive from this, so adding a backend is one edit.
TTS_ENGINES = ("elevenlabs", "openai")


def _build_tts_provider(tts_engine: str) -> TTSProvider:
    if tts_engine == "elevenlabs":
        from movie_subtitles.providers.elevenlabs import Speak

        return Speak()
    elif tts_engine == "openai":
        from movie_subtitles.providers.openai_ import OpenAISpeak

        return OpenAISpeak()
    else:
        raise ValueError(f"Unknown TTS engine: {tts_engine}")


def create_subtitles(
    fpath: str | Path,
    audio_lang: str = "en",
    srt_lang: str = "da",
    whisper_model_name: str = "large-v3",
    mt_model_name: str = "jbochi/madlad400-3b-mt",
    *,
    engine: str = "local",
    asr_engine: str | None = None,
    translation_engine: str | None = None,
    dub: bool = False,
    managed: bool = False,
    tts_engine: str | None = None,
    dub_workers: int = 1,
    dub_correction_passes: int = 3,
    voice_match: str = "auto",
    keep_cloned_voices: bool = False,
    clone_min_seconds: float = 30.0,
    clone_target_seconds: float = 60.0,
    voice_preset_table: str | Path | None = None,
    duck_level: float | None = None,
    separate_background: bool = False,
) -> None:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    if clone_min_seconds > clone_target_seconds:
        raise ValueError(
            f"--clone-min-seconds ({clone_min_seconds}) is greater than "
            f"--clone-target-seconds ({clone_target_seconds}), which makes cloning "
            "unreachable: no speaker could ever gather enough clean audio to pass the "
            "minimum before hitting the target. Lower --clone-min-seconds or raise "
            "--clone-target-seconds."
        )

    # Validate a --voice-preset-table up front, alongside the min/target check above,
    # rather than letting resolved_voices() reach load_preset_table() only after ASR and
    # full translation have already run (potentially minutes of paid API calls). The
    # parsed table is threaded straight into resolved_voices() below, so it is parsed
    # exactly once and the file cannot change between this validation and its use.
    presets = None
    if voice_preset_table is not None:
        from movie_subtitles.voices import load_preset_table

        presets = load_preset_table(voice_preset_table)

    # --engine is the shorthand that sets all stages; --asr-engine/--translation-engine/
    # --tts-engine override it per stage.
    resolved_asr_engine = asr_engine if asr_engine is not None else engine
    resolved_translation_engine = translation_engine if translation_engine is not None else engine
    resolved_tts_engine = tts_engine if tts_engine is not None else engine

    if managed and dub:
        raise ValueError(
            "--managed and --dub are mutually exclusive: --managed runs the ElevenLabs "
            "Dubbing job end to end instead of the local transcribe/translate/dub pipeline."
        )

    if managed and separate_background:
        raise ValueError(
            "--managed and --separate-background are mutually exclusive: the managed "
            "ElevenLabs Dubbing job already handles background preservation itself, so "
            "running source separation on top of it would be wasted work."
        )

    if not managed and translation_engine is None and engine == "elevenlabs":
        raise ValueError(
            "--engine elevenlabs does not set a translation stage: ElevenLabs has no "
            "standalone text-translation endpoint (translation exists only bundled "
            "inside the Dubbing job, see --managed). Pass --translation-engine "
            "{anthropic,openai,local} explicitly."
        )

    if dub and resolved_translation_engine == "local":
        logger.warning(
            "--dub with the local MADLAD400 translator: it ignores budget_chars, so "
            "timing-drift fitting degrades to TTS-rate-only (step 1 of the drift "
            "strategy is a silent no-op). Use --translation-engine anthropic/openai for "
            "length-budgeted translations if the dub timing is off."
        )

    if dub and resolved_tts_engine not in TTS_ENGINES:
        raise ValueError(
            f"--dub requires a usable TTS engine, but it resolved to '{resolved_tts_engine}'. "
            f"Pass --tts-engine {{{','.join(TTS_ENGINES)}}} explicitly."
        )

    if managed:
        _run_managed(fpath, audio_lang, srt_lang)
        return

    srt_file = fpath.with_suffix(".srt")

    # Build cheap-to-construct providers first so a missing API key or a missing ffmpeg
    # fails in milliseconds, rather than after the ASR model download and a full run of
    # paid per-segment translation calls.
    translator = _build_translation_provider(resolved_translation_engine, mt_model_name)
    tts = None
    if dub:
        _check_ffmpeg_tools()
        tts = _build_tts_provider(resolved_tts_engine)

    # Diarisation is a whole-file operation, so it runs once, up front, before the ASR
    # generator is consumed -- not as a post-pass -- since its results must already be
    # available while .srt cues are being emitted from the lazy ASR loop below.
    needs_diarization = voice_match != "off" and resolved_asr_engine != "elevenlabs"
    turns = _diarize_or_warn(fpath, resolved_asr_engine) if needs_diarization else []

    transcriber = _build_asr_provider(
        resolved_asr_engine, whisper_model_name, diarize=voice_match != "off"
    )
    segments = transcriber(fpath, audio_lang)
    if turns:
        segments = label_segments(segments, turns)

    srt_lines = []
    dub_segments: list[Segment] = []
    dub_translations: dict[int, str] = {}
    cue_id = 0

    # One-segment lookahead: a cue's padded end must not overlap the next emitted
    # cue's start, but that start is only known once the following segment arrives.
    # `pending` holds the most recently emitted (not-yet-written) cue's data, and is
    # flushed as soon as the next cue's start is known (or with next_start=None at
    # the end of the stream). This keeps `segments` a single lazy pass -- no
    # materializing it into a list -- so ASR inference and the progress bar still
    # drive off the same generator.
    def _flush_pending(pending: tuple[int, float, float, str], next_start: float | None) -> None:
        """Format and append `pending`'s cue, padding its end toward `next_start`."""
        p_id, p_start, p_end, p_text = pending
        srt_lines.append(format_block(p_id, p_start, pad_cue_end(p_end, next_start), p_text))

    pending: tuple[int, float, float, str] | None = None
    # Running aggregates for the degenerate-timings check (the whisper-1 fixed-cadence
    # vendor defect): two counters only, so `segments` stays a single lazy pass. Every
    # segment counts toward the total -- including ones the translator empties, whose
    # timings are just as much evidence -- and the tally uses the RAW segment times,
    # never the padded cue times (padding is a write-time .srt concern).
    segment_count = 0
    near_fixed_cadence = 0
    for segment in tqdm(segments, desc="Writing to srt file"):
        segment_count += 1
        duration = segment.end - segment.start
        if abs(duration - _DEGENERATE_CUE_SECONDS) <= _DEGENERATE_CUE_TOLERANCE:
            near_fixed_cadence += 1

        budget_chars = _budget_chars(
            segment.start, segment.end, segment.text, srt_lang, resolved_tts_engine
        )
        text = translator(segment.text, srt_lang, budget_chars=budget_chars)
        # Field names/order here are parsed by scripts/measure.py; changing them
        # silently breaks it (no import edge, no CI signal).
        logger.debug(
            f"[measure] measure=translate id={segment.id} start={segment.start:.3f} "
            f"end={segment.end:.3f} slot={segment.end - segment.start:.3f} "
            f"src_chars={len(segment.text)} tgt_chars={len(text)} budget={budget_chars} "
            f"lang={srt_lang}"
        )
        if not text:
            continue

        cue_id += 1

        if pending is not None:
            _flush_pending(pending, segment.start)
        pending = (cue_id, segment.start, segment.end, text)

        if dub:
            dub_segments.append(segment)
            dub_translations[segment.id] = text

    if pending is not None:
        _flush_pending(pending, None)

    if _warns_degenerate(segment_count, near_fixed_cadence):
        logger.warning(
            f"{near_fixed_cadence} of {segment_count} ASR segments have a duration of "
            f"~{_DEGENERATE_CUE_SECONDS:.3f}s: the ASR timestamps look degenerate, "
            "degrading to a fixed ~1s cadence (observed with --asr-engine openai's "
            "whisper-1 on music-heavy or dialogue-sparse audio). The .srt's cue "
            "timings, and any dub slots derived from them, are likely wrong. "
            "Re-run with --asr-engine elevenlabs on such material."
        )

    srt_file.write_text("".join(srt_lines), encoding="utf-8")
    logger.info(f"Saved srt file to {srt_file}")

    if tts is not None:
        with resolved_voices(
            fpath,
            dub_segments,
            tts_engine=resolved_tts_engine,
            mode=voice_match,
            clone_min_seconds=clone_min_seconds,
            clone_target_seconds=clone_target_seconds,
            presets=presets,
            keep=keep_cloned_voices,
        ) as voices_map:
            _dub_and_mux(
                fpath,
                dub_segments,
                dub_translations,
                tts,
                dub_workers=dub_workers,
                dub_correction_passes=dub_correction_passes,
                voices=voices_map,
                duck_level=duck_level,
                separate_background=separate_background,
            )


def _run_managed(fpath: Path, audio_lang: str, srt_lang: str) -> None:
    from movie_subtitles.dubbing import ManagedDub

    managed_dub = ManagedDub()
    out_path = managed_dub(fpath, audio_lang, srt_lang)
    logger.info(f"Saved managed dub to {out_path}")


def _check_ffmpeg_tools() -> None:
    """Raise a clear RuntimeError if ffmpeg or ffprobe is missing, before any work starts.

    dub.py uses ffprobe to measure synthesised clip duration and mux.py uses ffmpeg to
    assemble/mux; a missing ffprobe used to crash with a raw FileNotFoundError from
    subprocess deep inside dub.py. Checking both up front keeps the failure mode
    consistent with mux.py's existing ffmpeg check.
    """
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not on PATH. Install ffmpeg (which provides both "
            "ffmpeg and ffprobe) to use --dub."
        )


def _build_aligner() -> AlignmentProvider:
    """Build the Forced Alignment -> silencedetect -> container-duration degrade chain.

    Forced Alignment is a nice-to-have, not a hard requirement for --dub: a
    --tts-engine openai user may have no ELEVENLABS_API_KEY at all. So any failure
    building it (missing key, client construction error) is caught and logged, and the
    chain is built without it -- `FallbackAlign` degrades through the remaining tiers on
    a per-call basis, ending in the never-failing `DurationAlign`, so this never returns
    `None`.
    """
    from movie_subtitles.providers.fallback import FallbackAlign
    from movie_subtitles.providers.ffmpeg_align import DurationAlign, SilenceAlign

    providers: list[AlignmentProvider] = []

    if not os.environ.get("ELEVENLABS_API_KEY"):
        logger.info(
            "ELEVENLABS_API_KEY not set: speech-boundary measurement will fall back to "
            "ffmpeg silencedetect instead of Forced Alignment."
        )
    else:
        from movie_subtitles.providers.elevenlabs import Align

        try:
            providers.append(Align())
        except Exception as exc:
            logger.warning(
                f"Could not build the Forced Alignment aligner ({exc}); speech-boundary "
                "measurement will fall back to ffmpeg silencedetect."
            )

    providers.extend((SilenceAlign(), DurationAlign()))

    return FallbackAlign(providers)


def _dub_and_mux(
    fpath: Path,
    segments: list[Segment],
    translations: dict[int, str],
    tts: TTSProvider,
    dub_workers: int,
    dub_correction_passes: int,
    duck_level: float | None,
    voices: dict[str | None, str | None] = _NO_VOICES,
    separate_background: bool = False,
) -> None:
    from movie_subtitles.dub import synthesise_track
    from movie_subtitles.mux import mux_dub

    audio_track = fpath.with_name(f"{fpath.stem}.dub_audio.mp3")
    aligner = _build_aligner()

    _, speech_spans = synthesise_track(
        segments,
        translations,
        tts,
        audio_track,
        aligner=aligner,
        max_workers=dub_workers,
        correction_passes=dub_correction_passes,
        voices=voices,
    )

    # Everything from here on runs inside the `try` whose `finally` removes both
    # intermediates -- the whole-film TTS track just synthesised above, and the
    # accompaniment stem's temp dir. Separation is deliberately inside it too: an
    # exception escaping earlier would leak minutes of paid synthesis onto disk.
    stem_dir: str | None = None
    background_path: Path | None = None

    try:
        if separate_background:
            # A multi-hundred-MB intermediate only mux_dub reads, so it goes in a temp
            # dir rather than next to the user's video -- matching separate.py's own
            # discipline.
            stem_dir = tempfile.mkdtemp(prefix="movie-subtitles-dub-")

            from movie_subtitles.separate import Separate

            try:
                separator = Separate()
                background_path = separator(
                    fpath, Path(stem_dir) / f"{fpath.stem}.accompaniment.wav"
                )
            except ImportError:
                # demucs is a declared hard dependency (see separate.py's module
                # docstring), so an ImportError -- from the import above or from
                # Separate.__init__'s own lazy torch/demucs.api imports -- means a broken
                # install, not a runtime failure. Fail the run rather than silently
                # producing a worse output bed.
                raise
            except Exception as exc:
                # Separation is a nice-to-have on top of the #22 duck-and-mix path, not a
                # hard requirement of --dub: unreachable model weights, a bad input file,
                # or any decode/inference error must never fail the whole run.
                logger.warning(
                    f"Source separation failed ({exc}); falling back to ducking the "
                    "original audio instead of muxing over an accompaniment stem."
                )

        dubbed_path = mux_dub(
            fpath,
            audio_track,
            speech_spans=speech_spans,
            duck_level=duck_level,
            background_path=background_path,
        )
        logger.info(f"Saved dubbed video to {dubbed_path}")
    finally:
        audio_track.unlink()
        if stem_dir is not None:
            shutil.rmtree(stem_dir, ignore_errors=True)


def _load_env() -> None:
    """Load API keys from a .env file found by searching upward from the cwd.

    Called from main() only: importing this module stays side-effect free for library
    callers, who set the environment themselves. usecwd=True is deliberate -- the
    default resolves .env relative to *this file*, which finds the repo's own .env even
    when the user is working in some other project. Searching from the cwd instead means
    the .env belonging to the directory you invoked the CLI from is the one that wins.

    Already-exported variables are never overridden, so a real environment variable
    still beats the file. A missing .env is not an error: providers raise their own
    RuntimeError naming the specific key they need.
    """
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment from {dotenv_path}")


def main() -> None:
    _load_env()
    # Re-apply now that .env is loaded: MOVIE_SUBTITLES_LOG_LEVEL was read once at
    # import, before _load_env() ran, so a value living only in .env would otherwise
    # never take effect -- unlike every other variable that file holds.
    configure_logging()

    parser = ArgumentParser(
        "Command line interface for movie subtitles",
        usage="translation-cli <command> [<args>]",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The input file to transcribe",
    )
    parser.add_argument(
        "--audio-lang",
        type=str,
        default="en",
        help="The language of the audio in the movie",
    )
    parser.add_argument(
        "--srt-lang",
        type=str,
        default="da",
        help="The language of the srt file",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="The whisper model to use",
    )
    parser.add_argument(
        "--mt-model",
        type=str,
        default="jbochi/madlad400-3b-mt",
        help="The machine translation model to use",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["local", "elevenlabs", "openai"],
        default="local",
        help="The provider engine to use for transcription, translation and TTS "
        "(shorthand for --asr-engine, --translation-engine and --tts-engine when those "
        "are not set). --engine elevenlabs requires --translation-engine to be set "
        "explicitly, since ElevenLabs has no standalone translation endpoint",
    )
    parser.add_argument(
        "--asr-engine",
        type=str,
        choices=["local", "elevenlabs", "openai"],
        default=None,
        help="Override --engine for the ASR (transcription) stage only",
    )
    parser.add_argument(
        "--translation-engine",
        type=str,
        choices=["local", "anthropic", "openai"],
        default=None,
        help="Override --engine for the translation stage only. --dub with this resolving "
        "to 'local' still works, but timing-drift fitting degrades to TTS-rate-only "
        "since the local translator ignores the length budget",
    )
    parser.add_argument(
        "--tts-engine",
        type=str,
        choices=list(TTS_ENGINES),
        default=None,
        help="Override --engine for the TTS (dubbing) stage only",
    )
    parser.add_argument(
        "--dub",
        action="store_true",
        help=(
            "Synthesise the translated segments with TTS and mux them over the source "
            "video (requires ffmpeg and the API key for the resolved TTS engine)"
        ),
    )
    parser.add_argument(
        "--dub-workers",
        type=_positive_int,
        default=1,
        help=(
            "Maximum number of TTS/alignment calls to run concurrently during --dub "
            "synthesis. Defaults to 1 (serial), since vendor concurrency caps are "
            "per-subscription and low -- raise it to what your TTS plan allows"
        ),
    )
    parser.add_argument(
        "--dub-correction-passes",
        type=_positive_int,
        default=3,
        help=(
            "Maximum number of corrective re-synthesis passes a drifted group gets "
            "during --dub synthesis. Each pass costs paid TTS; 1 reproduces the old "
            "single-pass behaviour. Defaults to 3"
        ),
    )
    parser.add_argument(
        "--voice-match",
        type=str,
        choices=list(VOICE_MATCH_MODES),
        default="auto",
        help=(
            "How to pick TTS voices for diarized speakers during --dub: 'off' uses the "
            "single configured voice (today's behaviour), 'clone' instant-clones each "
            "speaker's voice via ElevenLabs IVC, 'preset' matches each speaker to a "
            "curated stock voice by gender/age, and 'auto' clones when enough clean "
            "audio is available for a speaker and falls back to a preset otherwise. "
            "On --asr-engine local/openai, any mode other than 'off' also runs a "
            "standalone pyannote.audio diarisation pass to label speakers (ElevenLabs "
            "diarizes natively). Defaults to 'auto'"
        ),
    )
    parser.add_argument(
        "--keep-cloned-voices",
        action="store_true",
        help=(
            "Do not delete ElevenLabs voices cloned for --voice-match clone/auto after "
            "the run finishes; the retained voice ids are logged"
        ),
    )
    parser.add_argument(
        "--clone-min-seconds",
        type=_positive_float,
        default=CLONE_MIN_SECONDS,
        help=(
            "Minimum seconds of clean (non-overlapping) speech a speaker must have "
            "before it is eligible for voice cloning; below this it falls back to a "
            f"preset. Defaults to {CLONE_MIN_SECONDS}"
        ),
    )
    parser.add_argument(
        "--clone-target-seconds",
        type=_positive_float,
        default=CLONE_TARGET_SECONDS,
        help=(
            "Maximum seconds of clean speech gathered per speaker to build a cloned "
            f"voice sample. Defaults to {CLONE_TARGET_SECONDS}"
        ),
    )
    parser.add_argument(
        "--voice-preset-table",
        type=str,
        default=None,
        help=(
            "Path to a JSON file overriding the built-in gender/age preset voice "
            "table used by --voice-match preset/auto (see voices.py PROFILE_KEYS for "
            "the recognised keys)"
        ),
    )
    parser.add_argument(
        "--duck-level",
        type=_unit_float,
        # None is a sentinel resolved by mux.mux_dub itself, not duplicated here as a
        # literal default per the repo's usual convention: which of the two numbers
        # applies depends on whether --separate-background succeeded, and argparse
        # cannot know that at parse time. Do not "fix" this back to 0.25.
        default=None,
        help=(
            "How much to attenuate the original audio (or, with --separate-background, "
            "the separated accompaniment stem) while the dub is speaking, during --dub "
            "muxing. 0.0 silences that bed entirely under the dub; 1.0 disables ducking "
            "altogether (bed plays at full volume under the dub). Must be between 0.0 "
            "and 1.0 inclusive. Defaults to 0.25, or 0.6 when --separate-background "
            "succeeded (the original dialogue is already gone, so only a little "
            "headroom over loud music is needed); an explicit value always wins over "
            "either default"
        ),
    )
    parser.add_argument(
        "--separate-background",
        action="store_true",
        help=(
            "Split the source audio into vocals/accompaniment stems (via Demucs) and mux "
            "the dub over the accompaniment stem only, removing the original dialogue "
            "instead of merely ducking it. Opt-in: this is CPU-minutes-to-tens-of-minutes "
            "on a feature-length film. Falls back to the default duck-and-mix path over "
            "the original audio if separation fails for any reason. Mutually exclusive "
            "with --managed"
        ),
    )
    parser.add_argument(
        "--managed",
        action="store_true",
        help=(
            "Use the managed ElevenLabs Dubbing job (create/poll/download) instead of "
            "the local transcribe/translate/dub pipeline (requires ELEVENLABS_API_KEY). "
            "Mutually exclusive with --dub."
        ),
    )
    args = parser.parse_args()

    try:
        create_subtitles(
            args.input,
            args.audio_lang,
            args.srt_lang,
            args.whisper_model,
            args.mt_model,
            engine=args.engine,
            asr_engine=args.asr_engine,
            translation_engine=args.translation_engine,
            dub=args.dub,
            managed=args.managed,
            tts_engine=args.tts_engine,
            dub_workers=args.dub_workers,
            dub_correction_passes=args.dub_correction_passes,
            voice_match=args.voice_match,
            keep_cloned_voices=args.keep_cloned_voices,
            clone_min_seconds=args.clone_min_seconds,
            clone_target_seconds=args.clone_target_seconds,
            voice_preset_table=args.voice_preset_table,
            duck_level=args.duck_level,
            separate_background=args.separate_background,
        )
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        TimeoutError,
        subprocess.CalledProcessError,
    ) as exc:
        logger.error(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
