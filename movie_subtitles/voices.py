"""Speaker -> TTS voice resolution: sample extraction, cloning, and preset matching.

Owns everything that maps a diarized `Segment.speaker` label to a TTS voice id, for
both the `elevenlabs` (instant voice cloning + curated presets) and `openai` (preset
only) TTS engines.
"""

import heapq
import json
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from movie_subtitles.ffmpeg import probe_duration
from movie_subtitles.ffmpeg import run as run_ffmpeg
from movie_subtitles.providers.base import Segment

logger = logging.getLogger("voices")

# Duplicated (deliberately, per repo convention) with argparse's --clone-min-seconds /
# --clone-target-seconds defaults and create_subtitles()'s keyword defaults in cli.py.
# Change every copy together.
CLONE_MIN_SECONDS = 30.0
CLONE_TARGET_SECONDS = 60.0

# Voice-match modes accepted by --voice-match.
VOICE_MATCH_MODES = ("off", "clone", "preset", "auto")

# TTS engines that support instant voice cloning.
_CLONE_CAPABLE_ENGINES = {"elevenlabs"}

_GENDERS = ("female", "male")
_AGE_BANDS = ("young", "adult", "elderly")

# The full recognised set of preset-table profile keys: every gender:age_band
# combination, plus the explicit "default" fallback used for an unknown profile.
PROFILE_KEYS = frozenset({f"{gender}:{band}" for gender in _GENDERS for band in _AGE_BANDS}) | {
    "default"
}

_KNOWN_TTS_ENGINES = frozenset({"elevenlabs", "openai"})

# Curated ElevenLabs stock voice ids (publicly documented default voices), plus the
# OpenAI voice set (alloy/echo/fable/onyx/nova/shimmer). "default" is used whenever
# classification degrades to an unknown profile.
_DEFAULT_PRESETS: dict[str, dict[str, str]] = {
    "elevenlabs": {
        "female:young": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "female:adult": "EXAVITQu4vr4xnSDxMaL",  # Sarah
        "female:elderly": "MF3mGyEYCl7XYWbV9V6O",  # Elli
        "male:young": "TxGEqnHWrfWFTfGW9XjX",  # Josh
        "male:adult": "pNInz6obpgDQGcFmaJgB",  # Adam
        "male:elderly": "VR6AewLTigWG4xSOukaG",  # Arnold
        "default": "EXAVITQu4vr4xnSDxMaL",  # Sarah -- matches elevenlabs.py's _DEFAULT_VOICE_ID
    },
    "openai": {
        "female:young": "nova",
        "female:adult": "nova",
        "female:elderly": "shimmer",
        "male:young": "echo",
        "male:adult": "onyx",
        "male:elderly": "fable",
        "default": "alloy",
    },
}


def load_preset_table(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a `--voice-preset-table` JSON file, replacing the built-in mapping.

    Validates strictly at load time rather than deferring failures into a mid-dub
    KeyError. Raises `ValueError` with a specific message for each of: unreadable or
    malformed JSON, a top-level key that isn't a known TTS engine, a profile key
    outside `PROFILE_KEYS`, a non-string voice value, or a provider block missing
    "default". There is no per-key merge with `_DEFAULT_PRESETS` -- a supplied file
    must be complete for the engines it names.
    """
    path = Path(path)
    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise ValueError(f"Could not read voice preset table {path}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Voice preset table {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Voice preset table {path} must be a JSON object mapping engine name -> "
            f"profile mapping, got {type(data).__name__}"
        )

    table: dict[str, dict[str, str]] = {}
    for engine, profiles in data.items():
        if engine not in _KNOWN_TTS_ENGINES:
            raise ValueError(
                f"Voice preset table {path} has top-level key {engine!r}, which is not a "
                f"known TTS engine (expected one of {sorted(_KNOWN_TTS_ENGINES)})"
            )

        if not isinstance(profiles, dict):
            raise ValueError(
                f"Voice preset table {path}: engine {engine!r} must map to an object, got "
                f"{type(profiles).__name__}"
            )

        engine_table: dict[str, str] = {}
        for profile_key, voice_id in profiles.items():
            if profile_key not in PROFILE_KEYS:
                raise ValueError(
                    f"Voice preset table {path}: engine {engine!r} has profile key "
                    f"{profile_key!r}, which is not recognised (expected one of "
                    f"{sorted(PROFILE_KEYS)})"
                )
            if not isinstance(voice_id, str):
                raise ValueError(
                    f"Voice preset table {path}: engine {engine!r} profile {profile_key!r} "
                    f"must be a string voice id, got {type(voice_id).__name__}"
                )
            engine_table[profile_key] = voice_id

        if "default" not in engine_table:
            raise ValueError(
                f"Voice preset table {path}: engine {engine!r} is missing a required "
                '"default" entry'
            )

        table[engine] = engine_table

    return table


def _preset_voice(
    presets: dict[str, dict[str, str]], tts_engine: str, profile: tuple[str, str] | None
) -> str | None:
    engine_table = presets.get(tts_engine)
    if engine_table is None:
        return None
    key = f"{profile[0]}:{profile[1]}" if profile is not None else "default"
    return engine_table.get(key, engine_table.get("default"))


def _clean_segments_by_speaker(segments: list[Segment]) -> dict[str, list[Segment]]:
    """Compute every speaker's "clean" (non-overlapped) segment set in one sweep.

    A segment disqualifies itself (for whichever speaker it belongs to) as soon as it
    overlaps in time with any segment carrying a *different* `speaker` value -- an
    unlabelled segment (`speaker is None`) counts as disqualifying too, since it may
    belong to an unrecognised speaker and must not be allowed to co-occur with a
    "clean" clip. That test is speaker-agnostic per segment (it only depends on the pair
    of labels involved), so it can be computed once for the whole segment list with a
    single start-sorted sweep over a min-heap of still-active segments, rather than
    rescanning every other segment for each of a speaker's own segments (the previous
    O(speakers * n^2) shape).
    """
    order = sorted(range(len(segments)), key=lambda i: segments[i].start)
    overlaps = [False] * len(segments)

    active: list[tuple[float, int]] = []  # (end, index), heap ordered by end
    for i in order:
        seg = segments[i]
        while active and active[0][0] <= seg.start:
            heapq.heappop(active)
        for _end, j in active:
            other = segments[j]
            if other.speaker != seg.speaker:
                overlaps[i] = True
                overlaps[j] = True
        heapq.heappush(active, (seg.end, i))

    by_speaker: dict[str, list[Segment]] = {}
    for i, seg in enumerate(segments):
        if seg.speaker is not None and not overlaps[i]:
            by_speaker.setdefault(seg.speaker, []).append(seg)
    for speaker_segments in by_speaker.values():
        speaker_segments.sort(key=lambda s: s.start)
    return by_speaker


def extract_speaker_sample(
    source: str | Path,
    speaker: str,
    clean_segments: list[Segment],
    *,
    target_seconds: float = CLONE_TARGET_SECONDS,
    out_dir: str | Path | None = None,
) -> tuple[Path | None, float]:
    """Extract up to `target_seconds` of clean audio for one speaker, in one ffmpeg pass.

    `clean_segments` is `speaker`'s already-computed, start-sorted, non-overlapping
    segment list (see `_clean_segments_by_speaker`). Clips are taken in start order
    until `target_seconds` is reached or the clean segments run out, and cut out of
    `source` with a single `atrim`/`concat` filtergraph over one input, written
    directly to a mono 16 kHz WAV -- avoiding a separate ffmpeg invocation (and a full
    decode of `source` from frame 0, since plain `-ss` after `-i` seeks by decoding)
    per clip, plus a second concat pass.

    Returns `(None, 0.0)` if no clean audio exists for this speaker at all. Otherwise
    returns the produced sample's path and the seconds actually gathered, measured from
    the produced file via `probe_duration` -- not the requested cut length, since an
    `-ss`/`-t` cut can run past the source's actual end.
    """
    source = Path(source)
    workdir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="voices-"))
    workdir.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[float, float]] = []
    requested = 0.0
    for seg in clean_segments:
        if requested >= target_seconds:
            break
        duration = seg.end - seg.start
        if duration <= 0:
            continue
        take = min(duration, target_seconds - requested)
        selected.append((seg.start, seg.start + take))
        requested += take

    if not selected:
        return None, 0.0

    atrim_filters = [
        f"[0:a]atrim=start={start:.3f}:end={end:.3f}[a{i}]"
        for i, (start, end) in enumerate(selected)
    ]
    concat_inputs = "".join(f"[a{i}]" for i in range(len(selected)))
    filter_complex = (
        ";".join(atrim_filters) + f";{concat_inputs}concat=n={len(selected)}:v=0:a=1[out]"
    )

    sample_path = workdir / f"{speaker}_sample.wav"
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(sample_path),
        ],
        what=f"extracting sample for speaker {speaker!r}",
    )
    return sample_path, probe_duration(sample_path)


# Bounded so classify_voice never decodes an entire (potentially long) sample: 15s is
# ample for a stable median F0 and coarse formant estimate.
_CLASSIFY_MAX_SECONDS = 15.0

# Below this the F0 estimate is treated as unreliable rather than as a very deep
# voice: pyin clamps at its own fmin, and music or rumble under the dialogue drags
# the median onto that floor. Kept above librosa's C2 (65.4 Hz) fmin for that reason.
_MIN_TRUSTED_F0 = 70.0

# F0 band where gender cannot be read off pitch alone (a low female or a high male);
# inside it the formant average decides, since a longer vocal tract lowers F1/F2.
_AMBIGUOUS_F0_LOW = 150.0
_AMBIGUOUS_F0_HIGH = 185.0
_AMBIGUOUS_FORMANT_AVG = 1300.0

# (young_cut, elderly_cut) per gender: at or above young_cut is "young", below
# elderly_cut is "elderly", between them is "adult". Female and male F0 ranges barely
# overlap, so the cuts are per gender rather than shared.
_AGE_CUTS = {"female": (250.0, 175.0), "male": (145.0, 95.0)}


def classify_voice(sample_path: str | Path) -> tuple[str, str] | None:
    """Classify a sample WAV into a coarse (gender, age_band) profile.

    Uses `librosa.pyin` for a noise-robust median F0 and `praat-parselmouth` for
    F1/F2 formants. Both libraries are imported lazily, inside this function, so
    `--voice-match off` (and any run that never classifies) never pays their import
    cost. The sample is decoded once (bounded to `_CLASSIFY_MAX_SECONDS`) and the
    resulting array is handed to both analyses -- `parselmouth.Sound` is built from
    the already-decoded samples rather than re-reading the file from disk itself.
    Degrades to `None` (an "unknown" profile) rather than raising if either analysis
    fails -- movie audio is noisy and a bad sample must fall back to the default
    preset, not abort the run.
    """
    try:
        import librosa
        import numpy as np
        import parselmouth
    except ImportError as exc:
        logger.warning(f"Voice classification unavailable ({exc}); using unknown profile")
        return None

    sample_path = Path(sample_path)
    try:
        y, sr = librosa.load(str(sample_path), sr=None, mono=True, duration=_CLASSIFY_MAX_SECONDS)
        if y.size == 0:
            raise ValueError("empty sample audio")

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=_MIN_TRUSTED_F0,
            fmax=float(librosa.note_to_hz("C6")),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if voiced_f0.size == 0:
            raise ValueError("no voiced frames detected")
        median_f0 = float(np.median(voiced_f0))

        snd = parselmouth.Sound(y, sampling_frequency=sr)
        formant = snd.to_formant_burg()
        times = np.arange(formant.xmin, formant.xmax, 0.01)
        f1_values = [formant.get_value_at_time(1, t) for t in times]
        f2_values = [formant.get_value_at_time(2, t) for t in times]
        f1_values = [v for v in f1_values if v == v]  # drop NaN
        f2_values = [v for v in f2_values if v == v]
        if not f1_values or not f2_values:
            raise ValueError("no formant frames detected")
        median_f1 = float(np.median(f1_values))
        median_f2 = float(np.median(f2_values))
    except Exception as exc:  # noqa: BLE001 -- classification must degrade, never raise
        logger.warning(
            f"Voice classification failed for {sample_path}: {exc}; using unknown profile"
        )
        return None

    # Coarse, heuristic thresholds, NOT tuned against a labelled dataset.
    #
    # F0 is the primary gender cue. Formants only disambiguate the band where F0
    # alone is unreliable (roughly a low female or a high male voice): a longer
    # vocal tract lowers F1/F2, so low formants there argue male. Formants are
    # deliberately NOT used as an age cue. An earlier revision banded age on
    # (F1+F2)/2, with <= 1400 meaning "elderly"; measured on real speech that
    # average sits around 1200-1350 for every speaker, so "elderly" swallowed
    # everyone whose F0 missed the "young" cut and "adult" -- the commonest case --
    # was unreachable.
    if _AMBIGUOUS_F0_LOW <= median_f0 < _AMBIGUOUS_F0_HIGH:
        gender = "female" if (median_f1 + median_f2) / 2.0 >= _AMBIGUOUS_FORMANT_AVG else "male"
    else:
        gender = "female" if median_f0 >= _AMBIGUOUS_F0_HIGH else "male"

    # Age bands are read off F0 relative to that gender's own range, since male and
    # female F0 distributions barely overlap. Pitch rises toward the child end and
    # falls with age, so each band is just a cut on the speaker's own scale.
    young_cut, elderly_cut = _AGE_CUTS[gender]
    if median_f0 >= young_cut:
        age_band = "young"
    elif median_f0 < elderly_cut:
        age_band = "elderly"
    else:
        age_band = "adult"

    logger.debug(
        f"Classified {sample_path.name}: F0={median_f0:.1f}Hz F1={median_f1:.1f}Hz "
        f"F2={median_f2:.1f}Hz -> {gender}:{age_band}"
    )
    return gender, age_band


def _cleanup_cloned_voices(cloned_voice_ids: list[str]) -> None:
    """Delete previously cloned voices, individually guarded.

    A failing delete logs a WARNING naming the leaked voice id and never aborts the
    remaining deletions, nor masks an exception already propagating from the caller.
    """
    if not cloned_voice_ids:
        return

    from movie_subtitles.providers.elevenlabs import build_client, delete_voice

    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001 -- must never mask the caller's exception
        logger.warning(
            f"Could not build ElevenLabs client to clean up cloned voices ({exc}); "
            f"the following voice ids leaked and must be deleted manually: "
            f"{', '.join(cloned_voice_ids)}"
        )
        return

    for voice_id in cloned_voice_ids:
        try:
            delete_voice(client, voice_id)
        except Exception as exc:  # noqa: BLE001 -- must not abort remaining deletions
            logger.warning(f"Failed to delete cloned voice {voice_id}: {exc}; delete it manually")


@contextmanager
def resolved_voices(
    source: str | Path,
    segments: list[Segment],
    *,
    tts_engine: str,
    mode: str = "auto",
    clone_min_seconds: float = CLONE_MIN_SECONDS,
    clone_target_seconds: float = CLONE_TARGET_SECONDS,
    presets: dict[str, dict[str, str]] | None = None,
    keep: bool = False,
) -> Iterator[dict[str | None, str | None]]:
    """Resolve a speaker -> TTS voice id mapping for `--dub`, honouring `--voice-match`.

    Yields the mapping for the caller to use for the duration of the `with` block. Any
    voice cloned along the way (ElevenLabs IVC) is deleted on exit unless `keep` is set
    -- whether the block completes normally, the resolution loop itself raises partway
    through, or the caller's `with` body (e.g. the dub) raises. A cloned voice succeeds
    by mutating the ElevenLabs account immediately (there is no way to "undo" that
    except an explicit delete), so a permanently leaked paid voice slot is the failure
    this context manager exists to prevent -- success and failure share this one
    cleanup path instead of being handled separately by the caller.

    - `mode="off"`: yields an all-empty mapping. Does no diarization work, sample
      extraction, or lazy import -- byte-for-byte today's (pre-diarization) behaviour.
    - `mode="clone"`: clones every eligible speaker (ElevenLabs IVC only). A speaker
      with fewer than `clone_min_seconds` of clean audio falls back to a preset, with
      a WARNING naming the speaker and the seconds actually found.
    - `mode="preset"`: classifies each speaker and picks a preset voice; never clones.
    - `mode="auto"`: clones when `tts_engine` supports cloning (elevenlabs), otherwise
      (and for any speaker ineligible to clone) falls back to preset matching.

    Cloning is never attempted for `tts_engine="openai"` in any mode. `presets`
    defaults to the built-in `_DEFAULT_PRESETS` table when not given.
    """
    if mode not in VOICE_MATCH_MODES:
        raise ValueError(f"Unknown voice-match mode {mode!r} (expected one of {VOICE_MATCH_MODES})")

    voices: dict[str | None, str | None] = {}
    cloned_ids: list[str] = []
    try:
        if mode != "off":
            speakers = sorted({s.speaker for s in segments if s.speaker is not None})
            if not speakers:
                logger.info("No segment carries a speaker label; dubbing with a single voice")
            else:
                presets = presets if presets is not None else _DEFAULT_PRESETS
                can_clone = tts_engine in _CLONE_CAPABLE_ENGINES
                want_clone = mode == "clone" or (mode == "auto" and can_clone)

                client = None
                if want_clone and can_clone:
                    from movie_subtitles.providers.elevenlabs import build_client

                    client = build_client()

                clean_by_speaker = _clean_segments_by_speaker(segments)

                for speaker in speakers:
                    sample_path, seconds_gathered = extract_speaker_sample(
                        source,
                        speaker,
                        clean_by_speaker.get(speaker, []),
                        target_seconds=clone_target_seconds,
                    )

                    if want_clone and can_clone and seconds_gathered >= clone_min_seconds:
                        from movie_subtitles.providers.elevenlabs import clone_voice

                        name = f"movie-subtitles {Path(source).stem} {speaker}"
                        try:
                            voice_id = clone_voice(client, name, [sample_path])
                        except Exception as exc:  # noqa: BLE001 -- degrade to preset
                            logger.warning(
                                f"Cloning voice for speaker {speaker!r} failed ({exc}); "
                                "falling back to a preset voice"
                            )
                        else:
                            voices[speaker] = voice_id
                            cloned_ids.append(voice_id)
                            continue
                    elif want_clone and can_clone:
                        logger.warning(
                            f"Speaker {speaker!r} has only {seconds_gathered:.1f}s of clean "
                            f"audio (need {clone_min_seconds:.1f}s to clone); falling back "
                            "to a preset voice"
                        )

                    profile = classify_voice(sample_path) if sample_path is not None else None
                    voices[speaker] = _preset_voice(presets, tts_engine, profile)

        yield voices
    finally:
        if cloned_ids:
            if keep:
                logger.info(
                    f"Keeping cloned voices (--keep-cloned-voices): {', '.join(cloned_ids)}"
                )
            else:
                _cleanup_cloned_voices(cloned_ids)
