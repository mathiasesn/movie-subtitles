"""Speaker -> TTS voice resolution: sample extraction, cloning, and preset matching.

Owns everything that maps a diarized `Segment.speaker` label to a TTS voice id, for
both the `elevenlabs` (instant voice cloning + curated presets) and `openai` (preset
only) TTS engines. See `specs/speaker-matched-dub-voices.md` for the design.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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


def _segments_overlap(a: Segment, b: Segment) -> bool:
    return a.start < b.end and b.start < a.end


def extract_speaker_sample(
    source: str | Path,
    speaker: str,
    segments: list[Segment],
    *,
    target_seconds: float = CLONE_TARGET_SECONDS,
    out_dir: str | Path | None = None,
) -> tuple[Path, float]:
    """Extract up to `target_seconds` of clean audio for one speaker.

    Selects `speaker`'s segments where no other speaker's segment overlaps in time,
    cuts each from `source` with ffmpeg as WAV, and concatenates them (via ffmpeg's
    concat demuxer) in start order until `target_seconds` is reached or the clean
    segments run out. Returns the concatenated sample path and the seconds actually
    gathered (which may be less than `target_seconds`).

    Produced once per speaker regardless of whether cloning or preset matching ends
    up consuming it.
    """
    source = Path(source)
    workdir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="voices-"))
    workdir.mkdir(parents=True, exist_ok=True)

    own_segments = sorted((s for s in segments if s.speaker == speaker), key=lambda s: s.start)
    other_segments = [s for s in segments if s.speaker is not None and s.speaker != speaker]

    clean = [
        s for s in own_segments if not any(_segments_overlap(s, other) for other in other_segments)
    ]

    gathered = 0.0
    clip_paths: list[Path] = []
    for i, seg in enumerate(clean):
        if gathered >= target_seconds:
            break
        duration = seg.end - seg.start
        if duration <= 0:
            continue

        take = min(duration, target_seconds - gathered)
        clip_path = workdir / f"{speaker}_{i}.wav"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-ss",
                f"{seg.start:.3f}",
                "-t",
                f"{take:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(clip_path),
            ],
            what=f"extracting sample clip for speaker {speaker!r}",
        )
        clip_paths.append(clip_path)
        gathered += take

    sample_path = workdir / f"{speaker}_sample.wav"
    if not clip_paths:
        # No clean audio at all: still produce an (empty) file so callers have a
        # uniform return type; classification/cloning downstream must handle 0.0
        # gathered seconds as "ineligible" rather than treat this as an error.
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.01",
                str(sample_path),
            ],
            what=f"writing empty sample placeholder for speaker {speaker!r}",
        )
        return sample_path, 0.0

    if len(clip_paths) == 1:
        clip_paths[0].rename(sample_path)
        return sample_path, gathered

    concat_list = workdir / f"{speaker}_concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths) + "\n")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(sample_path),
        ],
        what=f"concatenating sample clips for speaker {speaker!r}",
    )
    return sample_path, gathered


def classify_voice(sample_path: str | Path) -> tuple[str, str] | None:
    """Classify a sample WAV into a coarse (gender, age_band) profile.

    Uses `librosa.pyin` for a noise-robust median F0 and `praat-parselmouth` for
    F1/F2 formants. Both libraries are imported lazily, inside this function, so
    `--voice-match off` (and any run that never classifies) never pays their import
    cost. Degrades to `None` (an "unknown" profile) rather than raising if either
    analysis fails -- movie audio is noisy and a bad sample must fall back to the
    default preset, not abort the run.
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
        y, sr = librosa.load(str(sample_path), sr=None, mono=True)
        if y.size == 0:
            raise ValueError("empty sample audio")

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C6")),
            sr=sr,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if voiced_f0.size == 0:
            raise ValueError("no voiced frames detected")
        median_f0 = float(np.median(voiced_f0))

        snd = parselmouth.Sound(str(sample_path))
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

    # Coarse, heuristic thresholds: F0 is the primary gender cue, formants (which
    # track vocal-tract length) refine it and separate the age bands. These are not
    # tuned against a labelled dataset -- see specs/speaker-matched-dub-voices.md's
    # "Risks" section.
    gender = "female" if median_f0 >= 165.0 else "male"

    formant_avg = (median_f1 + median_f2) / 2.0
    if median_f0 >= 220.0 or formant_avg >= 2200.0:
        age_band = "young"
    elif median_f0 <= 110.0 or formant_avg <= 1400.0:
        age_band = "elderly"
    else:
        age_band = "adult"

    return gender, age_band


@dataclass
class VoiceResolution:
    """Result of `resolve_voices`: the speaker -> voice-id mapping, plus bookkeeping
    the caller needs to clean up any voices this call cloned."""

    voices: dict[str | None, str | None]
    cloned_voice_ids: list[str] = field(default_factory=list)


def resolve_voices(
    source: str | Path,
    segments: list[Segment],
    *,
    tts_engine: str,
    mode: str = "auto",
    clone_min_seconds: float = CLONE_MIN_SECONDS,
    clone_target_seconds: float = CLONE_TARGET_SECONDS,
    preset_table_path: str | Path | None = None,
    voice_name_prefix: str = "movie-subtitles",
) -> VoiceResolution:
    """Resolve a speaker -> TTS voice id mapping for `--dub`, honouring `--voice-match`.

    - `mode="off"`: returns an all-empty mapping. Does no diarization work, sample
      extraction, or lazy import -- byte-for-byte today's (pre-diarization) behaviour.
    - `mode="clone"`: clones every eligible speaker (ElevenLabs IVC only). A speaker
      with fewer than `clone_min_seconds` of clean audio falls back to a preset, with
      a WARNING naming the speaker and the seconds actually found.
    - `mode="preset"`: classifies each speaker and picks a preset voice; never clones.
    - `mode="auto"`: clones when `tts_engine` supports cloning (elevenlabs), otherwise
      (and for any speaker ineligible to clone) falls back to preset matching.

    Cloning is never attempted for `tts_engine="openai"` in any mode.

    Returns a `VoiceResolution` carrying both the mapping and the list of voice ids
    this call cloned, so the caller can delete them (via `cleanup_cloned_voices`) in a
    `finally` once the dub is done.
    """
    if mode not in VOICE_MATCH_MODES:
        raise ValueError(f"Unknown voice-match mode {mode!r} (expected one of {VOICE_MATCH_MODES})")

    if mode == "off":
        return VoiceResolution(voices={})

    speakers = sorted({s.speaker for s in segments if s.speaker is not None})
    if not speakers:
        logger.info("No segment carries a speaker label; dubbing with a single voice")
        return VoiceResolution(voices={})

    presets = (
        _DEFAULT_PRESETS if preset_table_path is None else load_preset_table(preset_table_path)
    )

    can_clone = tts_engine in _CLONE_CAPABLE_ENGINES
    want_clone = mode == "clone" or (mode == "auto" and can_clone)

    voices: dict[str | None, str | None] = {}
    cloned_ids: list[str] = []

    client = None
    if want_clone and can_clone:
        from movie_subtitles.providers.elevenlabs import build_client

        client = build_client()

    for speaker in speakers:
        if want_clone and can_clone:
            sample_path, seconds_gathered = extract_speaker_sample(
                source, speaker, segments, target_seconds=clone_target_seconds
            )
            if seconds_gathered >= clone_min_seconds:
                from movie_subtitles.providers.elevenlabs import clone_voice

                name = f"{voice_name_prefix} {Path(source).stem} {speaker}"
                try:
                    voice_id = clone_voice(client, name, [sample_path])
                except Exception as exc:  # noqa: BLE001 -- degrade to preset, never abort
                    logger.warning(
                        f"Cloning voice for speaker {speaker!r} failed ({exc}); falling "
                        "back to a preset voice"
                    )
                    profile = classify_voice(sample_path)
                    voices[speaker] = _preset_voice(presets, tts_engine, profile)
                    continue

                voices[speaker] = voice_id
                cloned_ids.append(voice_id)
                continue

            logger.warning(
                f"Speaker {speaker!r} has only {seconds_gathered:.1f}s of clean audio "
                f"(need {clone_min_seconds:.1f}s to clone); falling back to a preset voice"
            )
            profile = classify_voice(sample_path)
        else:
            sample_path, _seconds_gathered = extract_speaker_sample(
                source, speaker, segments, target_seconds=clone_target_seconds
            )
            profile = classify_voice(sample_path)

        voices[speaker] = _preset_voice(presets, tts_engine, profile)

    return VoiceResolution(voices=voices, cloned_voice_ids=cloned_ids)


def cleanup_cloned_voices(cloned_voice_ids: list[str]) -> None:
    """Delete previously cloned voices, individually guarded.

    A failing delete logs a WARNING naming the leaked voice id and never aborts the
    remaining deletions, nor masks an exception already propagating from the caller's
    `finally` block.
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
