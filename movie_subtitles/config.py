"""Strict YAML config-file loader for CLI defaults (issue #14).

`.env` covers secrets only, and the JSON preset table covers voice presets only --
neither is a place for "always run with `--srt-lang da --voice-match preset`" style
defaults. This module is that third piece: a single per-user YAML file whose keys mirror
`cli.py`'s argparse `dest` names, loaded once by `main()` and applied via
`parser.set_defaults(**resolved)` so precedence (explicit flag > config file > built-in
default) falls out of argparse itself rather than needing sentinel defaults everywhere.

Deliberately `main()`-only: importing this module (or any other) still reads no file from
disk, matching `cli.py:_load_env()`'s posture, and `create_subtitles()` keeps its own
signature/defaults for library callers that never touch the CLI.

Validation is strict and fails before any work starts, in the style of
`voices.py:load_preset_table`: an unknown key, wrong type, or out-of-choices value raises
`ValueError` naming the file and the key. This module does not import `cli.py` (cli.py
will import this module instead -- importing back would be circular).
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from movie_subtitles.voices import VOICE_MATCH_MODES

logger = logging.getLogger("config")


def default_config_path() -> Path:
    """The single per-user config file path, honouring XDG_CONFIG_HOME.

    Computed on each call rather than once at import time: cli.py imports this module
    before main() calls _load_env(), so an XDG_CONFIG_HOME (or HOME) set only in .env
    would be silently ignored by a module-level constant -- the same ordering trap
    cli.py already works around by re-calling configure_logging() after _load_env().

    There is deliberately no per-project/cwd config file: this is a per-user tool-
    preferences file, and a per-checkout one would make a run's behaviour depend on cwd.
    """
    return (
        Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        / "movie-subtitles"
        / "config.yaml"
    )


# Secret-like key names rejected with a tailored message pointing at .env, checked before
# the generic unknown-key error so the message teaches rather than just refusing. Compared
# case-insensitively against every config key.
_SECRET_KEYS = frozenset({"elevenlabs_api_key", "anthropic_api_key", "openai_api_key", "hf_token"})

# Engine choices, mirrored from cli.py's argparse `choices=` for the corresponding flags.
# Duplicated deliberately (repo convention -- see AGENTS.md "Defaults are duplicated"):
# config.py must not import cli.py (that would be circular), so these three are the one
# copy that can't instead be read off an argparse action.
_ENGINE_CHOICES = ("local", "elevenlabs", "openai")
_TRANSLATION_ENGINE_CHOICES = ("local", "anthropic", "openai")
_TTS_ENGINE_CHOICES = ("elevenlabs", "openai")

# Kinds of validation applied per key below.
_STR = "str"
_BOOL = "bool"
_POSITIVE_INT = "positive_int"
_POSITIVE_FLOAT = "positive_float"
_UNIT_FLOAT = "unit_float"
_CHOICE = "choice"

# Declared schema of every configurable key: every CLI option except --input and
# --config themselves. Each entry is (kind, choices-or-None, nullable). "nullable" keys
# mirror flags whose argparse default is None (an unresolved engine/table/duck-level),
# so an explicit `null` in YAML is accepted and passed through rather than rejected --
# it still means "not set", the same as omitting the key.
_SCHEMA: dict[str, tuple[str, tuple[str, ...] | None, bool]] = {
    "audio_lang": (_STR, None, False),
    "srt_lang": (_STR, None, False),
    "whisper_model": (_STR, None, False),
    "mt_model": (_STR, None, False),
    "engine": (_CHOICE, _ENGINE_CHOICES, False),
    "asr_engine": (_CHOICE, _ENGINE_CHOICES, True),
    "translation_engine": (_CHOICE, _TRANSLATION_ENGINE_CHOICES, True),
    "tts_engine": (_CHOICE, _TTS_ENGINE_CHOICES, True),
    "dub": (_BOOL, None, False),
    "dub_workers": (_POSITIVE_INT, None, False),
    "dub_correction_passes": (_POSITIVE_INT, None, False),
    "voice_match": (_CHOICE, VOICE_MATCH_MODES, False),
    "keep_cloned_voices": (_BOOL, None, False),
    "clone_min_seconds": (_POSITIVE_FLOAT, None, False),
    "clone_target_seconds": (_POSITIVE_FLOAT, None, False),
    "voice_preset_table": (_STR, None, True),
    "duck_level": (_UNIT_FLOAT, None, True),
    "separate_background": (_BOOL, None, False),
    "managed": (_BOOL, None, False),
}


def _validate_value(
    file: Path, key: str, value: Any, kind: str, choices: tuple[str, ...] | None, nullable: bool
) -> Any:
    if value is None:
        if nullable:
            return None
        raise ValueError(f"{file}: '{key}' cannot be null")

    if kind == _STR:
        if not isinstance(value, str):
            raise ValueError(f"{file}: '{key}' must be a string, got {value!r}")
        return value

    if kind == _CHOICE:
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{file}: '{key}' must be one of {list(choices)}, got {value!r}")
        return value

    if kind == _BOOL:
        # bool is a subclass of int in Python, so isinstance(True, int) is True too --
        # checked in this order deliberately, and this is exactly why YAML's `yes`/`no`/
        # `1`/`0` coercions can't be trusted here: they must be real YAML booleans.
        if not isinstance(value, bool):
            raise ValueError(f"{file}: '{key}' must be a boolean (true/false), got {value!r}")
        return value

    if kind == _POSITIVE_INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{file}: '{key}' must be a positive integer, got {value!r}")
        if value < 1:
            raise ValueError(f"{file}: '{key}' must be >= 1, got {value!r}")
        return value

    if kind == _POSITIVE_FLOAT:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{file}: '{key}' must be a positive number, got {value!r}")
        if value <= 0:
            raise ValueError(f"{file}: '{key}' must be > 0, got {value!r}")
        return float(value)

    if kind == _UNIT_FLOAT:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{file}: '{key}' must be a number in [0.0, 1.0], got {value!r}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{file}: '{key}' must be in [0.0, 1.0], got {value!r}")
        return float(value)

    raise AssertionError(f"unhandled schema kind {kind!r}")  # pragma: no cover


def _resolve_path(path: str | None) -> Path | None:
    """The file to load, or None if there is none to load.

    `path` given (from --config) must exist -- checked by the caller, which raises
    naming it explicitly. `path` None falls back to default_config_path(), where a
    missing file is not an error (mirrors _load_env()'s posture toward a missing .env).
    """
    if path is not None:
        return Path(path)
    default_path = default_config_path()
    return default_path if default_path.exists() else None


def load_config(path: str | None = None) -> dict[str, object]:
    """Load and strictly validate the YAML config file, returning a dict of overrides.

    `path` is the value of an explicit `--config <path>` flag; it must exist, else this
    raises ValueError naming it. With `path=None`, default_config_path() is used instead,
    and a missing default file returns {} rather than raising -- a config file is
    optional, same posture as .env.

    Validation order per key (checked in this order so the message teaches): (1) a
    known secret-key name (case-insensitively) -> tailored ValueError
    pointing at .env, (2) an unrecognised key -> ValueError listing the recognised ones,
    (3) wrong type / out-of-choices value -> ValueError naming file, key and the
    expected values. The returned dict is meant to be applied via
    parser.set_defaults(**resolved), so its keys are exactly the argparse dest names.
    """
    if path is not None and not Path(path).exists():
        raise ValueError(f"config file not found: {path}")

    file = _resolve_path(path)
    if file is None:
        return {}

    with file.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(f"{file}: top-level content must be a mapping of key: value, got {raw!r}")

    resolved: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{file}: config keys must be strings, got {key!r}")

        if key.lower() in _SECRET_KEYS:
            raise ValueError(
                f"{file}: '{key}' is an API key/secret and does not belong in the config "
                "file -- put it in .env instead"
            )

        if key not in _SCHEMA:
            raise ValueError(
                f"{file}: unrecognised config key '{key}'; recognised keys are: {sorted(_SCHEMA)}"
            )

        kind, choices, nullable = _SCHEMA[key]
        resolved[key] = _validate_value(file, key, value, kind, choices, nullable)

    logger.info(f"Loaded config from {file}")
    return resolved
