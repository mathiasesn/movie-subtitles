import logging
import os
import sys

# Every logger name this package uses. They are flat -- modules call
# `logging.getLogger("dub")`, not `logging.getLogger("movie_subtitles.dub")` -- so
# there is no common ancestor to set a level on and each one has to be named here.
# Add a new module's logger name when you add the module.
_PACKAGE_LOGGERS = (
    "cli",
    "dub",
    "dubbing",
    "elevenlabs",
    "fallback",
    "llm",
    "local",
    "mux",
    "openai_provider",
    "pyannote",
    "separate",
    "voices",
)


def _log_level() -> int:
    """The level named by MOVIE_SUBTITLES_LOG_LEVEL, or INFO if unset/unrecognised.

    `getLevelName` hands back a string for a name it does not know, which is the
    documented-as-discouraged reverse use of it; the isinstance check is what turns
    that into the INFO fallback. `basicConfig(level="DEBUG")` would accept a level
    name directly but raises on an unknown one, so the conversion cannot be dropped.
    """
    name = os.environ.get("MOVIE_SUBTITLES_LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else logging.INFO


def configure_logging() -> None:
    """Apply MOVIE_SUBTITLES_LOG_LEVEL to this package's own loggers.

    The root level deliberately stays at INFO. Raising it is what would also switch
    on every vendor library's DEBUG output -- on one measurement run httpcore,
    openai and anthropic together emitted 1968 lines against this package's 153,
    burying the `[measure]` lines the variable exists to surface. Setting the level
    per package logger keeps that noise off while still reaching every line we emit,
    since a record only has to clear its own logger's level before propagating to
    the root handler (which is left at NOTSET by `basicConfig`).

    Called once on import, so library callers that never touch `main()` still get
    it, and again from `cli.main()` after `_load_env()`, so the variable can be set
    in `.env` like every other one rather than needing to be exported beforehand.
    """
    level = _log_level()
    for name in _PACKAGE_LOGGERS:
        logging.getLogger(name).setLevel(level)


# MOVIE_SUBTITLES_LOG_LEVEL has no CLI flag by design: measurement runs (see
# specs/chars-per-second-measurement.md) need the per-segment DEBUG lines cli.py and
# dub.py emit, without every invocation growing a --log-level flag.
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%d/%m/%Y-%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
configure_logging()
