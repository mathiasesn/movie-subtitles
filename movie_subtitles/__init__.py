import logging
import os
import sys

# MOVIE_SUBTITLES_LOG_LEVEL: no CLI flag by design -- measurement runs (see
# specs/chars-per-second-measurement.md) need the per-segment DEBUG lines cli.py
# and dub.py emit, without every invocation growing a --log-level flag. An unset or
# unrecognised value falls back to INFO rather than crashing.
_LEVEL_NAME = os.environ.get("MOVIE_SUBTITLES_LOG_LEVEL", "INFO").upper()
_LEVEL = logging.getLevelName(_LEVEL_NAME)
if not isinstance(_LEVEL, int):
    _LEVEL = logging.INFO

logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%d/%m/%Y-%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=_LEVEL,
)
