"""pyannote.audio-backed speaker diarisation: whole-file turns for non-diarizing ASR.

Feeds #27 (`specs/diarisation-for-local-and-openai-asr.md`): `--asr-engine local` and
`--asr-engine openai` return no speaker information at all, so `Segment.speaker` stays
`None` for every cue and `--voice-match` degrades to a single narrator voice. This module
runs `pyannote/speaker-diarization-community-1` over the source audio once, up front, and
hands `cli.py` back a list of `Turn`s to merge onto ASR segments by temporal overlap.

Named `pyannote_.py`, trailing underscore, matching `providers/openai_.py`'s convention:
without it the module would shadow the installed `pyannote` package on `sys.path` and
break its own imports.

Every `pyannote.audio`/`torch` import is lazy -- inside `Diarize.__init__` and
`Diarize.diarize` -- per this repo's convention (see `separate.py`'s lazy `demucs`/`torch`
imports): a run that never diarizes (`--voice-match off`, or `--asr-engine elevenlabs`,
which diarizes natively via Scribe) must not pay this import cost.

`pyannote/speaker-diarization-community-1` is a gated model: first use requires a Hugging
Face account, accepting the model's conditions on its model page, and a token supplied at
`Pipeline.from_pretrained()` time. The token is read from `HF_TOKEN`, the `huggingface_hub`
standard. `Pipeline.from_pretrained` can fail two different ways when the token or
acceptance is missing -- raising, or (per the Hub client) returning `None` -- and both are
translated here into one actionable `RuntimeError` naming the env var, the model page, and
the required acceptance step, rather than letting an opaque 401/403 propagate.

This module never catches its own runtime/inference errors -- it raises, exactly like
`separate.py`; `cli.py` is what decides whether a diarisation failure degrades the run
(warn and continue with `speaker=None` everywhere) or, for an `ImportError`, propagates and
fails it.
"""

import logging
import os
import time
from pathlib import Path

from movie_subtitles.providers.base import Turn

logger = logging.getLogger("pyannote")

_MODEL_NAME = "pyannote/speaker-diarization-community-1"

_MODEL_PAGE = f"https://huggingface.co/{_MODEL_NAME}"

_TOKEN_ENV_VAR = "HF_TOKEN"


class Diarize:
    """Loads the community-1 diarisation pipeline once, then diarizes files with it.

    Follows this repo's model-wrapper convention: the pipeline is loaded in `__init__`,
    `diarize()` is the named method that does the work, and `__call__` delegates to it.
    """

    def __init__(self) -> None:
        # Lazy import: pyannote.audio (and therefore torch, if not already loaded) must
        # never be imported by a run that doesn't diarize.
        import torch
        from pyannote.audio import Pipeline

        # Defensive import: these live in huggingface_hub.errors on current versions
        # and huggingface_hub.utils on older ones. Only these exception types indicate
        # an auth/gating failure; anything else (network outage, corrupt cache, disk
        # error, ...) must propagate unchanged rather than being mislabelled.
        try:
            from huggingface_hub.errors import (
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            )
        except ImportError:
            from huggingface_hub.utils import (
                GatedRepoError,
                HfHubHTTPError,
                RepositoryNotFoundError,
            )

        token = os.environ.get(_TOKEN_ENV_VAR)
        try:
            pipeline = Pipeline.from_pretrained(_MODEL_NAME, token=token)
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
            raise RuntimeError(self._credential_message()) from exc
        if pipeline is None:
            raise RuntimeError(self._credential_message())

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {_MODEL_NAME} on {device}.")
        self._pipeline = pipeline.to(torch.device(device))

    @staticmethod
    def _credential_message() -> str:
        return (
            f"Failed to load {_MODEL_NAME}. This is a gated model: set the {_TOKEN_ENV_VAR} "
            f"environment variable to a Hugging Face access token, and make sure you have "
            f"accepted the model's conditions at {_MODEL_PAGE} (log in, then accept on that "
            "page) before it will load."
        )

    def __call__(self, fpath: str | Path) -> list[Turn]:
        return self.diarize(fpath)

    def diarize(self, fpath: str | Path) -> list[Turn]:
        """Diarize `fpath`'s audio, returning one `Turn` per contiguous speaker turn.

        Turns are sorted by start time. Raises whatever the pipeline raises -- nothing
        here is caught; `cli.py` owns the decision to degrade on a runtime failure.
        """
        logger.info(f"Diarizing {Path(fpath).name} with {_MODEL_NAME}.")
        start = time.monotonic()
        diarization = self._pipeline(str(fpath))
        elapsed = time.monotonic() - start

        turns = [
            Turn(start=segment.start, end=segment.end, speaker=str(speaker))
            for segment, _, speaker in diarization.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda turn: turn.start)

        speakers = {turn.speaker for turn in turns}
        logger.info(
            f"Diarization found {len(turns)} turn(s) across {len(speakers)} speaker(s) "
            f"in {elapsed:.1f}s."
        )
        return turns
