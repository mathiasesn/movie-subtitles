"""Demucs-backed source separation: split a source's audio into vocals/accompaniment.

Feeds #24 (`--separate-background`, `specs/separate-background-stem-for-dub.md`):
`mux_dub` wants to mix the dub over a bed that no longer contains the *original*
dialogue, rather than merely ducking that dialogue underneath it. Producing that bed
needs real source separation, not just an ffmpeg filter, hence Demucs.

Every demucs import is lazy -- `demucs.api` inside `Separate.__init__`, `demucs.audio`
inside `Separate.separate` -- per this repo's convention (see `voices.py:classify_voice`'s
lazy `librosa`/`parselmouth` imports): a run without `--separate-background`, the
overwhelming majority of runs, must not pay Demucs's (and by extension torch's) import
cost.

The work goes through `demucs.api.Separator` rather than a hand-rolled
`get_model`/`apply_model`/load/save pipeline. That is not just less code: `Separator`
already owns the decode (via ffmpeg, resampling to the model's own rate/channels itself),
the batching, `no_grad`, and the per-chunk device transfers, so this module neither
extracts an intermediate wav of its own -- a full-length, ~1.3 GB-for-a-feature disk
round-trip when it did -- nor needs a second audio-I/O library to read one back.

Demucs is a hard runtime dependency (see `pyproject.toml`); this module never swallows
an exception itself, so both kinds of failure propagate out of it. The two kinds are
handled differently by the caller: `cli.py` catches a *runtime* one -- unreachable model
weights, a bad input file, a decode/inference error -- and degrades to the #22
duck-and-mix path, but an `ImportError`/`ModuleNotFoundError` -- including from this
module's own lazy `demucs.api`/`torch` imports -- is re-raised and fails the run instead,
since a broken install must not silently produce a worse output audio bed.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger("separate")

# The only Demucs model this module supports. htdemucs is Demucs's default
# general-purpose 4-stem (vocals/drums/bass/other) model -- there's no need to expose a
# choice of model, since the only thing #24 wants out of it is "not vocals".
_MODEL_NAME = "htdemucs"

_VOCALS_STEM = "vocals"

# 16-bit is what the accompaniment is for: `mux_dub` reads it straight back through
# ffmpeg and re-encodes it into the output container anyway, so `save_audio`'s float
# default would double this file's size (~2.5 GB for a feature) to no audible end.
_OUTPUT_BITS_PER_SAMPLE = 16


class Separate:
    """Loads a Demucs model once, then splits a source file's audio into stems.

    Follows this repo's model-wrapper convention: the model is loaded in `__init__`,
    `separate()` is the named method that does the work, and `__call__` delegates to it.
    """

    def __init__(self) -> None:
        # Lazy import: demucs (and therefore torch, if not already loaded) must never
        # be imported by a run that doesn't pass --separate-background.
        import torch
        from demucs.api import Separator

        # Demucs defaults to the device its input tensor is on, i.e. the CPU, where
        # htdemucs is roughly an order of magnitude slower than on CUDA. torch is
        # already a dependency of this project and `providers/local.py` takes the same
        # posture with `device_map="auto"`, so prefer a GPU when the machine has one.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading the {_MODEL_NAME} separation model on {device}.")
        self._separator = Separator(model=_MODEL_NAME, device=device)

    def __call__(self, fpath: Path, out_path: Path) -> Path:
        return self.separate(fpath, out_path)

    def separate(self, fpath: Path, out_path: Path) -> Path:
        """Separate `fpath`'s audio and write the accompaniment stem to `out_path`.

        The accompaniment is every Demucs stem except `vocals`, summed -- equivalently
        Demucs's `--two-stems=vocals` complement -- keyed by stem name off what the model
        actually returns rather than a hardcoded stem order, since that set is a model
        property, not a constant this module should assume.

        The stems are summed in place, into a single clone of the first non-vocals stem.
        That matters at this size: the returned stems are float32 and full-length (~2.5 GB
        each for a 2-hour film at the model's 44.1 kHz stereo), so summing them with the
        obvious `sum(...)` would allocate a fresh full-length tensor per addition. The
        stem dict is dropped before the write, so the vocals stem -- the one thing here
        that exists only to be discarded -- isn't held resident across it.

        Demucs works internally at 44.1 kHz stereo, so a 5.1/48 kHz source is
        downmixed/resampled by `Separator` and does not keep its original layout in the
        accompaniment file -- `mux_dub` re-targets the final muxed output to the source's
        own format separately.

        This is CPU-minutes-to-tens-of-minutes on a feature-length film (less on a GPU;
        see `__init__`); callers should expect and communicate that, not treat a
        long-running call as hung.

        Raises whatever `Separator` or the write step raises -- nothing here is caught.
        """
        from demucs.audio import save_audio

        logger.info(
            f"Separating {fpath.name} into vocals/accompaniment stems via {_MODEL_NAME} "
            "-- this can take minutes to tens of minutes on CPU for a feature-length film."
        )
        start = time.monotonic()
        _, stems = self._separator.separate_audio_file(fpath)
        logger.info(f"Source separation finished in {time.monotonic() - start:.1f}s.")

        accompaniment = None
        for name, stem in stems.items():
            if name == _VOCALS_STEM:
                continue
            accompaniment = stem.clone() if accompaniment is None else accompaniment.add_(stem)
        if accompaniment is None:
            raise RuntimeError(
                f"{_MODEL_NAME} returned no stem other than '{_VOCALS_STEM}' "
                f"(got {sorted(stems)}), so there is no accompaniment to mix the dub over."
            )
        stems.clear()

        save_audio(
            accompaniment,
            out_path,
            self._separator.samplerate,
            bits_per_sample=_OUTPUT_BITS_PER_SAMPLE,
        )
        return out_path
