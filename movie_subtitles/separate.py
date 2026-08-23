"""Demucs-backed source separation: split a source's audio into vocals/accompaniment.

Feeds #24 (`--separate-background`, `specs/separate-background-stem-for-dub.md`):
`mux_dub` wants to mix the dub over a bed that no longer contains the *original*
dialogue, rather than merely ducking that dialogue underneath it. Producing that bed
needs real source separation, not just an ffmpeg filter, hence Demucs.

Both `demucs.pretrained` and `demucs.apply` are imported lazily, inside `Separate.__init__`
and `Separate.separate` respectively, per this repo's convention (see
`voices.py:classify_voice`'s lazy `librosa`/`parselmouth` imports): a run without
`--separate-background` -- the overwhelming majority of runs -- must not pay Demucs's
(and by extension torch's) import cost.

Demucs is a hard runtime dependency (see `pyproject.toml`), so the failure this module
must let propagate is a *runtime* one -- unreachable model weights, a bad input file, an
`apply_model`/ffmpeg error -- not a missing package. The caller (`cli.py`, not this
module) is responsible for catching that and degrading to the #22 duck-and-mix path;
this module never swallows an exception itself.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path

from movie_subtitles import ffmpeg

logger = logging.getLogger("separate")

# The only Demucs model this module supports. htdemucs is Demucs's default
# general-purpose 4-stem (vocals/drums/bass/other) model -- there's no need to expose a
# choice of model, since the only thing #24 wants out of it is "not vocals".
_MODEL_NAME = "htdemucs"

_VOCALS_STEM = "vocals"


class Separate:
    """Loads a Demucs model once, then splits a source file's audio into stems.

    Follows this repo's model-wrapper convention: the model is loaded in `__init__`,
    `separate()` is the named method that does the work, and `__call__` delegates to it.
    """

    def __init__(self) -> None:
        # Lazy import: demucs (and therefore torch, if not already loaded) must never
        # be imported by a run that doesn't pass --separate-background.
        from demucs.pretrained import get_model

        self._model = get_model(_MODEL_NAME)
        self._model.eval()

    def __call__(self, fpath: Path, out_path: Path) -> Path:
        return self.separate(fpath, out_path)

    def separate(self, fpath: Path, out_path: Path) -> Path:
        """Extract `fpath`'s audio, separate it, and write the accompaniment to `out_path`.

        The accompaniment is every Demucs stem except `vocals`, summed -- equivalently
        Demucs's `--two-stems=vocals` complement -- indexed via `self._model.sources`
        rather than a hardcoded stem order, since that order is a model property, not a
        constant this module should assume.

        Runs at the model's own expected sample rate/channel count
        (`self._model.samplerate`/`self._model.audio_channels`), so no resampling
        happens between ffmpeg's extraction and Demucs's forward pass. Demucs works
        internally at 44.1 kHz stereo, so a 5.1/48 kHz source is downmixed/resampled
        here and does not keep its original layout in the returned accompaniment file
        -- `mux_dub` re-targets the final output to the source's own format separately.

        This is CPU-minutes-to-tens-of-minutes on a feature-length film; callers should
        expect and communicate that, not treat a long-running call as hung.

        Raises whatever `ffmpeg.run`, `torch`/`demucs`, or the write step raises --
        nothing here is caught. Cleans up its own temp files in a `finally`, including
        on failure.
        """
        import torch
        from demucs.apply import apply_model

        tmp_dir = Path(tempfile.mkdtemp(prefix="movie-subtitles-separate-"))
        extracted_wav = tmp_dir / "extracted.wav"
        try:
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(fpath),
                    "-vn",
                    "-ac",
                    str(self._model.audio_channels),
                    "-ar",
                    str(self._model.samplerate),
                    "-c:a",
                    "pcm_s16le",
                    str(extracted_wav),
                ],
                what=f"Extracting audio from {fpath.name} for source separation",
            )

            import torchaudio

            waveform, _ = torchaudio.load(str(extracted_wav))
            # apply_model expects a batch dimension: (batch, channels, samples).
            mixture = waveform.unsqueeze(0)

            logger.info(
                f"Separating source audio into vocals/accompaniment stems via {_MODEL_NAME} -- "
                "this can take minutes to tens of minutes on CPU for a feature-length film."
            )
            start = time.monotonic()
            with torch.no_grad():
                stems = apply_model(self._model, mixture)[0]
            elapsed = time.monotonic() - start
            logger.info(f"Source separation finished in {elapsed:.1f}s.")

            accompaniment = sum(
                stems[i] for i, source in enumerate(self._model.sources) if source != _VOCALS_STEM
            )

            torchaudio.save(str(out_path), accompaniment, self._model.samplerate)
        finally:
            # Cleanup must happen even on failure, so this lives in `finally` rather
            # than after the `try` block.
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return out_path
