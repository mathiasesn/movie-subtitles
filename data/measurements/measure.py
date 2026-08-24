"""Analysis script behind specs/chars-per-second-measurement.md -- committed record.

Two logs sit next to this script: clip-translate-synth.measure.log and
trailer-translate-synth.measure.log, already filtered down to just the
`[measure]` lines from real runs. Parsing them offline (the default, no
flags) reproduces:
  - median source English chars/slot-second (from both logs' translate lines)
  - median tts-1 Danish chars/sec at 1.0x (from both logs' synth pass=0 lines)

Reproducing the third figure -- the median unconstrained tgt_chars/src_chars
en->da expansion ratio -- requires re-deriving English source text for the
45s clip via a fresh ElevenLabs ASR call, then translating each recovered
segment unconstrained (budget_chars=None) with the same Anthropic-backed
provider used in the real runs. That costs real money (~$0.01 ElevenLabs ASR
+ Anthropic translation of ~16 short segments) and is only run when this
script is invoked with --rerun-paid. Default (no flag) behaviour spends
nothing.

Usage:
    uv run python data/measurements/measure.py               # free, offline
    uv run python data/measurements/measure.py --rerun-paid   # + paid ASR/translation re-run
"""

import argparse
import re
import statistics
from pathlib import Path

# Stated once and reused by --rerun-paid's help text and its skip notice, so the
# three places that quote the price cannot drift apart.
_PAID_COST = "~$0.01 ElevenLabs ASR + Anthropic translation of ~16 short segments"

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

CLIP_LOG = _SCRIPT_DIR / "clip-translate-synth.measure.log"
TRAILER_LOG = _SCRIPT_DIR / "trailer-translate-synth.measure.log"
CLIP_MEDIA = _REPO_ROOT / "data" / "paw-patrol-the-dino-movie-clip.webm"

TRANSLATE_RE = re.compile(
    r"measure=translate id=(?P<id>\d+) start=(?P<start>[\d.]+) end=(?P<end>[\d.]+) "
    r"slot=(?P<slot>[\d.]+) src_chars=(?P<src_chars>\d+) tgt_chars=(?P<tgt_chars>\d+) "
    r"budget=(?P<budget>\d+) lang=(?P<lang>\w+)"
)
SYNTH_RE = re.compile(
    r"measure=synth id=(?P<id>\d+) pass=(?P<pass_>\d+) rate=(?P<rate>[\d.]+) "
    r"tgt_chars=(?P<tgt_chars>\d+) speech_len=(?P<speech_len>[\d.]+) "
    r"src_speech=(?P<src_speech>[\d.]+)"
)


def parse_logs(*paths):
    """Pooled (translate, synth) records from every named log, in the order given."""
    translates, synths = [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                m = TRANSLATE_RE.search(line)
                if m:
                    translates.append(m.groupdict())
                    continue
                m = SYNTH_RE.search(line)
                if m:
                    synths.append(m.groupdict())
    return translates, synths


def median_source_cps(all_translates):
    src_cps = []
    for t in all_translates:
        slot = float(t["slot"])
        src_chars = int(t["src_chars"])
        if slot <= 0 or src_chars == 0:
            continue
        src_cps.append(src_chars / slot)
    return statistics.median(src_cps), len(src_cps)


def median_tts_cps(all_synths):
    da_cps = []
    for s in all_synths:
        if s["pass_"] != "0":
            continue
        if abs(float(s["rate"]) - 1.0) > 1e-6:
            continue
        speech_len = float(s["speech_len"])
        tgt_chars = int(s["tgt_chars"])
        if speech_len <= 0 or tgt_chars == 0:
            continue
        da_cps.append(tgt_chars / speech_len)
    return statistics.median(da_cps), len(da_cps)


def rerun_paid_expansion_ratio():
    # Imported here, not at module scope, so the default (free) path pulls in no
    # vendor SDK, constructs no client and reads no .env. _load_env is the package's
    # single .env rule (find_dotenv(usecwd=True), never overriding exported vars);
    # calling it beats hand-rolling a second, divergent one here. No sys.path
    # manipulation: `uv run` already puts the package on the path.
    from movie_subtitles.cli import _load_env
    from movie_subtitles.providers.elevenlabs import ScribeTranscribe
    from movie_subtitles.providers.llm import LLMTranslate

    _load_env()

    print("Running ElevenLabs ASR on the 45s clip to recover English text...")
    transcribe = ScribeTranscribe(diarize=False)
    segments = list(transcribe(str(CLIP_MEDIA), "en"))
    print(f"Recovered {len(segments)} English segments from clip.")

    translator = LLMTranslate()
    ratios = []
    for seg in segments:
        src_text = seg.text.strip()
        if not src_text:
            continue
        tgt_text = translator(src_text, "da", budget_chars=None)
        src_chars = len(src_text)
        tgt_chars = len(tgt_text)
        if src_chars == 0:
            continue
        ratios.append(tgt_chars / src_chars)
        print(
            f"  id={seg.id} src_chars={src_chars} tgt_chars={tgt_chars} "
            f"ratio={tgt_chars / src_chars:.3f} src={src_text!r} tgt={tgt_text!r}"
        )

    median_ratio = statistics.median(ratios)
    print(
        f"\nmedian unconstrained en->da tgt_chars/src_chars ratio: "
        f"{median_ratio:.3f} (n={len(ratios)})"
    )
    return median_ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun-paid",
        action="store_true",
        help=(
            "Re-derive the unconstrained en->da expansion ratio via a fresh "
            "ElevenLabs ASR call + Anthropic translation of the 45s clip. "
            f"Costs real money ({_PAID_COST}). Default is off."
        ),
    )
    args = parser.parse_args()

    all_translates, all_synths = parse_logs(CLIP_LOG, TRAILER_LOG)

    median_src_cps, n_src = median_source_cps(all_translates)
    print(f"median source English chars/slot-second: {median_src_cps:.3f} (n={n_src})")

    median_da_cps, n_da = median_tts_cps(all_synths)
    print(f"median tts-1 Danish chars/sec @1.0x: {median_da_cps:.3f} (n={n_da})")

    if args.rerun_paid:
        median_ratio = rerun_paid_expansion_ratio()
    else:
        median_ratio = None
        print(
            f"\nSkipping the unconstrained en->da expansion-ratio figure "
            f"(requires --rerun-paid; costs {_PAID_COST})."
        )

    print("\n--- SUMMARY ---")
    print(f"median src chars/slot-second: {median_src_cps:.3f}")
    print(f"median da chars/sec @1.0x (tts-1): {median_da_cps:.3f}")
    if median_ratio is not None:
        print(f"median unconstrained en->da ratio: {median_ratio:.3f}")


if __name__ == "__main__":
    main()
