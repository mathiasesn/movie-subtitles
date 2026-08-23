"""Analysis script behind specs/chars-per-second-measurement.md -- committed record.

Recovers English source text for the 45s clip via one ElevenLabs ASR re-run
(cheap, ~$0.01), translates each segment unconstrained (budget_chars=None) with
the same Anthropic provider used in the real runs, and reports:
  - median unconstrained tgt_chars/src_chars ratio for en->da
  - median tts-1 Danish chars/sec at 1.0x (from run-clip.log + run-trailer.log
    synth pass=0 lines)
  - median source English chars/slot-second (from both logs' translate lines)
"""

import logging
import re
import statistics
import sys

sys.path.insert(0, "/home/mathias/code/movie-subtitles")

from dotenv import load_dotenv

from movie_subtitles.providers.elevenlabs import ScribeTranscribe
from movie_subtitles.providers.llm import LLMTranslate

load_dotenv("/home/mathias/code/movie-subtitles/.env")
logging.basicConfig(level=logging.WARNING)

_SCRATCHPAD = (
    "/tmp/claude-1000/-home-mathias-code-movie-subtitles/"
    "0dff05f5-9ab1-48ba-afea-bacd678e0d38/scratchpad"
)
CLIP_LOG = f"{_SCRATCHPAD}/run-clip.log"
TRAILER_LOG = f"{_SCRATCHPAD}/run-trailer.log"
CLIP_MEDIA = "/home/mathias/code/movie-subtitles/data/paw-patrol-the-dino-movie-clip.webm"

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


def parse_log(path):
    translates, synths = [], []
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


clip_translates, clip_synths = parse_log(CLIP_LOG)
trailer_translates, trailer_synths = parse_log(TRAILER_LOG)
all_translates = clip_translates + trailer_translates
all_synths = clip_synths + trailer_synths

# --- median source English chars/slot-second ---
src_cps = []
for t in all_translates:
    slot = float(t["slot"])
    src_chars = int(t["src_chars"])
    if slot <= 0 or src_chars == 0:
        continue
    src_cps.append(src_chars / slot)
median_src_cps = statistics.median(src_cps)
print(f"median source English chars/slot-second: {median_src_cps:.3f} (n={len(src_cps)})")

# --- median tts-1 Danish chars/sec at rate 1.0 (pass=0) ---
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
median_da_cps = statistics.median(da_cps)
print(f"median tts-1 Danish chars/sec @1.0x: {median_da_cps:.3f} (n={len(da_cps)})")

# --- unconstrained en->da expansion ratio ---
print("Running ElevenLabs ASR on the 45s clip to recover English text...")
transcribe = ScribeTranscribe(diarize=False)
segments = list(transcribe(CLIP_MEDIA, "en"))
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
    f"\nmedian unconstrained en->da tgt_chars/src_chars ratio: {median_ratio:.3f} (n={len(ratios)})"
)

print("\n--- SUMMARY ---")
print(f"median src chars/slot-second: {median_src_cps:.3f}")
print(f"median da chars/sec @1.0x (tts-1): {median_da_cps:.3f}")
print(f"median unconstrained en->da ratio: {median_ratio:.3f}")
