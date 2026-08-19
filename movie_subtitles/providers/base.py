from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str


class ASRProvider(Protocol):
    def __call__(self, fpath: str | Path, audio_lang: str) -> Iterable[Segment]: ...


class TranslationProvider(Protocol):
    def __call__(self, text: str, output_lang: str, budget_chars: int | None = None) -> str: ...


class TTSProvider(Protocol):
    def __call__(self, text: str, out_path: Path, speed: float = 1.0) -> Path: ...
