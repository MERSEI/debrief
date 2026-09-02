"""Контракт движка распознавания: файл на входе, сегменты с таймкодами на выходе."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..transcript import Segment


class SpeechToText(Protocol):
    def transcribe(self, path: Path, speaker: str) -> list[Segment]:
        """Распознать одну дорожку и разметить все сегменты как речь `speaker`."""
        ...
