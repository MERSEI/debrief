"""Локальное распознавание через faster-whisper.

Локальное, а не облачное, по двум причинам: запись созвона не покидает машину,
и стоимость часа разговора равна нулю. Цена — модель нужно один раз скачать
и держать VRAM/CPU под неё.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..transcript import Segment

log = logging.getLogger(__name__)


class WhisperEngine:
    """Обёртка над faster-whisper, отдающая сегменты в формате debrief.

    Модель загружается лениво и переиспользуется между дорожками: инициализация
    large-v3 занимает секунды, и делать её дважды за сессию незачем.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        *,
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Загружаю модель Whisper %s (%s)", self._model_name, self._device)
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def transcribe(self, path: Path, speaker: str) -> list[Segment]:
        model = self._load()

        # vad_filter выбрасывает тишину до распознавания. На дорожке loopback
        # это критично: пока собеседник молчит, там ровный ноль на десятки
        # минут, и без VAD модель галлюцинирует на нём текст.
        segments, info = model.transcribe(
            str(path),
            language=self._language,
            vad_filter=True,
            beam_size=5,
        )

        log.info(
            "Дорожка %s: язык %s (p=%.2f)",
            path.name,
            info.language,
            info.language_probability,
        )

        return [
            Segment(
                speaker=speaker,
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
            )
            for s in segments
            if s.text.strip()
        ]
