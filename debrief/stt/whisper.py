"""Локальное распознавание через faster-whisper.

Локальное, а не облачное, по двум причинам: запись созвона не покидает машину,
и стоимость часа разговора равна нулю. Цена — модель нужно один раз скачать
и держать VRAM/CPU под неё.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..transcript import Segment
from .devices import build_plan, cuda_device_count

log = logging.getLogger(__name__)


def _warm_up(model) -> None:
    """Прогнать секунду тишины, чтобы устройство отказало сейчас, а не потом.

    Генератор сегментов ленивый, поэтому его обязательно нужно проитерировать:
    без этого вычисление не запустится и проверка ничего не проверит.
    """
    import numpy as np

    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = model.transcribe(silence, vad_filter=False, beam_size=1)
    for _ in segments:
        break


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
        #: Заполняются при загрузке — на чём модель реально поднялась.
        self.device: str | None = None
        self.compute_type: str | None = None

    def _load(self):
        """Поднять модель, перебирая устройства по плану.

        GPU может быть виден и при этом непригоден — библиотеки CUDA ставятся
        отдельно от драйвера. Падать из-за этого нельзя: процессор медленнее,
        но работает, и молчаливый отказ хуже медленного успеха.
        """
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        plan = build_plan(
            self._device, self._compute_type, cuda_available=cuda_device_count() > 0
        )

        last_error: Exception | None = None
        for device, compute_type in plan:
            try:
                log.info(
                    "Загружаю Whisper %s на %s (%s)",
                    self._model_name,
                    device,
                    compute_type,
                )
                model = WhisperModel(
                    self._model_name, device=device, compute_type=compute_type
                )
                # Конструктор на непригодном GPU проходит молча: ctranslate2
                # подтягивает cuBLAS только на первом вычислении. Без прогрева
                # отказ всплыл бы посреди разбора часового созвона.
                _warm_up(model)

                self._model = model
                self.device = device
                self.compute_type = compute_type
                return self._model
            except Exception as exc:
                last_error = exc
                log.warning(
                    "%s не поднялось (%s), пробую дальше",
                    device,
                    str(exc).splitlines()[0][:120],
                )

        raise RuntimeError(
            f"Не удалось загрузить Whisper ни на одном устройстве из {plan}. "
            f"Последняя ошибка: {last_error}"
        )

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
