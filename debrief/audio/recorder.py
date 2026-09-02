"""Синхронная запись микрофона и системного звука в две отдельные дорожки.

Обе дорожки стартуют от одной точки отсчёта, поэтому таймкоды сегментов,
полученных от STT независимо по каждому файлу, ложатся на общую ось времени.
Именно это делает диаризацию бесплатной — см. debrief.transcript.

Ось времени удерживается не сама собой: пропуски в потоке заполняются
тишиной, иначе молчащий loopback сдвинул бы всю дорожку собеседника
к нулю. Подробности в debrief.audio.timeline.
"""

from __future__ import annotations

import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .devices import Device, find_loopback, find_microphone
from .timeline import missing_frames, silence

log = logging.getLogger(__name__)

CHUNK = 1024


@dataclass(frozen=True)
class Recording:
    """Результат сессии записи."""

    mic_path: Path
    loopback_path: Path
    duration: float
    mic_device: Device
    loopback_device: Device
    mic_frames: int = 0
    loopback_frames: int = 0
    #: Сколько секунд тишины пришлось дописать в каждую дорожку.
    mic_padded: float = 0.0
    loopback_padded: float = 0.0


class _TrackWriter:
    """Пишет один поток WASAPI в WAV, удерживая позицию на общей оси времени."""

    def __init__(self, pa, device: Device, path: Path, sample_format: int, started_at: float):
        self._path = path
        self._device = device
        self._started_at = started_at
        self._sample_width = pa.get_sample_size(sample_format)
        self._frames_written = 0
        self._frames_padded = 0

        self._wave = wave.open(str(path), "wb")
        self._wave.setnchannels(device.channels)
        self._wave.setsampwidth(self._sample_width)
        self._wave.setframerate(device.sample_rate)

        self._stream = pa.open(
            format=sample_format,
            channels=device.channels,
            rate=device.sample_rate,
            frames_per_buffer=CHUNK,
            input=True,
            input_device_index=device.index,
            stream_callback=self._on_chunk,
        )

    def _pad_to_now(self, now: float) -> None:
        """Дописать тишину за время, за которое устройство не прислало ничего."""
        gap = missing_frames(
            now - self._started_at, self._device.sample_rate, self._frames_written
        )
        if gap <= 0:
            return
        self._wave.writeframes(
            silence(gap, self._device.channels, self._sample_width)
        )
        self._frames_written += gap
        self._frames_padded += gap

    def _on_chunk(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        # Тишина дописывается перед приходящим блоком: сам блок относится к
        # «сейчас», а всё, что устройство промолчало до него, — к прошлому.
        self._pad_to_now(time.monotonic())
        self._wave.writeframes(in_data)
        self._frames_written += frame_count
        return (in_data, pyaudio.paContinue)

    def close(self, now: float) -> None:
        self._stream.stop_stream()
        self._stream.close()
        # Хвост записи тоже может быть тишиной — например, собеседник замолчал
        # за минуту до конца созвона.
        self._pad_to_now(now)
        self._wave.close()

    @property
    def frames(self) -> int:
        return self._frames_written

    @property
    def padded_seconds(self) -> float:
        return self._frames_padded / self._device.sample_rate


class DualTrackRecorder:
    """Контекст-менеджер на одну сессию записи.

    Микрофон и loopback открываются двумя независимыми потоками WASAPI. Если
    второй не открылся, первый обязательно закрывается — иначе устройство
    остаётся захваченным до перезапуска процесса.
    """

    def __init__(self, out_dir: Path, session_name: str):
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._session_name = session_name
        self._pa = None
        self._writers: dict[str, _TrackWriter] = {}
        # Статистика переживает закрытие потоков: сами writer'ы после
        # teardown недоступны, а отчёт нужен уже после выхода из with.
        self._closed: dict[str, tuple[int, float]] = {}
        self._started_at = 0.0
        self.duration = 0.0
        self.mic_path = self._out_dir / f"{session_name}.mic.wav"
        self.loopback_path = self._out_dir / f"{session_name}.them.wav"

    def __enter__(self) -> "DualTrackRecorder":
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        try:
            self.mic_device = find_microphone(self._pa)
            self.loopback_device = find_loopback(self._pa)

            # Общий ноль берётся до открытия потоков: они открываются не
            # мгновенно, и разницу в старте каждая дорожка добьёт тишиной сама.
            self._started_at = time.monotonic()

            self._writers = {
                "mic": _TrackWriter(
                    self._pa,
                    self.mic_device,
                    self.mic_path,
                    pyaudio.paInt16,
                    self._started_at,
                ),
                "loopback": _TrackWriter(
                    self._pa,
                    self.loopback_device,
                    self.loopback_path,
                    pyaudio.paInt16,
                    self._started_at,
                ),
            }
        except Exception:
            self._teardown(time.monotonic())
            raise

        return self

    def __exit__(self, *exc_info) -> None:
        ended = time.monotonic()
        self.duration = ended - self._started_at
        self._teardown(ended)

    def _teardown(self, now: float) -> None:
        for name, writer in self._writers.items():
            try:
                writer.close(now)
            except Exception:  # закрываем остальные, даже если один упал
                log.exception("Дорожка %s закрылась с ошибкой", name)
            finally:
                self._closed[name] = (writer.frames, writer.padded_seconds)
        self._writers = {}
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def result(self) -> Recording:
        mic = self._closed.get("mic")
        loopback = self._closed.get("loopback")
        return Recording(
            mic_path=self.mic_path,
            loopback_path=self.loopback_path,
            duration=self.duration,
            mic_device=self.mic_device,
            loopback_device=self.loopback_device,
            mic_frames=mic[0] if mic else 0,
            loopback_frames=loopback[0] if loopback else 0,
            mic_padded=mic[1] if mic else 0.0,
            loopback_padded=loopback[1] if loopback else 0.0,
        )
