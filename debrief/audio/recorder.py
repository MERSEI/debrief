"""Синхронная запись микрофона и системного звука в две отдельные дорожки.

Обе дорожки стартуют от одной точки отсчёта, поэтому таймкоды сегментов,
полученных от STT независимо по каждому файлу, ложатся на общую ось времени.
Именно это делает диаризацию бесплатной — см. debrief.transcript.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .devices import Device, find_loopback, find_microphone

CHUNK = 1024


@dataclass(frozen=True)
class Recording:
    """Результат сессии записи."""

    mic_path: Path
    loopback_path: Path
    duration: float
    mic_device: Device
    loopback_device: Device


class _TrackWriter:
    """Пишет один поток WASAPI в WAV через неблокирующий callback."""

    def __init__(self, pa, device: Device, path: Path, sample_format: int):
        self._path = path
        self._sample_width = pa.get_sample_size(sample_format)
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

    def _on_chunk(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pyaudio

        self._wave.writeframes(in_data)
        return (in_data, pyaudio.paContinue)

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._wave.close()


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
        self._writers: list[_TrackWriter] = []
        self._started_at = 0.0
        self.mic_path = self._out_dir / f"{session_name}.mic.wav"
        self.loopback_path = self._out_dir / f"{session_name}.them.wav"

    def __enter__(self) -> "DualTrackRecorder":
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        try:
            self.mic_device = find_microphone(self._pa)
            self.loopback_device = find_loopback(self._pa)

            self._writers = [
                _TrackWriter(self._pa, self.mic_device, self.mic_path, pyaudio.paInt16),
                _TrackWriter(
                    self._pa, self.loopback_device, self.loopback_path, pyaudio.paInt16
                ),
            ]
        except Exception:
            self._teardown()
            raise

        self._started_at = time.monotonic()
        return self

    def __exit__(self, *exc_info) -> None:
        self.duration = time.monotonic() - self._started_at
        self._teardown()

    def _teardown(self) -> None:
        for writer in self._writers:
            try:
                writer.close()
            except Exception:  # закрываем остальные, даже если один упал
                pass
        self._writers = []
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def result(self) -> Recording:
        return Recording(
            mic_path=self.mic_path,
            loopback_path=self.loopback_path,
            duration=getattr(self, "duration", 0.0),
            mic_device=self.mic_device,
            loopback_device=self.loopback_device,
        )
