"""Поиск устройств WASAPI: микрофон и loopback системного вывода.

Loopback — это способ Windows отдать нам то, что уходит в колонки, без
виртуальных кабелей и без бота-участника в конференции. Обычный PyAudio
такого не умеет, поэтому зависимость именно PyAudioWPatch.
"""

from __future__ import annotations

from dataclasses import dataclass


class DeviceNotFound(RuntimeError):
    """Не удалось найти подходящее устройство ввода."""


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.channels}ch @ {self.sample_rate} Hz)"


def _as_device(info: dict) -> Device:
    return Device(
        index=int(info["index"]),
        name=str(info["name"]),
        channels=int(info["maxInputChannels"]),
        sample_rate=int(info["defaultSampleRate"]),
    )


def find_microphone(pa) -> Device:
    """Устройство ввода по умолчанию — дорожка «я»."""
    try:
        return _as_device(pa.get_default_input_device_info())
    except OSError as exc:  # PyAudio бросает OSError, когда устройства нет
        raise DeviceNotFound("Микрофон по умолчанию не найден") from exc


def find_loopback(pa) -> Device:
    """Loopback-дубль текущего устройства вывода — дорожка «собеседник».

    Идём от устройства вывода по умолчанию и ищем loopback с тем же именем:
    так запись всегда следует за тем, куда Windows реально играет звук, даже
    если пользователь переключил гарнитуру посреди созвона.
    """
    import pyaudiowpatch as pyaudio

    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise DeviceNotFound("WASAPI недоступен на этой системе") from exc

    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])

    if speakers.get("isLoopbackDevice"):
        return _as_device(speakers)

    for candidate in pa.get_loopback_device_info_generator():
        if speakers["name"] in candidate["name"]:
            return _as_device(candidate)

    raise DeviceNotFound(
        f"Не нашёл loopback для устройства вывода «{speakers['name']}». "
        "Проверьте, что звук идёт через WASAPI-совместимое устройство."
    )


def list_input_devices(pa) -> list[Device]:
    """Все устройства, с которых физически можно читать (для диагностики)."""
    devices = []
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if int(info.get("maxInputChannels", 0)) > 0:
            devices.append(_as_device(info))
    return devices
