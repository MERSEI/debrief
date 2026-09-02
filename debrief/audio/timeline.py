"""Удержание дорожек на общей оси времени.

WASAPI loopback молчит буквально: пока через устройство вывода ничего не
играет, поток не отдаёт буферов вообще — не нули, а ничего. Наивная запись
«что пришло, то и записали» даёт файл, в котором первая реплика собеседника
стоит в позиции 0, хотя прозвучала на третьей минуте.

Для Debrief это фатально: таймкоды сегментов от STT считаются от начала
файла, и на них держится вся склейка дорожек. Поэтому пропуски заполняются
тишиной — позиция в файле обязана соответствовать реальному времени.
"""

from __future__ import annotations


def missing_frames(
    elapsed: float,
    sample_rate: int,
    frames_written: int,
    *,
    tolerance: float = 0.1,
) -> int:
    """Сколько фреймов тишины не хватает, чтобы догнать реальное время.

    :param elapsed: секунд прошло с общего нуля записи
    :param sample_rate: частота дискретизации дорожки
    :param frames_written: сколько фреймов уже в файле
    :param tolerance: расхождение, которое считаем дрожанием и не трогаем

    Допуск обязателен. Буферы приходят пачками по 1024 фрейма, и мгновенное
    расхождение в пределах пары буферов — норма, а не пропуск. Без допуска
    дорожка обрастала бы миллисекундными вставками на каждом чанке.
    """
    if elapsed <= 0 or sample_rate <= 0:
        return 0

    expected = int(elapsed * sample_rate)
    gap = expected - frames_written

    if gap <= int(tolerance * sample_rate):
        return 0
    return gap


def silence(frames: int, channels: int, sample_width: int) -> bytes:
    """Блок цифровой тишины на нужное число фреймов."""
    if frames <= 0:
        return b""
    return b"\x00" * (frames * channels * sample_width)
