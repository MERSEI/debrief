"""Выбор устройства для Whisper с откатом на процессор.

Наличие видеокарты не означает, что на ней получится считать. ctranslate2
рапортует об устройстве через драйвер, а грузит вычисления через cuBLAS и
cuDNN, которые ставятся отдельно от драйвера. Машина с GPU, но без CUDA
Toolkit, отвечает «устройство есть» и падает на первой же загрузке модели:

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

Поэтому «auto» здесь — не «спросить у ctranslate2», а «попробовать по
очереди и взять первое, что реально поднялось».
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: На процессоре int8 быстрее float32 в разы и почти не теряет качества.
CPU_COMPUTE = "int8"
GPU_COMPUTE = "float16"


def cuda_device_count() -> int:
    """Сколько CUDA-устройств видит ctranslate2. Ноль, если он сам не загрузился."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def build_plan(
    device: str, compute_type: str, *, cuda_available: bool
) -> list[tuple[str, str]]:
    """Очередь попыток «(устройство, тип вычислений)».

    Явно названное устройство остаётся единственной попыткой: если человек
    написал cuda, он хочет знать, что она не работает, а не получить молча
    десятикратно более медленный прогон на процессоре.
    """
    if device != "auto":
        resolved = compute_type
        if compute_type == "default":
            resolved = GPU_COMPUTE if device == "cuda" else CPU_COMPUTE
        return [(device, resolved)]

    plan: list[tuple[str, str]] = []
    if cuda_available:
        plan.append(("cuda", GPU_COMPUTE if compute_type == "default" else compute_type))
    plan.append(("cpu", CPU_COMPUTE if compute_type == "default" else compute_type))
    return plan
