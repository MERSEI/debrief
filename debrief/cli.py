"""Командный интерфейс Debrief."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import Config, load_dotenv


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def _engine(config: Config):
    from .stt.whisper import WhisperEngine

    return WhisperEngine(
        config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        language=config.language,
    )


def _report(outcome) -> int:
    """Общий хвост для record и process."""
    if outcome.summary:
        print()
        print(outcome.summary.context)
        print()
    else:
        print(
            "Разбор не выполнен, сохранена только стенограмма.",
            file=sys.stderr,
        )

    if outcome.note_path:
        print(f"Заметка: {outcome.note_path}")

    # Ненулевой код, когда протокола нет: так вызывающий скрипт узнает,
    # что работа сделана наполовину, а файлы всё же на месте.
    return 0 if outcome.summary else 1


def cmd_devices(args, config: Config) -> int:
    """Показать, что именно будет записано. Первая команда при настройке."""
    import pyaudiowpatch as pyaudio

    from .audio.devices import DeviceNotFound, find_loopback, find_microphone

    with pyaudio.PyAudio() as pa:
        try:
            print(f"Микрофон  (дорожка «я»):          {find_microphone(pa)}")
        except DeviceNotFound as exc:
            print(f"Микрофон: {exc}", file=sys.stderr)
        try:
            print(f"Loopback  (дорожка «собеседник»): {find_loopback(pa)}")
        except DeviceNotFound as exc:
            print(f"Loopback: {exc}", file=sys.stderr)
    return 0


def cmd_record(args, config: Config) -> int:
    """Записать созвон, затем сразу разобрать его."""
    from .audio.recorder import DualTrackRecorder
    from .pipeline import EmptyTranscript, process

    session = args.name or datetime.now().strftime("%Y-%m-%d %H-%M созвон")

    with DualTrackRecorder(config.recordings_dir, session) as recorder:
        print(f"Запись идёт: {session}")
        print(f"  я:          {recorder.mic_device.name}")
        print(f"  собеседник: {recorder.loopback_device.name}")

        try:
            if args.duration:
                print(f"Остановлюсь сама через {args.duration} с.")
                time.sleep(args.duration)
            else:
                print("Ctrl+C — остановить и разобрать.")
                # Держим процесс живым, пока пишут потоки WASAPI. sleep, а не
                # input(): при запуске из скрипта или планировщика stdin закрыт,
                # и input() вернул бы EOF мгновенно, оборвав запись на нуле.
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nОстанавливаю запись...")

    result = recorder.result()
    print(f"Записано {result.duration:.0f} с.")

    # Тишина, дописанная в дорожку, — это не дефект, а признак того, что
    # устройство молчало. Много тишины на своей дорожке при живом разговоре
    # означает, что микрофон взят не тот.
    if result.loopback_padded > 1:
        print(f"  собеседник молчал: {result.loopback_padded:.0f} с")
    if result.mic_padded > 1:
        print(f"  микрофон молчал:   {result.mic_padded:.0f} с")

    if result.duration < 1.0:
        print("Запись слишком короткая, разбирать нечего.", file=sys.stderr)
        return 1

    if args.no_process:
        return 0

    print("Распознаю...")
    try:
        outcome = process(
            result.mic_path,
            result.loopback_path,
            session_name=session,
            config=config,
            engine=_engine(config),
            hint=args.hint,
        )
    except EmptyTranscript as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return _report(outcome)


def cmd_process(args, config: Config) -> int:
    """Разобрать уже записанную пару дорожек."""
    from .pipeline import EmptyTranscript, process

    mic = Path(args.mic)
    them = Path(args.them) if args.them else mic.with_suffix("").with_suffix(".them.wav")

    for path in (mic, them):
        if not path.exists():
            print(f"Файл не найден: {path}", file=sys.stderr)
            return 1

    session = args.name or mic.name.split(".")[0]
    try:
        outcome = process(
            mic,
            them,
            session_name=session,
            config=config,
            engine=_engine(config),
            hint=args.hint,
        )
    except EmptyTranscript as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return _report(outcome)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="debrief",
        description="Двухдорожечная запись созвонов с разбором в протокол встречи.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    p_devices = sub.add_parser("devices", help="проверить устройства записи")
    p_devices.set_defaults(func=cmd_devices)

    p_record = sub.add_parser("record", help="записать созвон и разобрать его")
    p_record.add_argument("-n", "--name", help="имя сессии (по умолчанию дата и время)")
    p_record.add_argument("--hint", help="контекст встречи, например «собес на Python»")
    p_record.add_argument(
        "--no-process", action="store_true", help="только записать, без разбора"
    )
    p_record.add_argument(
        "-d",
        "--duration",
        type=int,
        help="остановиться самой через N секунд (для запуска из скрипта)",
    )
    p_record.set_defaults(func=cmd_record)

    p_process = sub.add_parser("process", help="разобрать готовые дорожки")
    p_process.add_argument("mic", help="WAV с микрофоном")
    p_process.add_argument("--them", help="WAV с системным звуком")
    p_process.add_argument("-n", "--name", help="имя сессии")
    p_process.add_argument("--hint", help="контекст встречи")
    p_process.set_defaults(func=cmd_process)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    load_dotenv()
    return args.func(args, Config.from_env())
