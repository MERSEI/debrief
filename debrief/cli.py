"""Командный интерфейс Debrief."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import Config


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
    from .pipeline import process

    session = args.name or datetime.now().strftime("%Y-%m-%d %H-%M созвон")

    with DualTrackRecorder(config.recordings_dir, session) as recorder:
        print(f"Запись идёт: {session}")
        print(f"  я:          {recorder.mic_device.name}")
        print(f"  собеседник: {recorder.loopback_device.name}")
        print("Ctrl+C — остановить и разобрать.")
        try:
            while True:
                input()
        except (KeyboardInterrupt, EOFError):
            print("\nОстанавливаю запись...")

    result = recorder.result()
    print(f"Записано {result.duration:.0f} с. Распознаю...")

    if args.no_process:
        return 0

    outcome = process(
        result.mic_path,
        result.loopback_path,
        session_name=session,
        config=config,
        engine=_engine(config),
        hint=args.hint,
    )
    print(f"\n{outcome.summary.context}\n")
    if outcome.note_path:
        print(f"Заметка: {outcome.note_path}")
    return 0


def cmd_process(args, config: Config) -> int:
    """Разобрать уже записанную пару дорожек."""
    from .pipeline import process

    mic = Path(args.mic)
    them = Path(args.them) if args.them else mic.with_suffix("").with_suffix(".them.wav")

    for path in (mic, them):
        if not path.exists():
            print(f"Файл не найден: {path}", file=sys.stderr)
            return 1

    session = args.name or mic.name.split(".")[0]
    outcome = process(
        mic,
        them,
        session_name=session,
        config=config,
        engine=_engine(config),
        hint=args.hint,
    )
    print(f"\n{outcome.summary.context}\n")
    if outcome.note_path:
        print(f"Заметка: {outcome.note_path}")
    return 0


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
    return args.func(args, Config.from_env())
