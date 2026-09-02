"""Конфигурация из окружения. Ничего не читается в момент импорта."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> None:
    """Подтянуть .env в окружение.

    Своя реализация вместо python-dotenv: нужен разбор `KEY=value` и ничего
    больше, а лишняя зависимость в списке — лишняя причина, по которой
    установка может не пройти.

    Уже заданные переменные окружения не перетираются: то, что передано
    явно при запуске, всегда сильнее файла.
    """
    file = Path(path)
    if not file.exists():
        return

    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    recordings_dir: Path
    vault_dir: Path | None
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    language: str | None
    telegram_token: str | None
    telegram_chat_id: str | None
    keep_audio: bool

    @classmethod
    def from_env(cls) -> "Config":
        vault = os.environ.get("DEBRIEF_VAULT_DIR")
        language = os.environ.get("DEBRIEF_LANGUAGE")
        return cls(
            recordings_dir=Path(
                os.environ.get("DEBRIEF_RECORDINGS_DIR", "./recordings")
            ).expanduser(),
            vault_dir=Path(vault).expanduser() if vault else None,
            whisper_model=os.environ.get("DEBRIEF_WHISPER_MODEL", "large-v3"),
            whisper_device=os.environ.get("DEBRIEF_WHISPER_DEVICE", "auto"),
            whisper_compute_type=os.environ.get(
                "DEBRIEF_WHISPER_COMPUTE_TYPE", "default"
            ),
            # "auto" означает автоопределение языка Whisper'ом.
            language=None if language in (None, "", "auto") else language,
            telegram_token=os.environ.get("DEBRIEF_TELEGRAM_TOKEN") or None,
            telegram_chat_id=os.environ.get("DEBRIEF_TELEGRAM_CHAT_ID") or None,
            keep_audio=_bool("DEBRIEF_KEEP_AUDIO", True),
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)
