"""Конфигурация из окружения. Ничего не читается в момент импорта."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
