"""Отправка карточки протокола в Telegram через Bot API.

Здесь достаточно одного HTTP-запроса, поэтому aiogram не подключается —
бот ничего не слушает, он только пишет владельцу.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"
LIMIT = 4096


def _chunks(text: str, size: int = LIMIT) -> list[str]:
    """Порезать текст по границам абзацев, не разрывая строки посередине."""
    if len(text) <= size:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > size and current:
            parts.append(current)
            current = ""
        # Одна строка длиннее лимита — режем принудительно.
        while len(line) > size:
            parts.append(line[:size])
            line = line[size:]
        current += line
    if current:
        parts.append(current)
    return parts


def send(token: str, chat_id: str, text: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        for part in _chunks(text):
            response = client.post(
                API.format(token=token),
                json={
                    "chat_id": chat_id,
                    "text": part,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
            )
            if response.status_code != 200:
                log.error("Telegram отклонил сообщение: %s", response.text)
                response.raise_for_status()
