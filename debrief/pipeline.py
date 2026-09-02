"""Оркестрация: две дорожки -> транскрипт -> протокол -> файлы и Telegram."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .stt.base import SpeechToText
from .summarize import MeetingSummary, summarize
from .summarize import to_markdown as summary_to_markdown
from .transcript import ME, THEM, Segment, build_transcript
from .transcript import to_markdown as transcript_to_markdown

log = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    segments: list[Segment]
    summary: MeetingSummary
    note_path: Path | None


def process(
    mic_path: Path,
    loopback_path: Path,
    *,
    session_name: str,
    config: Config,
    engine: SpeechToText,
    hint: str | None = None,
) -> ProcessResult:
    """Прогнать записанную сессию через распознавание и разбор."""
    log.info("Распознаю дорожку микрофона: %s", mic_path.name)
    mine = engine.transcribe(mic_path, ME)

    log.info("Распознаю дорожку собеседника: %s", loopback_path.name)
    theirs = engine.transcribe(loopback_path, THEM)

    segments = build_transcript(mine, theirs)
    log.info(
        "Транскрипт собран: %d реплик (%d моих, %d собеседника)",
        len(segments),
        sum(1 for s in segments if s.speaker == ME),
        sum(1 for s in segments if s.speaker == THEM),
    )

    summary = summarize(segments, hint=hint)

    summary_md = summary_to_markdown(summary, title=session_name)
    note_path = None
    if config.vault_dir:
        from .sinks.obsidian import write_note

        note_path = write_note(
            config.vault_dir,
            session_name,
            summary_md,
            transcript_to_markdown(segments),
            hint=hint,
        )
        log.info("Заметка записана: %s", note_path)

    if config.telegram_enabled:
        from .sinks import telegram

        telegram.send(config.telegram_token, config.telegram_chat_id, summary_md)
        log.info("Протокол отправлен в Telegram")

    return ProcessResult(segments=segments, summary=summary, note_path=note_path)
