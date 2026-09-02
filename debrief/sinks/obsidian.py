"""Сохранение протокола в Markdown-хранилище."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def write_note(
    vault_dir: Path,
    session_name: str,
    summary_md: str,
    transcript_md: str,
    *,
    hint: str | None = None,
) -> Path:
    """Записать заметку с протоколом и свёрнутой стенограммой.

    Стенограмма кладётся в тот же файл под <details>: она нужна редко, но
    когда нужна — искать её во втором файле неудобно.
    """
    vault_dir.mkdir(parents=True, exist_ok=True)
    path = vault_dir / f"{session_name}.md"

    frontmatter = "\n".join(
        [
            "---",
            f"created: {datetime.now().isoformat(timespec='seconds')}",
            "type: meeting",
            f"context: {hint}" if hint else "context:",
            "tags: [debrief]",
            "---",
            "",
        ]
    )

    body = (
        f"{frontmatter}{summary_md}\n"
        "## Стенограмма\n\n"
        "<details>\n<summary>Развернуть</summary>\n\n"
        f"{transcript_md}\n\n"
        "</details>\n"
    )

    path.write_text(body, encoding="utf-8")
    return path
