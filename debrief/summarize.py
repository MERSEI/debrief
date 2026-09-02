"""Разбор транскрипта в структурированный протокол встречи.

Схема ответа задана Pydantic-моделью и валидируется API, а не парсингом
markdown из текста. Главное требование к промпту — не выдумывать: пустой
список договорённостей честнее, чем правдоподобно звучащий выдуманный.
"""

from __future__ import annotations

from typing import Sequence

import anthropic
from pydantic import BaseModel, Field

from .transcript import Segment, to_plain_text

MODEL = "claude-opus-5"

SYSTEM = """\
Ты разбираешь стенограмму рабочего созвона (собеседование, продажа, синк с клиентом).
Стенограмма размечена по спикерам: «Я» — владелец записи, «Собеседник» — вторая сторона.

Правила:
1. Опирайся только на сказанное. Ничего не додумывай и не достраивай по смыслу.
2. Если в разговоре чего-то не было — оставь список пустым. Пустой раздел это
   нормальный результат, выдуманный — брак.
3. Обязательство — это то, что сторона явно взяла на себя, а не то, что было
   упомянуто как возможность.
4. В цитатах приводи слова дословно и всегда с таймкодом из стенограммы.
5. Красный флаг — конкретное наблюдение из разговора (уклонение от вопроса о
   бюджете, смена сроков, противоречие сказанному ранее), а не общая тревога.
6. Пиши по-русски, кратко, без вводных оборотов.
"""


class Commitment(BaseModel):
    what: str = Field(description="Что именно обещано, формулировкой из разговора")
    due: str | None = Field(
        default=None, description="Срок, если он назван вслух. Иначе null."
    )


class Quote(BaseModel):
    timestamp: str = Field(description="Таймкод из стенограммы, формат ЧЧ:ММ:СС")
    speaker: str = Field(description="«Я» или «Собеседник»")
    text: str = Field(description="Дословная цитата")
    why: str = Field(description="Почему эта фраза важна")


class MeetingSummary(BaseModel):
    """Протокол встречи. Каждый список может быть пустым."""

    context: str = Field(description="О чём был разговор, одно-два предложения")
    decisions: list[str] = Field(description="Договорённости, к которым пришли")
    my_commitments: list[Commitment] = Field(description="Что пообещал «Я»")
    their_commitments: list[Commitment] = Field(description="Что пообещал «Собеседник»")
    open_questions: list[str] = Field(
        description="Вопросы, оставшиеся без ответа, включая заданные мне и не отвеченные"
    )
    red_flags: list[str] = Field(description="Тревожные наблюдения из разговора")
    key_quotes: list[Quote] = Field(description="Фразы, к которым стоит вернуться")
    next_step: str = Field(description="Одно конкретное действие, которое логично сделать первым")


class SummarizationRefused(RuntimeError):
    """Модель отказалась обрабатывать запрос."""


def summarize(
    segments: Sequence[Segment],
    *,
    client: anthropic.Anthropic | None = None,
    hint: str | None = None,
) -> MeetingSummary:
    """Превратить транскрипт в структурированный протокол.

    `hint` — необязательный контекст встречи («собес на позицию Python-разработчика»,
    «созвон с клиентом по смете»). Он заметно улучшает разбор: без него модель не
    знает, чья сторона какая, и трактует разговор нейтральнее, чем нужно.
    """
    if not segments:
        raise ValueError("Пустой транскрипт: нечего разбирать")

    client = client or anthropic.Anthropic()

    header = f"Контекст встречи: {hint}\n\n" if hint else ""
    prompt = f"{header}Стенограмма:\n\n{to_plain_text(segments)}"

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
        output_format=MeetingSummary,
    )

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise SummarizationRefused(
            f"Модель отклонила разбор стенограммы: {getattr(details, 'explanation', 'без пояснения')}"
        )

    return response.parsed_output


def to_markdown(summary: MeetingSummary, *, title: str) -> str:
    """Отрендерить протокол в Markdown под Obsidian."""

    def bullets(items, render=lambda x: x) -> str:
        return "\n".join(f"- {render(i)}" for i in items) if items else "- —"

    def commitment(c: Commitment) -> str:
        return f"{c.what}" + (f" — до {c.due}" if c.due else "")

    def quote(q: Quote) -> str:
        return f"`{q.timestamp}` **{q.speaker}:** «{q.text}»\n  - {q.why}"

    return f"""# {title}

{summary.context}

## Договорённости
{bullets(summary.decisions)}

## Мои обязательства
{bullets(summary.my_commitments, commitment)}

## Обязательства собеседника
{bullets(summary.their_commitments, commitment)}

## Открытые вопросы
{bullets(summary.open_questions)}

## Красные флаги
{bullets(summary.red_flags)}

## Ключевые цитаты
{bullets(summary.key_quotes, quote)}

## Следующий шаг
{summary.next_step}
"""
