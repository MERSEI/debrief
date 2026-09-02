"""Слияние двух моно-дорожек в один размеченный по спикерам транскрипт.

Диаризация здесь не вероятностная. Дорожки пишутся с физически разных
источников — микрофон это всегда я, WASAPI loopback это всегда собеседник, —
поэтому спикер известен точно, без pyannote и порогов похожести голосов.

Взамен появляется другая проблема: акустическое эхо. Если звук идёт в колонки,
микрофон переспрашивает то же самое, и одна фраза собеседника попадает в обе
дорожки. Здесь это лечится на уровне текста, а не DSP: см. drop_echo.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

ME = "me"
THEM = "them"

SPEAKER_LABELS = {ME: "Я", THEM: "Собеседник"}


@dataclass(frozen=True)
class Segment:
    """Реплика одного спикера на общей оси времени записи (в секундах)."""

    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _overlap(a: Segment, b: Segment) -> float:
    """Длительность пересечения двух сегментов по времени."""
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _similarity(a: str, b: str) -> float:
    left, right = a.strip().casefold(), b.strip().casefold()
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def drop_echo(
    segments: Sequence[Segment],
    *,
    keep: str = THEM,
    min_overlap_ratio: float = 0.5,
    min_similarity: float = 0.72,
) -> list[Segment]:
    """Убрать реплики, просочившиеся из колонок в микрофон.

    Сегмент выбрасывается, когда он одновременно:
      * принадлежит не тому спикеру, чью дорожку мы считаем оригиналом (`keep`);
      * перекрыт по времени сегментом оригинала более чем на `min_overlap_ratio`
        собственной длительности;
      * текстуально похож на него не меньше `min_similarity`.

    Все три условия обязательны: перекрытие без похожести — это обычное
    перебивание, когда оба говорят разное одновременно, и его терять нельзя.
    """
    originals = [s for s in segments if s.speaker == keep]
    result: list[Segment] = []

    for seg in segments:
        if seg.speaker == keep:
            result.append(seg)
            continue
        if seg.duration <= 0:
            continue

        is_echo = any(
            _overlap(seg, other) / seg.duration >= min_overlap_ratio
            and _similarity(seg.text, other.text) >= min_similarity
            for other in originals
        )
        if not is_echo:
            result.append(seg)

    return result


def merge_adjacent(
    segments: Sequence[Segment], *, max_gap: float = 1.2
) -> list[Segment]:
    """Склеить подряд идущие реплики одного спикера, если пауза меньше max_gap.

    Whisper режет речь по интонационным паузам, поэтому без склейки транскрипт
    превращается в лестницу из «Я:» на каждую половину предложения.
    """
    merged: list[Segment] = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        prev = merged[-1]
        if prev.speaker == seg.speaker and seg.start - prev.end <= max_gap:
            merged[-1] = Segment(
                speaker=prev.speaker,
                start=prev.start,
                end=max(prev.end, seg.end),
                text=f"{prev.text.rstrip()} {seg.text.lstrip()}".strip(),
            )
        else:
            merged.append(seg)

    return merged


def build_transcript(
    mine: Iterable[Segment],
    theirs: Iterable[Segment],
    *,
    max_gap: float = 1.2,
    min_duration: float = 0.25,
    suppress_echo: bool = True,
) -> list[Segment]:
    """Собрать финальный транскрипт из двух независимо распознанных дорожек."""
    pool = [s for s in (*mine, *theirs) if s.text.strip() and s.duration >= min_duration]
    pool.sort(key=lambda s: (s.start, s.speaker))

    if suppress_echo:
        pool = drop_echo(pool)

    return merge_adjacent(pool, max_gap=max_gap)


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def to_markdown(segments: Sequence[Segment], *, with_timestamps: bool = True) -> str:
    lines: list[str] = []
    for seg in segments:
        label = SPEAKER_LABELS.get(seg.speaker, seg.speaker)
        prefix = f"`{format_timestamp(seg.start)}` " if with_timestamps else ""
        lines.append(f"{prefix}**{label}:** {seg.text}")
    return "\n\n".join(lines)


def to_plain_text(segments: Sequence[Segment]) -> str:
    """Компактный вид для отправки в LLM — без markdown-шума."""
    return "\n".join(
        f"[{format_timestamp(seg.start)}] {SPEAKER_LABELS.get(seg.speaker, seg.speaker)}: {seg.text}"
        for seg in segments
    )
