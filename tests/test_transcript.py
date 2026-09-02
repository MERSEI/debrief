from debrief.transcript import (
    ME,
    THEM,
    Segment,
    build_transcript,
    drop_echo,
    format_timestamp,
    merge_adjacent,
    to_plain_text,
)


def seg(speaker, start, end, text):
    return Segment(speaker=speaker, start=start, end=end, text=text)


class TestMergeAdjacent:
    def test_joins_same_speaker_within_gap(self):
        out = merge_adjacent(
            [seg(ME, 0.0, 2.0, "Смотрите,"), seg(ME, 2.5, 4.0, "какая идея.")],
            max_gap=1.2,
        )
        assert len(out) == 1
        assert out[0].text == "Смотрите, какая идея."
        assert (out[0].start, out[0].end) == (0.0, 4.0)

    def test_keeps_separate_when_gap_too_large(self):
        out = merge_adjacent(
            [seg(ME, 0.0, 2.0, "Раз"), seg(ME, 10.0, 11.0, "Два")], max_gap=1.2
        )
        assert len(out) == 2

    def test_never_joins_across_speakers(self):
        out = merge_adjacent(
            [seg(ME, 0.0, 2.0, "Вопрос"), seg(THEM, 2.1, 3.0, "Ответ")], max_gap=5.0
        )
        assert [s.speaker for s in out] == [ME, THEM]


class TestDropEcho:
    def test_removes_speaker_echo_from_mic(self):
        # Собеседник говорит в колонки, микрофон ловит то же самое.
        segments = [
            seg(THEM, 5.0, 8.0, "Бюджет у нас примерно триста тысяч"),
            seg(ME, 5.1, 7.9, "бюджет у нас примерно триста тысяч"),
        ]
        out = drop_echo(sorted(segments, key=lambda s: s.start))
        assert len(out) == 1
        assert out[0].speaker == THEM

    def test_keeps_genuine_interruption(self):
        # Перекрытие по времени есть, но текст другой — это перебивание.
        segments = [
            seg(THEM, 5.0, 8.0, "Мы хотели бы запустить до конца квартала"),
            seg(ME, 5.2, 6.0, "Секунду, уточню по срокам"),
        ]
        out = drop_echo(sorted(segments, key=lambda s: s.start))
        assert len(out) == 2

    def test_keeps_same_phrase_said_at_a_different_time(self):
        # Одинаковый текст, но без перекрытия — я действительно это повторил.
        segments = [
            seg(THEM, 5.0, 8.0, "давайте зафиксируем сроки"),
            seg(ME, 40.0, 42.0, "давайте зафиксируем сроки"),
        ]
        out = drop_echo(sorted(segments, key=lambda s: s.start))
        assert len(out) == 2


class TestBuildTranscript:
    def test_interleaves_tracks_by_time(self):
        mine = [seg(ME, 0.0, 2.0, "Привет"), seg(ME, 6.0, 7.0, "Понял")]
        theirs = [seg(THEM, 2.5, 5.0, "Привет, начнём")]
        out = build_transcript(mine, theirs)
        assert [s.speaker for s in out] == [ME, THEM, ME]

    def test_drops_micro_segments(self):
        mine = [seg(ME, 0.0, 0.1, "э"), seg(ME, 1.0, 3.0, "Итак")]
        out = build_transcript(mine, [], min_duration=0.25)
        assert [s.text for s in out] == ["Итак"]

    def test_drops_blank_text(self):
        out = build_transcript([seg(ME, 0.0, 2.0, "   ")], [])
        assert out == []

    def test_plain_text_render_is_llm_ready(self):
        out = build_transcript([seg(ME, 65.0, 67.0, "Начнём")], [])
        assert to_plain_text(out) == "[00:01:05] Я: Начнём"


def test_format_timestamp_crosses_hour():
    assert format_timestamp(3725) == "01:02:05"
