from debrief.summarize import Commitment, MeetingSummary, Quote, to_markdown


def make_summary(**overrides):
    base = dict(
        context="Созвон по смете на интеграцию.",
        decisions=["Работаем по фиксу"],
        my_commitments=[Commitment(what="Прислать смету", due="пятницу")],
        their_commitments=[Commitment(what="Дать доступ к репозиторию")],
        open_questions=["Кто принимает работу"],
        red_flags=["Ушёл от вопроса про бюджет"],
        key_quotes=[
            Quote(
                timestamp="00:12:40",
                speaker="Собеседник",
                text="Бюджет обсудим позже",
                why="Единственное упоминание денег за весь созвон",
            )
        ],
        next_step="Отправить смету до пятницы",
    )
    base.update(overrides)
    return MeetingSummary(**base)


def test_renders_all_sections():
    md = to_markdown(make_summary(), title="Клиент X")
    assert md.startswith("# Клиент X")
    for heading in ("Договорённости", "Мои обязательства", "Красные флаги", "Следующий шаг"):
        assert f"## {heading}" in md


def test_commitment_without_due_has_no_dangling_dash():
    md = to_markdown(make_summary(), title="t")
    assert "- Дать доступ к репозиторию\n" in md
    assert "Прислать смету — до пятницу" in md


def test_empty_sections_render_a_placeholder_not_a_blank():
    md = to_markdown(
        make_summary(decisions=[], red_flags=[], key_quotes=[]), title="t"
    )
    assert md.count("- —") == 3


def test_optional_due_defaults_to_none():
    assert Commitment(what="что-то").due is None
