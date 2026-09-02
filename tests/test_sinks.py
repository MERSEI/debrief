from debrief.sinks.telegram import LIMIT, _chunks


def test_short_text_stays_one_message():
    assert _chunks("привет") == ["привет"]


def test_splits_on_line_boundaries():
    text = "\n".join(f"строка {i}" for i in range(2000))
    parts = _chunks(text)
    assert len(parts) > 1
    assert all(len(p) <= LIMIT for p in parts)
    assert "".join(parts) == text


def test_splits_a_single_overlong_line():
    text = "x" * (LIMIT * 2 + 17)
    parts = _chunks(text)
    assert all(len(p) <= LIMIT for p in parts)
    assert "".join(parts) == text
