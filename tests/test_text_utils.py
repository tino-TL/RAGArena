from __future__ import annotations

from ragarena.utils.text import sanitize_text


def test_sanitize_text_removes_null_byte() -> None:
    assert sanitize_text("abc\x00def") == "abcdef"


def test_sanitize_text_preserves_unicode() -> None:
    value = "中文 αβγ ∑ √ RAG"
    assert sanitize_text(value) == value


def test_sanitize_text_preserves_newlines_and_tabs() -> None:
    value = "line 1\n\tline 2\r\nline 3"
    assert sanitize_text(value) == value
