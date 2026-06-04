from __future__ import annotations

import unicodedata


PRESERVED_CONTROL_CHARS = {"\n", "\r", "\t"}


def sanitize_text(value: str) -> str:
    """Remove database-invalid control characters while preserving readable text."""
    return "".join(
        char
        for char in value
        if char in PRESERVED_CONTROL_CHARS or not is_invalid_control_char(char)
    )


def is_invalid_control_char(char: str) -> bool:
    return unicodedata.category(char) == "Cc"
