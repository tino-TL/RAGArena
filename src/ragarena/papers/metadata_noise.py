from __future__ import annotations

import re

from ragarena.utils.text import sanitize_text

MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)


def is_metadata_noise(content: str, section_name: str | None = None, order_index: int | None = None) -> bool:
    """Return True for standalone parser blocks that are metadata/header noise."""
    text = sanitize_text(content).strip()
    if not text:
        return True
    if looks_like_table_or_multiline_body(text):
        return False
    if is_page_number(text):
        return True
    if is_running_header_footer(text):
        return True
    if is_standalone_date(text):
        return True
    if is_arxiv_header_footer(text):
        return True
    if is_short_footnote_marker(text):
        return True
    return False


def is_standalone_boundary_noise_line(content: str) -> bool:
    text = sanitize_text(content).strip()
    return bool(text) and (is_page_number(text) or is_running_header_footer(text) or is_short_footnote_marker(text))


def looks_like_table_or_multiline_body(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return True
    return "|" in text or "\t" in text


def is_page_number(text: str) -> bool:
    normalized = text.strip()
    if re.fullmatch(r"\d{1,3}", normalized):
        return True
    return bool(re.fullmatch(r"(?i)(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)", normalized))


def is_standalone_date(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if re.fullmatch(r"(?:19|20)\d{2}", normalized):
        return True
    month_year = rf"(?i){MONTH_RE}\.?\s+(?:19|20)\d{{2}}"
    day_month_year = rf"(?i)\d{{1,2}}\s+{MONTH_RE}\.?\s+(?:19|20)\d{{2}}"
    month_day_year = rf"(?i){MONTH_RE}\.?\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}"
    iso_date = r"(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?"
    return bool(
        re.fullmatch(month_year, normalized)
        or re.fullmatch(day_month_year, normalized)
        or re.fullmatch(month_day_year, normalized)
        or re.fullmatch(iso_date, normalized)
    )


def is_arxiv_header_footer(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith(("arxiv:", "preprint", "submitted to", "under review"))
        or "arxiv.org" in lowered
        or "all rights reserved" in lowered
        or "creative commons" in lowered
        or "copyright" in lowered
        or "license" in lowered
    )


def is_running_header_footer(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    normalized = re.sub(r"^#{1,6}\s*", "", normalized).strip()
    if re.fullmatch(r"\d{1,3}\s*[\u00b7\u2022]\s+.+", normalized):
        return True
    if re.fullmatch(r".+\s*[\u00b7\u2022]\s+\d{1,3}", normalized):
        return True
    if re.fullmatch(r"(?i)page\s+\d{1,3}(?:\s+of\s+\d{1,3})?", normalized):
        return True
    return False


def is_short_footnote_marker(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) > 24:
        return False
    compact = re.sub(r"\s+", "", normalized)
    if re.fullmatch(r"[\d,*†‡§]+", compact):
        return True
    if re.fullmatch(r"(?:\d+[,\s]*){1,6}", normalized):
        return True
    return bool(re.fullmatch(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+", normalized))
