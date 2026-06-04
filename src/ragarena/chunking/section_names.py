from __future__ import annotations

import re
from html import unescape

from ragarena.utils.text import sanitize_text

HEADING_SECTION_RE = re.compile(r"^##\s+(.+)$")


def derive_section_name_from_content(content: str) -> str | None:
    first_line = first_non_empty_line(content)
    if first_line is None:
        return None
    match = HEADING_SECTION_RE.match(first_line)
    if not match:
        return None
    return unescape(match.group(1)).strip() or None


def resolve_section_name(
    content: str,
    planner_section_name: str | None,
    fallback_section_name: str | None,
) -> str | None:
    derived = derive_section_name_from_content(content)
    if derived:
        return derived
    if planner_section_name:
        return planner_section_name
    return fallback_section_name


def first_non_empty_line(content: str) -> str | None:
    for line in sanitize_text(content).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def extract_section_name_from_heading(heading: str) -> str:
    return re.sub(r"^##\s+", "", unescape(heading).strip()).strip()
