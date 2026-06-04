from __future__ import annotations

from datetime import datetime
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests

from ragarena.papers.models import PaperMetadata

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_papers(
    query: str,
    *,
    max_results: int = 20,
    start: int = 0,
    timeout: int = 30,
) -> list[PaperMetadata]:
    params = urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    response = requests.get(f"{ARXIV_API_URL}?{params}", timeout=timeout)
    response.raise_for_status()
    return parse_arxiv_feed(response.text)


def parse_arxiv_feed(xml_text: str) -> list[PaperMetadata]:
    root = ElementTree.fromstring(xml_text)
    papers: list[PaperMetadata] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        source_url = _text(entry, "atom:id")
        arxiv_id = source_url.rsplit("/", 1)[-1]
        title = _normalize_text(_text(entry, "atom:title"))
        abstract = _normalize_text(_text(entry, "atom:summary"))
        authors = [
            _normalize_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib["term"]
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        pdf_url = _pdf_url(entry)
        papers.append(
            PaperMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=[author for author in authors if author],
                abstract=abstract,
                categories=categories,
                published_at=_parse_datetime(_text(entry, "atom:published")),
                updated_at=_parse_datetime(_text(entry, "atom:updated")),
                pdf_url=pdf_url,
                source_url=source_url,
            )
        )
    return papers


def _text(entry: ElementTree.Element, path: str) -> str:
    return entry.findtext(path, default="", namespaces=ATOM_NS).strip()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _pdf_url(entry: ElementTree.Element) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" and link.attrib.get("href"):
            return link.attrib["href"]
    source_url = _text(entry, "atom:id")
    return source_url.replace("/abs/", "/pdf/")
