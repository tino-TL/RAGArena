from __future__ import annotations

from ragarena.papers.arxiv_client import parse_arxiv_feed
from ragarena.papers.models import paper_to_document
from ragarena.papers.text_parser import normalize_pdf_text


SAMPLE_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Retrieval Augmented Generation for Research Assistants</title>
    <summary>
      This paper studies retrieval augmented generation for research workflows.
    </summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Engineer</name></author>
    <category term="cs.CL" />
    <category term="cs.AI" />
    <link href="http://arxiv.org/pdf/2401.00001v1" rel="related" title="pdf" />
  </entry>
</feed>
"""


def test_parse_arxiv_feed() -> None:
    papers = parse_arxiv_feed(SAMPLE_ARXIV_FEED)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.00001v1"
    assert papers[0].title == "Retrieval Augmented Generation for Research Assistants"
    assert papers[0].authors == ["Alice Researcher", "Bob Engineer"]
    assert papers[0].categories == ["cs.CL", "cs.AI"]
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.00001v1"


def test_paper_to_document() -> None:
    paper = parse_arxiv_feed(SAMPLE_ARXIV_FEED)[0]
    document = paper_to_document(paper)

    assert document.title == paper.title
    assert document.source == paper.source_url
    assert paper.abstract in document.content
    assert document.content_hash


def test_normalize_pdf_text() -> None:
    assert normalize_pdf_text(" hello   world \n\n page   two ") == "hello world\npage two"
