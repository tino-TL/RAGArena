from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    document_id: int
    content: str
    score: float
    model_name: str
    source_scores: dict[str, float]
    section_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ElasticsearchVectorStore:
    def __init__(
        self,
        url: str,
        index_name: str = "ragarena_chunks",
        embedding_dims: int = 2560,
    ) -> None:
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.embedding_dims = embedding_dims
        self.session = requests.Session()

    def create_index(self) -> None:
        if self.index_exists():
            self.update_mapping()
            return

        response = self.session.put(
            f"{self.url}/{self.index_name}",
            json={
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "long"},
                        "document_id": {"type": "long"},
                        "model_name": {"type": "keyword"},
                        "content": {"type": "text"},
                        "document_title": {"type": "text"},
                        "document_source": {"type": "keyword"},
                        "arxiv_id": {"type": "keyword"},
                        "paper_id": {"type": "long"},
                        "paper_title": {"type": "text"},
                        "paper_authors": {"type": "keyword"},
                        "paper_categories": {"type": "keyword"},
                        "paper_pdf_url": {"type": "keyword"},
                        "paper_source_url": {"type": "keyword"},
                        "paper_section": {"type": "keyword"},
                        "section_name": {"type": "keyword"},
                        "chunk_type": {"type": "keyword"},
                        "semantic_chunk_type": {"type": "keyword"},
                        "visual_refs": {"type": "keyword"},
                        "source_chunk_ids": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "chunking_strategy": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                }
            },
            timeout=10,
        )
        response.raise_for_status()

    def recreate_index(self) -> bool:
        deleted = False
        if self.index_exists():
            response = self.session.delete(f"{self.url}/{self.index_name}", timeout=10)
            response.raise_for_status()
            deleted = True
        self.create_index()
        return deleted

    def update_mapping(self) -> None:
        response = self.session.put(
            f"{self.url}/{self.index_name}/_mapping",
            json={
                "properties": {
                    "section_name": {"type": "keyword"},
                    "chunk_type": {"type": "keyword"},
                    "semantic_chunk_type": {"type": "keyword"},
                    "visual_refs": {"type": "keyword"},
                    "source_chunk_ids": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunking_strategy": {"type": "keyword"},
                    "paper_id": {"type": "long"},
                    "arxiv_id": {"type": "keyword"},
                }
            },
            timeout=10,
        )
        response.raise_for_status()

    def index_exists(self) -> bool:
        response = self.session.head(f"{self.url}/{self.index_name}", timeout=10)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def index_chunk(
        self,
        chunk_id: int,
        document_id: int,
        model_name: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if len(embedding) != self.embedding_dims:
            raise ValueError(
                f"Expected {self.embedding_dims} dimensions, got {len(embedding)}"
            )

        document_id_value = self.document_id(chunk_id, model_name)
        document = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "model_name": model_name,
            "content": content,
            "embedding": embedding,
        }
        if metadata:
            document.update(metadata)

        response = self.session.put(
            f"{self.url}/{self.index_name}/_doc/{document_id_value}",
            json=document,  # type: ignore[arg-type]
            timeout=10,
        )
        response.raise_for_status()

    def refresh(self) -> None:
        response = self.session.post(f"{self.url}/{self.index_name}/_refresh", timeout=10)
        response.raise_for_status()

    def count(self, model_name: str | None = None) -> int:
        body: dict[str, Any] = {}
        if model_name:
            body["query"] = {"term": {"model_name": model_name}}

        response = self.session.get(
            f"{self.url}/{self.index_name}/_count",
            json=body,
            timeout=10,
        )
        response.raise_for_status()
        return int(response.json()["count"])

    def knn_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        model_name: str | None = None,
        chunking_strategy: str | None = None,
        paper_id: int | None = None,
        arxiv_id: str | None = None,
    ) -> list[SearchResult]:
        if len(query_vector) != self.embedding_dims:
            raise ValueError(
                f"Expected {self.embedding_dims} dimensions, got {len(query_vector)}"
            )

        knn: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(50, top_k),
        }
        filters: list[dict[str, Any]] = []
        if model_name:
            filters.append({"term": {"model_name": model_name}})
        if chunking_strategy:
            filters.append(chunking_strategy_filter(chunking_strategy))
        if paper_id is not None:
            filters.append({"term": {"paper_id": paper_id}})
        if arxiv_id:
            filters.append({"term": {"arxiv_id": arxiv_id}})
        if filters:
            knn["filter"] = filters if len(filters) > 1 else filters[0]

        response = self.session.post(
            f"{self.url}/{self.index_name}/_search",
            json={
                "knn": knn,
                "_source": SEARCH_SOURCE_FIELDS,
            },
            timeout=10,
        )
        response.raise_for_status()

        results = []
        for hit in response.json()["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                SearchResult(
                    chunk_id=source["chunk_id"],
                    document_id=source["document_id"],
                    content=source["content"],
                    score=hit["_score"],
                    model_name=source["model_name"],
                    source_scores={"vector": hit["_score"]},
                    section_name=source.get("section_name"),
                    metadata=extract_metadata(source),
                )
            )

        return dedupe_search_results(results, top_k=top_k)

    def bm25_search(
        self,
        query: str,
        top_k: int = 5,
        model_name: str | None = None,
        chunking_strategy: str | None = None,
        paper_id: int | None = None,
        arxiv_id: str | None = None,
    ) -> list[SearchResult]:
        filters: list[dict[str, Any]] = []
        if model_name:
            filters.append({"term": {"model_name": model_name}})
        if chunking_strategy:
            filters.append(chunking_strategy_filter(chunking_strategy))
        if paper_id is not None:
            filters.append({"term": {"paper_id": paper_id}})
        if arxiv_id:
            filters.append({"term": {"arxiv_id": arxiv_id}})

        if filters:
            query_body: dict[str, Any] = {
                "bool": {
                    "must": {"match": {"content": query}},
                    "filter": filters,
                }
            }
        else:
            query_body = {"match": {"content": query}}

        response = self.session.post(
            f"{self.url}/{self.index_name}/_search",
            json={
                "query": query_body,
                "size": top_k,
                "_source": SEARCH_SOURCE_FIELDS,
            },
            timeout=10,
        )
        response.raise_for_status()

        results = []
        for hit in response.json()["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                SearchResult(
                    chunk_id=source["chunk_id"],
                    document_id=source["document_id"],
                    content=source["content"],
                    score=hit["_score"],
                    model_name=source["model_name"],
                    source_scores={"bm25": hit["_score"]},
                    section_name=source.get("section_name"),
                    metadata=extract_metadata(source),
                )
            )

        return dedupe_search_results(results, top_k=top_k)

    @staticmethod
    def document_id(chunk_id: int, model_name: str) -> str:
        return str(chunk_id)


def dedupe_search_results(results: list[SearchResult], *, top_k: int | None = None) -> list[SearchResult]:
    by_chunk_id: dict[int, SearchResult] = {}
    for result in results:
        current = by_chunk_id.get(result.chunk_id)
        if current is None or result.score > current.score:
            by_chunk_id[result.chunk_id] = result
    deduped = sorted(by_chunk_id.values(), key=lambda item: item.score, reverse=True)
    return deduped if top_k is None else deduped[:top_k]


def chunking_strategy_filter(chunking_strategy: str) -> dict[str, Any]:
    if chunking_strategy == "agentic":
        return {"terms": {"chunking_strategy": ["agentic", "agentic_fusion"]}}
    return {"term": {"chunking_strategy": chunking_strategy}}


SEARCH_SOURCE_FIELDS = [
    "chunk_id",
    "document_id",
    "model_name",
    "content",
    "document_title",
    "document_source",
    "arxiv_id",
    "paper_id",
    "paper_title",
    "paper_authors",
    "paper_categories",
    "paper_pdf_url",
    "paper_source_url",
    "paper_section",
    "section_name",
    "chunk_type",
    "semantic_chunk_type",
    "visual_refs",
    "source_chunk_ids",
    "source_body_block_ids",
    "source_visual_block_ids",
    "page_number",
    "chunking_strategy",
]


METADATA_FIELDS = [
    "document_title",
    "document_source",
    "arxiv_id",
    "paper_id",
    "paper_title",
    "paper_authors",
    "paper_categories",
    "paper_pdf_url",
    "paper_source_url",
    "paper_section",
    "section_name",
    "chunk_type",
    "semantic_chunk_type",
    "visual_refs",
    "source_chunk_ids",
    "source_body_block_ids",
    "source_visual_block_ids",
    "page_number",
    "chunking_strategy",
]


def extract_metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {key: source[key] for key in METADATA_FIELDS if key in source}
