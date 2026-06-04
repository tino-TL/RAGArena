from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from ragarena.retrieval.vector_store import SearchResult


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_content_chars: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_content_chars = max_content_chars
        self.model = None
        self.load_error: str | None = None
        self._load_model()

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not results:
            return []

        if self.model is None:
            return results[:top_k]

        pairs = [(query, self._content_for_rerank(result.content)) for result in results]
        scores = self.model.predict(pairs)
        reranked = [
            replace(
                result,
                score=float(score),
                source_scores={**result.source_scores, "rerank": float(score)},
            )
            for result, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    def _content_for_rerank(self, content: str) -> str:
        if self.max_content_chars is None or self.max_content_chars <= 0:
            return content
        return content[: self.max_content_chars]

    def _load_model(self) -> None:
        model_path = _get_cached_model_path(self.model_name)
        if model_path:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            model_name_or_path = str(model_path)
        else:
            self.load_error = f"Reranker model is not cached locally: {self.model_name}"
            return

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name_or_path,
                local_files_only=True,
            )
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)


def _get_cached_model_path(model_name: str) -> Path | None:
    cache_root = _get_huggingface_cache_root()
    model_dir = cache_root / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        return None

    snapshots = sorted(
        (
            path
            for path in model_dir.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _get_huggingface_cache_root() -> Path:
    if hf_hub_cache := os.getenv("HF_HUB_CACHE"):
        return Path(hf_hub_cache)
    if hf_home := os.getenv("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"
