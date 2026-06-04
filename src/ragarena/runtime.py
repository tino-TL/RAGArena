from __future__ import annotations

from functools import lru_cache

from ragarena.config import settings
from ragarena.embedding.encoder import BGEEncoder
from ragarena.generation.generator import DeepSeekGenerator, OllamaDecisionGenerator
from ragarena.cache.redis_cache import RedisCache
from ragarena.observability.langfuse_tracer import LangfuseTracer, get_langfuse_tracer
from ragarena.retrieval.vector_store import ElasticsearchVectorStore
from ragarena.reranking.reranker import BGEReranker


@lru_cache(maxsize=4)
def get_bge_encoder(
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    expected_dimension: int | None = None,
) -> BGEEncoder:
    return BGEEncoder(
        model_name,
        expected_dimension=expected_dimension or settings.embedding_dimensions,
    )


@lru_cache(maxsize=8)
def get_elasticsearch_vector_store(
    url: str,
    index_name: str = "ragarena_chunks",
    embedding_dims: int | None = None,
) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(
        url=url,
        index_name=index_name,
        embedding_dims=embedding_dims or settings.embedding_dimensions,
    )


@lru_cache(maxsize=8)
def get_deepseek_generator(
    api_key: str | None,
    model: str = "deepseek-chat",
) -> DeepSeekGenerator:
    return DeepSeekGenerator(api_key=api_key, model=model)


@lru_cache(maxsize=4)
def get_ollama_decision_generator(
    url: str,
    model: str,
    timeout: int,
    keep_alive: str,
) -> OllamaDecisionGenerator:
    return OllamaDecisionGenerator(
        url=url,
        model=model,
        timeout=timeout,
        keep_alive=keep_alive,
    )


@lru_cache(maxsize=4)
def get_redis_cache(url: str) -> RedisCache:
    return RedisCache(url=url)


@lru_cache(maxsize=1)
def get_observability_tracer() -> LangfuseTracer:
    return get_langfuse_tracer()


@lru_cache(maxsize=4)
def get_bge_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    max_content_chars: int | None = None,
) -> BGEReranker:
    return BGEReranker(
        model_name,
        max_content_chars=max_content_chars or settings.rerank_max_content_chars,
    )
