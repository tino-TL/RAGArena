from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_dsn: str = "postgresql://ragarena:ragarena@localhost:5432/ragarena"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "ragarena_chunks"
    ollama_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"
    agent_decision_model: str = "qwen3.5:4b"
    agent_decision_enabled: bool = True
    agent_decision_timeout: int = 30
    redis_url: str = "redis://localhost:6379/0"
    rag_cache_ttl_seconds: int = 3600
    langfuse_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dimensions: int = 2560
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_candidate_limit: int = 20
    rerank_max_content_chars: int = 1000
    hyde_enabled: bool = True
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    retrieval_candidate_multiplier: int = 4
    retrieval_rrf_k: int = 60
    pipeline_retry_attempts: int = 3
    pipeline_retry_backoff_seconds: float = 2.0
    pipeline_schedule_interval_minutes: int = 60
    pipeline_schedule_cron: str | None = None
    evaluation_output_dir: str = "reports"
    agentic_chunk_enabled: bool = True
    agentic_chunk_provider: str = "ollama"
    agentic_chunk_model: str = "qwen3.5:4b"
    agentic_chunk_max_tokens: int = 800
    agentic_chunk_embed_low_value: bool = False
    agentic_chunk_min_tokens: int = 180
    agentic_chunk_target_tokens: int = 400
    agentic_chunk_max_chunks_per_section: int = 3
    log_level: str = "INFO"
    log_format: str = "plain"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
