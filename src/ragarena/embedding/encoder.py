from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast


class BGEEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        expected_dimension: int = 2560,
    ) -> None:
        cached_model_path = _get_cached_model_path(model_name)
        model_path = str(cached_model_path) if cached_model_path else model_name
        if cached_model_path:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.device = resolve_torch_device()
        print_embedding_device_info(self.device)
        try:
            self.model = SentenceTransformer(
                model_path,
                local_files_only=bool(cached_model_path),
                device=self.device,
            )
        except Exception:
            self.model = SentenceTransformer(model_name, local_files_only=True, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []

        print(
            "before encode: "
            f"text_count={len(texts)} batch_size={batch_size} device={self.device} model={self.model_name}"
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        print(f"after encode: text_count={len(texts)} device={self.device}")

        vectors = embeddings.tolist()
        for vector in vectors:
            if len(vector) != self.expected_dimension:
                raise ValueError(
                    f"Expected {self.expected_dimension} dimensions, got {len(vector)}"
                )

        return vectors


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


def resolve_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def print_embedding_device_info(device: str) -> None:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        print(f"torch version: {torch.__version__}")
        print(f"cuda available: {cuda_available}")
        print(f"embedding device: {device}")
        print(f"GPU name: {gpu_name}")
    except Exception as exc:
        print("torch version: unavailable")
        print("cuda available: false")
        print(f"embedding device: {device}")
        print("GPU name: None")
        print(f"torch diagnostics error: {cast(Any, exc)}")
