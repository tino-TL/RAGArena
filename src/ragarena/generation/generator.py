from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import requests

from ragarena.observability import get_langfuse_tracer

DEFAULT_SYSTEM_PROMPT = (
    "You are RAGArena's RAG answer generator. Answer only from the provided context."
)


@dataclass(frozen=True)
class GenerationResult:
    model: str
    answer: str


class DeepSeekGenerator:
    def __init__(
        self,
        api_key: str | None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> GenerationResult:
        if not self.is_configured():
            raise ValueError(
                "DEEPSEEK_API_KEY is not configured. Add it to .env before running ragarena-ask."
            )

        tracer = get_langfuse_tracer()
        with tracer.generation(
            "deepseek.generate",
            model=self.model,
            input={"system": system_prompt, "prompt": prompt},
            metadata={"provider": "deepseek", "stream": False},
        ) as observation:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0,
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            result = GenerationResult(
                model=payload.get("model", self.model),
                answer=payload["choices"][0]["message"]["content"].strip(),
            )
            observation.update(output=result.answer)
            return result

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Iterator[str]:
        if not self.is_configured():
            raise ValueError(
                "DEEPSEEK_API_KEY is not configured. Add it to .env before running generation."
            )

        tracer = get_langfuse_tracer()
        with tracer.generation(
            "deepseek.stream_generate",
            model=self.model,
            input={"system": system_prompt, "prompt": prompt},
            metadata={"provider": "deepseek", "stream": True},
        ) as observation:
            chunks: list[str] = []
            with self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0,
                    "stream": True,
                },
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if not line or not line.startswith("data: "):
                        continue

                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break

                    chunk = parse_stream_delta(data)
                    if chunk:
                        chunks.append(chunk)
                        yield chunk
            observation.update(output="".join(chunks))

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


class OllamaDecisionGenerator:
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: int = 30,
        keep_alive: str = "30m",
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        *,
        json_mode: bool = True,
    ) -> GenerationResult:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "think": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"

        tracer = get_langfuse_tracer()
        with tracer.generation(
            "ollama.decision",
            model=self.model,
            input={"system": system_prompt, "prompt": prompt},
            metadata={"provider": "ollama", "json_mode": json_mode, "think": False},
        ) as observation:
            response = self.session.post(
                f"{self.url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            answer = str((data.get("message") or {}).get("content") or "").strip()
            result = GenerationResult(model=str(data.get("model") or self.model), answer=answer)
            observation.update(output=result.answer)
            return result


def parse_stream_delta(data: str) -> str:
    import json

    payload = json.loads(data)
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""
