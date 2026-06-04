from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ragarena.config import settings


class NoopObservation(AbstractContextManager["NoopObservation"]):
    def __enter__(self) -> "NoopObservation":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def update(self, **kwargs: Any) -> None:
        return None


@dataclass
class LangfuseObservation(AbstractContextManager["LangfuseObservation"]):
    context_manager: Any
    tracer: "LangfuseTracer"
    observation: Any = None

    def __enter__(self) -> "LangfuseObservation":
        try:
            self.observation = self.context_manager.__enter__()
            self.tracer._push_observation(self.observation)
        except Exception:
            self.observation = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self.context_manager.__exit__(*exc_info)
        except Exception:
            return None
        finally:
            self.tracer._pop_observation(self.observation)

    def update(self, **kwargs: Any) -> None:
        if self.observation is None:
            return
        try:
            self.observation.update(**kwargs)
        except Exception:
            return


class LangfuseTracer:
    def __init__(
        self,
        *,
        enabled: bool,
        public_key: str | None,
        secret_key: str | None,
        host: str,
    ) -> None:
        self.enabled = enabled and bool(public_key and secret_key)
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._client: Any | None = None
        self._observations: list[Any] = []
        self._trace_id: str | None = None
        self._trace_url: str | None = None

    def start_trace(
        self,
        name: str,
        input: Any,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        return self.observation(
            name,
            as_type="trace",
            input=input,
            metadata=metadata,
        )

    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> AbstractContextManager[Any]:
        if not self.enabled:
            return self._noop()

        client = self._get_client()
        if client is None:
            return self._noop()

        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
        }
        if input is not None:
            kwargs["input"] = input
        if metadata is not None:
            kwargs["metadata"] = metadata
        if model is not None:
            kwargs["model"] = model

        try:
            return LangfuseObservation(client.start_as_current_observation(**kwargs), self)
        except Exception:
            if as_type == "trace":
                kwargs["as_type"] = "span"
                try:
                    return LangfuseObservation(client.start_as_current_observation(**kwargs), self)
                except Exception:
                    return self._noop()
            return self._noop()

    def span(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Any]:
        return self.observation(
            name,
            as_type="span",
            input=input,
            metadata=metadata,
        )

    def generation(
        self,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> AbstractContextManager[Any]:
        observation = self.observation(
            name,
            as_type="generation",
            input=input,
            metadata=metadata,
            model=model,
        )
        if output is not None and isinstance(observation, NoopObservation):
            return observation
        return observation

    def event(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        client = self._get_client()
        if client is None:
            return
        try:
            if hasattr(client, "create_event"):
                client.create_event(name=name, metadata=metadata)
            elif hasattr(client, "event"):
                client.event(name=name, metadata=metadata)
        except Exception:
            return

    def _noop(self) -> NoopObservation:
        return NoopObservation()

    def get_trace_id(self) -> str | None:
        return self._trace_id

    def get_trace_url(self) -> str | None:
        return self._trace_url

    def flush(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.flush()
        except Exception:
            return

    def _push_observation(self, observation: Any) -> None:
        if observation is None:
            return
        self._observations.append(observation)
        trace_id = (
            getattr(observation, "trace_id", None)
            or getattr(observation, "traceId", None)
            or getattr(observation, "id", None)
        )
        if trace_id and self._trace_id is None:
            self._trace_id = str(trace_id)
            self._trace_url = self._build_trace_url(self._trace_id)

    def _pop_observation(self, observation: Any) -> None:
        if observation is None:
            return
        if self._observations and self._observations[-1] is observation:
            self._observations.pop()
            return
        self._observations = [item for item in self._observations if item is not observation]

    def _build_trace_url(self, trace_id: str) -> str | None:
        client = self._client
        if client is not None and hasattr(client, "get_trace_url"):
            try:
                return str(client.get_trace_url(trace_id))
            except Exception:
                pass
        if not self.host:
            return None
        return f"{self.host.rstrip('/')}/trace/{trace_id}"

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client

        try:
            from langfuse import Langfuse
        except ImportError:
            self.enabled = False
            return None

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", self.secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", self.host)

        try:
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                base_url=self.host,
            )
        except Exception:
            self.enabled = False
            return None
        return self._client


@lru_cache(maxsize=1)
def get_langfuse_tracer() -> LangfuseTracer:
    return LangfuseTracer(
        enabled=settings.langfuse_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
