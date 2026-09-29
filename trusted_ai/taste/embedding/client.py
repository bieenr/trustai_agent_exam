from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import httpx
import numpy as np

from trusted_ai.taste.config import EmbeddingSettings


logger = logging.getLogger(__name__)
RETRYABLE_STATUS = {408, 429}


class EmbeddingError(RuntimeError):
    """Raised when the embedding server cannot return vectors for a request."""


def format_query(text: str, instruction: str | None) -> str:
    """Qwen3-Embedding expects an instruction prefix on queries only, never on documents."""
    return f"Instruct: {instruction}\nQuery: {text}" if instruction else text


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class EmbeddingClient:
    """Minimal client for an OpenAI-compatible `/v1/embeddings` endpoint (vLLM, Xinference)."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._http = httpx.Client(
            base_url=settings.base_url, headers=headers,
            timeout=settings.timeout_s, transport=transport,
        )
        self._sleep = sleep

    def embed_texts(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray:
        """Return L2-normalised vectors with shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        inputs = [format_query(text, instruction) for text in texts]
        size = self.settings.batch_size
        batches = [self._embed_batch(inputs[start:start + size]) for start in range(0, len(inputs), size)]
        return l2_normalize(np.vstack(batches).astype(np.float32))

    def close(self) -> None:
        self._http.close()

    def _embed_batch(self, inputs: list[str]) -> np.ndarray:
        # The `dimensions` field is deliberately omitted: some vLLM versions reject it
        # with HTTP 400 ("does not support matryoshka representation").
        payload = {"model": self.settings.model, "input": inputs}
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._http.post("/embeddings", json=payload)
            except httpx.TransportError as exc:  # timeouts, refused connections, resets
                error: Exception = exc
            else:
                if response.status_code < 400:
                    return self._parse(response.json(), len(inputs))
                if response.status_code < 500 and response.status_code not in RETRYABLE_STATUS:
                    raise EmbeddingError(
                        f"Embedding request rejected ({response.status_code}): {response.text[:300]}"
                    )
                error = EmbeddingError(f"Embedding server error {response.status_code}")
            if attempt < attempts - 1:
                delay = 2.0 ** attempt
                logger.warning("Embedding request failed (%s); retrying in %.0fs", error, delay)
                self._sleep(delay)
        raise EmbeddingError(f"Embedding request failed after {attempts} attempts: {error}")

    @staticmethod
    def _parse(body: dict, expected: int) -> np.ndarray:
        data = sorted(body.get("data", []), key=lambda item: item["index"])
        if len(data) != expected:
            raise EmbeddingError(f"Expected {expected} embeddings, got {len(data)}")
        return np.asarray([item["embedding"] for item in data], dtype=np.float32)
