from __future__ import annotations

import json
import tempfile
import unittest

import httpx
import numpy as np

from trusted_ai.taste.config import EmbeddingSettings
from trusted_ai.taste.embedding import EmbeddingClient, EmbeddingError, VectorCache


def settings(**overrides: object) -> EmbeddingSettings:
    values = {"base_url": "http://embed.test/v1", "api_key": "secret", "batch_size": 2}
    return EmbeddingSettings(**{**values, **overrides})  # type: ignore[arg-type]


class Server:
    """Records requests and replays scripted responses; vectors are [len(text), 1, 0]."""

    def __init__(self, failures: list[int] | None = None) -> None:
        self.failures = list(failures or [])
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"body": body, "auth": request.headers.get("authorization")})
        if self.failures:
            return httpx.Response(self.failures.pop(0), text="boom")
        data = [{"index": i, "embedding": [float(len(text)), 1.0, 0.0]} for i, text in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data))})


def client_for(server: Server, sleeps: list[float], **overrides: object) -> EmbeddingClient:
    return EmbeddingClient(settings(**overrides), transport=httpx.MockTransport(server), sleep=sleeps.append)


class EmbeddingClientTest(unittest.TestCase):
    def test_instruction_is_prefixed_and_batches_are_split(self) -> None:
        server, sleeps = Server(), []
        vectors = client_for(server, sleeps).embed_texts(["a", "bb", "ccc"], instruction="Find films")

        self.assertEqual(len(server.requests), 2)
        self.assertEqual(server.requests[0]["body"]["input"], ["Instruct: Find films\nQuery: a",
                                                                "Instruct: Find films\nQuery: bb"])
        self.assertNotIn("dimensions", server.requests[0]["body"])
        self.assertEqual(server.requests[0]["auth"], "Bearer secret")
        self.assertEqual(vectors.shape, (3, 3))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)

    def test_documents_are_sent_without_instruction_and_kept_in_order(self) -> None:
        server = Server()
        vectors = client_for(server, []).embed_texts(["a", "bbbb"])

        self.assertEqual(server.requests[0]["body"]["input"], ["a", "bbbb"])
        # response data was reversed; parsing must restore order by index
        self.assertGreater(vectors[1, 0], vectors[0, 0])

    def test_server_errors_are_retried_with_backoff(self) -> None:
        server, sleeps = Server(failures=[503, 500]), []
        vectors = client_for(server, sleeps).embed_texts(["a"])

        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(vectors.shape, (1, 3))

    def test_client_errors_are_not_retried(self) -> None:
        server, sleeps = Server(failures=[400]), []
        with self.assertRaises(EmbeddingError):
            client_for(server, sleeps).embed_texts(["a"])
        self.assertEqual((len(server.requests), sleeps), (1, []))

    def test_gives_up_after_three_retries(self) -> None:
        server, sleeps = Server(failures=[500] * 10), []
        with self.assertRaises(EmbeddingError):
            client_for(server, sleeps).embed_texts(["a"])
        self.assertEqual(sleeps, [1.0, 2.0, 4.0])
        self.assertEqual(len(server.requests), 4)

    def test_connection_errors_are_retried(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        sleeps: list[float] = []
        client = EmbeddingClient(settings(), transport=httpx.MockTransport(refuse), sleep=sleeps.append)
        with self.assertRaises(EmbeddingError):
            client.embed_texts(["a"])
        self.assertEqual(sleeps, [1.0, 2.0, 4.0])


class VectorCacheTest(unittest.TestCase):
    def test_interrupted_run_resumes_with_only_missing_texts(self) -> None:
        texts = [f"movie {i}" for i in range(5)]
        calls: list[list[str]] = []

        def flaky(batch: list[str]) -> np.ndarray:
            calls.append(batch)
            if len(calls) == 2:
                raise EmbeddingError("server went away")
            return np.ones((len(batch), 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            cache = VectorCache(directory, namespace="model-a")
            with self.assertRaises(EmbeddingError):
                cache.get_or_embed(texts, flaky, chunk_size=2)
            self.assertEqual(cache.missing(texts), [2, 3, 4])

            vectors = cache.get_or_embed(texts, flaky, chunk_size=2)
            self.assertEqual(calls[2:], [["movie 2", "movie 3"], ["movie 4"]])
            self.assertEqual(vectors.shape, (5, 4))
            self.assertEqual(VectorCache(directory, namespace="model-b").missing(texts), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
