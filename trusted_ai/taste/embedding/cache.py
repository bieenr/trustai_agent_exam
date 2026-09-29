from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def text_key(text: str, namespace: str = "") -> str:
    return hashlib.sha1(f"{namespace}\x00{text}".encode("utf-8")).hexdigest()


class VectorCache:
    """One `.npy` file per text hash, so an interrupted encode resumes where it stopped.

    The namespace (model name, and instruction for queries) is part of the hash, so
    switching models never serves stale vectors.
    """

    def __init__(self, directory: str | Path, namespace: str = "") -> None:
        self.directory = Path(directory)
        self.namespace = namespace

    def get(self, text: str) -> np.ndarray | None:
        path = self._path(text)
        return np.load(path) if path.exists() else None

    def put(self, text: str, vector: np.ndarray) -> None:
        path = self._path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, vector.astype(np.float32))
        os.replace(temporary, path)  # atomic: a crash never leaves a half-written vector

    def missing(self, texts: Sequence[str]) -> list[int]:
        return [index for index, text in enumerate(texts) if not self._path(text).exists()]

    def get_or_embed(
        self,
        texts: Sequence[str],
        embed: Callable[[list[str]], np.ndarray],
        *,
        chunk_size: int = 64,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embed only uncached texts, persisting each chunk before requesting the next."""
        todo = self.missing(texts)
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start:start + chunk_size]
            vectors = embed([texts[index] for index in chunk])
            for index, vector in zip(chunk, vectors):
                self.put(texts[index], vector)
            if on_progress:
                on_progress(min(start + chunk_size, len(todo)), len(todo))
        return np.vstack([self.get(text) for text in texts])

    def _path(self, text: str) -> Path:
        key = text_key(text, self.namespace)
        return self.directory / key[:2] / f"{key}.npy"
