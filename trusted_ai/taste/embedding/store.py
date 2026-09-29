from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MOVIE_EMBEDDINGS = "movie_embeddings.npy"
MOVIE_IDS = "movie_ids.npy"
META = "meta.json"
USER_TASTE = "user_taste_embeddings.npy"
USER_NEGATIVE = "user_negative_embeddings.npy"
USER_IDS = "user_ids.npy"


@dataclass(frozen=True)
class MovieEmbeddings:
    vectors: np.ndarray  # float32, L2-normalised, one row per movie
    movie_ids: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class UserEmbeddings:
    taste: np.ndarray
    negative: np.ndarray
    user_ids: np.ndarray


def save_movie_embeddings(directory: Path, vectors: np.ndarray, movie_ids: np.ndarray,
                          meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / MOVIE_EMBEDDINGS, vectors.astype(np.float16))
    np.save(directory / MOVIE_IDS, movie_ids.astype(np.int64))
    (directory / META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_movie_embeddings(directory: Path) -> MovieEmbeddings | None:
    if not (directory / MOVIE_EMBEDDINGS).exists():
        return None
    vectors = np.load(directory / MOVIE_EMBEDDINGS).astype(np.float32)
    # float16 storage loses a little precision; renormalise so dot product stays cosine.
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    meta_path = directory / META
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return MovieEmbeddings(vectors, np.load(directory / MOVIE_IDS), meta)


def save_user_embeddings(directory: Path, embeddings: UserEmbeddings) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / USER_TASTE, embeddings.taste.astype(np.float32))
    np.save(directory / USER_NEGATIVE, embeddings.negative.astype(np.float32))
    np.save(directory / USER_IDS, embeddings.user_ids.astype(np.int64))


def load_user_embeddings(directory: Path) -> UserEmbeddings | None:
    if not (directory / USER_TASTE).exists():
        return None
    return UserEmbeddings(
        np.load(directory / USER_TASTE), np.load(directory / USER_NEGATIVE),
        np.load(directory / USER_IDS),
    )
