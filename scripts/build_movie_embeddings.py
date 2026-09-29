#!/usr/bin/env python3
"""Encode every movie once through the embedding server; safe to interrupt and re-run.

Usage: python -m scripts.build_movie_embeddings [--check] [--limit N]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from trusted_ai.taste.config import EmbeddingSettings, TastePaths
from trusted_ai.taste.content import embedding_text
from trusted_ai.taste.dataset import load_dataset
from trusted_ai.taste.embedding import (
    EmbeddingClient, VectorCache, load_movie_embeddings, save_movie_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=TastePaths().data_root)
    parser.add_argument("--output-dir", type=Path, help="defaults to <data-root>/embeddings")
    parser.add_argument("--chunk-size", type=int, default=64, help="texts persisted per cache flush")
    parser.add_argument("--batch-size", type=int, default=16, help="texts per HTTP request")
    parser.add_argument("--limit", type=int, help="encode only the first N movies (smoke test)")
    parser.add_argument("--check", action="store_true", help="embed two sentences, print the dimension, exit")
    return parser.parse_args()


def health_check(client: EmbeddingClient) -> None:
    started = time.perf_counter()
    vectors = client.embed_texts(["A heist goes wrong.", "A toy cowboy feels replaced."])
    print(f"OK: {vectors.shape[0]} vectors of dim {vectors.shape[1]} "
          f"in {time.perf_counter() - started:.2f}s from {client.settings.base_url}")


def main() -> None:
    args = parse_args()
    settings = EmbeddingSettings.from_env()
    if settings is None:
        raise SystemExit("Set EMBEDDING_BASE_URL (and EMBEDDING_API_KEY) in .env first.")
    client = EmbeddingClient(replace(settings, batch_size=args.batch_size))
    if args.check:
        health_check(client)
        return

    paths = TastePaths(args.data_root)
    dataset = load_dataset(paths)
    movies = dataset.movies.head(args.limit) if args.limit else dataset.movies
    texts = [embedding_text(row, settings.max_chars) for _, row in movies.iterrows()]
    truncated = sum(len(embedding_text(row)) > settings.max_chars for _, row in movies.iterrows())
    cache = VectorCache(paths.embedding_cache_dir / "movies", namespace=settings.model)
    pending = len(cache.missing(texts))
    print(f"{len(texts)} movies, {len(texts) - pending} cached, {pending} to encode via {settings.base_url}")

    started = time.perf_counter()
    vectors = cache.get_or_embed(
        texts, client.embed_texts, chunk_size=args.chunk_size,
        on_progress=lambda done, total: print(f"  encoded {done}/{total}", flush=True),
    )
    elapsed = time.perf_counter() - started
    output_dir = args.output_dir or paths.embeddings_dir
    # Keep timings of earlier (possibly interrupted) runs; a fully cached re-run adds nothing.
    previous = load_movie_embeddings(output_dir)
    runs = previous.meta.get("encode_runs", []) if previous else []
    if pending:
        runs.append({"finished_at": datetime.now(timezone.utc).isoformat(), "encoded": pending,
                     "seconds": round(elapsed, 2), "seconds_per_movie": round(elapsed / pending, 4)})
    meta = {
        "model": settings.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dimensions": int(vectors.shape[1]),
        "n_movies": len(texts),
        "encode_runs": runs,
        "batch_size": args.batch_size,
        "text_template": "{title} ({year}). Genres: {genres}. {plot}",
        "max_chars": settings.max_chars,
        "n_truncated": int(truncated),
        "dtype": "float16",
    }
    save_movie_embeddings(output_dir, vectors, movies["movieId"].to_numpy(), meta)
    print(f"Wrote {vectors.shape} embeddings to {output_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
