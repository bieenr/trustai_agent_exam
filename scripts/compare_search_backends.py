#!/usr/bin/env python3
"""Compare TF-IDF and embedding description search, and time live query encoding.

Usage: python -m scripts.compare_search_backends [--latency-runs 30]
Writes evaluation/search_backend_comparison.json for the report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from trusted_ai.taste import TasteService
from trusted_ai.taste.config import QUERY_INSTRUCTION, ROOT_DIR


USERS = (1, 15, 30)
QUERIES = [
    {"query": "dark psychological thriller with a twist"},
    {"query": "I liked Toy Story but I'm tired of animated movies", "exclude_genres": ["Animation"]},
    {"query": "feel-good romantic comedy set in New York"},
    {"query": "epic space battle with rebels fighting an empire"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latency-runs", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "evaluation" / "search_backend_comparison.json")
    return parser.parse_args()


def measure_latency(service: TasteService, runs: int) -> dict[str, float]:
    """Uncached single-query round trips (network + model), as the agent sees them at runtime."""
    client = service.search.encoder.client
    timings = []
    for index in range(runs):
        started = time.perf_counter()
        client.embed_texts([f"latency probe {index}: a tense heist movie"], instruction=QUERY_INSTRUCTION)
        timings.append((time.perf_counter() - started) * 1000)
    return {"runs": runs, "p50_ms": round(float(np.percentile(timings, 50)), 1),
            "p95_ms": round(float(np.percentile(timings, 95)), 1),
            "max_ms": round(max(timings), 1)}


def main() -> None:
    args = parse_args()
    service = TasteService()
    if service.search.encoder is None:
        raise SystemExit("Embedding search unavailable: build movie embeddings and set EMBEDDING_BASE_URL.")
    comparisons = []
    for user_id in USERS:
        for spec in QUERIES:
            filters = {key: value for key, value in spec.items() if key != "query"}
            row = {"user_id": user_id, **spec}
            for backend in ("tfidf", "embedding"):
                # the raw search ranking: verdict filtering is not part of the backend comparison
                result = service.search.search(user_id, spec["query"], top_n=args.top_n,
                                               backend=backend, **filters)
                row[backend] = [f"{item['title']} ({item['year']}) [{'|'.join(item['genres'])}] "
                                f"q={item['query_similarity']:.3f}" for item in result["results"]]
            comparisons.append(row)
            print(f"\nuser {user_id} | {spec['query']}")
            for backend in ("tfidf", "embedding"):
                print(f"  {backend}:")
                for line in row[backend]:
                    print(f"    - {line}")

    latency = measure_latency(service, args.latency_runs)
    print(f"\nQuery encoding latency: {latency}")
    payload = {"embedding_meta": service.embedding.meta, "query_latency": latency, "comparisons": comparisons}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
