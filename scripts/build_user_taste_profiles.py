#!/usr/bin/env python3
"""Build user taste artifacts from the training split only.

Writes the user-similarity matrix, user taste embeddings (when movie embeddings exist)
and data/user_taste_profiles.json. Re-run whenever the training split changes.

Usage: python -m scripts.build_user_taste_profiles [--summary-model openai:openai/gpt-4.1-mini]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from trusted_ai.taste.collaborative import compute_user_similarity
from trusted_ai.taste.config import ROOT_DIR, TasteConfig, TastePaths
from trusted_ai.taste.content import EmbeddingSpace, compute_user_embeddings
from trusted_ai.taste.dataset import assert_no_test_leak, load_dataset
from trusted_ai.taste.embedding import load_movie_embeddings, save_user_embeddings
from trusted_ai.taste.genre_profile import catalog_genre_share
from trusted_ai.taste.profile import build_profile
from trusted_ai.taste.summary import SUMMARY_PROMPT_VERSION, LLMSummariser, template_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=TastePaths().data_root)
    parser.add_argument("--output", type=Path, help="defaults to <data-root>/user_taste_profiles.json")
    parser.add_argument("--summary-model", help="LangChain model id; omit to use the template summary")
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)


def main() -> None:
    args = parse_args()
    paths = TastePaths(args.data_root)
    config = TasteConfig(paths=paths)
    dataset = load_dataset(paths)
    assert_no_test_leak(dataset, paths.test_ratings)

    similarity = compute_user_similarity(dataset)
    similarity.save(paths.user_similarity, paths.user_similarity_ids)
    print(f"Saved {similarity.matrix.shape} user similarity to {paths.user_similarity}")

    embedding_rows: dict[int, int] = {}
    movie_embeddings = load_movie_embeddings(paths.embeddings_dir)
    if movie_embeddings is None:
        print("No movie embeddings found; skipping user taste embeddings "
              "(run `python -m scripts.build_movie_embeddings` first)")
    else:
        users = compute_user_embeddings(EmbeddingSpace(dataset, movie_embeddings, None, config))
        save_user_embeddings(paths.embeddings_dir, users)
        embedding_rows = {int(uid): row for row, uid in enumerate(users.user_ids)}
        print(f"Saved {users.taste.shape} user taste embeddings to {paths.embeddings_dir}")

    summarise = (LLMSummariser(args.summary_model, paths.data_root / "summary_cache")
                 if args.summary_model else template_summary)
    share = catalog_genre_share(dataset.movies)
    profiles = []
    for user_id in dataset.user_ids:
        profile = build_profile(dataset, similarity, config, user_id, share, embedding_rows.get(user_id))
        profile["taste_summary"] = summarise(profile)
        profiles.append(profile)

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ratings_source": relative(dataset.ratings_path),
            "movies_source": relative(paths.movies),
            "user_count": len(profiles),
            "genre_scoring": "deviation = genre avg_rating - user avg_rating; "
                             "confidence low <3, medium 3-9, high >=10 ratings",
            "avoid_rule": f"deviation <= {config.avoid_deviation} and confidence != low",
            "similar_users": f"centered cosine, top {config.similar_users_k}, "
                             f"n_common_ratings >= {config.min_common_ratings}",
            "taste_summary_source": args.summary_model or "template",
            "taste_summary_prompt_version": SUMMARY_PROMPT_VERSION if args.summary_model else None,
            "embedding_model": movie_embeddings.meta.get("model") if movie_embeddings else None,
            "config": {key: value for key, value in asdict(config).items() if key != "paths"},
        },
        "profiles": profiles,
    }
    output = args.output or paths.profiles
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"Wrote {len(profiles)} profiles to {output}")


if __name__ == "__main__":
    main()
