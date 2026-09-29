from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
QUERY_INSTRUCTION = (
    "Given a user's description of the movie they want, "
    "retrieve movies whose plot, tone and genre match"
)


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    base_url: str
    api_key: str | None = None
    model: str = "Qwen/Qwen3-Embedding-4B"
    timeout_s: float = 60.0
    batch_size: int = 16
    max_retries: int = 3
    # Documents longer than this are cut at a word boundary before encoding. 6,000 chars
    # (~1.5k tokens) fits a server started with the default 2048-token context; raise it
    # together with vLLM's --max-model-len to embed full plots.
    max_chars: int = 6000

    @classmethod
    def from_env(cls) -> "EmbeddingSettings | None":
        """Return None when no embedding server is configured."""
        base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
        if not base_url:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=os.getenv("EMBEDDING_API_KEY") or None,
            # slots=True turns class attributes into descriptors, so defaults are spelled out.
            model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"),
            timeout_s=float(os.getenv("EMBEDDING_TIMEOUT_S", "60")),
            max_chars=int(os.getenv("EMBEDDING_MAX_CHARS", "6000")),
        )


@dataclass(frozen=True, slots=True)
class TastePaths:
    """Every artifact the taste module reads or writes, relative to one data root."""

    data_root: Path = ROOT_DIR / "data"

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "ml-latest-small-filtered"

    @property
    def train_ratings(self) -> Path:
        return self.dataset_dir / "ranking_split" / "ratings_train.csv"

    @property
    def test_ratings(self) -> Path:
        return self.dataset_dir / "ranking_split" / "ratings_test.csv"

    @property
    def movies(self) -> Path:
        return self.dataset_dir / "movies_with_plots.csv"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_root / "embeddings"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_root / "embedding_cache"

    @property
    def profiles(self) -> Path:
        return self.data_root / "user_taste_profiles.json"

    @property
    def user_similarity(self) -> Path:
        return self.data_root / "user_similarity.npy"

    @property
    def user_similarity_ids(self) -> Path:
        return self.data_root / "user_similarity_user_ids.npy"

    @property
    def score_model(self) -> Path:
        """Learned scorer artifacts written by scripts.train_score_model."""
        return self.data_root / "score_model"


@dataclass(frozen=True, slots=True)
class TasteConfig:
    liked_rating: float = 4.0
    disliked_rating: float = 2.5
    avoid_deviation: float = -0.5
    similar_users_k: int = 20
    min_common_ratings: int = 5
    min_similar_raters: int = 3
    # Item bias = sum of residuals / (n + shrinkage): rarely rated movies stay near the user's mean.
    baseline_shrinkage: float = 25.0
    query_weight: float = 0.6
    taste_weight: float = 0.4
    # Optional penalty from the "negative taste" vector; 0 keeps the plan's formula.
    negative_weight: float = 0.0
    # recommend_for_user blend; weights are renormalised when a signal is missing.
    cf_weight: float = 0.5
    content_weight: float = 0.35
    genre_weight: float = 0.15
    # Shrinks peer-based predictions from few raters toward the user's own mean.
    cf_shrinkage: float = 2.0
    # Users with no ratings get the best-liked movies overall: each mean rating is shrunk toward
    # the catalog mean as if this many extra ratings sat at that mean.
    cold_start_prior: float = 10.0
    # recommend_for_user / search_by_description rank this many candidates, check each with
    # score_candidate, then keep top_n: "pass_first" drops fails and puts passes first (in their
    # ranked order), "pass_only" keeps passes only, "off" keeps the ranking unchanged.
    candidate_pool: int = 20
    verdict_filter: str = "pass_first"
    # Offline dataset: a user's "now" is their latest train rating, so newer movies did not exist yet.
    cap_year_at_last_rating: bool = True
    paths: TastePaths = field(default_factory=TastePaths)
