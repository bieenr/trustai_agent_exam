"""User taste module: genre profile, similar users and content embeddings served over an API."""

from .config import EmbeddingSettings, TasteConfig, TastePaths
from .dataset import TasteDataset, load_dataset
from .service import TasteService

__all__ = ["EmbeddingSettings", "TasteConfig", "TasteDataset", "TastePaths", "TasteService", "load_dataset"]
