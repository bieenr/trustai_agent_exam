from __future__ import annotations

import pandas as pd

from trusted_ai.taste.dataset import TasteDataset


def item_biases(dataset: TasteDataset, shrinkage: float) -> pd.DataFrame:
    """How far each movie lands from its raters' own means, shrunk toward 0 when few rated it.

    Returns a frame indexed by movieId with columns `bias` and `n_ratings`. Residuals are
    taken against each rater's mean so a movie loved by harsh critics still reads as good.
    """
    ratings = dataset.ratings
    residual = ratings["rating"] - ratings["userId"].map(dataset.user_means)
    grouped = residual.groupby(ratings["movieId"]).agg(["sum", "size"])
    return pd.DataFrame({
        "bias": grouped["sum"] / (grouped["size"] + shrinkage),
        "n_ratings": grouped["size"].astype(int),
    })
