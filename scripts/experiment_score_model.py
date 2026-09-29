#!/usr/bin/env python3
"""Experiment: can a small learned model beat the rule cascade of `score_candidate`?

Usage: python -m scripts.experiment_score_model [--n-test 2000 | 0 for the whole test split] [--seed 42]
Features come from the train split only (the same signals the scorer computes). Models are
fit on ratings_validation.csv; thresholds are picked on out-of-fold validation predictions
(folds grouped by user). The current cascade ("B", `score_candidate`) is scored on the same
test pairs. With the default sample the pairs match scripts.eval_score_candidate.
Writes evaluation/score_candidate/model_experiment.md unless --output says otherwise.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trusted_ai.taste import TasteService
from trusted_ai.taste.config import ROOT_DIR
from trusted_ai.taste.learned import ALL_RULE, RULE_FEATURES


FEATURES, ALL = RULE_FEATURES, ALL_RULE
TARGET_PASS_PRECISION = 0.80
TARGET_FAIL_PRECISION = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-test", type=int, default=2000, help="0 = every row of ratings_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT_DIR / "evaluation" / "score_candidate" / "model_experiment.md")
    return parser.parse_args()


def build_features(service: TasteService, pairs: pd.DataFrame) -> pd.DataFrame:
    """The 13 rule features per pair, computed by the scorer's own code so they match the evidence."""
    return service.scorer.features.frame(pairs, item_cf=False)


def models() -> dict[str, object]:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "gbm": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=40, random_state=0),
    }


def threshold_for(scores: np.ndarray, positives: np.ndarray, target: float) -> float:
    """Lowest threshold whose out-of-fold precision still reaches the target."""
    order = np.argsort(-scores)
    precision = np.cumsum(positives[order]) / np.arange(1, len(order) + 1)
    ok = np.where(precision >= target)[0]
    ok = ok[ok >= 29]  # ignore the noisy head of the ranking
    return float(scores[order][ok.max()]) if len(ok) else float("inf")


def cascade_verdicts(service: TasteService, pairs: pd.DataFrame) -> pd.DataFrame:
    """Verdicts of the current rule cascade (B) with the predictions it based them on."""
    rows = []
    for user_id, movie_id in zip(pairs["userId"].astype(int), pairs["movieId"].astype(int)):
        result = service.score_candidate(user_id, movie_id)
        rows.append({"verdict": result["verdict"], "source": result["source"],
                     "peer_predicted_rating": result["evidence"].get("peer_predicted_rating"),
                     "baseline_predicted_rating": result["evidence"].get("baseline_predicted_rating")})
    return pd.DataFrame(rows, index=pairs.index, dtype=object).astype(
        {"peer_predicted_rating": float, "baseline_predicted_rating": float})


def pct_ci(precision: float, n: int) -> str:
    """Precision with a 95% normal-approximation half-width, in percentage points."""
    if not n:
        return "-"
    half = 1.96 * np.sqrt(precision * (1 - precision) / n)
    return f"{precision * 100:.1f}% ±{half * 100:.1f}"


def precision_at(scores: np.ndarray, positives: np.ndarray, k: int) -> float:
    return float(positives[np.argsort(-scores)[:k]].mean()) if k else float("nan")


def pct(value: float) -> str:
    return "-" if value != value else f"{value * 100:.1f}%"


def run(service: TasteService, args: argparse.Namespace) -> str:
    paths = service.config.paths
    validation = pd.read_csv(paths.train_ratings.parent / "ratings_validation.csv")
    validation = validation[validation["userId"].isin(service.dataset.user_means.index)].reset_index(drop=True)
    test = pd.read_csv(paths.test_ratings)
    n_test_rows = len(test)
    if args.n_test > 0:
        test = test.sample(n=args.n_test, random_state=args.seed)
    test = test[test["userId"].isin(service.dataset.user_means.index)
                & test["movieId"].isin(service.dataset.movie_position)]
    b_pairs = cascade_verdicts(service, test)

    x_val, x_test = build_features(service, validation), build_features(service, test)
    liked_cut, disliked_cut = service.config.liked_rating, service.config.disliked_rating
    y_val = {"like": (validation["rating"] >= liked_cut).to_numpy(), "dislike": (validation["rating"] <= disliked_cut).to_numpy()}
    y_test = {"like": (test["rating"] >= liked_cut).to_numpy(), "dislike": (test["rating"] <= disliked_cut).to_numpy()}
    groups = validation["userId"].to_numpy()

    # Current cascade (B) on the same pairs, with and without the genre branch.
    b_pass = (b_pairs["verdict"] == "pass").to_numpy()
    b_fail = (b_pairs["verdict"] == "fail").to_numpy()
    b_fail_no_genre = b_fail & (b_pairs["source"] != "genre_deviation").to_numpy()
    b_peer = b_pairs["peer_predicted_rating"].fillna(b_pairs["baseline_predicted_rating"]).to_numpy()

    lines = [
        "# Thí nghiệm: mô hình học được vs. luồng quy tắc (B)",
        "",
        f"- Train: {len(validation)} cặp `ratings_validation.csv` (đặc trưng tính từ train). "
        f"Test: {len(test)} / {n_test_rows} cặp của `ratings_test.csv` "
        f"({'toàn bộ' if args.n_test <= 0 else f'mẫu ngẫu nhiên, seed {args.seed}'}), "
        f"{test['userId'].nunique()} user.",
        "- Độ chính xác ghi kèm ± nửa khoảng tin cậy 95% (điểm phần trăm).",
        f"- Tỉ lệ nền test: liked {pct(y_test['like'].mean())}, disliked {pct(y_test['dislike'].mean())}.",
        f"- Hai bộ phân loại: P(liked = rating ≥ {liked_cut}) và P(disliked = rating ≤ {disliked_cut}).",
        "",
        "## 1. AUC trên test",
        "",
        "| model | features | AUC liked | AUC disliked |",
        "|---|---|---|---|",
    ]
    if not np.isnan(b_peer).all():
        mask = ~np.isnan(b_peer)
        lines.append(f"| B: peer_predicted (fallback baseline) | 1 | "
                     f"{roc_auc_score(y_test['like'][mask], b_peer[mask]):.3f} | "
                     f"{roc_auc_score(y_test['dislike'][mask], -b_peer[mask]):.3f} |")
    ablations = {"all": ALL, "no content": [f for f in ALL if f not in FEATURES["content"]],
                 "no genre": [f for f in ALL if f not in FEATURES["genre"]],
                 "peer + baseline": FEATURES["peer"] + FEATURES["baseline"]}
    fitted: dict[tuple[str, str], object] = {}
    for model_name in models():
        for label, columns in ablations.items():
            if model_name == "gbm" and label != "all":
                continue
            aucs = []
            for target in ("like", "dislike"):
                model = models()[model_name].fit(x_val[columns], y_val[target])
                aucs.append(roc_auc_score(y_test[target], model.predict_proba(x_test[columns])[:, 1]))
                if label == "all":
                    fitted[(model_name, target)] = model
            lines.append(f"| {model_name} | {label} ({len(columns)}) | {aucs[0]:.3f} | {aucs[1]:.3f} |")

    lines += ["", "## 2. Cùng độ phủ với B (xếp hạng test theo xác suất, lấy top-k = số verdict của B)", "",
              "| | k | B | logreg | gbm |", "|---|---|---|---|---|"]
    probs = {(m, t): fitted[(m, t)].predict_proba(x_test[ALL])[:, 1] for m in models() for t in ("like", "dislike")}
    for name, k, b_mask, target in (("pass", b_pass.sum(), b_pass, "like"),
                                    ("fail (không tính genre)", b_fail_no_genre.sum(), b_fail_no_genre, "dislike"),
                                    ("fail (gồm genre)", b_fail.sum(), b_fail, "dislike")):
        cells = [pct_ci(precision_at(probs[(m, target)], y_test[target], int(k)), int(k)) for m in models()]
        lines.append(f"| {name} | {k} | {pct_ci(y_test[target][b_mask].mean(), int(k))} | " + " | ".join(cells) + " |")

    lines += ["", f"## 3. Quy tắc triển khai được: ngưỡng chọn trên validation (out-of-fold, GroupKFold theo user) "
              f"để precision pass ≥ {pct(TARGET_PASS_PRECISION)}, fail ≥ {pct(TARGET_FAIL_PRECISION)}", "",
              "| model | ngưỡng P(liked) | pass n | pass đúng | ngưỡng P(disliked) | fail n | fail đúng | tổng verdict | đúng tổng |",
              "|---|---|---|---|---|---|---|---|---|"]
    b_total = int(b_pass.sum() + b_fail.sum())
    b_right = y_test["like"][b_pass].sum() + y_test["dislike"][b_fail].sum()
    lines.append(f"| B (hiện tại) | - | {b_pass.sum()} | {pct_ci(y_test['like'][b_pass].mean(), int(b_pass.sum()))} | - | "
                 f"{b_fail.sum()} | {pct_ci(y_test['dislike'][b_fail].mean(), int(b_fail.sum()))} | {b_total} | "
                 f"{pct_ci(b_right / b_total, b_total)} |")
    folds = GroupKFold(n_splits=5)
    for model_name in models():
        cuts = {}
        for target, goal in (("like", TARGET_PASS_PRECISION), ("dislike", TARGET_FAIL_PRECISION)):
            oof = cross_val_predict(models()[model_name], x_val[ALL], y_val[target], groups=groups,
                                    cv=folds, method="predict_proba")[:, 1]
            cuts[target] = threshold_for(oof, y_val[target], goal)
        passed = probs[(model_name, "like")] >= cuts["like"]
        failed = ~passed & (probs[(model_name, "dislike")] >= cuts["dislike"])
        total = passed.sum() + failed.sum()
        right = y_test["like"][passed].sum() + y_test["dislike"][failed].sum()
        lines.append(f"| {model_name} | {cuts['like']:.3f} | {passed.sum()} | "
                     f"{pct_ci(y_test['like'][passed].mean(), int(passed.sum()))} | {cuts['dislike']:.3f} | "
                     f"{failed.sum()} | {pct_ci(y_test['dislike'][failed].mean(), int(failed.sum()))} | "
                     f"{total} | {pct_ci(right / total if total else 0.0, int(total))} |")

    lines += ["", "## 4. Hệ số logreg (đặc trưng đã chuẩn hoá; dương = tăng xác suất)", "",
              "| feature | P(liked) | P(disliked) |", "|---|---|---|"]
    coef = {t: fitted[("logreg", t)][-1].coef_[0] for t in ("like", "dislike")}
    for index, feature in enumerate(ALL):
        lines.append(f"| {feature} | {coef['like'][index]:+.3f} | {coef['dislike'][index]:+.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    # macOS Accelerate BLAS raises spurious FP warnings inside sklearn's matmuls; results are finite.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    args = parse_args()
    report = run(TasteService(use_score_model=False), args)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
