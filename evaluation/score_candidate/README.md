# `score_candidate` evaluation

Checks whether `TasteService.score_candidate(user_id, movie_id)` predicts a user's taste correctly, using held-out
ratings the scorer never sees. Detailed experiment history is in [`experiment_note.md`](experiment_note.md)
(Vietnamese).

## 1. What is evaluated

`score_candidate` returns one of:

| verdict | meaning |
|---|---|
| `pass` | the user is predicted to like the movie |
| `fail` | the user is predicted to dislike the movie |
| no verdict (`insufficient_signal`) | not confident enough; in production the pair goes to `llm_judge` |

### Data

`data/ml-latest-small-filtered/ranking_split/`, 610 users, 5135 movies. The split is **temporal per user**: every
validation rating comes after that user's train ratings, and every test rating after validation.

| split | ratings | used for |
|---|---|---|
| `ratings_train.csv` | 59,023 | all features, user history, MLP training |
| `ratings_validation.csv` | 7,377 | fitting the GBM, picking thresholds |
| `ratings_test.csv` | 7,664 | ground-truth labels only |

### Labels

| label | rule | share of test |
|---|---|---|
| liked | rating ≥ 4.0 | 46.4% |
| disliked | rating ≤ 2.5 | 20.2% |
| middle | 2.5 < rating < 4.0 | 33.4%, not graded |

A `pass` is correct when the rating is ≥ 4.0. A `fail` is correct when it is ≤ 2.5.

### Metrics

| metric | what it shows |
|---|---|
| pass / fail precision | how often a verdict is right |
| overall precision | precision over all pass + fail verdicts |
| coverage | share of pairs that get a verdict; the rest cost an LLM call |
| recall liked / disliked | share of liked (disliked) pairs that get a pass (fail) |
| AUC liked / disliked | ranking quality of the score, independent of the threshold |
| within-user AUC | AUC computed per user, then averaged: ranking movies for the same person |
| precision at recall 10 / 25 / 50 / 75% | models compared at the same number of verdicts |
| users with ≥ 1 pass | whether passes reach all users or pile up on a few |

All numbers below are on the **full test set (7,664 pairs, 610 users)** unless stated otherwise.

## 2. Current method: gbm_stack

Code: `trusted_ai/taste/learned/`. Training: `python -m scripts.train_score_model`. Artifacts: `data/score_model/`.

### Architecture

```
(user, movie) ──► 13 features ──► GBM "liked"    ──► P(liked)    ≥ 0.703 ──► pass
                     │                                                        │ otherwise
                     └────────► GBM "disliked" ──► P(disliked) ≥ 0.702 ──► fail
                                                                               │ otherwise
                                                                               ▼
                                                                  no verdict → llm_judge
```

- Two `HistGradientBoostingClassifier` models: 200 iterations, learning rate 0.05, 15 leaves, at least 40 samples per
  leaf.
- Both are fit on the 7,377 validation pairs. Their features are computed from train only.
- Thresholds are picked on out-of-fold predictions (GroupKFold by user, 5 folds). Each threshold is the loosest cut that
  keeps **pass precision ≥ 80%** and **fail precision ≥ 70%**.
- Unknown users, or a missing `data/score_model/`, fall back to the rule chain (peer kNN → baseline → genre avoidance).

### Features (13, all from train)

| group | features | meaning |
|---|---|---|
| peer | `peer_offset`, `log_n_raters` | how the 20 most similar users who rated the movie deviate from their own mean |
| baseline | `user_mean`, `movie_bias`, `log_movie_n` | how generous the user is; how well the movie is rated overall |
| user | `user_std`, `log_user_n` | spread and size of the user's history |
| genre | `genre_dev_mean`, `genre_dev_min` | the user's rating deviation on the movie's genres |
| content | `content_pos_minus_neg` | embedding similarity to the user's liked movies minus similarity to their disliked movies |
| item-CF | `item_knn_offset`, `item_knn_weight` | how the user rated the 20 train movies most similar to this one by co-rating (adjusted cosine) |
| stack | `mlp_pred` | rating predicted by an MLP (see below) |

The **MLP** takes two inputs: a PCA-128 projection of the movie's `Qwen3-Embedding-4B` vector, and the user's
rating history. Its layers are 256 → 64 → 1, and it predicts the residual over `mu + b_u + b_i`. It is trained for 11
epochs on train only and never sees validation, so its predictions are out of sample for the GBM. Its ratings are
precomputed as a 610 × 5135 matrix, so serving does not need torch.

### Results

| AUC liked / disliked | within-user AUC liked / disliked | pass n (prec.) | fail n (prec.) | coverage | overall precision | recall liked / disliked | users with ≥ 1 pass |
|---|---|---|---|---|---|---|---|
| 0.801 / 0.841 | 0.678 / 0.681 | 1823 (80.7%) | 459 (85.8%) | 29.8% | 81.7% | 41.4% / 25.4% | 401 / 610 |

| precision at recall | 10% | 25% | 50% | 75% |
|---|---|---|---|---|
| liked | 90–92% | 86–87% | 77% | 68% |
| disliked | 93–95% | 84–87% | 65–66% | 45–46% |

Ranges are the spread over 2 MLP seeds. Recall is computed from the precision counts and the test base rates.

- **Calibration**: every 0.1 probability bin is within 0.04 of the observed rate. For example, P(liked) ≈ 0.85 means
  84% of those pairs are actually liked.
- **Severe errors are rare**: 66 of 1823 passes (3.6%) are movies rated ≤ 2.5, and 12 of 459 fails are rated ≥ 4.
  Most wrong passes are 3.0–3.5 ratings (286).
- **Runtime**: running the live `TasteService.score_candidate` on all test pairs reproduces the same counts. Start-up
  takes about 3.5 s, and each call takes about 14 ms.

By user group (quintiles of `user_mean`, seed 0):

| group | `user_mean` | AUC liked / disliked | pass n (prec.) | fail n (prec.) | coverage | users with ≥ 1 pass |
|---|---|---|---|---|---|---|
| Q1 harsh | 1.40–3.20 | 0.836 / 0.863 | 51 (86.3%) | 389 (87.9%) | 27.5% | 17 / 88 |
| Q2 | 3.20–3.42 | 0.729 / 0.796 | 100 (78.0%) | 34 (79.4%) | 9.1% | 37 / 86 |
| Q3 | 3.42–3.63 | 0.768 / 0.807 | 270 (78.1%) | 34 (67.6%) | 19.7% | 64 / 111 |
| Q4 | 3.63–3.91 | 0.729 / 0.760 | 391 (77.7%) | 2 (100%) | 25.9% | 97 / 134 |
| Q5 generous | 3.91–5.00 | 0.748 / 0.766 | 1011 (82.5%) | 0 | 66.0% | 186 / 191 |

The known weakness is that harsh users rarely get a pass. The passes they do get are reliable, but 71 of 88 Q1 users
have none. Separate thresholds per group were tried and made this worse: Q1 and Q2 cannot reach 80% pass precision on
validation, so Q1 lost all 51 passes, and overall precision fell to 78.6%.

## 3. Comparison with other methods (absolute labels)

Same labels, same test set, same thresholding targets.

| method | AUC liked | AUC disliked | pass n (prec.) | fail n (prec.) | coverage | overall precision |
|---|---|---|---|---|---|---|
| rule chain (fallback) | – | – | 1864 (76.9%) | 852 (53.8%) | 35.4% | 69.6% |
| logistic regression, 13 rule features | 0.779 | 0.817 | 1648 (79.3%) | 456 (71.3%) | 27.5% | 77.6% |
| MLP alone | 0.774 | 0.807 | 1824 (79.1%) | 568 (73.8%) | 31.2% | 77.8% |
| GBM regressor, rating ≥ 4.0 / ≤ 2.5 | 0.786 | 0.824 | 1436 (81.7%) | 696 (74.7%) | 27.8% | 79.4% |
| GBM, 13 rule features | 0.789 | 0.828 | 1453 (81.5%) | 307 (86.6%) | 23.0% | 82.4% |
| GBM, rule features + item-CF | 0.797 | 0.837 | 1725 (81.2%) | 435 (81.8%) | 28.2% | 81.3% |
| **gbm_stack** (current) | **0.801** | **0.841** | 1823 (80.7%) | 459 (85.8%) | 29.8% | 81.7% |

- gbm_stack has the best AUC on both sides. It keeps precision close to the best single GBM (81.7% vs 82.4%) while
  covering 6.8 points more pairs, about 520 extra verdicts.
- The rule chain has the highest coverage, but its fails are wrong almost half the time. It is kept only as a fallback.
- The MLP alone reaches similar coverage, but its fail precision is 12 points lower than gbm_stack's.

## 4. Why not relative labels

A fixed 4.0 cut can look unfair: for a harsh rater, 4.0 is a strong like, while for a generous rater it is an ordinary
movie. The alternative is labels relative to each user's train mean: **liked ⇔ rating − `user_mean` ≥ +0.5**,
**disliked ⇔ ≤ −0.5**.

The table uses the same columns as §3. Rows graded on relative labels predict the offset (predicted rating −
`user_mean`) and give a verdict when it is ≥ +δ or ≤ −δ. The first row repeats the absolute result of the same model for
reference. gbm_stack was not retrained on relative labels, so the GBM regressor is the comparison point: it is the
closest model that was run under both label types.

| method | labels / verdict rule | AUC liked | AUC disliked | pass n (prec.) | fail n (prec.) | coverage | overall precision |
|---|---|---|---|---|---|---|---|
| GBM regressor | absolute, ≥ 4.0 / ≤ 2.5 | 0.786 | 0.824 | 1436 (81.7%) | 696 (74.7%) | 27.8% | 79.4% |
| GBM regressor | relative, offset ±0.5 | 0.737 | 0.759 | 669 (67.3%) | 1199 (66.2%) | 24.4% | 66.6% |
| GBM regressor | relative, offset ±1.0 | 0.737 | 0.759 | 82 (89.0%) | 262 (76.3%) | 4.5% | 79.4% |
| GBM regressor | relative, thresholds picked on validation (80% / 70%) | 0.737 | 0.759 | 0 | 520 (75.2%) | 6.8% | 75.2% |
| ridge | relative, offset ±0.5 | 0.714 | 0.734 | 564 (60.8%) | 1140 (63.1%) | 22.2% | 62.3% |
| ordinal regression | relative, offset ±0.5 | 0.720 | 0.731 | 508 (64.8%) | 1141 (63.7%) | 21.5% | 64.0% |

Source: `regression_models.md`.

- **Harder to predict.** With the same model, AUC drops by about 0.05–0.07. Most of the signal is knowing which users
  rate high (`user_mean` is the top feature), and relative labels remove exactly that signal.
- **Less accurate at similar coverage.** Overall precision falls from 79.4% to 66.6% at a comparable coverage of about
  25%.
- **Precision can only be recovered by giving up coverage.** Matching the absolute setup's 79.4% requires δ = ±1.0,
  which drops coverage to 4.5%. Picking thresholds with the same 80% target as the absolute setup gives **no pass at
  all**.
- **The labels themselves are noisy.** A relative "disliked" is a real rating ≤ 2.5 only 62.2% of the time. For
  generous users this falls to 29.9%, because their relative dislikes are mostly 3.0–3.5 ratings.

**Decision:** keep absolute labels (≥ 4.0 / ≤ 2.5) with one global threshold. The cost is the Q1 gap described in §2.
Closing it means accepting lower precision for harsh users (a product decision) or ranking top-k per user.

## 5. Limitations

- The GBM is fit on only 7,377 validation pairs, because its features must be computed from train.
- The MLP's epoch count (11) comes from an earlier early-stopping run on validation, and validation is also used for
  thresholds. Both may make the thresholds slightly optimistic.
- gbm_stack was run with one GBM seed and two MLP seeds. Treat AUC differences below about 0.005 as noise.
- Within-user AUC is only about 0.68: ranking movies for the same person is still the hardest part.

## 6. Reproduce and files

| command | output |
|---|---|
| `python -m scripts.train_score_model` | trains gbm_stack → `data/score_model/` |
| `python -m scripts.eval_score_candidate [--n 2000] [--seed 42]` | live `score_candidate` on a test sample → `report.md`, `pairs.csv` |
| `python -m scripts.eval_score_models_full` | kNN / logreg / GBM on the full test set → `model_eval_full.md` |
| `python -m scripts.eval_regression_models` | regression models under absolute and relative labels → `regression_models.md` |

| file | content |
|---|---|
| `README.md` | this overview |
| `experiment_note.md` | curated log of every experiment (rule chain, logreg, GBM, kNN, MLP, item-CF, gbm_stack) |
| `regression_models.md` | raw output of `scripts.eval_regression_models`, the source of §4 (regenerated on each run) |
