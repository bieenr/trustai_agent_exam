# Recommendation evaluation

Each recommended movie is judged on three **independent** criteria for a `(movie, user, conversation)` input. A failure on one criterion never skips the others.

| Criterion | Judged on | Verdict | `None` when |
|---|---|---|---|
| `hard_constraints` | every movie | genres, year range, not previously rated in `ratings_train` | never |
| `semantic` | every movie | LLM judge: does the movie fit the conversation? | the case has `semantic_requirement: false` |
| `taste` | movies with a held-out rating in `ratings_test` for this user | rating ≥ 4.0 is a hit, ≤ 2.5 is a dislike | the user has no held-out rating for the movie |

An unknown `movie_id` fails `hard_constraints` and `semantic` and is skipped for `taste`.

Taste uses only held-out ratings as ground truth. There is no LLM taste judge and no `TasteService` fallback: the agent recommends with `TasteService`, so scoring with it would grade the system against itself.

## Metrics

`ranking_metrics.py` computes Precision@K, NDCG@K and hit rate **separately for each criterion**. The top-K list is truncated first, then movies with a `None` verdict are dropped, and metrics are computed over the remaining `judged_count` movies in their original order. NDCG is binary and normalizes against the best ordering of the hits within that list, so it needs no full candidate pool.

`taste` also reports `negative_hits`, `dislike_rate` and `negative_hit_rate`. A criterion that judged nothing in a case (semantic disabled, or no held-out ratings) has `null` metrics; skip those cases when averaging. Duplicate movie IDs are removed before the top-K list is taken.

## Structure

- `catalog.py`: loads movies, train and test ratings into a `Catalog`.
- `criteria.py`: the three criterion checks.
- `evaluator.py`: `ConversationEvaluator` runs the checks for each case, and `load_cases` reads test cases.
- `llm_judge.py`: cached semantic LLM judge with audit logging.
- `ranking_metrics.py`: per-criterion metrics and catalog coverage.
- `schemas.py`: Pydantic input and output models.
- `data/recommend_test.json`: conversation test cases.

## Run

```bash
python -m scripts.run_conversation_eval recommendations.json \
  --cases evaluation/recommendation/data/recommend_test.json \
  -k 10
```

Each case result contains per-movie verdicts and a `metrics` object with `hard_constraints`, `semantic` and `taste` entries. By default, results are written to `evaluation_results.json`, judge responses are cached in `.eval_cache/`, and judge inputs and outputs are appended to `evaluation_audit.jsonl`.
