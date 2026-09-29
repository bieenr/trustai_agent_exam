# Explanation groundedness

Does the agent's answer say only what its tools returned? Recommendation quality is measured in
`../recommendation/`; this checks the *explanation*: numbers, titles, ratings, genre preferences,
and whether descriptive statements come from the data or from the LLM's own movie knowledge.

## Data

Everything the pipeline reads is a **transcript** (`trusted_ai/transcripts.py`): the format of the
Streamlit chat logs (`logs/conversations/`) and of `scripts.run_agent_cases` runs (one file per case).
It holds the setup (user, model, agent config, system prompt) and every turn: user message, tool
calls with the exact output the LLM received, answer, token usage and any history compaction.
Turns marked `replayed` were given as context, not produced in that run, and are not judged.

| file | content |
|---|---|
| `data/explain_cases.json` | *(to generate)* follow-up cases: the transcript of a turn-1 run + one follow-up question written by an LLM playing the user (`scripts.build_explain_cases --base-run <run dir>`), 5 intents rotated: `why_movie`, `peer_opinion`, `own_history`, `challenge_claim`, `profile` |
| `results/agent_pass_first_<version>/` | transcripts of the agent's answers to `recommend_test.json` (turn 1) and `explain_cases.json` (turn 2) |
| `results/groundedness_<version>.{json,md}` | per-claim labels and the summary |

Turn 1 is the 60 `recommend_test.json` conversations; turn 2 replays the same turn-1 transcript for
every agent version, so versions are compared on the same follow-ups.

## Method

1. **Evidence** for an answer = the earlier turns still in the agent's context (user messages, tool
   outputs, answers; turns folded by history compaction are replaced by their summary) + this
   turn's user message and tool outputs (`evidence.py`).
2. **Claims**: an LLM splits the answer into atomic claims with a type (`movie_ref`, `user_rating`, `user_stat`,
   `genre_pref`, `peer_stat`, `prediction`, `qualitative`) and the stated number (`llm.py`).
3. **Judge**: an LLM labels every claim `supported`, `contradicted`, `unsupported` or `external` (general movie
   knowledge not in the data — true or not, it did not come from the dataset), and, when a lookup found nothing,
   whether the answer says so.
4. **Leakage** (regex in `evaluator.py`): model/probability/percentile/embedding words, decimals next to
   "score/similarity", movie IDs, snake_case field or tool names.

LLM steps are cached in `.eval_cache/groundedness/` (key: prompt version, model, step, input) and audited in
`evaluation_audit.jsonl`. Extractor and judge: `deepseek-v4.1-flash`, temperature 0. Keep the judge model fixed
when comparing agent models.

```bash
PY=python  # environment with langchain
R=evaluation/explanation_groundedness/results
$PY -m scripts.run_agent_cases --tag v3 --results-dir $R --no-recs                   # turn 1
$PY -m scripts.build_explain_cases --base-run $R/agent_pass_first_v3                  # follow-ups (review them)
$PY -m scripts.run_agent_cases --tag v3b --results-dir $R --no-recs \
    --cases evaluation/explanation_groundedness/data/explain_cases.json               # turn 2
$PY -m scripts.run_groundedness_eval --transcripts $R/agent_pass_first_v3 $R/agent_pass_first_v3b \
    --output $R/groundedness_v3.json
$PY -m scripts.run_groundedness_eval --transcripts logs/conversations/ \
    --output $R/groundedness_logs.json                                                # chat logs
```

## Results: chat logs (2026-09-29)

Three Streamlit conversations (users 1, 10 and 30; 26 turns) from `logs/conversations/`; full output in
`results/groundedness_logs.{json,md}`. 24 answers judged, 2 lost to judge errors (truncated or incomplete
JSON output). 15.2 claims per answer; 16.7% of answers have every claim supported, 91.7% cite at least one
data number, none leak internals.

| claim type | claims | supported | contradicted | unsupported | external |
|---|---|---|---|---|---|
| genre_pref | 47 | 91.5% | 4.3% | 4.3% | 0.0% |
| movie_ref | 56 | 94.6% | 0.0% | 1.8% | 3.6% |
| peer_stat | 54 | 88.9% | 5.6% | 5.6% | 0.0% |
| prediction | 24 | 91.7% | 0.0% | 8.3% | 0.0% |
| qualitative | 169 | 68.0% | 0.6% | 5.3% | 26.0% |
| user_rating | 6 | 100.0% | 0.0% | 0.0% | 0.0% |
| user_stat | 9 | 100.0% | 0.0% | 0.0% | 0.0% |
| **overall** | **365** | **81.1%** | **1.6%** | **4.7%** | **12.6%** |

Numbers hold up; almost all `external` claims are `qualitative` tone words the plot does not state. Real
errors: a movie's story told from memory when the tool returned no plot (Mary Poppins via `get_peer_opinion`),
peer counts inflated ("a handful" for one rater), a near-average genre called one of the user's strongest, and
wrong superlatives ("the strongest peer rating so far"). Some `external` labels are judge misses: most of the 9 on
the Chitty Chitty Bang Bang turn are story details (Caractacus Potts, the Baron, the Child Catcher) that are in
the plot returned in turn 1.

## Limitations

- The judge is an LLM and has not been checked against human labels for the current agent. With long
  evidence (full plots over several turns) it can miss details that are there and label them `external`.
- `external` mostly counts tone words ("fun", "warm", "a classic") that the plot text does not state.
- `states_not_found` rarely applies (the agent looks up reference movies itself); a set of trap questions (movies not in
  the dataset, false premises) is still to be written.
