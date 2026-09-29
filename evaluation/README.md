# Evaluation

Evaluation is split by concern:

- `recommendation/`: recommendation correctness and ranking quality.
- `explanation_groundedness/`: claims in the agent's answers checked against its tool outputs (rules + LLM judge).
- `score_candidate/`: pass/fail taste verdicts of `TasteService.score_candidate` checked against held-out test ratings, plus the label-choice experiments.


See each folder's README for its scope and status.
