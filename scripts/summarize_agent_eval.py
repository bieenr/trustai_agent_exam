#!/usr/bin/env python3
"""Compare the conversation evaluation of several agent runs (hard constraints and semantic fit).

Usage: python -m scripts.summarize_agent_eval off pass_first pass_only [eval_name:agent_name ...]
Reads evaluation/recommendation/results/eval_<name>.json (scripts.run_conversation_eval output for
agent_<name>_recs.json) and the transcripts in agent_<name>/ (scripts.run_agent_cases), and writes
summary.md there.
`eval_name:agent_name` pairs an evaluation with the agent run it judged (e.g. a re-judged run).
Taste is not reported: `score_candidate` is evaluated on the full test split instead.
"""

from __future__ import annotations

import argparse
import json
from statistics import mean

from evaluation.recommendation import load_cases
from scripts.run_agent_cases import CASES, RESULTS_DIR
from trusted_ai import transcripts


def load_run(name: str) -> dict[str, dict]:
    """{case_id: {"movie_ids": [...]} or {"error": ...}} from the last turn of each transcript."""
    rows = {}
    for _, transcript in transcripts.load([RESULTS_DIR / f"agent_{name}"]):
        turn = transcript["turns"][-1]
        rows[transcript["case_id"]] = ({"error": turn["error"]} if "error" in turn else
                                       {"movie_ids": [movie["movie_id"] for movie in turn.get("recommendations", [])]})
    return rows


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def criterion(cases: list[dict], name: str) -> dict:
    judged = [case["metrics"][name] for case in cases
              if case["metrics"].get(name) and case["metrics"][name]["judged_count"]]
    movies = [movie for case in cases for movie in case["movies"] if movie.get(name) is not None]
    return {
        "cases": len(judged),
        "movies": len(movies),
        "movie_pass": mean(movie[name]["passed"] for movie in movies) if movies else None,
        "precision": mean(m["precision"] for m in judged) if judged else None,
        "all_pass": mean(m["precision"] == 1.0 for m in judged) if judged else None,
        "hit_rate": mean(m["hit_rate"] for m in judged) if judged else None,
        "ndcg": mean(m["ndcg"] for m in judged) if judged else None,
    }


PASS_COUNTS = (1, 2, 3, 4, 5)


def passes(movie: dict, name: str) -> bool:
    """`both` = hard constraints pass and semantic passes or is not required."""
    if name == "both":
        return movie["hard_constraints"]["passed"] and (movie.get("semantic") is None or movie["semantic"]["passed"])
    return bool(movie.get(name)) and movie[name]["passed"]


def pass_distribution(cases: list[dict], name: str, semantic_cases: set[str]) -> str:
    """Share of cases with at least n passing movies; semantic counts only cases that require it
    (including those where the agent returned nothing)."""
    scope = cases if name != "semantic" else [case for case in cases if case["case_id"] in semantic_cases]
    counts = [sum(passes(movie, name) for movie in case["movies"]) for case in scope]
    return " | ".join(f"{sum(c >= n for c in counts)} ({pct(mean(c >= n for c in counts))})" for n in PASS_COUNTS) \
        if counts else " | ".join("-" for _ in PASS_COUNTS)


def failures(cases: list[dict], name: str) -> list[str]:
    lines = []
    for case in cases:
        for movie in case["movies"]:
            result = movie.get(name)
            if result is not None and not result["passed"]:
                lines.append(f"- `{case['case_id']}` #{movie['rank']} {movie['title']}: {'; '.join(result['reasons'])}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    pairs = [name.split(":", 1) if ":" in name else (name, name) for name in parser.parse_args().names]
    runs = {name: (json.loads((RESULTS_DIR / f"eval_{name}.json").read_text(encoding="utf-8")),
                   load_run(agent))
            for name, agent in pairs}
    semantic_cases = {case.case_id for case in load_cases(CASES) if case.expected.semantic_requirement}

    lines = ["# Agent trên recommend_test.json: ràng buộc cứng và độ hợp ngữ cảnh", "",
             "- `precision` = trung bình theo case của tỉ lệ phim đạt; `all pass` = tỉ lệ case mà mọi phim đều đạt; "
             "`hit rate` = tỉ lệ case có ≥ 1 phim đạt. Semantic chỉ tính các case có `semantic_requirement: true`.",
             "- Judge semantic: LLM, cache trong `.eval_cache/` (dùng chung giữa các luồng).", "",
             "| luồng | case có gợi ý | phim / case | case 0 phim | lỗi agent | ràng buộc: phim đạt | ràng buộc: all pass | "
             "semantic: phim đạt | semantic: precision | semantic: hit rate | semantic: NDCG |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (cases, agent) in runs.items():
        hard, semantic = criterion(cases, "hard_constraints"), criterion(cases, "semantic")
        counts = [len(row.get("movie_ids", [])) for row in agent.values()]
        lines.append(f"| {name} | {sum(c > 0 for c in counts)} / {len(agent)} | {mean(counts):.1f} | "
                     f"{sum(c == 0 for c in counts)} | {sum('error' in row for row in agent.values())} | "
                     f"{pct(hard['movie_pass'])} | {pct(hard['all_pass'])} | {pct(semantic['movie_pass'])} "
                     f"({semantic['movies']} phim) | {pct(semantic['precision'])} | {pct(semantic['hit_rate'])} | "
                     f"{pct(semantic['ndcg'])} |")
    header = " | ".join(f"≥ {n} phim" for n in PASS_COUNTS)
    lines += ["", "## Số case có ≥ n phim đạt", "",
              "“cả hai” = đạt ràng buộc cứng và đạt semantic (hoặc case không yêu cầu semantic). Semantic tính trên các "
              "case có yêu cầu semantic.", "",
              f"| luồng | tiêu chí | {header} |", "|---|---|" + "---|" * len(PASS_COUNTS)]
    for name, (cases, _) in runs.items():
        for label, key in (("ràng buộc", "hard_constraints"), ("semantic", "semantic"), ("cả hai", "both")):
            lines.append(f"| {name} | {label} | {pass_distribution(cases, key, semantic_cases)} |")
    for name, (cases, agent) in runs.items():
        empty = [case_id for case_id, row in agent.items() if not row.get("movie_ids")]
        lines += ["", f"## {name}", "", f"Case không có phim: {', '.join(empty) or 'không'}.", "",
                  "Phim trượt ràng buộc cứng:", "", *(failures(cases, "hard_constraints") or ["- không"]), "",
                  "Phim trượt semantic:", "", *(failures(cases, "semantic") or ["- không"])]
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
