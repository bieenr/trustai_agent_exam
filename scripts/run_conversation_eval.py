from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from evaluation.recommendation import CachedLLMJudge, ConversationEvaluator, load_catalog, load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked movie recommendations.")
    parser.add_argument("recommendations", type=Path, help="JSON mapping case_id to movie_id list")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/recommendation/data/recommend_test.json"),
    )
    parser.add_argument("--case-id", help="Evaluate only one case")
    parser.add_argument("-k", type=int, default=10, help="Maximum recommendations per case")
    parser.add_argument("--model", default="openai:deepseek/deepseek-v4.1-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="conversation-eval-v2")
    parser.add_argument("--cache-dir", type=Path, default=Path(".eval_cache"))
    parser.add_argument("--audit-path", type=Path, default=Path("evaluation_audit.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    recommendations = json.loads(args.recommendations.read_text(encoding="utf-8"))
    cases = load_cases(args.cases)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise ValueError(f"Unknown case_id: {args.case_id}")
    judge = CachedLLMJudge(
        model=args.model, temperature=args.temperature,
        prompt_version=args.prompt_version, cache_dir=args.cache_dir,
        audit_path=args.audit_path,
    )
    evaluator = ConversationEvaluator(judge, load_catalog())
    results = []
    missing = [case.case_id for case in cases if case.case_id not in recommendations]
    if missing:
        print(f"Skipping {len(missing)} case(s) without recommendations: {', '.join(missing)}")
    for case in cases:
        if case.case_id in missing:
            continue
        movie_ids = recommendations[case.case_id]
        if not isinstance(movie_ids, list):
            raise ValueError(f"Recommendations for {case.case_id!r} must be a JSON list")
        results.append(evaluator.evaluate(case, movie_ids, k=args.k).model_dump())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} case result(s) to {args.output}")


if __name__ == "__main__":
    main()
