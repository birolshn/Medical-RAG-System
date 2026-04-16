from __future__ import annotations

import argparse
from pathlib import Path

from medical_rag.config import (
    DEFAULT_BM25_ANALYSIS_PATH,
    DEFAULT_CORPUS_PATH,
    DEFAULT_EVALUATION_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_RAG_PATH,
    DEFAULT_TERMS_PATH,
)
from medical_rag.utils import read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--terms-path", type=Path, default=DEFAULT_TERMS_PATH)
    fetch_parser.add_argument("--output-path", type=Path, default=DEFAULT_CORPUS_PATH)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    eval_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    eval_parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    eval_parser.add_argument("--bm25-analysis-path", type=Path, default=DEFAULT_BM25_ANALYSIS_PATH)

    rag_parser = subparsers.add_parser("rag")
    rag_parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    rag_parser.add_argument("--method", choices=["bm25", "semantic", "hybrid"], default="hybrid")
    rag_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    rag_parser.add_argument("--output-path", type=Path, default=DEFAULT_RAG_PATH)
    rag_parser.add_argument("--query", action="append")

    full_parser = subparsers.add_parser("run-all")
    full_parser.add_argument("--terms-path", type=Path, default=DEFAULT_TERMS_PATH)
    full_parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    full_parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    full_parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    full_parser.add_argument("--bm25-analysis-path", type=Path, default=DEFAULT_BM25_ANALYSIS_PATH)
    full_parser.add_argument("--output-path", type=Path, default=DEFAULT_RAG_PATH)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch":
        from medical_rag.pipeline import build_corpus

        payload = build_corpus(args.terms_path, args.output_path)
        print_summary(payload)
        return

    if args.command == "evaluate":
        from medical_rag.evaluation import evaluate_methods

        payload = read_json(args.corpus_path)
        output = evaluate_methods(payload, args.model_name, args.evaluation_path, args.bm25_analysis_path)
        print_evaluation_summary(output)
        print(f'best model is: {output["best_method"]}')
        return

    if args.command == "rag":
        from medical_rag.evaluation import EVAL_QUERIES
        from medical_rag.rag import run_rag_demo

        payload = read_json(args.corpus_path)
        queries = args.query or [EVAL_QUERIES[0]["query"], EVAL_QUERIES[1]["query"]]
        try:
            run_rag_demo(payload, args.method, args.model_name, queries, args.output_path)
            print(args.output_path)
        except RuntimeError as exc:
            print(str(exc))
        return

    if args.command == "run-all":
        from medical_rag.evaluation import EVAL_QUERIES, evaluate_methods
        from medical_rag.pipeline import build_corpus
        from medical_rag.rag import run_rag_demo

        corpus_payload = build_corpus(args.terms_path, args.corpus_path)
        print_summary(corpus_payload)
        evaluation_payload = evaluate_methods(
            corpus_payload,
            args.model_name,
            args.evaluation_path,
            args.bm25_analysis_path,
        )
        print_evaluation_summary(evaluation_payload)
        print(f'best model is: {evaluation_payload["best_method"]}')
        try:
            run_rag_demo(
                corpus_payload,
                evaluation_payload["best_method"],
                args.model_name,
                [EVAL_QUERIES[0]["query"], EVAL_QUERIES[1]["query"]],
                args.output_path,
            )
            print(args.output_path)
        except RuntimeError as exc:
            print(str(exc))
        return


def print_summary(payload: dict) -> None:
    stats = payload["stats"]
    print(f'terms processed: {stats["terms_processed"]}')
    print(f'unique articles: {stats["unique_articles"]}')
    print(f'duplicates removed: {stats["duplicates_removed"]}')
    print(f'errors: {stats["errors"]}')


def print_evaluation_summary(payload: dict) -> None:
    print("evaluation summary:")
    for method, metrics in payload["summary"].items():
        print(
            f"{method}: "
            f"recall@5={metrics['avg_recall_at_5']:.4f}, "
            f"mrr@5={metrics['avg_mrr_at_5']:.4f}, "
            f"ndcg@5={metrics['avg_ndcg_at_5']:.4f}"
        )


if __name__ == "__main__":
    main()
