from __future__ import annotations

from math import log2
from statistics import mean
from typing import Any

from medical_rag.retrieval import BM25Retriever, Corpus, HybridRetriever, SearchResult, SemanticRetriever
from medical_rag.utils import write_json

EVAL_QUERIES = [
    {
        "id": "q1",
        "query": "What are the latest guidelines for managing type 2 diabetes?",
        "target_term": "type 2 diabetes mellitus",
    },
    {
        "id": "q2",
        "query": "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "target_term": "acute otitis media",
    },
    {
        "id": "q3",
        "query": "Iron supplementation dosing for anemia during pregnancy",
        "target_term": "iron deficiency anemia",
    },
    {
        "id": "q4",
        "query": "Çölyak hastalığı tanı kriterleri nelerdir?",
        "target_term": "celiac disease diagnosis",
    },
    {
        "id": "q5",
        "query": "Antibiotic resistance patterns in community acquired pneumonia",
        "target_term": "community acquired pneumonia",
    },
]


def evaluate_methods(
    payload: dict[str, Any],
    model_name: str,
    evaluation_path,
    bm25_analysis_path,
) -> dict[str, Any]:
    corpus = Corpus(payload)
    bm25 = BM25Retriever(corpus)
    semantic = SemanticRetriever(corpus, model_name=model_name)
    hybrid = HybridRetriever(bm25, semantic)
    methods = {"bm25": bm25, "semantic": semantic, "hybrid": hybrid}

    summary: dict[str, dict[str, float]] = {}
    outputs: dict[str, list[dict[str, Any]]] = {}

    for method_name, retriever in methods.items():
        query_outputs = []
        recalls = []
        mrrs = []
        ndcgs = []
        for query in EVAL_QUERIES:
            gold_pmids = gold_set(corpus, query["target_term"])
            results = retriever.search(query["query"], top_k=5)
            metric_block = metrics_for_results(results, gold_pmids)
            query_outputs.append(
                {
                    "query_id": query["id"],
                    "query": query["query"],
                    "target_term": query["target_term"],
                    "gold_size": len(gold_pmids),
                    "metrics": metric_block,
                    "results": [serialize_result(result) for result in results],
                }
            )
            recalls.append(metric_block["recall_at_5"])
            mrrs.append(metric_block["mrr_at_5"])
            ndcgs.append(metric_block["ndcg_at_5"])
        summary[method_name] = {
            "avg_recall_at_5": mean(recalls),
            "avg_mrr_at_5": mean(mrrs),
            "avg_ndcg_at_5": mean(ndcgs),
        }
        outputs[method_name] = query_outputs

    best_method = max(summary, key=lambda item: (summary[item]["avg_ndcg_at_5"], summary[item]["avg_mrr_at_5"]))
    payload_out = {
        "best_method": best_method,
        "summary": summary,
        "queries": outputs,
    }
    write_json(evaluation_path, payload_out)
    write_json(bm25_analysis_path, run_bm25_analysis(corpus))
    return payload_out


def gold_set(corpus: Corpus, target_term: str) -> set[str]:
    return {article["pmid"] for article in corpus.articles if target_term in article["matched_terms"]}


def metrics_for_results(results: list[SearchResult], gold_pmids: set[str]) -> dict[str, float]:
    if not gold_pmids:
        return {"precision_at_5": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0, "ndcg_at_5": 0.0}
    hits = [1 if result.pmid in gold_pmids else 0 for result in results]
    precision = sum(hits) / len(results) if results else 0.0
    recall = sum(hits) / len(gold_pmids)
    reciprocal_rank = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1.0 / idx
            break
    dcg = sum(hit / log2(idx + 1) for idx, hit in enumerate(hits, start=1))
    ideal_hits = [1] * min(len(gold_pmids), len(results))
    idcg = sum(hit / log2(idx + 1) for idx, hit in enumerate(ideal_hits, start=1))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "mrr_at_5": reciprocal_rank,
        "ndcg_at_5": ndcg,
    }


def serialize_result(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "pmid": result.pmid,
        "title": result.title,
        "score": result.score,
        "matched_terms": result.matched_terms,
    }


def run_bm25_analysis(corpus: Corpus) -> dict[str, Any]:
    query = "What are the latest guidelines for managing type 2 diabetes?"
    configs = [
        {"k1": 0.6, "b": 0.25},
        {"k1": 1.5, "b": 0.75},
        {"k1": 2.2, "b": 0.95},
    ]
    outputs = []
    for config in configs:
        retriever = BM25Retriever(corpus, k1=config["k1"], b=config["b"])
        outputs.append(
            {
                **config,
                "results": [serialize_result(result) for result in retriever.search(query, top_k=5)],
            }
        )
    return {"query": query, "configs": outputs}
