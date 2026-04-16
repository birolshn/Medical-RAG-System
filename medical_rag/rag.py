from __future__ import annotations

import os
from typing import Any

import requests

from medical_rag.retrieval import BM25Retriever, Corpus, HybridRetriever, SemanticRetriever
from medical_rag.utils import write_json


def build_retriever(payload: dict[str, Any], method: str, model_name: str):
    corpus = Corpus(payload)
    bm25 = BM25Retriever(corpus)
    semantic = SemanticRetriever(corpus, model_name=model_name)
    if method == "bm25":
        return bm25
    if method == "semantic":
        return semantic
    return HybridRetriever(bm25, semantic)


def run_rag_demo(
    corpus_payload: dict[str, Any],
    method: str,
    model_name: str,
    queries: list[str],
    output_path,
    top_k: int = 5,
) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")
    retriever = build_retriever(corpus_payload, method=method, model_name=model_name)
    outputs = []
    for query in queries:
        results = retriever.search(query, top_k=top_k)
        context = "\n\n".join(
            [
                f'PMID: {result.pmid}\nTitle: {retriever.corpus.pmid_to_article[result.pmid]["title"]}\n'
                f'Abstract: {retriever.corpus.pmid_to_article[result.pmid]["abstract"]}'
                for result in results
            ]
        )
        answer = call_groq(api_key, query, context)
        outputs.append(
            {
                "query": query,
                "retrieved_documents": [
                    {
                        "rank": result.rank,
                        "pmid": result.pmid,
                        "title": result.title,
                        "matched_terms": result.matched_terms,
                    }
                    for result in results
                ],
                "answer": answer,
            }
        )
    payload = {"method": method, "queries": outputs}
    write_json(output_path, payload)
    return payload


def call_groq(api_key: str, query: str, context: str) -> str:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the provided medical context. If the context is insufficient, say so. "
                            "Cite every substantive claim with PMID or article title."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext:\n{context}",
                    },
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()
