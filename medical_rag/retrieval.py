from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from medical_rag.utils import expand_query, tokenize


@dataclass
class SearchResult:
    pmid: str
    title: str
    score: float
    rank: int
    matched_terms: list[str]


class Corpus:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.articles = payload["articles"]
        self.pmid_to_article = {article["pmid"]: article for article in self.articles}


class BM25Retriever:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.tokens = [tokenize(article["text"]) for article in corpus.articles]
        self.model = BM25Okapi(self.tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        prepared_query = expand_query(query)
        scores = self.model.get_scores(tokenize(prepared_query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                pmid=self.corpus.articles[index]["pmid"],
                title=self.corpus.articles[index]["title"],
                score=float(scores[index]),
                rank=rank,
                matched_terms=self.corpus.articles[index]["matched_terms"],
            )
            for rank, index in enumerate(order, start=1)
        ]


class SemanticRetriever:
    def __init__(self, corpus: Corpus, model_name: str) -> None:
        self.corpus = corpus
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self._normalize(self.model.encode([article["text"] for article in corpus.articles], convert_to_numpy=True))

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        prepared_query = expand_query(query)
        query_embedding = self._normalize(self.model.encode([prepared_query], convert_to_numpy=True))[0]
        scores = self.doc_embeddings @ query_embedding
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                pmid=self.corpus.articles[index]["pmid"],
                title=self.corpus.articles[index]["title"],
                score=float(scores[index]),
                rank=rank,
                matched_terms=self.corpus.articles[index]["matched_terms"],
            )
            for rank, index in enumerate(order, start=1)
        ]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, semantic: SemanticRetriever, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.corpus = bm25.corpus

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=len(self.corpus.articles))
        semantic_results = self.semantic.search(query, top_k=len(self.corpus.articles))
        scores: dict[str, float] = {}
        for results in [bm25_results, semantic_results]:
            for result in results:
                scores[result.pmid] = scores.get(result.pmid, 0.0) + (1.0 / (self.rrf_k + result.rank))
        ranked_pmids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        output = []
        for rank, pmid in enumerate(ranked_pmids, start=1):
            article = self.corpus.pmid_to_article[pmid]
            output.append(
                SearchResult(
                    pmid=pmid,
                    title=article["title"],
                    score=float(scores[pmid]),
                    rank=rank,
                    matched_terms=article["matched_terms"],
                )
            )
        return output
