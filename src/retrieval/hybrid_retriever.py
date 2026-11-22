"""
src/retrieval/hybrid_retriever.py
Dense (ChromaDB) + Sparse (BM25) fused with Reciprocal Rank Fusion.
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.utils import doc_key, tokenize


class HybridRetriever:
    def __init__(self, vsm: VectorStoreManager, documents: List[Document],
                 rrf_k: int = 60, top_n: int = 10):
        self.vsm = vsm
        self.documents = documents
        self.rrf_k = rrf_k
        self.top_n = top_n
        self._bm25 = None
        self._build_bm25()

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [tokenize(d.page_content) for d in self.documents]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            print("[hybrid] rank_bm25 not installed — dense only")

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        dense = self.vsm.similarity_search(query, k=k)
        if not self._bm25:
            return dense
        scores = self._bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # Drop zero-score hits: they share no terms with the query and would
        # otherwise still earn an RRF bonus.
        sparse = [self.documents[i] for i in top_idx if scores[i] > 0]
        return self._rrf(dense, sparse)

    def _rrf(self, a: List[Document], b: List[Document]) -> List[Document]:
        scores: dict = {}
        doc_map: dict = {}
        for rank, doc in enumerate(a):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0) + 1 / (self.rrf_k + rank + 1)
            doc_map[key] = doc
        for rank, doc in enumerate(b):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0) + 1 / (self.rrf_k + rank + 1)
            doc_map[key] = doc
        return [doc_map[k] for k in sorted(scores, key=lambda k: scores[k], reverse=True)][:self.top_n]
