"""
src/retrieval/reranker.py
CrossEncoder (local) and Cohere (API) re-rankers with auto-fallback.
"""
from __future__ import annotations
from typing import List, Literal
from langchain_core.documents import Document
from src.config import get_settings


def get_reranker(backend: str | None = None, top_n: int = 5):
    """Factory — returns the right reranker based on config."""
    settings = get_settings()
    b = backend or settings.reranker_backend
    if b == "cohere" and settings.cohere_api_key:
        return CohereReranker(top_n=top_n)
    elif b == "none":
        return PassthroughReranker(top_n=top_n)
    return CrossEncoderReranker(top_n=top_n)


class CrossEncoderReranker:
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        self._model = None

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs: return docs
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL)
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        for score, doc in ranked[:self.top_n]:
            doc.metadata["rerank_score"] = round(float(score), 4)
            doc.metadata["reranker"] = "cross_encoder"
            result.append(doc)
        return result


class CohereReranker:
    def __init__(self, top_n: int = 5):
        self.settings = get_settings()
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            import cohere
            co = cohere.Client(self.settings.cohere_api_key)
            texts = [d.page_content for d in docs]
            res = co.rerank(query=query, documents=texts,
                            model="rerank-english-v3.0", top_n=self.top_n)
            reranked = [docs[r.index] for r in res.results]
            for i, doc in enumerate(reranked):
                doc.metadata["rerank_score"] = res.results[i].relevance_score
                doc.metadata["reranker"] = "cohere"
            return reranked
        except Exception as e:
            print(f"[cohere reranker] failed: {e}, falling back to CrossEncoder")
            return CrossEncoderReranker(top_n=self.top_n).rerank(query, docs)


class PassthroughReranker:
    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        return docs[:self.top_n]
