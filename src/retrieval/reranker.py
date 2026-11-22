"""
src/retrieval/reranker.py
CrossEncoder (local) and Cohere (API) re-rankers with auto-fallback.
"""
from __future__ import annotations
from typing import List
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
    _shared_models: dict = {}  # model name -> loaded CrossEncoder (load once per process)

    def __init__(self, top_n: int = 5, model_name: str | None = None):
        self.top_n = top_n
        self.model_name = model_name or get_settings().cross_encoder_model

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs: return docs
        model = self._shared_models.get(self.model_name)
        if model is None:
            from sentence_transformers import CrossEncoder
            model = self._shared_models[self.model_name] = CrossEncoder(self.model_name)
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = [Document(page_content=d.page_content,
                   metadata={**d.metadata, "rerank_score": round(float(s), 4),
                             "reranker": "cross_encoder"})
                  for s, d in ranked[:self.top_n]]
        return result


class CohereReranker:
    def __init__(self, top_n: int = 5):
        self.settings = get_settings()
        self.top_n = top_n
        self._client = None
        self._fallback = None

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        try:
            import cohere
            if self._client is None:
                self._client = cohere.Client(self.settings.cohere_api_key)
            co = self._client
            texts = [d.page_content for d in docs]
            res = co.rerank(query=query, documents=texts,
                            model=self.settings.cohere_rerank_model, top_n=self.top_n)
            result = [Document(page_content=docs[r.index].page_content,
                               metadata={**docs[r.index].metadata,
                                         "rerank_score": round(float(r.relevance_score), 4),
                                         "reranker": "cohere"})
                      for r in res.results]
            return result
        except Exception as e:
            print(f"[cohere reranker] failed: {e}, falling back to CrossEncoder")
            if self._fallback is None:
                self._fallback = CrossEncoderReranker(top_n=self.top_n)
            return self._fallback.rerank(query, docs)


class PassthroughReranker:
    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        return docs[:self.top_n]
