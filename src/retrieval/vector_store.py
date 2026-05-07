"""
src/retrieval/vector_store.py
ChromaDB + Pinecone abstraction with similarity, MMR, and filtered search.
"""
from __future__ import annotations
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from src.config import get_settings


class VectorStoreManager:
    def __init__(self, collection_name: str = "unified_rag"):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        self._store = self._init()

    def _init(self):
        if self.settings.vector_db == "chroma":
            return Chroma(collection_name=self.collection_name,
                          embedding_function=self.embeddings,
                          persist_directory=self.settings.chroma_persist_dir)
        elif self.settings.vector_db == "pinecone":
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore
            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            if self.settings.pinecone_index_name not in [i.name for i in pc.list_indexes()]:
                pc.create_index(name=self.settings.pinecone_index_name, dimension=1536,
                                metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            return PineconeVectorStore(index_name=self.settings.pinecone_index_name,
                                       embedding=self.embeddings)
        raise ValueError(f"Unknown vector_db: {self.settings.vector_db}")

    def add_documents(self, docs: List[Document], batch_size: int = 100) -> None:
        for i in range(0, len(docs), batch_size):
            self._store.add_documents(docs[i:i+batch_size])
        print(f"[vector_store] upserted {len(docs)} chunks")

    def similarity_search(self, query: str, k: int = 5,
                          filter: Optional[dict] = None) -> List[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        return self._store.similarity_search_with_score(query, k=k)

    def mmr_search(self, query: str, k: int = 5, fetch_k: int = 20) -> List[Document]:
        return self._store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    def as_retriever(self, k: int = 5):
        return self._store.as_retriever(search_kwargs={"k": k})
