"""
src/ingestion/chunking.py
Four chunking strategies: recursive, semantic, token, hierarchical.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from src.config import get_settings


class ChunkStrategy(str, Enum):
    RECURSIVE    = "recursive"
    SEMANTIC     = "semantic"
    TOKEN        = "token"
    HIERARCHICAL = "hierarchical"


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_breakpoint_percentile: float = 95.0
    parent_chunk_size: int = 2048
    child_chunk_size: int = 256


class DocumentChunker:
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        self.settings = get_settings()

    def split(self, documents: List[Document]) -> List[Document]:
        fn = {
            ChunkStrategy.RECURSIVE:    self._recursive,
            ChunkStrategy.SEMANTIC:     self._semantic,
            ChunkStrategy.TOKEN:        self._token,
            ChunkStrategy.HIERARCHICAL: self._hierarchical,
        }[self.config.strategy]
        return fn(documents)

    def _recursive(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        return self._tag(splitter.split_documents(docs), "recursive")

    def _semantic(self, docs):
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model=self.settings.openai_embedding_model,
                                   openai_api_key=self.settings.openai_api_key)
            splitter = SemanticChunker(emb, breakpoint_threshold_type="percentile",
                                       breakpoint_threshold_amount=self.config.semantic_breakpoint_percentile)
            return self._tag(splitter.split_documents(docs), "semantic")
        except ImportError:
            return self._recursive(docs)

    def _token(self, docs):
        splitter = TokenTextSplitter(chunk_size=self.config.chunk_size,
                                     chunk_overlap=self.config.chunk_overlap,
                                     encoding_name="cl100k_base")
        return self._tag(splitter.split_documents(docs), "token")

    def _hierarchical(self, docs):
        p_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.parent_chunk_size,
                                                    chunk_overlap=self.config.chunk_overlap)
        c_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.child_chunk_size,
                                                    chunk_overlap=32)
        all_chunks = []
        for doc in docs:
            parents = p_splitter.split_documents([doc])
            for pi, parent in enumerate(parents):
                pid = f"{doc.metadata.get('source','doc')}::p{pi}"
                parent.metadata.update({"chunk_level": "parent", "chunk_id": pid,
                                        "chunk_strategy": "hierarchical"})
                all_chunks.append(parent)
                for ci, child in enumerate(c_splitter.split_documents([parent])):
                    child.metadata.update({"chunk_level": "child",
                                           "chunk_id": f"{pid}::c{ci}",
                                           "parent_id": pid,
                                           "chunk_strategy": "hierarchical"})
                    all_chunks.append(child)
        return all_chunks

    @staticmethod
    def _tag(chunks, strategy):
        for i, c in enumerate(chunks):
            c.metadata.update({"chunk_strategy": strategy,
                                "chunk_index": i,
                                "chunk_length": len(c.page_content)})
        return chunks
