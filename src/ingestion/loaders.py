"""
src/ingestion/loaders.py
Multi-source loader: PDF · Web · Text · REST API → List[Document]
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import httpx
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from src.ingestion.chunking import ChunkConfig, ChunkStrategy, DocumentChunker


class MultiSourceLoader:
    def __init__(self, chunk_config: ChunkConfig | None = None):
        self.chunker = DocumentChunker(chunk_config or ChunkConfig())

    def load_and_chunk(
        self,
        pdfs: List[str] | None = None,
        urls: List[str] | None = None,
        texts: List[str] | None = None,
        api_endpoints: List[dict] | None = None,
    ) -> List[Document]:
        raw: List[Document] = []
        if pdfs:           raw.extend(self._pdfs(pdfs))
        if urls:           raw.extend(self._urls(urls))
        if texts:          raw.extend(self._texts(texts))
        if api_endpoints:  raw.extend(self._apis(api_endpoints))
        chunks = self.chunker.split(raw)
        print(f"[loader] {len(raw)} raw docs → {len(chunks)} chunks")
        return chunks

    def _pdfs(self, paths):
        docs = []
        for p in paths:
            pages = PyPDFLoader(p).load()
            for page in pages:
                page.metadata.update({"source_type": "pdf", "source": Path(p).name})
            docs.extend(pages)
        return docs

    def _urls(self, urls):
        docs = []
        for url in urls:
            try:
                pages = WebBaseLoader(url).load()
                for page in pages:
                    page.metadata.update({"source_type": "web",
                                          "source": urlparse(url).netloc, "url": url})
                docs.extend(pages)
            except Exception as e:
                print(f"[loader] url failed {url}: {e}")
        return docs

    def _texts(self, paths):
        docs = []
        for p in paths:
            pages = TextLoader(p, encoding="utf-8").load()
            for page in pages:
                page.metadata.update({"source_type": "text", "source": Path(p).name})
            docs.extend(pages)
        return docs

    def _apis(self, endpoints):
        docs = []
        for ep in endpoints:
            try:
                resp = httpx.get(ep["url"], headers=ep.get("headers", {}),
                                 params=ep.get("params", {}), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                field = ep.get("field", "")
                content = self._dig(data, field)
                items = content if isinstance(content, list) else [content]
                for item in items:
                    text = json.dumps(item) if not isinstance(item, str) else item
                    docs.append(Document(page_content=text,
                                         metadata={"source_type": "api", "source": ep["url"]}))
            except Exception as e:
                print(f"[loader] api failed {ep['url']}: {e}")
        return docs

    @staticmethod
    def _dig(data, field):
        if not field: return data
        for part in field.split("."):
            if isinstance(data, dict): data = data.get(part, {})
        return data
