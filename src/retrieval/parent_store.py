"""
src/retrieval/parent_store.py
Small JSON-backed store for hierarchical *parent* chunks.

Only child chunks are embedded/indexed (precise hits, no duplicate text in
retrieval). After re-ranking, a child is swapped for its parent so the LLM
sees the full surrounding context.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_core.documents import Document


class ParentStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, parents: Iterable[Document]) -> int:
        n = 0
        with self._lock:
            for p in parents:
                self._data[p.metadata["chunk_id"]] = {
                    "page_content": p.page_content, "metadata": p.metadata}
                n += 1
            if n:
                self._save()
        return n

    def delete_by_origin(self, origins: List[str]) -> None:
        if not origins:
            return
        wanted = set(origins)
        with self._lock:
            before = len(self._data)
            self._data = {k: v for k, v in self._data.items()
                          if v["metadata"].get("origin") not in wanted}
            if len(self._data) != before:
                self._save()

    def get(self, parent_id: str) -> Optional[Document]:
        d = self._data.get(parent_id)
        return Document(page_content=d["page_content"], metadata=dict(d["metadata"])) if d else None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
