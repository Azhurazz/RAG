"""
src/knowledge_graph/entity_extractor.py
src/knowledge_graph/graph_builder.py
GPT-4o extracts entities + triples from chunks → upserts into Neo4j.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config import get_settings


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str

@dataclass
class Triple:
    subject: str
    relation: str
    obj: str


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract entities and relationships from text.
Return ONLY valid JSON (no markdown):
{"entities":[{"name":"...","type":"...","description":"..."}],
 "triples":[{"subject":"...","relation":"...","object":"..."}]}
Entity types: Person,Organization,Concept,Technology,Method,Dataset,Framework,Tool
Max 15 entities, 20 triples. Skip generic words. Relations: 2-4 word verb phrases."""),
    ("human", "Text:\n{text}"),
])


class EntityExtractor:
    def __init__(self):
        s = get_settings()
        self._chain = EXTRACT_PROMPT | ChatOpenAI(
            model=s.openai_chat_model, openai_api_key=s.openai_api_key,
            temperature=0, response_format={"type": "json_object"}
        ) | StrOutputParser()

    def extract(self, text: str) -> Tuple[List[Entity], List[Triple]]:
        try:
            raw = self._chain.invoke({"text": text[:2500]})
            data = json.loads(raw)
            entities = [Entity(e["name"].strip(), e.get("type","Concept"), e.get("description",""))
                        for e in data.get("entities", []) if e.get("name")]
            triples = [Triple(t["subject"].strip(), t["relation"].strip(), t["object"].strip())
                       for t in data.get("triples", [])
                       if t.get("subject") and t.get("relation") and t.get("object")]
            return entities, triples
        except Exception as e:
            print(f"[extractor] error: {e}")
            return [], []

    def extract_names(self, text: str) -> List[str]:
        entities, _ = self.extract(text)
        return [e.name for e in entities]


# ── Graph Builder ─────────────────────────────────────────────────────────────

from langchain_core.documents import Document
from rich.progress import track
from src.knowledge_graph.neo4j_client import Neo4jClient


class GraphBuilder:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.neo4j = Neo4jClient()

    def build(self, chunks: List[Document]) -> dict:
        total_entities = total_triples = errors = 0
        for chunk in track(chunks, description="[KG] Building graph..."):
            chunk_id = str(chunk.metadata.get("chunk_id", chunk.metadata.get("chunk_index", "?")))
            source   = chunk.metadata.get("source", "unknown")
            self.neo4j.upsert_chunk(chunk_id, chunk.page_content, source)
            try:
                entities, triples = self.extractor.extract(chunk.page_content)
                for e in entities:
                    self.neo4j.upsert_entity(e.name, e.entity_type, e.description, source)
                    self.neo4j.link_entity_to_chunk(e.name, chunk_id)
                    total_entities += 1
                for t in triples:
                    self.neo4j.upsert_entity(t.subject, "Concept", source=source)
                    self.neo4j.upsert_entity(t.obj,     "Concept", source=source)
                    self.neo4j.upsert_relationship(t.subject, t.obj, t.relation, source_doc=source)
                    total_triples += 1
            except Exception as e:
                print(f"[graph_builder] chunk {chunk_id}: {e}")
                errors += 1
        stats = self.neo4j.get_stats()
        stats.update({"chunks_processed": len(chunks), "triples_extracted": total_triples, "errors": errors})
        return stats

    def close(self): self.neo4j.close()
