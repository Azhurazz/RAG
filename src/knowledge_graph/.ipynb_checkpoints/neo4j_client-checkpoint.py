"""
src/knowledge_graph/neo4j_client.py
Neo4j driver wrapper — all Cypher queries in one place.
Schema: (:Entity)-[:RELATED_TO]->(:Entity), (:Entity)-[:MENTIONED_IN]->(:Chunk)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase
from src.config import get_settings


class Neo4jClient:
    def __init__(self):
        s = get_settings()
        self._driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        self._init_constraints()

    def _init_constraints(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            s.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            s.run("CREATE INDEX chunk_id    IF NOT EXISTS FOR (c:Chunk)  ON (c.chunk_id)")

    def close(self): self._driver.close()

    def upsert_entity(self, name: str, entity_type: str,
                      description: str = "", source: str = "") -> None:
        with self._driver.session() as s:
            s.run("MERGE (e:Entity {name:$name}) SET e.type=$type, e.description=$desc, e.source=$src",
                  name=name, type=entity_type, desc=description, src=source)

    def upsert_relationship(self, src: str, tgt: str, relation: str,
                            weight: float = 1.0, source_doc: str = "") -> None:
        with self._driver.session() as s:
            s.run("""
                MATCH (a:Entity {name:$src}) MATCH (b:Entity {name:$tgt})
                MERGE (a)-[r:RELATED_TO {relation:$rel}]->(b)
                SET r.weight=$w, r.source=$src_doc
            """, src=src, tgt=tgt, rel=relation, w=weight, src_doc=source_doc)

    def upsert_chunk(self, chunk_id: str, text: str, source: str) -> None:
        with self._driver.session() as s:
            s.run("MERGE (c:Chunk {chunk_id:$id}) SET c.text=$text, c.source=$src",
                  id=chunk_id, text=text, src=source)

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str) -> None:
        with self._driver.session() as s:
            s.run("""
                MATCH (e:Entity {name:$name}) MATCH (c:Chunk {chunk_id:$id})
                MERGE (e)-[:MENTIONED_IN]->(c)
            """, name=entity_name, id=chunk_id)

    def get_neighbors(self, name: str, depth: int = 2,
                      relation_filter: Optional[str] = None, limit: int = 20) -> List[Dict]:
        filter_clause = "AND r.relation=$rel" if relation_filter else ""
        with self._driver.session() as s:
            result = s.run(f"""
                MATCH path=(start:Entity {{name:$name}})-[r:RELATED_TO*1..{depth}]->(end:Entity)
                WHERE start<>end {filter_clause}
                RETURN DISTINCT end.name AS entity, end.type AS type,
                       end.description AS description, length(path) AS hops,
                       [rel IN relationships(path) | rel.relation] AS relations
                ORDER BY hops ASC LIMIT $limit
            """, name=name, rel=relation_filter, limit=limit)
            return [dict(r) for r in result]

    def get_entity_chunks(self, name: str, depth: int = 2) -> List[Dict]:
        with self._driver.session() as s:
            result = s.run(f"""
                MATCH (start:Entity {{name:$name}})-[:RELATED_TO*0..{depth}]->(e:Entity)
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text,
                       c.source AS source, collect(DISTINCT e.name) AS entities
                LIMIT 20
            """, name=name)
            return [dict(r) for r in result]

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        with self._driver.session() as s:
            result = s.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($q)
                   OR toLower(e.description) CONTAINS toLower($q)
                RETURN e.name AS name, e.type AS type, e.description AS description
                LIMIT $limit
            """, q=query, limit=limit)
            return [dict(r) for r in result]

    def get_subgraph_context(self, entity_names: List[str], depth: int = 2) -> str:
        lines = []
        for name in entity_names:
            neighbors = self.get_neighbors(name, depth=depth, limit=10)
            if neighbors:
                lines.append(f"Entity: {name}")
                for n in neighbors:
                    lines.append(f"  {' → '.join(n['relations'])} → {n['entity']} ({n['type']}): {n['description']}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, int]:
        with self._driver.session() as s:
            return {
                "entities":      s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"],
                "relationships": s.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS n").single()["n"],
                "chunks":        s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"],
            }
