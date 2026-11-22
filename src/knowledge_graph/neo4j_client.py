"""
src/knowledge_graph/neo4j_client.py
Neo4j driver wrapper — all Cypher queries in one place.
Schema: (:Entity)-[:RELATED_TO]->(:Entity), (:Entity)-[:MENTIONED_IN]->(:Chunk)
"""
from __future__ import annotations
from typing import Dict, List, Optional
from neo4j import GraphDatabase
from src.config import get_settings


def _key(name: str) -> str:
    return " ".join(name.split()).lower()


class Neo4jClient:
    _schema_ready = False  # run the constraint/index DDL once per process, not per client

    def __init__(self):
        s = get_settings()
        self._driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        if not Neo4jClient._schema_ready:
            self._init_constraints()
            Neo4jClient._schema_ready = True

    def _init_constraints(self):
        with self._driver.session() as s:
            s.run("DROP CONSTRAINT entity_name IF EXISTS")
            s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
            s.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            s.run("CREATE INDEX chunk_id    IF NOT EXISTS FOR (c:Chunk)  ON (c.chunk_id)")

    def close(self): self._driver.close()

    def upsert_entity(self, name: str, entity_type: str,
                      description: str = "", source: str = "") -> None:
        with self._driver.session() as s:
            s.run("""
                MERGE (e:Entity {key:$key})
                ON CREATE SET e.name=$name, e.type=$type, e.description=$desc, e.source=$src
                ON MATCH SET  e.type = CASE WHEN $type <> 'Concept' THEN $type ELSE e.type END,
                              e.description = CASE WHEN $desc <> '' THEN $desc ELSE e.description END
            """, key=_key(name), name=name, type=entity_type, desc=description, src=source)

    def upsert_relationship(self, src: str, tgt: str, relation: str,
                            weight: float = 1.0, source_doc: str = "") -> None:
        with self._driver.session() as s:
            s.run("""
                MATCH (a:Entity {key:$src}) MATCH (b:Entity {key:$tgt})
                MERGE (a)-[r:RELATED_TO {relation:$rel}]->(b)
                SET r.weight=$w, r.source=$src_doc
            """, src=_key(src), tgt=_key(tgt), rel=relation, w=weight, src_doc=source_doc)
            
    def upsert_chunk(self, chunk_id: str, text: str, source: str) -> None:
        with self._driver.session() as s:
            s.run("MERGE (c:Chunk {chunk_id:$id}) SET c.text=$text, c.source=$src",
                  id=chunk_id, text=text, src=source)

    def link_entity_to_chunk(self, entity_name: str, chunk_id: str) -> None:
        with self._driver.session() as s:
            s.run("""
                MATCH (e:Entity {key:$key}) MATCH (c:Chunk {chunk_id:$id})
                MERGE (e)-[:MENTIONED_IN]->(c)
            """, key=_key(entity_name), id=chunk_id)

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Remove Chunk nodes (and their MENTIONED_IN links), e.g. when a document is re-ingested."""
        if not chunk_ids:
            return
        with self._driver.session() as s:
            s.run("MATCH (c:Chunk) WHERE c.chunk_id IN $ids DETACH DELETE c", ids=list(chunk_ids))

    def get_neighbors(self, name: str, depth: int = 2,
                      relation_filter: Optional[str] = None, limit: int = 20) -> List[Dict]:
        depth = max(1, int(depth))
        # r is a *list* of relationships in a variable-length match
        filter_clause = "AND ALL(x IN r WHERE x.relation = $rel)" if relation_filter else ""
        with self._driver.session() as s:
            result = s.run(f"""
                MATCH path=(start:Entity {{key:$key}})-[r:RELATED_TO*1..{depth}]->(end:Entity)
                WHERE start<>end {filter_clause}
                RETURN DISTINCT end.name AS entity, end.type AS type,
                       end.description AS description, length(path) AS hops,
                       [rel IN relationships(path) | rel.relation] AS relations
                ORDER BY hops ASC LIMIT $limit
            """, key=_key(name), rel=relation_filter, limit=limit)
            return [dict(r) for r in result]

    def get_entity_chunks(self, name: str, depth: int = 2) -> List[Dict]:
        depth = max(0, int(depth))
        with self._driver.session() as s:
            # Closest chunks first: the caller keeps only the first few rows,
            # so chunks mentioning the entity itself must not lose to 2-hop neighbours.
            result = s.run(f"""
                MATCH path=(start:Entity {{key:$key}})-[:RELATED_TO*0..{depth}]->(e:Entity)
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
                WITH c, e, length(path) AS hops
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.source AS source,
                       collect(DISTINCT e.name) AS entities, min(hops) AS hops
                ORDER BY hops ASC
                LIMIT 20
            """, key=_key(name))
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
