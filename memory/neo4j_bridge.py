"""Neo4j bridge — syncs the existing SQLite+NetworkX memory graph to Neo4j.

Uses the circuit breaker pattern from memory/momento_adapter.py for
graceful degradation when Neo4j is unavailable.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class _CircuitBreaker:
    """Simple circuit breaker to avoid hammering a failing Neo4j instance.

    Same pattern as memory/momento_adapter.py.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._last_failure_time = 0.0
        self._state = "closed"  # closed, open, half_open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self._reset_timeout:
                self._state = "half_open"
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._state = "open"


class Neo4jBridge:
    """Bridge between AURA's memory system and Neo4j graph database.

    Provides:
    - Sync from SQLite+NetworkX brain to Neo4j
    - Cypher query interface
    - Codebase graph import
    - Graceful fallback when Neo4j is unavailable
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None
        self._breaker = _CircuitBreaker()

    def _get_driver(self):
        """Lazy-load the Neo4j driver."""
        if self._driver is not None:
            return self._driver

        try:
            import neo4j
            self._driver = neo4j.GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._breaker.record_success()
            return self._driver
        except ImportError:
            raise RuntimeError(
                "neo4j package not installed. Install with: pip install neo4j"
            )
        except Exception as exc:
            self._breaker.record_failure()
            raise RuntimeError(f"Failed to connect to Neo4j at {self._uri}: {exc}")

    @property
    def available(self) -> bool:
        """Check if Neo4j is reachable."""
        if self._breaker.is_open:
            return False
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            self._breaker.record_success()
            return True
        except Exception:
            self._breaker.record_failure()
            return False

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def query_knowledge(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a read-only Cypher query.

        Args:
            cypher: Cypher query string.
            params: Optional query parameters.

        Returns:
            List of result records as dicts.
        """
        if self._breaker.is_open:
            return []

        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(cypher, params or {})
                records = [dict(record) for record in result]
                self._breaker.record_success()
                return records
        except Exception as exc:
            self._breaker.record_failure()
            return []

    def execute_write(self, cypher: str, params: Optional[Dict] = None) -> Dict:
        """Execute a write Cypher query.

        Args:
            cypher: Cypher query string.
            params: Optional query parameters.

        Returns:
            Summary dict with counters.
        """
        if self._breaker.is_open:
            return {"error": "circuit_breaker_open"}

        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(cypher, params or {})
                summary = result.consume()
                self._breaker.record_success()
                return {
                    "nodes_created": summary.counters.nodes_created,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set,
                }
        except Exception as exc:
            self._breaker.record_failure()
            return {"error": str(exc)}

    def import_codebase_graph(
        self, nodes: List[Dict], relationships: List[Dict]
    ) -> Dict:
        """Import a codebase structure into the Neo4j graph.

        Args:
            nodes: List of dicts with keys: label, properties.
            relationships: List of dicts with keys: from_label, from_key,
                from_value, to_label, to_key, to_value, rel_type, properties.

        Returns:
            Summary with counts.
        """
        if self._breaker.is_open:
            return {"error": "circuit_breaker_open"}

        nodes_created = 0
        rels_created = 0

        try:
            driver = self._get_driver()
            with driver.session() as session:
                # Import nodes
                for node in nodes:
                    label = node.get("label", "Node")
                    props = node.get("properties", {})
                    cypher = f"MERGE (n:{label} {{{_props_clause(props)}}})"
                    session.run(cypher, props)
                    nodes_created += 1

                # Import relationships
                for rel in relationships:
                    cypher = (
                        f"MATCH (a:{rel['from_label']} {{{rel['from_key']}: $from_val}}) "
                        f"MATCH (b:{rel['to_label']} {{{rel['to_key']}: $to_val}}) "
                        f"MERGE (a)-[r:{rel['rel_type']}]->(b)"
                    )
                    params = {
                        "from_val": rel["from_value"],
                        "to_val": rel["to_value"],
                    }
                    if rel.get("properties"):
                        props_set = ", ".join(
                            f"r.{k} = ${k}" for k in rel["properties"]
                        )
                        cypher += f" SET {props_set}"
                        params.update(rel["properties"])
                    session.run(cypher, params)
                    rels_created += 1

            self._breaker.record_success()
            return {
                "nodes_created": nodes_created,
                "relationships_created": rels_created,
            }
        except Exception as exc:
            self._breaker.record_failure()
            return {"error": str(exc)}

    def sync_from_brain(self, brain) -> Dict:
        """Sync memories from the existing Brain (SQLite+NetworkX) to Neo4j.

        Args:
            brain: A memory.brain.Brain instance.

        Returns:
            Summary with sync counts.
        """
        if self._breaker.is_open:
            return {"error": "circuit_breaker_open"}

        try:
            memories = brain.recall_with_budget(max_tokens=10000)
            synced = 0
            for mem in memories:
                content = str(mem)
                content_hash = str(hash(content))
                self.execute_write(
                    "MERGE (m:Memory {id: $id}) SET m.content_hash = $hash, m.tier = 'brain', m.timestamp = $ts",
                    {"id": content_hash, "hash": content_hash, "ts": time.time()},
                )
                synced += 1

            self._breaker.record_success()
            return {"synced_memories": synced}
        except Exception as exc:
            self._breaker.record_failure()
            return {"error": str(exc)}

    def schema_info(self) -> Dict:
        """Return current graph schema information."""
        labels = self.query_knowledge("CALL db.labels() YIELD label RETURN label")
        rel_types = self.query_knowledge(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        constraints = self.query_knowledge("SHOW CONSTRAINTS")

        return {
            "labels": [r.get("label") for r in labels],
            "relationship_types": [r.get("relationshipType") for r in rel_types],
            "constraints": constraints,
        }


def _props_clause(props: Dict) -> str:
    """Build a Cypher properties clause from a dict."""
    return ", ".join(f"{k}: ${k}" for k in props)
