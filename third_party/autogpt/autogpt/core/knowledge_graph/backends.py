from __future__ import annotations

"""Pluggable graph database backends used by :mod:`graph_store`."""

import logging
import os
import re
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class GraphBackend:
    """Abstract interface for persistent graph backends."""

    def add_node(self, node_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        raise NotImplementedError

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    def remove_node(self, node_id: str) -> None:
        raise NotImplementedError

    def remove_edge(self, source: str, target: str, relation_type: Optional[str]) -> None:
        raise NotImplementedError

    def query(
        self,
        *,
        node_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional for many backends
        """Close any outstanding connections."""

    def reconnect(self) -> None:  # pragma: no cover - optional
        """Attempt to re-establish connections after failures."""


class NoOpGraphBackend(GraphBackend):
    """Fallback backend that performs no operations."""

    def add_node(self, node_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        logger.debug("NoOpGraphBackend.add_node(%s, %s) noop", node_id, entity_type)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any],
    ) -> None:
        logger.debug(
            "NoOpGraphBackend.add_edge(%s, %s, %s) noop", source, target, relation_type
        )

    def remove_node(self, node_id: str) -> None:
        logger.debug("NoOpGraphBackend.remove_node(%s) noop", node_id)

    def remove_edge(self, source: str, target: str, relation_type: Optional[str]) -> None:
        logger.debug(
            "NoOpGraphBackend.remove_edge(%s, %s, %s) noop", source, target, relation_type
        )

    def query(
        self,
        *,
        node_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {"nodes": [], "edges": []}


class Neo4jGraphBackend(GraphBackend):
    """Neo4j-backed implementation."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: Optional[str] = None,
        **driver_kwargs: Any,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Neo4j driver not installed. Install neo4j-driver to enable this backend."
            ) from exc

        self._uri = uri
        self._auth = (user, password)
        self._driver_kwargs = dict(driver_kwargs)
        self._driver = GraphDatabase.driver(uri, auth=(user, password), **driver_kwargs)
        self._database = database

    # -- Internal helpers -------------------------------------------------
    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    @staticmethod
    def _label(name: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name or "Entity")
        return safe or "Entity"

    @staticmethod
    def _relationship(name: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name or "RELATED_TO")
        return safe or "RELATED_TO"

    # -- GraphBackend implementation --------------------------------------
    def add_node(self, node_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        label = self._label(entity_type)
        props = dict(properties or {})
        props["id"] = node_id
        props["type"] = entity_type
        query = f"""
        MERGE (n:{label} {{id:$id}})
        SET n += $props
        """
        with self._session() as session:
            session.execute_write(lambda tx: tx.run(query, id=node_id, props=props))

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any],
    ) -> None:
        rel = self._relationship(relation_type)
        props = dict(properties or {})
        query = f"""
        MERGE (src {{id:$source}})
        MERGE (dst {{id:$target}})
        MERGE (src)-[r:{rel}]->(dst)
        SET r += $props, r.type = $type
        """
        with self._session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    query,
                    source=source,
                    target=target,
                    props=props,
                    type=relation_type,
                )
            )

    def remove_node(self, node_id: str) -> None:
        query = "MATCH (n {id:$id}) DETACH DELETE n"
        with self._session() as session:
            session.execute_write(lambda tx: tx.run(query, id=node_id))

    def remove_edge(self, source: str, target: str, relation_type: Optional[str]) -> None:
        rel_filter = ""
        params: Dict[str, Any] = {"source": source, "target": target}
        if relation_type:
            rel = self._relationship(relation_type)
            rel_filter = f":{rel}"
        query = f"""
        MATCH (src {{id:$source}})-[r{rel_filter}]->(dst {{id:$target}})
        DELETE r
        """
        with self._session() as session:
            session.execute_write(lambda tx: tx.run(query, **params))

    def query(
        self,
        *,
        node_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        node_conditions = []
        params: Dict[str, Any] = {}
        if node_id:
            node_conditions.append("n.id = $node_id")
            params["node_id"] = node_id
        if entity_type:
            node_conditions.append("n.type = $entity_type")
            params["entity_type"] = entity_type
        node_where = ""
        if node_conditions:
            node_where = "WHERE " + " AND ".join(node_conditions)
        node_limit = f"LIMIT {int(limit)}" if limit else ""
        node_query = f"MATCH (n) {node_where} RETURN n {node_limit}"

        edge_conditions = []
        if node_id:
            edge_conditions.append("(src.id = $node_id OR dst.id = $node_id)")
        if relation_type:
            edge_conditions.append("type(r) = $relation_type")
            params["relation_type"] = relation_type
        edge_where = ""
        if edge_conditions:
            edge_where = "WHERE " + " AND ".join(edge_conditions)
        edge_limit = f"LIMIT {int(limit)}" if limit else ""
        edge_query = f"""
        MATCH (src)-[r]->(dst)
        {edge_where}
        RETURN src.id AS source, dst.id AS target, type(r) AS type, r AS rel {edge_limit}
        """

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        with self._session() as session:
            for record in session.execute_read(lambda tx: tx.run(node_query, **params)):
                node = record["n"]
                nodes.append(
                    {
                        "id": node.get("id"),
                        "type": node.get("type"),
                        "properties": {k: v for k, v in node.items() if k not in {"id", "type"}},
                    }
                )
            for record in session.execute_read(lambda tx: tx.run(edge_query, **params)):
                rel_props = dict(record["rel"])
                rel_props.pop("type", None)
                edges.append(
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": rel_props,
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self._driver.close()

    def reconnect(self) -> None:  # pragma: no cover - best effort
        try:
            self._driver.verify_connectivity()
            return
        except Exception:
            logger.debug("Neo4j connectivity verification failed; attempting reconnect.")
        try:
            self._driver.close()
        except Exception:
            logger.debug("Neo4j driver close during reconnect failed.", exc_info=True)
        try:
            from neo4j import GraphDatabase  # type: ignore

            self._driver = GraphDatabase.driver(
                self._uri, auth=self._auth, **self._driver_kwargs
            )
            self._driver.verify_connectivity()
        except Exception:
            logger.warning("Neo4j reconnect attempt failed.", exc_info=True)


class TigerGraphBackend(GraphBackend):
    """TigerGraph backed implementation using pyTigerGraph REST APIs."""

    def __init__(
        self,
        *,
        host: str,
        graph: str,
        username: str,
        password: Optional[str] = None,
        secret: Optional[str] = None,
        api_token: Optional[str] = None,
        restpp_port: int = 9000,
        gs_port: int = 14240,
        use_tls: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            from pyTigerGraph import TigerGraphConnection  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyTigerGraph package not installed. Install pyTigerGraph to enable this backend."
            ) from exc

        protocol = "https" if use_tls else "http"
        self._conn = TigerGraphConnection(
            host=host,
            graphname=graph,
            username=username,
            password=password,
            restppPort=restpp_port,
            gsPort=gs_port,
            useCert=use_tls,
            protocol=protocol,
            **kwargs,
        )
        self._node_types: Dict[str, str] = {}
        self._token: Optional[str] = api_token
        if not self._token:
            if not secret:
                raise RuntimeError(
                    "TIGERGRAPH_SECRET or TIGERGRAPH_TOKEN must be provided to authenticate."
                )
            try:
                token_info = self._conn.getToken(secret)
            except Exception as exc:
                raise RuntimeError("Failed to retrieve TigerGraph API token.") from exc
            if isinstance(token_info, (list, tuple)) and token_info:
                self._token = token_info[0]
            elif isinstance(token_info, dict) and "token" in token_info:
                self._token = token_info["token"]
            else:
                raise RuntimeError("Unexpected response retrieving TigerGraph token.")
        self._conn._defaultToken = self._token

    @staticmethod
    def _sanitize_identifier(value: str | None, fallback: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in (value or ""))
        safe = safe.strip("_") or fallback
        return safe

    def add_node(self, node_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        vertex_type = self._sanitize_identifier(entity_type, "Entity")
        payload = dict(properties or {})
        payload["type"] = entity_type
        try:
            self._conn.upsertVertex(vertex_type, node_id, payload)
            self._node_types[node_id] = vertex_type
        except Exception:
            logger.warning("TigerGraph add_node failed for %s", node_id, exc_info=True)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any],
    ) -> None:
        edge_type = self._sanitize_identifier(relation_type, "RELATED_TO")
        props = dict(properties or {})
        source_type = props.pop("_source_type", None) or self._node_types.get(source) or "Entity"
        target_type = props.pop("_target_type", None) or self._node_types.get(target) or "Entity"
        try:
            self._conn.upsertEdge(
                source_type,
                source,
                edge_type,
                target_type,
                target,
                props,
            )
        except Exception:
            logger.warning(
                "TigerGraph add_edge failed for %s->%s (%s)",
                source,
                target,
                relation_type,
                exc_info=True,
            )

    def remove_node(self, node_id: str) -> None:
        vertex_type = self._node_types.get(node_id, "Entity")
        try:
            self._conn.delVertices(vertex_type, where=f'id="{node_id}"')
            self._node_types.pop(node_id, None)
        except Exception:
            logger.warning("TigerGraph remove_node failed for %s", node_id, exc_info=True)

    def remove_edge(self, source: str, target: str, relation_type: Optional[str]) -> None:
        edge_type = self._sanitize_identifier(relation_type or "RELATED_TO", "RELATED_TO")
        source_type = self._node_types.get(source)
        target_type = self._node_types.get(target)
        where_clause = f'source_id="{source}" and target_id="{target}"'
        try:
            self._conn.delEdges(
                source_type or "",
                source if source_type else "",
                edge_type,
                target_type or "",
                target if target_type else "",
                where=where_clause,
            )
        except Exception:
            logger.warning("TigerGraph remove_edge failed", exc_info=True)

    def query(
        self,
        *,
        node_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        try:
            if node_id:
                vertex_type = self._node_types.get(node_id)
                candidates = [vertex_type] if vertex_type else self._conn.getVertexTypes()
                for vtype in candidates:
                    data = self._conn.getVerticesById(vtype, [node_id])
                    for entry in data or []:
                        attrs = dict(entry.get("attributes", {}))
                        nodes.append(
                            {
                                "id": entry.get("v_id"),
                                "type": entry.get("v_type"),
                                "properties": attrs,
                            }
                        )
            else:
                vertex_types = (
                    [self._sanitize_identifier(entity_type, "Entity")]
                    if entity_type
                    else self._conn.getVertexTypes()
                )
                for vtype in vertex_types:
                    data = self._conn.getVertices(vtype, limit=limit or 50)
                    for entry in data or []:
                        attrs = dict(entry.get("attributes", {}))
                        nodes.append(
                            {
                                "id": entry.get("v_id"),
                                "type": entry.get("v_type"),
                                "properties": attrs,
                            }
                        )
            if relation_type or node_id:
                edge_types = (
                    [self._sanitize_identifier(relation_type, "RELATED_TO")]
                    if relation_type
                    else self._conn.getEdgeTypes()
                )
                for etype in edge_types:
                    data = self._conn.getEdges(
                        "", "", etype, limit=limit or 50, sort="", where=None
                    )
                    for entry in data or []:
                        edges.append(
                            {
                                "source": entry.get("from_id"),
                                "target": entry.get("to_id"),
                                "type": entry.get("e_type"),
                                "properties": dict(entry.get("attributes", {})),
                            }
                        )
        except Exception:
            logger.debug("TigerGraph query failed.", exc_info=True)
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:  # pragma: no cover - pyTigerGraph has no explicit close
        try:
            self._conn.close()
        except Exception:
            logger.debug("TigerGraph connection close failed.", exc_info=True)

    def reconnect(self) -> None:  # pragma: no cover - best effort
        if not self._token:
            return
        try:
            self._conn.refreshToken(self._token)
        except Exception:
            logger.debug("TigerGraph token refresh failed.", exc_info=True)


class NeptuneGraphBackend(GraphBackend):
    """Amazon Neptune backend using the openCypher HTTP endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        port: int = 8182,
        scheme: str = "https",
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        session_token: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "requests package not installed. Install requests to enable Neptune backend."
            ) from exc

        self._requests = requests
        self._endpoint = endpoint
        self._port = port
        self._scheme = scheme
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._session_token = session_token
        self._session = requests.Session()
        self._timeout = timeout
        self._url = f"{self._scheme}://{self._endpoint}:{self._port}/openCypher"
        self._apply_auth()

    @staticmethod
    def _label(name: str | None) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in (name or "Entity"))
        safe = safe.strip("_") or "Entity"
        return safe

    def _post(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"query": query}
        if parameters:
            payload["parameters"] = parameters
        try:
            response = self._session.post(self._url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except self._requests.RequestException as exc:
            raise RuntimeError(f"Neptune request failed: {exc}") from exc

    def _extract_rows(self, payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        results = []
        for record in payload.get("results", []):
            row = record.get("row")
            if isinstance(row, list):
                results.append(row)
        return results

    def add_node(self, node_id: str, entity_type: str, properties: Dict[str, Any]) -> None:
        label = self._label(entity_type)
        props = dict(properties or {})
        props.pop("id", None)
        statement = f"MERGE (n:{label} {{id:$id}}) SET n.type = $type SET n += $props"
        params = {"id": node_id, "type": entity_type, "props": props}
        self._post(statement, params)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Dict[str, Any],
    ) -> None:
        rel = self._label(relation_type)
        props = dict(properties or {})
        props.pop("type", None)
        statement = (
            f"MATCH (s {{id:$source}}), (t {{id:$target}}) "
            f"MERGE (s)-[r:{rel}]->(t) "
            f"SET r.type = $type "
            f"SET r += $props"
        )
        params = {"source": source, "target": target, "type": relation_type, "props": props}
        self._post(statement, params)

    def remove_node(self, node_id: str) -> None:
        statement = "MATCH (n {id:$id}) DETACH DELETE n"
        self._post(statement, {"id": node_id})

    def remove_edge(self, source: str, target: str, relation_type: Optional[str]) -> None:
        rel_clause = ""
        if relation_type:
            rel_clause = f":{self._label(relation_type)}"
        statement = (
            f"MATCH (s {{id:$source}})-[r{rel_clause}]->(t {{id:$target}}) "
            f"DELETE r"
        )
        self._post(statement, {"source": source, "target": target})

    def query(
        self,
        *,
        node_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_limit = f" LIMIT {int(limit)}" if limit else ""
        node_statement = "MATCH (n)"
        where_clauses = []
        params: Dict[str, Any] = {}
        if node_id:
            where_clauses.append("n.id = $node_id")
            params["node_id"] = node_id
        if entity_type:
            where_clauses.append("n.type = $entity_type")
            params["entity_type"] = entity_type
        if where_clauses:
            node_statement += " WHERE " + " AND ".join(where_clauses)
        node_statement += f" RETURN n{node_limit}"

        try:
            node_payload = self._post(node_statement, params or None)
            for row in self._extract_rows(node_payload):
                if not row:
                    continue
                node = row[0]
                if isinstance(node, dict):
                    nodes.append(
                        {
                            "id": node.get("id"),
                            "type": node.get("type"),
                            "properties": {
                                k: v for k, v in node.items() if k not in {"id", "type"}
                            },
                        }
                    )
        except Exception:
            logger.debug("Neptune node query failed.", exc_info=True)

        edge_statement = "MATCH (s)-[r]->(t)"
        edge_clauses = []
        edge_params: Dict[str, Any] = {}
        if node_id:
            edge_clauses.append("s.id = $edge_node OR t.id = $edge_node")
            edge_params["edge_node"] = node_id
        if relation_type:
            edge_clauses.append(f"type(r) = $edge_rel")
            edge_params["edge_rel"] = relation_type
        if edge_clauses:
            edge_statement += " WHERE " + " AND ".join(edge_clauses)
        edge_statement += " RETURN s.id, t.id, type(r), r"
        if limit:
            edge_statement += f" LIMIT {int(limit)}"

        try:
            edge_payload = self._post(edge_statement, edge_params or None)
            for row in self._extract_rows(edge_payload):
                if len(row) < 4:
                    continue
                props = row[3] if isinstance(row[3], dict) else {}
                edges.append(
                    {
                        "source": row[0],
                        "target": row[1],
                        "type": row[2],
                        "properties": props,
                    }
                )
        except Exception:
            logger.debug("Neptune edge query failed.", exc_info=True)

        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self._session.close()

    def reconnect(self) -> None:  # pragma: no cover - best effort
        try:
            self._session.close()
        except Exception:
            logger.debug("Failed to close Neptune session during reconnect.", exc_info=True)
        self._session = self._requests.Session()
        self._apply_auth()

    def _apply_auth(self) -> None:
        if self._region and self._access_key and self._secret_key:
            try:
                from requests_aws4auth import AWS4Auth  # type: ignore
            except ImportError:
                logger.warning(
                    "requests-aws4auth not available; Neptune requests will be unauthenticated."
                )
                self._session.auth = None
            else:
                self._session.auth = AWS4Auth(
                    self._access_key,
                    self._secret_key,
                    self._region,
                    "neptune-db",
                    session_token=self._session_token,
                )
        else:
            self._session.auth = None


def build_backend_from_env() -> Optional[GraphBackend]:
    """Instantiate a backend based on ``GRAPH_BACKEND`` environment configuration."""

    backend_name = os.getenv("GRAPH_BACKEND")
    if not backend_name:
        return None

    backend_name = backend_name.lower()
    if backend_name == "neo4j":
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE")
        if not password:
            logger.warning("NEO4J_PASSWORD not set; skipping Neo4j backend initialisation.")
            return None
        try:
            backend = Neo4jGraphBackend(uri, user, password, database=database)
            logger.info("Initialised Neo4j graph backend at %s", uri)
            return backend
        except Exception as exc:
            logger.warning("Failed to initialise Neo4j backend: %s", exc, exc_info=True)
            return None
    if backend_name == "tigergraph":
        host = os.getenv("TIGERGRAPH_HOST")
        graph = os.getenv("TIGERGRAPH_GRAPH")
        username = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
        password = os.getenv("TIGERGRAPH_PASSWORD")
        secret = os.getenv("TIGERGRAPH_SECRET")
        token = os.getenv("TIGERGRAPH_TOKEN")
        restpp_port = int(os.getenv("TIGERGRAPH_RESTPP_PORT", "9000") or 9000)
        gs_port = int(os.getenv("TIGERGRAPH_GS_PORT", "14240") or 14240)
        use_tls = os.getenv("TIGERGRAPH_USE_TLS", "false").lower() in {"1", "true", "yes", "on"}
        if not host or not graph:
            logger.warning(
                "TIGERGRAPH_HOST and TIGERGRAPH_GRAPH must be set to use the TigerGraph backend."
            )
            return None
        try:
            backend = TigerGraphBackend(
                host=host,
                graph=graph,
                username=username,
                password=password,
                secret=secret,
                api_token=token,
                restpp_port=restpp_port,
                gs_port=gs_port,
                use_tls=use_tls,
            )
            logger.info("Initialised TigerGraph backend for graph '%s' on %s", graph, host)
            return backend
        except Exception as exc:
            logger.warning("Failed to initialise TigerGraph backend: %s", exc, exc_info=True)
            return None
    if backend_name == "neptune":
        endpoint = os.getenv("NEPTUNE_ENDPOINT")
        if not endpoint:
            logger.warning("NEPTUNE_ENDPOINT must be set to use the Neptune backend.")
            return None
        port = int(os.getenv("NEPTUNE_PORT", "8182") or 8182)
        scheme = os.getenv("NEPTUNE_SCHEME", "https")
        region = os.getenv("NEPTUNE_REGION")
        access_key = os.getenv("NEPTUNE_ACCESS_KEY_ID")
        secret_key = os.getenv("NEPTUNE_SECRET_ACCESS_KEY")
        session_token = os.getenv("NEPTUNE_SESSION_TOKEN")
        timeout = float(os.getenv("NEPTUNE_TIMEOUT", "10") or 10)
        try:
            backend = NeptuneGraphBackend(
                endpoint=endpoint,
                port=port,
                scheme=scheme,
                region=region,
                access_key=access_key,
                secret_key=secret_key,
                session_token=session_token,
                timeout=timeout,
            )
            logger.info("Initialised Neptune graph backend at %s://%s:%s", scheme, endpoint, port)
            return backend
        except Exception as exc:
            logger.warning("Failed to initialise Neptune backend: %s", exc, exc_info=True)
            return None

    logger.warning("Unsupported graph backend '%s'; using in-memory GraphStore only.", backend_name)
    return None


__all__ = [
    "GraphBackend",
    "NoOpGraphBackend",
    "Neo4jGraphBackend",
    "TigerGraphBackend",
    "NeptuneGraphBackend",
    "build_backend_from_env",
]
