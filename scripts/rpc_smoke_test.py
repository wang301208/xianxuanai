from __future__ import annotations

"""Smoke-test the RPC integration stack (skill gateway, Neo4j, Qdrant)."""

import os
import sys
import time
from typing import Any, Dict

import numpy as np
import requests
import structlog

from common.logging_config import configure_logging
from backend.autogpt.autogpt.core.knowledge_graph.backends import build_backend_from_env
from modules.memory.backends import QdrantANNBackend
from modules.skills.rpc_client import SkillRPCClient

configure_logging()
logger = structlog.get_logger("rpc-smoke-test")


def wait_for_http(url: str, *, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code < 400:
                logger.info("service_ready", url=url, status=response.status_code)
                return
            logger.warning("service_unready", url=url, status=response.status_code)
        except requests.RequestException as exc:
            logger.debug("service_probe_error", url=url, error=str(exc))
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for {url}")


def run_skill_rpc_probe(client: SkillRPCClient) -> None:
    echo = client.invoke(
        "demo.echo",
        {"message": "hello rpc"},
        metadata={"execution_mode": "rpc"},
    )
    assert isinstance(echo, Dict)
    assert echo.get("status") == "ok", f"Unexpected echo response: {echo}"

    search = client.invoke(
        "demo.search",
        {"query": "graph databases"},
        metadata={"execution_mode": "rpc"},
    )
    assert search.get("status") == "ok", f"Search failed: {search}"
    assert search.get("results"), "Search returned no results."
    logger.info("skill_rpc_probe_passed", echo_status=echo.get("status"), search_results=len(search["results"]))


def run_qdrant_probe() -> None:
    backend = QdrantANNBackend(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333") or 6333),
        collection="skills_smoke",
        wait_result=True,
    )
    try:
        vector = np.asarray([0.1, 0.3, 0.5], dtype=np.float32)
        backend.upsert("demo-1", vector, {"text": "demo"})
        hits = backend.query(vector, top_k=1)
        assert hits, "Qdrant query returned no results."
        backend.delete("demo-1")
    finally:
        backend.close()
    logger.info("qdrant_probe_passed")


def run_neo4j_probe() -> None:
    backend = build_backend_from_env()
    if backend is None:
        raise RuntimeError("Neo4j backend not configured.")

    node_id = "skill:rpc_smoke"
    backend.add_node(node_id, "Skill", {"title": "RPC Smoke"})
    view = backend.query(node_id=node_id)
    assert any(node.get("id") == node_id for node in view.get("nodes", [])), "Neo4j node missing."
    backend.remove_node(node_id)
    backend.close()
    logger.info("neo4j_probe_passed")


def main() -> None:
    wait_for_http(os.getenv("SKILL_RPC_BASE_URL", "http://skill-gateway:8300") + "/healthz", timeout=90)
    wait_for_http("http://qdrant:6333/healthz", timeout=90)
    wait_for_http("http://neo4j:7474/", timeout=90)

    client = SkillRPCClient.from_env()
    run_skill_rpc_probe(client)
    run_qdrant_probe()
    run_neo4j_probe()
    logger.info("smoke_test_complete", status="ok")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("smoke_test_failed")
        sys.exit(1)
