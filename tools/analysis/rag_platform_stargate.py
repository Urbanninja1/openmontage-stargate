"""Stargate RAG Platform research provider — Projects as knowledge source.

Fork-owned module — never synced from upstream.
NEW capability not in upstream: leverages Lane 5 RAG Platform.

Provides pipeline-side research stage backed by Stargate Projects
(Hebbia-tier multi-doc RAG — dense+sparse hybrid + Qwen3-Reranker-4B).
When a brief includes `project: <slug>`, this shim queries the
`{slug}--content` Qdrant collection at the RAG Platform (:8104) instead
of hitting the web.

Result: documentary / explainer pipelines become knowledge-grounded
from ingested PDFs, with proper citations in the output sidecar JSON.

Fallback chain:
  project-bound query → rag_platform_stargate (this)
  no project or no hits → searxng_stargate → crawl4ai_stargate
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

RAG_URL = "http://127.0.0.1:8104"
RAG_API_KEY_ENV = "STARGATE_RAG_API_KEY"


class RagPlatformStargate(BaseTool):
    name = "rag_platform_stargate"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "research"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = [
        "project_knowledge_search", "hybrid_dense_sparse", "reranked",
        "citation_output", "offline_research",
    ]
    supports = {
        "project_binding": True, "hybrid_search": True, "reranking": True,
        "citation_emission": True,
    }
    best_for = [
        "documentary / explainer pipelines with a pre-ingested project",
        "knowledge-grounded fact retrieval with citations",
        "long-document citation-heavy research",
    ]
    not_good_for = [
        "general web search (use searxng_stargate)",
        "projects not yet ingested",
    ]

    quality_score = 0.92
    historical_success_rate = 0.97
    latency_p50_seconds = 2.5

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=512, vram_mb=0, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "project": {"type": "string", "description": "Stargate project slug — required for routing here."},
            "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            "collection_url": {"type": "string",
                               "default": "http://127.0.0.1:6333",
                               "description": "Qdrant URL; normally default."},
            "qdrant_api_key": {"type": "string", "description": "Qdrant api-key. Falls back to QDRANT_API_KEY env."},
            "num_results": {"type": "integer", "default": 10},  # selector-compat alias
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
            "preferred_provider": {"type": "string"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "object"}},
            "query": {"type": "string"},
            "num_results": {"type": "integer"},
            "project": {"type": "string"},
        },
    }

    fallback = "searxng_stargate"
    fallback_tools = ["searxng_stargate", "crawl4ai_stargate"]
    agent_skills = ["stargate-rag", "hybrid-search", "project-knowledge"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            # Without a project, we can't add value — return low rank
            if not inputs.get("project"):
                return ToolResult(success=True, data={"ranked": True, "quality_score": 0.0, "cost_usd": 0.0})
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        project = (inputs.get("project") or "").strip()
        if not project:
            return ToolResult(
                success=False,
                error="rag_platform_stargate requires 'project' param; use searxng_stargate for general search",
            )

        query = (inputs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, error="query is required")

        top_k = min(int(inputs.get("top_k", inputs.get("num_results", 10))), 50)
        api_key = os.environ.get(RAG_API_KEY_ENV)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        # Qdrant api-key: explicit param > QDRANT_API_KEY env > none.
        qdrant_api_key = inputs.get("qdrant_api_key") or os.environ.get("QDRANT_API_KEY")
        search_body = {
            "collection_url": inputs.get("collection_url", "http://127.0.0.1:6333"),
            "collection_name": f"{project}--content",
            "query": query,
            "top_k": top_k,
        }
        if qdrant_api_key:
            search_body["qdrant_api_key"] = qdrant_api_key

        try:
            r = requests.post(
                f"{RAG_URL}/v1/search",
                headers=headers,
                json=search_body,
                timeout=30,
            )
            if r.status_code == 404:
                return ToolResult(
                    success=False,
                    error=f"project collection '{project}--content' not found; fallback to searxng",
                    duration_seconds=time.time() - t_start,
                )
            if r.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"RAG Platform HTTP {r.status_code}: {r.text[:200]}",
                    duration_seconds=time.time() - t_start,
                )
            raw = r.json()
            raw_results = raw if isinstance(raw, list) else raw.get("results", [])

            results = [
                {
                    "title": hit.get("payload", {}).get("section_title")
                             or hit.get("payload", {}).get("document_title", ""),
                    "url": f"project://{project}/{hit.get('payload', {}).get('document_id', '')}/p{hit.get('payload', {}).get('page', '?')}",
                    "snippet": hit.get("payload", {}).get("text", "")[:800],
                    "score": hit.get("score", 0.0),
                    "citation": {
                        "document_id": hit.get("payload", {}).get("document_id"),
                        "page": hit.get("payload", {}).get("page"),
                        "section": hit.get("payload", {}).get("section_title"),
                    },
                }
                for hit in raw_results
            ]
        except requests.RequestException as exc:
            return ToolResult(
                success=False,
                error=f"RAG Platform HTTP error: {exc}",
                duration_seconds=time.time() - t_start,
            )

        return ToolResult(
            success=True,
            data={
                "results": results,
                "query": query,
                "num_results": len(results),
                "project": project,
            },
            duration_seconds=time.time() - t_start,
        )
