"""Stargate SearXNG web-search provider.

Fork-owned module — never synced from upstream.

Routes OpenMontage web_search to local SearXNG at :8888.
No mode switch needed (always-on).
"""

from __future__ import annotations

import time
from typing import Any

import requests

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

SEARXNG_URL = "http://127.0.0.1:8888"


class SearxngStargate(BaseTool):
    # Lane 6: register under TWO names so pipeline YAMLs that call the
    # phantom upstream `web_search` tool (e.g. cinematic.yaml) resolve
    # through capability-based discovery. `name` is the identity used
    # by registry.get(); `aliases` get re-registered post-discovery via
    # the aliasing block at bottom of this module.
    name = "searxng_stargate"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "web_search"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = ["web_search", "privacy_preserving", "multi_engine_meta"]
    supports = {"multi_engine": True, "rate_limit_safe": True}
    best_for = ["general web research", "privacy-preserving search", "offline-friendly"]
    not_good_for = ["Google-exclusive knowledge graph results", "proprietary AI overviews"]

    quality_score = 0.75
    historical_success_rate = 0.98
    latency_p50_seconds = 1.5

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=256, vram_mb=0, network_required=True)
    retry_policy = RetryPolicy(max_retries=2, backoff_seconds=2.0, retryable_errors=["timeout", "503", "429"])

    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            "engines": {"type": "array", "items": {"type": "string"},
                        "description": "SearXNG engine list, e.g. ['google', 'duckduckgo']"},
            "format": {"type": "string", "enum": ["json"], "default": "json"},
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
        },
    }

    fallback = "crawl4ai_stargate"
    fallback_tools = ["crawl4ai_stargate"]
    agent_skills = ["web-search", "searxng"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        query = (inputs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, error="query is required")

        num_results = min(int(inputs.get("num_results", 10)), 50)

        params = {"q": query, "format": "json", "safesearch": "0"}
        engines = inputs.get("engines")
        if engines:
            params["engines"] = ",".join(engines)

        try:
            r = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=30)
            if r.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"SearXNG HTTP {r.status_code}: {r.text[:200]}",
                    duration_seconds=time.time() - t_start,
                )
            data = r.json()
            raw_results = data.get("results") or []

            results = [
                {
                    "title": r_.get("title", ""),
                    "url": r_.get("url", ""),
                    "snippet": r_.get("content") or r_.get("snippet", ""),
                    "engine": r_.get("engine", ""),
                    "score": r_.get("score", 0.0),
                }
                for r_ in raw_results[:num_results]
            ]
        except requests.RequestException as exc:
            return ToolResult(
                success=False,
                error=f"SearXNG HTTP error: {exc}",
                duration_seconds=time.time() - t_start,
            )

        return ToolResult(
            success=True,
            data={
                "results": results,
                "query": query,
                "num_results": len(results),
            },
            duration_seconds=time.time() - t_start,
        )


class WebSearchStargate(SearxngStargate):
    """Alias registration as `web_search` so pipeline YAMLs that list the
    phantom upstream `web_search` tool name (e.g. cinematic.yaml) resolve.

    Everything else is inherited from SearxngStargate — same execute logic,
    same rank probe, same schema. Only the registry name + provider tag
    change so both rows appear in registry.provider_catalog()['stargate'].
    """
    name = "web_search"
    provider = "stargate-alias"
