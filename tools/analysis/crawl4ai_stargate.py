"""Stargate Crawl4AI URL-extract provider.

Fork-owned module — never synced from upstream.

Deep extract of a specific URL (JS-rendered if needed) via Crawl4AI
at :11235. Fallback when SearXNG snippet isn't enough and pipeline needs
the full page text (e.g. documentary-montage citing an academic paper).
"""

from __future__ import annotations

import time
from typing import Any

import requests

from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

CRAWL4AI_URL = "http://127.0.0.1:11235"


class Crawl4aiStargate(BaseTool):
    name = "crawl4ai_stargate"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "url_extract"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = ["url_extract", "js_rendering", "markdown_output"]
    supports = {"js_rendering": True, "pdf_extract": True}
    best_for = ["deep extraction of academic papers / long-form articles", "JS-heavy pages"]
    not_good_for = ["bulk search (use searxng_stargate)"]

    quality_score = 0.88
    historical_success_rate = 0.90
    latency_p50_seconds = 8.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=1024, vram_mb=0, network_required=True)

    input_schema = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string"},
            "js_render": {"type": "boolean", "default": True},
            "output_format": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "text": {"type": "string"},
            "markdown": {"type": "string"},
            "length_chars": {"type": "integer"},
        },
    }

    fallback_tools = []
    agent_skills = ["web-extract", "crawl4ai"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        url = (inputs.get("url") or "").strip()
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(success=False, error="valid url required")

        try:
            r = requests.post(
                f"{CRAWL4AI_URL}/crawl",
                json={
                    "urls": [url],
                    "priority": 10,
                    "crawler_config": {"cache_mode": "bypass"},
                },
                timeout=60,
            )
            if r.status_code != 200:
                return ToolResult(success=False,
                                  error=f"Crawl4AI HTTP {r.status_code}: {r.text[:200]}",
                                  duration_seconds=time.time() - t_start)
            data = r.json()
            # Crawl4AI returns { results: [{url, markdown, cleaned_html, extracted_content, ...}] }
            first = (data.get("results") or [{}])[0]
            markdown = first.get("markdown", "") or ""
            title = first.get("metadata", {}).get("title", "")
            text = first.get("cleaned_html", "") or markdown

        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"Crawl4AI error: {exc}",
                              duration_seconds=time.time() - t_start)

        return ToolResult(
            success=True,
            data={
                "url": url,
                "title": title,
                "text": text,
                "markdown": markdown,
                "length_chars": len(markdown),
            },
            duration_seconds=time.time() - t_start,
        )
