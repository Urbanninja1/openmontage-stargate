"""Stargate ensemble multi-speaker TTS provider (VibeVoice-Large 9B).

Fork-owned module — never synced from upstream.
NEW capability — no upstream equivalent.

Wraps Stargate's render_ensemble_scene (4-speaker VibeVoice-Large) via
HTTP call to MCP server. Script format:
    [Tony] You know what bothers me?
    [Silvio] What?
    [Tony] The coffee here.

Used for podcast-repurpose (multi-host dialogue), documentary-montage
(interview re-enactment), cinematic (dialogue scenes).

Mode lease: audio_studio (3090 #2 for VibeVoice-Large).
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from tools._stargate_character import load_character
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

# Use MCP HTTP transport for render_ensemble_scene. If MCP server exposes
# a dedicated HTTP route, call it; else we invoke via Agent API adapter.
# For portability, we go through Agent API at :8096/characters/render_ensemble.
ENSEMBLE_URL = "http://127.0.0.1:8096/characters/render_ensemble"

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_ensemble")


class EnsembleStargate(BaseTool):
    name = "ensemble_stargate"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "multi_speaker_tts"
    provider = "stargate"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    capabilities = [
        "multi_speaker_tts", "dialogue_synthesis", "character_ensemble",
        "anti_voice_bleed",
    ]
    supports = {
        "multi_speaker": True, "up_to_4_speakers": True, "character_binding": True,
    }
    best_for = [
        "podcast dialogue (2-4 speakers)",
        "documentary interview scenes",
        "Sopranos-4 / Lannisters-2 group-chat rendering",
    ]
    not_good_for = [
        "single-speaker narration (use fish_stargate)",
        "> 4 speakers (VibeVoice limit)",
    ]

    quality_score = 0.89
    historical_success_rate = 0.88
    latency_p50_seconds = 20.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2048, vram_mb=16000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1, backoff_seconds=5.0, retryable_errors=["timeout"])

    input_schema = {
        "type": "object",
        "required": ["script"],
        "properties": {
            "script": {"type": "string",
                       "description": "Multi-speaker script. Lines of form '[speaker_id] text'."},
            "characters": {"type": "array", "items": {"type": "string"},
                           "description": "Stargate character ids present in script. Required for voice binding."},
            "output_path": {"type": "string"},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
            "preferred_provider": {"type": "string"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output_path": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "speakers": {"type": "array", "items": {"type": "string"}},
            "engine": {"type": "string"},
        },
    }

    fallback = "fish_stargate"
    fallback_tools = []  # Fish-per-line fallback is a pipeline-side decision, not selector chain
    agent_skills = ["multi-speaker-tts", "vibevoice", "character-ensemble"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        script = (inputs.get("script") or "").strip()
        if not script:
            return ToolResult(success=False, error="script is required")

        # Extract speakers from script
        script_speakers = sorted(set(re.findall(r"^\[(\w+[-\w]*)\]", script, re.MULTILINE)))
        if not script_speakers:
            return ToolResult(success=False,
                              error="script has no [speaker] prefixes; expected '[tony] hello'")
        if len(script_speakers) > 4:
            return ToolResult(success=False, error=f"VibeVoice max 4 speakers; got {len(script_speakers)}")

        characters = inputs.get("characters") or script_speakers
        # Validate all characters load
        missing = [c for c in characters if load_character(c) is None]
        if missing:
            return ToolResult(success=False,
                              error=f"characters not found: {missing}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"ensemble_{uuid.uuid4().hex[:8]}.wav")

        try:
            with stargate_mode_lease("audio_studio", duration_minutes=15):
                r = requests.post(
                    ENSEMBLE_URL,
                    json={
                        "script": script,
                        "characters": characters,
                        "output_path": output_path,
                    },
                    timeout=300,
                )
                if r.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"ensemble HTTP {r.status_code}: {r.text[:200]}",
                        duration_seconds=time.time() - t_start,
                    )
                body = r.json()
                server_output = body.get("output_path", output_path)
                if server_output != output_path and Path(server_output).is_file():
                    Path(output_path).write_bytes(Path(server_output).read_bytes())
        except RuntimeError as exc:
            return ToolResult(success=False, error=f"mode_lease failed: {exc}",
                              duration_seconds=time.time() - t_start)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"ensemble HTTP error: {exc}",
                              duration_seconds=time.time() - t_start)

        duration = self._audio_duration(output_path)
        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "duration_seconds": duration,
                "speakers": characters,
                "engine": "vibevoice-large-9b",
            },
            artifacts=[output_path],
            model="vibevoice-large-9b",
            duration_seconds=time.time() - t_start,
        )

    @staticmethod
    def _audio_duration(path: str) -> float:
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=5,
            )
            return float(r.stdout.strip() or "0")
        except (subprocess.SubprocessError, ValueError):
            return 0.0
