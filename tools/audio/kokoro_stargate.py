"""Stargate Kokoro TTS provider (persistent P6000 #1 service shelf).

Fork-owned module — never synced from upstream.
Upstream candidate — answers calesthio/OpenMontage#37.

Fast (~100ms) CPU TTS via Kokoro at http://127.0.0.1:8880.
OpenAI-compat endpoint. No mode switch needed (always-on under Mode C).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import requests

from tools._stargate_character import voice_binding_for
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

KOKORO_URL = "http://127.0.0.1:8880"
OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_tts")

VALID_VOICES = {
    "af_heart", "af_nicole", "af_sky", "af_sarah",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
}


class KokoroStargateTTS(BaseTool):
    name = "kokoro_stargate"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    capabilities = ["text_to_speech", "offline_generation", "persistent_service"]
    supports = {"voice_cloning": False, "emotion_tags": False, "multilingual": False, "streaming": False}
    best_for = ["ultra-fast narration", "high-throughput pipelines", "non-character-bound audio"]
    not_good_for = ["voice cloning", "emotion tags (strips them as literal text)"]

    quality_score = 0.80
    historical_success_rate = 0.99
    latency_p50_seconds = 0.5

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=512, vram_mb=0, network_required=False)
    retry_policy = RetryPolicy(max_retries=2, backoff_seconds=1.0, retryable_errors=["timeout", "503"])

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text (max 5000 chars). Emotion tags stripped to literal text."},
            "character": {"type": "string", "description": "Stargate character id (optional) — maps to voice_id."},
            "voice_id": {"type": "string", "default": "af_heart", "enum": sorted(VALID_VOICES)},
            "output_format": {"type": "string", "enum": ["wav"], "default": "wav"},
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
            "sample_rate": {"type": "integer"},
            "engine": {"type": "string"},
            "voice_id": {"type": "string"},
        },
    }

    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    agent_skills = ["text-to-speech", "kokoro"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()

        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        text = (inputs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="text is required")
        if len(text) > 5000:
            return ToolResult(success=False, error=f"text too long ({len(text)}>5000 chars)")

        # Strip upstream emotion tags that Kokoro speaks as literal text
        text = self._strip_emotion_tags(text)

        voice_id = inputs.get("voice_id") or "af_heart"

        # Character → voice_id override
        character_id = inputs.get("character")
        if character_id:
            binding = voice_binding_for(character_id, "kokoro")
            if binding and binding.voice_id and binding.voice_id in VALID_VOICES:
                voice_id = binding.voice_id

        if voice_id not in VALID_VOICES:
            return ToolResult(success=False, error=f"invalid voice_id {voice_id!r}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"kokoro_{uuid.uuid4().hex[:8]}.wav")

        try:
            r = requests.post(
                f"{KOKORO_URL}/v1/audio/speech",
                json={"input": text, "voice": voice_id, "model": "kokoro", "response_format": "wav"},
                timeout=60,
            )
            if r.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"Kokoro HTTP {r.status_code}: {r.text[:200]}",
                    duration_seconds=time.time() - t_start,
                )
            Path(output_path).write_bytes(r.content)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"Kokoro HTTP error: {exc}",
                              duration_seconds=time.time() - t_start)

        duration = self._audio_duration(output_path)
        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "duration_seconds": duration,
                "sample_rate": 24000,
                "engine": "kokoro",
                "voice_id": voice_id,
            },
            artifacts=[output_path],
            model="kokoro-82m",
            duration_seconds=time.time() - t_start,
        )

    @staticmethod
    def _strip_emotion_tags(text: str) -> str:
        """Remove [laugh] [sigh] [etc.] — Kokoro speaks them literally otherwise."""
        import re
        return re.sub(r"\[[^\]]+\]", "", text).strip()

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
