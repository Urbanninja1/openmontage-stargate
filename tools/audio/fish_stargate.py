"""Stargate Fish S2-PRO SGLang-Omni TTS provider.

Fork-owned module — never synced from upstream.

Routes OpenMontage TTS calls to Stargate's Fish S2-PRO SGLang-Omni
service on http://127.0.0.1:8882, replacing ElevenLabs / OpenAI TTS /
Google TTS in pipelines.

Features carried from Stargate:
  - Character binding via _stargate_character (14-character roster)
  - Emotion-tag native support ([laugh], [sigh], [Jersey Italian], etc.)
  - Voice cloning from reference_path (per-character WAV)
  - Mode-switch via _stargate_mode_lease (audio_studio mode required)
  - COMPILE=1 warmup tax paid once per session (21s → 3s after)

See docs/specs/openmontage.md and docs/plans/2026-04-20-feat-lane-6-openmontage-fork-plan.md §Phase 3.1.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import requests

from tools._stargate_character import voice_binding_for, load_character
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)


FISH_URL = "http://127.0.0.1:8882"
OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_tts")


class FishStargateTTS(BaseTool):
    name = "fish_stargate"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []  # HTTP-only; service availability checked at execute

    capabilities = [
        "text_to_speech",
        "voice_cloning",
        "emotion_tags",
        "character_bound",
        "offline_generation",
    ]
    supports = {
        "voice_cloning": True,
        "emotion_tags": True,
        "multilingual": True,
        "streaming": False,
        "character_binding": True,
    }
    best_for = [
        "character-voiced narration (11 of 14 Stargate characters)",
        "emotion-heavy dialogue",
        "cross-lingual synthesis",
    ]
    not_good_for = [
        "ultra-low-latency real-time conversation (use Kokoro)",
        "disentangled emotion register (use IndexTTS-2)",
    ]

    quality_score = 0.95
    historical_success_rate = 0.98
    latency_p50_seconds = 4.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=1024, vram_mb=12000, network_required=False)
    retry_policy = RetryPolicy(max_retries=2, backoff_seconds=2.0, retryable_errors=["timeout", "503"])

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize. Emotion tags supported."},
            "character": {"type": "string", "description": "Stargate character id (optional). Overrides voice_id."},
            "voice_id": {"type": "string", "description": "Direct reference path. Ignored if character set."},
            "reference_path": {"type": "string", "description": "Explicit voice reference WAV."},
            "output_format": {"type": "string", "enum": ["wav"], "default": "wav"},
            "sample_rate": {"type": "integer", "enum": [48000], "default": 48000},
            "output_path": {"type": "string", "description": "Destination path. Auto-generated if omitted."},
            # Selector-compat params — accepted but passed through unchanged
            "preferred_provider": {"type": "string"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output_path": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "sample_rate": {"type": "integer"},
            "engine": {"type": "string"},
            "character": {"type": "string"},
            "rtf": {"type": "number", "description": "Real-time factor — gen_time / audio_duration"},
        },
    }

    fallback = "kokoro_stargate"
    fallback_tools = ["kokoro_stargate", "piper_tts"]
    agent_skills = ["text-to-speech", "fish-audio", "character-voice"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()

        # operation=rank is just availability probe
        if inputs.get("operation") == "rank":
            return ToolResult(
                success=True,
                data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0},
            )

        text = inputs.get("text", "").strip()
        if not text:
            return ToolResult(success=False, error="text is required")

        # Character binding — resolve voice reference via YAML
        character_id = inputs.get("character")
        reference_path = inputs.get("reference_path") or inputs.get("voice_id")
        if character_id and not inputs.get("reference_path"):
            binding = voice_binding_for(character_id, "fish-s2")
            if binding and binding.reference_path:
                reference_path = binding.reference_path

        # Voice+Character Phase 9 guard: character binding is nominal-only when
        # the reference WAV doesn't exist on disk — the service silently falls
        # back to its default voice, which violates the Decision Communication
        # Contract. Fail loud instead. See docs/solutions/integration-issues/
        # 2026-04-21-lane6-voice-references-missing.md for remediation options.
        if character_id and reference_path and not Path(reference_path).is_file():
            return ToolResult(
                success=False,
                error=(
                    f"voice_reference_missing: character={character_id} "
                    f"reference_path={reference_path} does not exist on disk. "
                    "Voice+Character Phase 9 (ref-capture) was cut from roadmap; "
                    "drop a 15s WAV at the path or pass explicit reference_path."
                ),
                duration_seconds=time.time() - t_start,
            )

        # Output path
        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"fish_{uuid.uuid4().hex[:8]}.wav")

        # Mode lease (audio_studio needed for 3090-side Fish engine)
        try:
            with stargate_mode_lease("audio_studio", duration_minutes=10):
                # Fish SGLang exposes the OpenAI-compat /v1/audio/speech endpoint.
                payload = {
                    "input": text,
                    "model": "s2-pro",
                    "response_format": "wav",
                }
                # Reference audio: fish SGLang accepts `voice` as a path or
                # reference ID. Attenborough binding provides an absolute path.
                if reference_path:
                    payload["voice"] = reference_path

                r = requests.post(
                    f"{FISH_URL}/v1/audio/speech",
                    json=payload,
                    timeout=180,
                )
                if r.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"Fish returned HTTP {r.status_code}: {r.text[:200]}",
                        duration_seconds=time.time() - t_start,
                    )

                # /v1/audio/speech always returns the audio bytes (OpenAI-compat).
                Path(output_path).write_bytes(r.content)

        except RuntimeError as exc:
            return ToolResult(
                success=False,
                error=f"mode_lease failed: {exc}",
                duration_seconds=time.time() - t_start,
            )
        except requests.RequestException as exc:
            return ToolResult(
                success=False,
                error=f"Fish HTTP error: {exc}",
                duration_seconds=time.time() - t_start,
            )

        # Audio duration via ffprobe (best-effort)
        duration = self._audio_duration(output_path)
        gen_time = time.time() - t_start
        rtf = gen_time / duration if duration > 0 else 0.0

        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "duration_seconds": duration,
                "sample_rate": 48000,
                "engine": "fish-s2-pro",
                "character": character_id or "",
                "rtf": rtf,
            },
            artifacts=[output_path],
            model="fish-s2-pro-sglang",
            duration_seconds=gen_time,
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
