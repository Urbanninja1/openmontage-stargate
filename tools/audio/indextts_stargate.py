"""Stargate IndexTTS-2 disentanglement TTS provider.

Fork-owned module — never synced from upstream.

Disentangled timbre + emotion via IndexTTS-2 on http://127.0.0.1:8885.
Stargate routes 2 of 14 characters to IndexTTS (Tywin Lannister, Dr. House)
for its distinctive sarcastic-register capability.

Mode lease: audio_studio required (3090 #2/#3 via mode_manager).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import requests

from tools._stargate_character import voice_binding_for
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)


INDEXTTS_URL = "http://127.0.0.1:8885"
OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_tts")


class IndexTTSStargate(BaseTool):
    name = "indextts_stargate"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    capabilities = [
        "text_to_speech", "voice_cloning", "disentangled_emotion",
        "character_bound", "offline_generation",
    ]
    supports = {
        "voice_cloning": True,
        "emotion_tags": False,  # not native; strip before invoke
        "disentangled_emotion": True,  # separate timbre + emotion refs
        "multilingual": False,
        "streaming": False,
    }
    best_for = [
        "characters needing sarcastic / distinctive register (Dr. House, Tywin)",
        "disentangled emotion control (timbre_ref + emotion_ref)",
    ]
    not_good_for = [
        "native emotion tag support (use Fish)",
        "ultra-low latency (use Kokoro)",
    ]

    quality_score = 0.92
    historical_success_rate = 0.96
    latency_p50_seconds = 5.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=1024, vram_mb=8000, network_required=False)
    retry_policy = RetryPolicy(max_retries=2, backoff_seconds=2.0, retryable_errors=["timeout", "503"])

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "character": {"type": "string", "description": "Stargate character id — loads timbre + emotion refs."},
            "voice_id": {"type": "string", "description": "Explicit timbre reference path (WAV)."},
            "emotion_ref_path": {"type": "string", "description": "Explicit emotion reference path (WAV)."},
            "emotion_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7},
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
            "engine": {"type": "string"},
            "character": {"type": "string"},
            "emotion_strength": {"type": "number"},
        },
    }

    fallback = "fish_stargate"
    fallback_tools = ["fish_stargate", "kokoro_stargate", "piper_tts"]
    agent_skills = ["text-to-speech", "indextts", "disentangled-voice"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        text = (inputs.get("text") or "").strip()
        if not text:
            return ToolResult(success=False, error="text is required")

        # Strip emotion tags (IndexTTS doesn't parse them)
        import re
        text = re.sub(r"\[[^\]]+\]", "", text).strip()

        reference_path = inputs.get("voice_id")
        emotion_ref_path = inputs.get("emotion_ref_path")

        character_id = inputs.get("character")
        if character_id:
            binding = voice_binding_for(character_id, "indextts-2")
            if binding:
                reference_path = reference_path or binding.reference_path
                emotion_ref_path = emotion_ref_path or binding.emotion_ref_path

        if not reference_path:
            return ToolResult(success=False,
                              error="IndexTTS-2 requires timbre reference (character or voice_id)")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"indextts_{uuid.uuid4().hex[:8]}.wav")

        # IndexTTS container mounts ./data/voice_references:/voices:ro.
        # Translate host-relative data/voice_references/... → /voices/...
        def _container_path(p: str) -> str:
            if p.startswith("/voices/") or p.startswith("http"):
                return p
            if p.startswith("data/voice_references/"):
                return "/voices/" + p[len("data/voice_references/"):]
            if "/data/voice_references/" in p:
                return "/voices/" + p.split("/data/voice_references/", 1)[1]
            return p

        timbre = _container_path(reference_path)
        emo = _container_path(emotion_ref_path) if emotion_ref_path else None

        try:
            with stargate_mode_lease("audio_studio", duration_minutes=10):
                # IndexTTS-2 native endpoint: /v2/tts
                payload = {
                    "text": text,
                    "timbre_ref_path": timbre,
                }
                if emo:
                    payload["emo_audio_prompt_path"] = emo

                r = requests.post(f"{INDEXTTS_URL}/v2/tts", json=payload, timeout=300)
                if r.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"IndexTTS-2 HTTP {r.status_code}: {r.text[:200]}",
                        duration_seconds=time.time() - t_start,
                    )
                ct = r.headers.get("content-type", "")
                if "audio" in ct:
                    Path(output_path).write_bytes(r.content)
                else:
                    data = r.json()
                    # IndexTTS returns {"path": "/output/indextts/XXX.wav", ...}
                    # container /output/indextts/ maps to host /home/edson/stargate/output/indextts/
                    src_container = data.get("path") or data.get("output_path") or data.get("audio_path")
                    if not src_container:
                        return ToolResult(
                            success=False,
                            error=f"IndexTTS response missing path: {data}",
                            duration_seconds=time.time() - t_start,
                        )
                    if src_container.startswith("/output/indextts/"):
                        src_host = "/home/edson/stargate/output/indextts/" + src_container[len("/output/indextts/"):]
                    else:
                        src_host = src_container
                    if Path(src_host).is_file():
                        Path(output_path).write_bytes(Path(src_host).read_bytes())
                    else:
                        return ToolResult(
                            success=False,
                            error=f"IndexTTS wrote to {src_container} but not found at {src_host}",
                            duration_seconds=time.time() - t_start,
                        )
        except RuntimeError as exc:
            return ToolResult(success=False, error=f"mode_lease failed: {exc}",
                              duration_seconds=time.time() - t_start)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"IndexTTS HTTP error: {exc}",
                              duration_seconds=time.time() - t_start)

        duration = self._audio_duration(output_path)
        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "duration_seconds": duration,
                "engine": "indextts-2",
                "character": character_id or "",
                "emotion_strength": inputs.get("emotion_strength", 0.7),
            },
            artifacts=[output_path],
            model="indextts-2",
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
