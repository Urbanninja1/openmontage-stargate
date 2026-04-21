"""Stargate ACE-Step music generation provider.

Fork-owned module — never synced from upstream.
NEW capability shim: replaces paid upstream `music_gen` (ElevenLabs) and
`suno_music` (Suno API) with Stargate's local ACE-Step ComfyUI workflow.

Routes `music_generation` calls to ComfyUI at :8188 using
`comfyui-workflows/music/ace-step-music.json` (mounted from the Stargate
repo). Placeholders: `LYRICS` + `TAGS`.

Quality tiers:
  fast     → ACE-Step turbo 8-step (~15 s)  [default — runs on 3090 #3]
  balanced → ACE-Step SFT 50-step (~30 s)
  quality  → SongGen v2 (music_studio mode — EVICTS 3090 #2 LLM)

Mode lease: music_studio ONLY for quality tier; fast/balanced run
alongside other workflows via image_studio.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from tools._stargate_comfy import (
    AUDIO_OUTPUT_CLASSES, ComfyError, download_artifacts, poll_until_done,
    render_placeholders, resolve_workflow, submit_prompt, validate_output_nodes,
    comfyui_healthy,
)
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_music")

WORKFLOW_BY_QUALITY = {
    "fast": "ace-step-music",
    "balanced": "ace-step-music",  # same workflow, callers may tune STEPS later
    "quality": "ace-step-music",   # SongGen v2 path deferred — see songgen_stargate TODO
}


class MusicStargate(BaseTool):
    name = "music_stargate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    install_instructions = (
        "Already provisioned on Stargate — ACE-Step SFT checkpoint at "
        "/models/comfyui/checkpoints/ + workflow in comfyui-workflows/music/. "
        "No agent-side setup."
    )

    capabilities = [
        "music_generation", "lyric_to_song", "tag_controlled_music",
        "offline_generation",
    ]
    supports = {
        "prompt_music": True, "lyric_control": True, "tag_control": True,
        "streaming": False, "offline": True,
    }
    best_for = [
        "royalty-free background music for explainer / cinematic videos",
        "tag-controlled mood music (calm, cinematic, ambient, uplifting)",
        "lyric-driven short songs (under 60 s)",
    ]
    not_good_for = [
        "perfect vocal mimicry (use songgen_stargate when shipped)",
        "multi-minute continuous pieces (hard cap ~60-90 s per call)",
    ]

    quality_score = 0.85
    historical_success_rate = 0.94
    latency_p50_seconds = 20.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=1024, vram_mb=4000, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Free-form music description. Mapped to TAGS placeholder."},
            "tags": {"type": "string", "description": "Explicit ACE-Step tag string (overrides prompt→tags mapping)."},
            "lyrics": {"type": "string", "description": "Lyrics text. Empty = instrumental."},
            "quality": {"type": "string", "enum": ["fast", "balanced", "quality"], "default": "fast"},
            "duration_seconds": {"type": "number", "default": 20.0, "minimum": 5.0, "maximum": 60.0},
            "seed": {"type": "integer", "default": 0},
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
            "workflow": {"type": "string"},
            "model": {"type": "string"},
        },
    }

    fallback = None
    fallback_tools = []
    agent_skills = ["music", "acestep"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt and not inputs.get("lyrics") and not inputs.get("tags"):
            return ToolResult(success=False, error="prompt, tags, or lyrics required")

        if not comfyui_healthy():
            return ToolResult(success=False, error="ComfyUI not healthy")

        quality = inputs.get("quality", "fast")
        workflow_name = WORKFLOW_BY_QUALITY.get(quality)
        if workflow_name is None:
            return ToolResult(success=False, error=f"invalid quality={quality}")

        # TAGS default: if no explicit tags, fall back to the prompt verbatim.
        # ACE-Step accepts tag-like phrases well enough ("cinematic ambient calm piano").
        tags = inputs.get("tags") or prompt
        lyrics = inputs.get("lyrics") or ""

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"ace_{uuid.uuid4().hex[:8]}.wav")

        # Quality tier decides mode: fast/balanced = image_studio suffices
        # (ACE-Step is light, fits alongside other workflows on 3090 #3).
        # "quality" tier reserves music_studio for SongGen v2 (not yet shimmed).
        mode = "music_studio" if quality == "quality" else "image_studio"

        try:
            with stargate_mode_lease(mode, duration_minutes=15):
                wf = resolve_workflow(workflow_name)
                validate_output_nodes(wf, AUDIO_OUTPUT_CLASSES)
                params: dict[str, Any] = {
                    "TAGS": tags,
                    "LYRICS": lyrics,
                    "SEED": inputs.get("seed", 0),
                }
                wf = render_placeholders(wf, params)

                prompt_id = submit_prompt(wf)
                # 15 min — ACE-Step 50-step at 30s + cold load margin.
                entry = poll_until_done(prompt_id, timeout_s=900)
                artifacts = download_artifacts(entry, Path(output_path).parent)
                if not artifacts:
                    return ToolResult(
                        success=False,
                        error=f"no artifacts from prompt {prompt_id}",
                        duration_seconds=time.time() - t_start,
                    )
                # Prefer WAV/MP3/FLAC audio over spectrogram PNG.
                audio = next(
                    (p for p in artifacts
                     if p.endswith((".wav", ".mp3", ".flac", ".ogg"))),
                    None,
                )
                if audio is None:
                    return ToolResult(
                        success=False,
                        error=(
                            "ACE-Step returned only non-audio artifacts "
                            f"(first={artifacts[0]!r}); workflow may need a SaveAudio node"
                        ),
                        duration_seconds=time.time() - t_start,
                    )
                if audio != output_path:
                    Path(output_path).write_bytes(Path(audio).read_bytes())

        except ComfyError as exc:
            return ToolResult(success=False, error=f"ComfyUI error: {exc}",
                              duration_seconds=time.time() - t_start)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"unexpected: {exc}",
                              duration_seconds=time.time() - t_start)

        duration = self._audio_duration(output_path)
        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "duration_seconds": duration,
                "workflow": workflow_name,
                "model": "ace-step-sft",
            },
            artifacts=[output_path],
            model="ace-step",
            duration_seconds=time.time() - t_start,
            seed=inputs.get("seed", 0),
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
