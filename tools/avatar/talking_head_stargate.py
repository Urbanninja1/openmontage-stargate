"""Stargate FLOAT + InfiniteTalk talking-head provider — SOTA replacement.

Fork-owned module — never synced from upstream.

Routes OpenMontage `avatar` capability (talking_head side) calls to:
  - FLOAT workflow (`float-portrait.json`) by default — animates a still
    portrait from audio, best for cropped face photos, Disney-12-principle
    motion.
  - InfiniteTalk workflow (`infinitetalk-dub.json`) via `mode="infinitetalk"`
    — handles long-form dubbing of a static image + audio → video, with
    smart pose + expression inference.

Drop-in replacement for upstream `talking_head` (sadtalker/musetalk):
input = (face image path, audio path) → output = video with animated
face synced to audio.

Why FLOAT + InfiniteTalk over sadtalker:
  - FLOAT: Disney-12-principle motion, fewer uncanny-valley artifacts
    around blinks + brows. Keeps natural head tilt.
  - InfiniteTalk: handles arbitrary-length audio without the 10-second
    cap that SadTalker falls over at.
  - Both run offline in our ComfyUI — no external clone / model dl.

Mode lease: image_studio (3090 #2).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from tools._stargate_comfy import (
    ComfyError, VIDEO_OUTPUT_CLASSES, download_artifacts, poll_until_done,
    render_placeholders, resolve_workflow, submit_prompt, validate_output_nodes,
    comfyui_healthy,
)
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolStatus, ToolTier,
)

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_talking_head")
WORKFLOW_BY_MODE = {
    "float": "float-portrait",          # default — shorter clips, natural motion
    "infinitetalk": "infinitetalk-dub", # long-form, ~every length supported
}


class TalkingHeadStargate(BaseTool):
    name = "talking_head_stargate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "avatar"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    install_instructions = (
        "Already provisioned on Stargate — FLOAT + InfiniteTalk custom "
        "nodes and weights preinstalled in ComfyUI. No agent-side setup."
    )

    capabilities = [
        "talking_head", "face_animation", "audio_driven_motion",
        "offline_generation",
    ]
    supports = {
        "photo_to_video": True, "audio_driven": True,
        "long_form": True,  # via mode="infinitetalk"
        "offline": True,
    }
    best_for = [
        "single-photo presenter videos from narration audio",
        "corporate spokesperson clips (1-60 s with FLOAT)",
        "long-form dubbing (InfiniteTalk, arbitrary length)",
    ]
    not_good_for = [
        "full-body avatar (face crop only)",
        "multi-face scenes (use avatar_spokesperson pipeline with explicit selection)",
    ]

    quality_score = 0.91
    historical_success_rate = 0.93
    latency_p50_seconds = 120.0

    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=4096, vram_mb=14000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1, backoff_seconds=10.0, retryable_errors=["timeout", "cuda_oom"])

    input_schema = {
        "type": "object",
        "required": ["image_path", "audio_path"],
        "properties": {
            "image_path": {"type": "string", "description": "Source face photo."},
            "audio_path": {"type": "string", "description": "Driving audio."},
            "mode": {"type": "string", "enum": ["float", "infinitetalk"], "default": "float",
                     "description": "float=<60s portrait (Disney-12); infinitetalk=any length dub."},
            "output_path": {"type": "string"},
            "seed": {"type": "integer", "default": 0},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
            "preferred_provider": {"type": "string"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "output_path": {"type": "string"},
            "engine": {"type": "string"},
            "workflow": {"type": "string"},
            "mode": {"type": "string"},
        },
    }

    fallback = "talking_head"
    fallback_tools = ["talking_head"]
    agent_skills = ["avatar-video", "talking-head"]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        image = (inputs.get("image_path") or "").strip()
        audio = (inputs.get("audio_path") or "").strip()
        if not image or not audio:
            return ToolResult(success=False, error="image_path and audio_path are both required")
        if not Path(image).is_file():
            return ToolResult(success=False, error=f"image not found: {image}")
        if not Path(audio).is_file():
            return ToolResult(success=False, error=f"audio not found: {audio}")

        if not comfyui_healthy():
            return ToolResult(success=False, error="ComfyUI not healthy")

        mode = inputs.get("mode", "float")
        workflow_name = WORKFLOW_BY_MODE.get(mode)
        if workflow_name is None:
            return ToolResult(success=False, error=f"invalid mode={mode}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"{mode}_{uuid.uuid4().hex[:8]}.mp4")

        try:
            with stargate_mode_lease("image_studio", duration_minutes=20):
                wf = resolve_workflow(workflow_name)
                validate_output_nodes(wf, VIDEO_OUTPUT_CLASSES)
                params = {
                    "INPUT_IMAGE": image,
                    "INPUT_AUDIO": audio,
                    "SEED": inputs.get("seed", 0),
                }
                wf = render_placeholders(wf, params)
                prompt_id = submit_prompt(wf)
                entry = poll_until_done(prompt_id, timeout_s=2400)
                artifacts = download_artifacts(entry, Path(output_path).parent)

                if not artifacts:
                    return ToolResult(
                        success=False,
                        error=f"no artifacts from prompt {prompt_id}",
                        duration_seconds=time.time() - t_start,
                    )
                final = next(
                    (p for p in artifacts if p.endswith((".mp4", ".webm"))),
                    artifacts[0],
                )
                if final != output_path:
                    Path(output_path).write_bytes(Path(final).read_bytes())

        except ComfyError as exc:
            return ToolResult(success=False, error=f"ComfyUI error: {exc}",
                              duration_seconds=time.time() - t_start)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"unexpected: {exc}",
                              duration_seconds=time.time() - t_start)

        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "engine": "float" if mode == "float" else "infinitetalk",
                "workflow": workflow_name,
                "mode": mode,
            },
            artifacts=[output_path],
            model=mode,
            duration_seconds=time.time() - t_start,
            seed=inputs.get("seed", 0),
        )
