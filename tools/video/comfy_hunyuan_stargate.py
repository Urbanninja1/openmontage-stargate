"""Stargate ComfyUI HunyuanVideo 1.5 video generation provider.

Fork-owned module — never synced from upstream.

Specialized for face-heavy / multi-person scenes (Lane 3b flagged
HunyuanVideo 1.5 as SOTA for faces). Uses hunyuan15-t2v.json /
hunyuan15-i2v.json workflows.

Lane 3b gotchas captured: DualCLIPLoader(type="hunyuan_video_15"),
qwen_2.5_vl_7b_fp8_scaled.safetensors text encoder, dedicated
hunyuan_video_15_vae.safetensors. All 16-channel.

Mode lease: image_studio.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from tools._stargate_comfy import (
    ComfyError, download_artifacts, inject_lora_stack, poll_until_done,
    render_placeholders, resolve_workflow, submit_prompt, validate_output_nodes,
    comfyui_healthy, VIDEO_OUTPUT_CLASSES,
)
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_video")

WORKFLOW_ROUTING = {
    "t2v": "hunyuan15-t2v",
    "i2v": "hunyuan15-i2v",
}


class ComfyHunyuanStargate(BaseTool):
    name = "comfy_hunyuan_stargate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_gen"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    capabilities = [
        "text_to_video", "image_to_video", "face_heavy_scenes", "multi_person",
        "lora_stack", "offline_generation",
    ]
    supports = {
        "text_to_video": True, "image_to_video": True, "lora_stack": True,
        "face_generation": True, "seed_control": True,
    }
    best_for = [
        "face-heavy / multi-person dialogue scenes",
        "talking-head-style productions (before full lipsync)",
        "cinematic close-ups",
    ]
    not_good_for = [
        "abstract / non-human scenes (use Wan 2.2)",
        "ControlNet V2V (use Wan Fun Control)",
    ]

    quality_score = 0.90
    historical_success_rate = 0.92
    latency_p50_seconds = 120.0

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=8192, vram_mb=18000, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "kind": {"type": "string", "enum": ["t2v", "i2v"], "default": "t2v"},
            "reference_image": {"type": "string"},
            "duration_frames": {"type": "integer", "default": 81, "minimum": 16, "maximum": 240},
            "seed": {"type": "integer", "default": 0},
            "lora_stack": {
                "type": "array",
                "items": {"type": "object", "required": ["name", "weight"],
                          "properties": {"name": {"type": "string"}, "weight": {"type": "number"}}},
            },
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
            "frames": {"type": "integer"},
            "workflow": {"type": "string"},
            "model": {"type": "string"},
        },
    }

    fallback = "comfy_wan_stargate"
    fallback_tools = ["comfy_wan_stargate"]
    agent_skills = ["comfyui", "hunyuan-video", "video-generation"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        if not comfyui_healthy():
            return ToolResult(success=False, error="ComfyUI not healthy")

        kind = inputs.get("kind", "t2v")
        workflow_name = WORKFLOW_ROUTING.get(kind)
        if workflow_name is None:
            return ToolResult(success=False, error=f"no workflow for kind={kind}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"hunyuan_{uuid.uuid4().hex[:8]}.mp4")

        try:
            with stargate_mode_lease("image_studio", duration_minutes=30):
                wf = resolve_workflow(workflow_name)
                validate_output_nodes(wf, VIDEO_OUTPUT_CLASSES)
                params: dict[str, Any] = {
                    "PROMPT": prompt,
                    "NEGATIVE_PROMPT": inputs.get("negative_prompt", ""),
                    "DURATION_FRAMES": inputs.get("duration_frames", 81),
                    "SEED": inputs.get("seed", 0),
                }
                if inputs.get("reference_image"):
                    params["INPUT_IMAGE"] = inputs["reference_image"]
                wf = render_placeholders(wf, params)

                lora_stack = inputs.get("lora_stack") or []
                if lora_stack:
                    stack = [(L["name"], float(L["weight"])) for L in lora_stack]
                    wf = inject_lora_stack(wf, stack)

                prompt_id = submit_prompt(wf)
                # 30 min — HunyuanVideo 1.5 cold load (~3 min for 6GB encoder + VAE) +
                # 33-frame sample at 60-120s/frame can easily cross 15 min baseline.
                entry = poll_until_done(prompt_id, timeout_s=1800)
                artifacts = download_artifacts(entry, Path(output_path).parent)

                if not artifacts:
                    return ToolResult(success=False,
                                      error=f"no artifacts from prompt {prompt_id}",
                                      duration_seconds=time.time() - t_start)
                final = next((p for p in artifacts if p.endswith(".mp4") or p.endswith(".webm")), artifacts[0])
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
                "duration_seconds": inputs.get("duration_frames", 81) / 24.0,
                "frames": inputs.get("duration_frames", 81),
                "workflow": workflow_name,
                "model": "hunyuanvideo-1.5",
            },
            artifacts=[output_path],
            model="hunyuanvideo-1.5",
            duration_seconds=time.time() - t_start,
            seed=inputs.get("seed", 0),
        )
