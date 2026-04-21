"""Stargate ComfyUI Wan 2.2 video generation provider.

Fork-owned module — never synced from upstream.

Routes video_gen calls to ComfyUI at :8188 using Stargate's Wan 2.2 SOTA
workflows (Lightning 4-step + 8-step 2-stage split, Fun Control ControlNet
V2V, SVI 2.0 Pro extend, etc.). Replaces Runway / Kling / Veo / MiniMax.

Mode lease: image_studio (both 3090s — evicts persistent LLMs).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from tools._stargate_comfy import (
    ComfyError, download_artifacts, inject_lora_stack, patch_node_inputs,
    poll_until_done, render_placeholders, resolve_workflow, submit_prompt,
    validate_output_nodes, comfyui_healthy, VIDEO_OUTPUT_CLASSES,
)
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_video")

# Quality+kind → workflow routing
WORKFLOW_ROUTING = {
    ("fast", "t2v"): "wan22-t2v-lightning",
    ("fast", "i2v"): "wan22-i2v-lightning",
    ("balanced", "t2v"): "wan22-t2v",
    ("balanced", "i2v"): "wan22-i2v",
    ("best", "t2v"): "wan22-t2v",
    ("best", "i2v"): "wan22-i2v",
    ("fast", "control"): "wan22-funcontrol-v2v",
    ("balanced", "control"): "wan22-funcontrol-v2v",
    ("fast", "long"): "wan22-svi-extend-no-freelong",
    ("balanced", "long"): "wan22-svi-extend",
}


class ComfyWanStargate(BaseTool):
    name = "comfy_wan_stargate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"  # matches upstream video_selector discovery
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    capabilities = [
        "text_to_video", "image_to_video", "controlnet_v2v", "long_clip_extend",
        "lora_stack", "offline_generation",
    ]
    supports = {
        "text_to_video": True, "image_to_video": True, "controlnet": True,
        "lora_stack": True, "seed_control": True, "multilingual": True,
    }
    best_for = [
        "14B Wan 2.2 with Lightning LoRA (4-step, 90s for 5s clip)",
        "I2V with SOTA Kijai 260412 rank-256 LoRA",
        "SVI 2.0 Pro long-clip extension",
    ]
    not_good_for = [
        "face-heavy / multi-person scenes (use comfy_hunyuan_stargate)",
        "real-time preview (offline only)",
    ]

    quality_score = 0.93
    historical_success_rate = 0.94
    latency_p50_seconds = 90.0

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=8192, vram_mb=20000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1, backoff_seconds=10.0, retryable_errors=["timeout", "cuda_oom"])

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "quality": {"type": "string", "enum": ["fast", "balanced", "best"], "default": "fast"},
            "kind": {"type": "string", "enum": ["t2v", "i2v", "control", "long"], "default": "t2v"},
            "reference_image": {"type": "string", "description": "Image path (for kind=i2v or long/SVI-extend)."},
            "reference_video": {"type": "string", "description": "Video path (for kind=control V2V)."},
            "control_type": {"type": "string", "enum": ["depth", "openpose", "canny"]},
            "duration_frames": {"type": "integer", "default": 81, "minimum": 16, "maximum": 360},
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

    fallback = "comfy_hunyuan_stargate"
    fallback_tools = ["comfy_hunyuan_stargate"]
    agent_skills = ["comfyui", "wan-2-2", "video-generation"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()

        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        if not comfyui_healthy():
            return ToolResult(success=False, error=f"ComfyUI not healthy at {'8188'}")

        quality = inputs.get("quality", "fast")
        kind = inputs.get("kind", "t2v")
        workflow_name = WORKFLOW_ROUTING.get((quality, kind))
        if workflow_name is None:
            return ToolResult(success=False, error=f"no workflow for quality={quality} kind={kind}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"wan_{uuid.uuid4().hex[:8]}.mp4")

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
                if inputs.get("reference_video"):
                    params["INPUT_VIDEO"] = inputs["reference_video"]
                elif kind == "control" and inputs.get("reference_image"):
                    # Fun Control V2V wants a video — reject with a clear error
                    # rather than silently let the workflow fail on missing placeholder.
                    return ToolResult(
                        success=False,
                        error="kind=control requires reference_video (path to input clip); reference_image is for kind=i2v/long",
                        duration_seconds=time.time() - t_start,
                    )

                wf = render_placeholders(wf, params)

                # LoRA stack injection
                lora_stack = inputs.get("lora_stack") or []
                if lora_stack:
                    stack = [(L["name"], float(L["weight"])) for L in lora_stack]
                    wf = inject_lora_stack(wf, stack)

                prompt_id = submit_prompt(wf)
                # 30 min — accommodates Wan 2.2 14B cold load + generation matrix.
                entry = poll_until_done(prompt_id, timeout_s=1800)
                artifacts = download_artifacts(entry, Path(output_path).parent)

                if not artifacts:
                    return ToolResult(
                        success=False,
                        error=f"workflow completed but no artifacts (prompt_id={prompt_id})",
                        duration_seconds=time.time() - t_start,
                    )

                # Use the first MP4 as canonical output
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
                "model": "wan-2.2-14b" if "14b" in workflow_name.lower() or "lightning" in workflow_name else "wan-2.2",
            },
            artifacts=[output_path],
            model="wan-2.2",
            duration_seconds=time.time() - t_start,
            seed=inputs.get("seed", 0),
        )
