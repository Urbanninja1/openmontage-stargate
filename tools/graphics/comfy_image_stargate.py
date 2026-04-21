"""Stargate ComfyUI FLUX 2 + Klein image generation provider.

Fork-owned module — never synced from upstream.
WATCH: calesthio/OpenMontage PR #29 (native ComfyUI provider) — if merged,
this shim pivots to a thin URL/quality-tier override on top of the upstream
class.

Routes image_gen to ComfyUI at :8188 using FLUX 2 workflows.
Quality tiers:
  fast     → flux-klein-4b-Q8 (~15s, balanced quality)
  balanced → flux-klein-9b-Q8 (~52s, default)
  best     → flux2-dev-Q8 20-step (~104s, SOTA)

Character LoRA routing: brief.character → /models/comfyui/loras/flux2-dev/character/<id>.safetensors

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
    comfyui_healthy, IMAGE_OUTPUT_CLASSES,
)
from tools._stargate_mode_lease import stargate_mode_lease
from tools.base_tool import (
    BaseTool, Determinism, ExecutionMode, ResourceProfile, RetryPolicy,
    ToolResult, ToolRuntime, ToolStability, ToolTier,
)

OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_image")

WORKFLOW_BY_QUALITY = {
    "fast": "flux2-klein-4b-t2i",
    "balanced": "flux2-klein-9b-t2i",
    "best": "flux2-dev-t2i",
}


class ComfyImageStargate(BaseTool):
    name = "comfy_image_stargate"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"  # matches upstream image_selector discovery
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    capabilities = [
        "text_to_image", "character_lora", "quality_tiered",
        "offline_generation",
    ]
    supports = {
        "text_to_image": True, "lora_stack": True, "character_binding": True,
        "seed_control": True,
    }
    best_for = [
        "storyboard frames",
        "character portraits with trained LoRAs",
        "illustrated panel art",
    ]
    not_good_for = [
        "photorealistic faces with perfect identity (use PuLID workflow variant)",
        "vector / CAD output",
    ]

    quality_score = 0.94
    historical_success_rate = 0.96
    latency_p50_seconds = 50.0

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=4096, vram_mb=16000, network_required=False)

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "quality": {"type": "string", "enum": ["fast", "balanced", "best"], "default": "balanced"},
            "width": {"type": "integer", "default": 1024},
            "height": {"type": "integer", "default": 1024},
            "seed": {"type": "integer", "default": 0},
            "character": {"type": "string", "description": "Stargate character id — auto-loads character LoRA if exists."},
            "lora_stack": {
                "type": "array",
                "items": {"type": "object", "required": ["name", "weight"]},
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
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "workflow": {"type": "string"},
            "model": {"type": "string"},
        },
    }

    fallback_tools = []
    agent_skills = ["comfyui", "flux-2", "image-generation"]

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()
        if inputs.get("operation") == "rank":
            return ToolResult(success=True, data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0})

        prompt = (inputs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        if not comfyui_healthy():
            return ToolResult(success=False, error="ComfyUI not healthy")

        quality = inputs.get("quality", "balanced")
        workflow_name = WORKFLOW_BY_QUALITY.get(quality)
        if workflow_name is None:
            return ToolResult(success=False, error=f"invalid quality={quality}")

        output_path = inputs.get("output_path")
        if not output_path:
            OUTPUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)
            output_path = str(OUTPUT_DIR_DEFAULT / f"flux_{uuid.uuid4().hex[:8]}.png")

        # Character LoRA auto-routing — check disk before injecting so missing
        # LoRAs don't cause workflow validation to fail (char LoRAs were cut
        # from Voice+Character Stack Phase 9 per roadmap).
        lora_stack = list(inputs.get("lora_stack") or [])
        character_id = inputs.get("character")
        if character_id:
            lora_rel = f"flux2-dev/character/{character_id}.safetensors"
            # Try a few ways character folders are named (dr-house vs dr_house)
            for cid in (character_id, character_id.replace("-", "_")):
                candidate_abs = Path(f"/models/comfyui/loras/flux2-dev/character/{cid}/lora.safetensors")
                candidate_flat = Path(f"/models/comfyui/loras/flux2-dev/character/{cid}.safetensors")
                if candidate_flat.is_file():
                    lora_rel = f"flux2-dev/character/{cid}.safetensors"
                    break
                if candidate_abs.is_file():
                    lora_rel = f"flux2-dev/character/{cid}/lora.safetensors"
                    break
            else:
                # no file found — skip LoRA injection entirely
                lora_rel = None
            if lora_rel and not any(L.get("name") == lora_rel for L in lora_stack):
                lora_stack.insert(0, {"name": lora_rel, "weight": 0.85})

        try:
            with stargate_mode_lease("image_studio", duration_minutes=15):
                wf = resolve_workflow(workflow_name)
                validate_output_nodes(wf, IMAGE_OUTPUT_CLASSES)
                params: dict[str, Any] = {
                    "PROMPT": prompt,
                    "NEGATIVE_PROMPT": inputs.get("negative_prompt", ""),
                    "WIDTH": inputs.get("width", 1024),
                    "HEIGHT": inputs.get("height", 1024),
                    "SEED": inputs.get("seed", 0),
                }
                wf = render_placeholders(wf, params)

                if lora_stack:
                    stack = [(L["name"], float(L["weight"])) for L in lora_stack]
                    try:
                        wf = inject_lora_stack(wf, stack)
                    except ComfyError as exc:
                        # Character LoRA missing → strip and retry without
                        if character_id and "character" in str(lora_stack[0].get("name", "")):
                            lora_stack = lora_stack[1:]
                            if lora_stack:
                                wf = inject_lora_stack(wf, [(L["name"], float(L["weight"])) for L in lora_stack])
                        else:
                            raise

                prompt_id = submit_prompt(wf)
                # 20 min — accommodates FLUX 2 Dev Q8 (18 GB) cold load + 20-step sampling + VAE decode.
                # Klein 9B "balanced" finishes in 60-90s once loaded, but first run per-session pays 60-120s
                # model-load tax. "best" tier routinely runs 200-400s end-to-end.
                entry = poll_until_done(prompt_id, timeout_s=1200)
                artifacts = download_artifacts(entry, Path(output_path).parent)

                if not artifacts:
                    return ToolResult(success=False,
                                      error=f"no artifacts from prompt {prompt_id}",
                                      duration_seconds=time.time() - t_start)
                final = next((p for p in artifacts if p.endswith(".png") or p.endswith(".jpg")), artifacts[0])
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
                "width": inputs.get("width", 1024),
                "height": inputs.get("height", 1024),
                "workflow": workflow_name,
                "model": workflow_name.split("-")[0],
            },
            artifacts=[output_path],
            model="flux-2" if "flux2" in workflow_name else "flux-klein",
            duration_seconds=time.time() - t_start,
            seed=inputs.get("seed", 0),
        )
