"""Shared ComfyUI dispatch helpers for Stargate OpenMontage shims.

Fork-owned helper module — never synced from upstream.

Ports the core _comfyui_generate pattern from Stargate's
services/mcp-server/tools/media_shared.py:19-196 to a dependency-light
TypeScript-free pure-Python module callable from the 3 comfy_*_stargate
shims (comfy_wan, comfy_hunyuan, comfy_image).

Contract:
  1. resolve_workflow(name) -> dict              Load ComfyUI API-format JSON
  2. render_placeholders(wf, params) -> dict     Substitute {{KEY}} tokens
  3. inject_lora_stack(wf, stack) -> dict        Chain LoraLoaderModelOnly nodes
  4. submit_prompt(wf) -> prompt_id              POST :8188/prompt
  5. poll_until_done(prompt_id, ...) -> outputs  Adaptive backoff
  6. download_artifacts(outputs, dst_dir) -> list Collect MP4/PNG paths

COMFYUI_URL overridable via env; default http://127.0.0.1:8188.

See docs/specs/openmontage.md §"Shim inventory" + Lane 3b's
past-solution: docs/solutions/integration-issues/video-studio-comfyui-node-api-mismatches.md.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger("stargate_comfy")

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_PATH", str(Path.home() / "ComfyUI")))
COMFYUI_OUTPUT = COMFYUI_ROOT / "output"

STARGATE_WORKFLOWS = Path("/home/edson/stargate/comfyui-workflows")

# Model-loader node class_types for inject_lora_stack chain start detection
MODEL_LOADER_CLASSES = {
    "UNETLoader", "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "CheckpointLoaderSimple",
}

# Output node class_types — shim pre-validation guards against SaveImage/VHS mismatch
VIDEO_OUTPUT_CLASSES = {"VHS_VideoCombine", "VHS_VideoSave"}
IMAGE_OUTPUT_CLASSES = {"SaveImage", "Save Image w/Metadata", "SaveImageExtended"}


class ComfyError(Exception):
    """ComfyUI submission / polling error."""


def resolve_workflow(name: str) -> dict:
    """Find and load a ComfyUI API-format workflow JSON by name.

    Searches comfyui-workflows/**/<name> and **<name>.json.
    """
    # Accept "wan22-t2v-lightning" or "wan22-t2v-lightning.json"
    fname = name if name.endswith(".json") else f"{name}.json"
    candidates = list(STARGATE_WORKFLOWS.rglob(fname))
    if not candidates:
        raise ComfyError(f"workflow not found: {name} (searched {STARGATE_WORKFLOWS})")
    with candidates[0].open("r", encoding="utf-8") as fh:
        return json.load(fh)


def render_placeholders(wf: dict, params: dict[str, Any]) -> dict:
    """Substitute {{KEY}} tokens throughout the workflow JSON.

    Fail-loudly contract (past-solution: ACE-Step {{TAGS}} silent-fail bug):
    raises ComfyError if any placeholder remains unsubstituted.
    """
    s = json.dumps(wf)
    for key, value in params.items():
        token = "{{" + key.upper() + "}}"
        if token in s:
            # numeric values serialize as JSON, strings quoted
            if isinstance(value, (int, float)):
                s = s.replace(f'"{token}"', str(value)).replace(token, str(value))
            else:
                s = s.replace(token, str(value))

    # Any remaining placeholders?
    remaining = set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", s))
    if remaining:
        raise ComfyError(
            f"unsubstituted placeholders in workflow: {sorted(remaining)}. "
            f"Provide values for: {', '.join(k.lower() for k in remaining)}"
        )
    return json.loads(s)


def patch_node_inputs(wf: dict, node_patches: dict[str, dict[str, Any]]) -> dict:
    """Patch specific node inputs by node_id.

    node_patches = {"42": {"seed": 42, "steps": 8}} → sets
    wf["42"]["inputs"]["seed"] = 42, etc.
    """
    for node_id, patches in node_patches.items():
        if node_id not in wf:
            raise ComfyError(f"node_id {node_id!r} not in workflow")
        wf[node_id].setdefault("inputs", {}).update(patches)
    return wf


def inject_lora_stack(wf: dict, stack: list[tuple[str, float]]) -> dict:
    """Insert LoraLoaderModelOnly nodes between the model loader and its consumers.

    Ported from Stargate services/mcp-server/tools/lora.py:168-247.
    Security invariant: class_type is ALWAYS hard-coded "LoraLoaderModelOnly",
    never derived from user input. Matches Lane 3b Phase 4.1 test contract.
    """
    if not stack:
        return wf

    loader_id: Optional[str] = None
    for nid, node in wf.items():
        if isinstance(node, dict) and node.get("class_type") in MODEL_LOADER_CLASSES:
            loader_id = nid
            break
    if loader_id is None:
        raise ComfyError("workflow has no model-loader node; cannot inject LoRA stack")

    # Find consumers of loader's MODEL output (slot 0)
    consumers: list[tuple[str, str]] = []
    for nid, node in wf.items():
        if nid == loader_id or not isinstance(node, dict):
            continue
        for input_key, link in node.get("inputs", {}).items():
            if isinstance(link, list) and len(link) == 2 and link[0] == loader_id and link[1] == 0:
                consumers.append((nid, input_key))

    prev_node = loader_id
    prev_slot = 0
    for i, (name, weight) in enumerate(stack, start=1):
        new_id = f"lora_stack_{i}"
        wf[new_id] = {
            "inputs": {
                "model": [prev_node, prev_slot],
                "lora_name": name,
                "strength_model": weight,
            },
            # SECURITY INVARIANT — hard-coded, never user-derived
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"LoRA #{i} ({name})"},
        }
        prev_node = new_id
        prev_slot = 0

    for consumer_id, input_key in consumers:
        wf[consumer_id]["inputs"][input_key] = [prev_node, prev_slot]

    return wf


def submit_prompt(wf: dict) -> str:
    """POST /prompt and return prompt_id.

    Filters out top-level keys starting with underscore (Stargate-local
    documentation metadata like `_lane3b`, `_comment`, `_install_required`) —
    ComfyUI treats every top-level key as a node and chokes on string values.
    """
    clean = {k: v for k, v in wf.items() if not k.startswith("_") and isinstance(v, dict)}
    r = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": clean, "client_id": f"stargate_{uuid.uuid4().hex[:8]}"},
        timeout=30,
    )
    if r.status_code != 200:
        raise ComfyError(f"/prompt returned {r.status_code}: {r.text[:200]}")
    data = r.json()
    if "prompt_id" not in data:
        raise ComfyError(f"/prompt response missing prompt_id: {data}")
    return data["prompt_id"]


def poll_until_done(prompt_id: str, *, timeout_s: int = 900, start_poll: float = 2.0,
                    max_poll: float = 10.0, backoff: float = 1.3) -> dict:
    """Adaptive exponential-backoff poll of /history/{prompt_id}.

    Returns the entry dict with "outputs" key.
    """
    deadline = time.monotonic() + timeout_s
    delay = start_poll
    while time.monotonic() < deadline:
        time.sleep(delay)
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=15)
        if r.status_code == 200:
            body = r.json() or {}
            entry = body.get(prompt_id)
            if entry and "outputs" in entry:
                status = (entry.get("status") or {}).get("status_str")
                if status == "error":
                    msgs = (entry.get("status") or {}).get("messages", [])
                    raise ComfyError(f"workflow failed: {msgs}")
                if entry["outputs"] or status == "success":
                    return entry
        delay = min(delay * backoff, max_poll)
    raise ComfyError(f"timed out waiting for prompt {prompt_id}")


def download_artifacts(entry: dict, dst_dir: Path) -> list[str]:
    """Scan entry.outputs for video/image artifacts; copy into dst_dir.

    Returns list of destination file paths.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for node_outputs in (entry.get("outputs") or {}).values():
        for kind in ("videos", "gifs", "images", "audio"):
            for item in (node_outputs.get(kind) or []):
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                file_type = item.get("type", "output")
                if not filename:
                    continue
                # ComfyUI writes to {COMFYUI_OUTPUT}/<subfolder>/<filename>
                src = COMFYUI_OUTPUT / subfolder / filename if file_type == "output" \
                    else COMFYUI_ROOT / file_type / subfolder / filename
                if src.is_file():
                    dst = dst_dir / filename
                    if dst != src:
                        shutil.copy2(src, dst)
                    paths.append(str(dst))
    return paths


def validate_output_nodes(wf: dict, expected: set[str]) -> None:
    """Ensure workflow contains at least one output node of the expected kind.

    Past-solution: SaveImage-when-video-expected silent mismatch (Lane 3b
    docs/solutions/integration-issues/video-studio-comfyui-node-api-mismatches.md).
    """
    found = {
        n["class_type"] for n in wf.values()
        if isinstance(n, dict) and "class_type" in n
    }
    if not (expected & found):
        raise ComfyError(
            f"workflow has no output node of expected kinds {expected}. "
            f"Has: {sorted(c for c in found if 'Save' in c or 'VHS' in c)}"
        )


def comfyui_healthy() -> bool:
    """Quick health-check — used by shim status/dependency probe."""
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False
