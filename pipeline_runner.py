"""Stargate-local pipeline runner — CLI for OpenMontage pipelines.

Fork-owned module (no upstream equivalent). Supports:
  - Tools-available stages (calls tool_registry shims)
  - llm_step stages (direct OpenAI-compat call to llama-swap)
  - iterate_over stages (fan-out per-item, e.g. panel × 6)
  - Remotion compose stages (npx remotion render with calculated props)
  - Sidecar metadata + POST /output/record on success

Usage:
    python -m pipeline_runner <pipeline> --brief "..." \\
        [--character <id>] [--project <slug>] [--output-dir <dir>]

Every pipeline run writes:
    {output_dir}/{pipeline}_{run_id}.mp4          (if compose succeeded)
    {output_dir}/{pipeline}_{run_id}.meta.json    (always — provenance)
    ~/stargate/logs/openmontage-shims.log         (structured per-tool)

Last stdout line is parseable JSON — OpenClaw subprocess relies on this.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required — pip install pyyaml", file=sys.stderr)
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "pipeline_defs"
AGENT_API_URL = "http://127.0.0.1:8096"
LLAMASWAP_URL = "http://127.0.0.1:8080"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline_runner")

sys.path.insert(0, str(ROOT))
from tools.tool_registry import registry  # noqa: E402
from tools._stargate_mode_lease import stargate_mode_lease  # noqa: E402
registry.discover()


# Capabilities that require image_studio / audio_studio GPU modes.
# Pipeline_runner pre-acquires the lease around iterate stages so multi-item
# stages don't thrash mode-switch 6x.
_MODE_FOR_CAPABILITY = {
    "video_gen": "image_studio",
    "image_gen": "image_studio",
    # Shims declare the upstream selector-compatible capability names
    # ("image_generation"/"video_generation"), so map those too — otherwise
    # the stage-level lease never fires and each iteration cold-starts ComfyUI
    # (the shim leases per-call and restores `off` after every image).
    "image_generation": "image_studio",
    "video_generation": "image_studio",
    "tts": None,             # Kokoro on P6000 = no switch; Fish/IndexTTS handled by shim
    "multi_speaker_tts": "audio_studio",
    "research": None,
    "url_extract": None,
    "web_search": None,
}


def _mode_for_stage(stage: dict) -> Optional[str]:
    """Determine if stage needs a GPU mode lease (returns mode name or None)."""
    tools = stage.get("tools_available") or []
    modes_needed: set[str] = set()
    for tn in tools:
        tool_cls = registry._tools.get(tn)
        if tool_cls is None:
            continue
        cap = getattr(tool_cls, "capability", "")
        mode = _MODE_FOR_CAPABILITY.get(cap)
        if mode:
            modes_needed.add(mode)
    if len(modes_needed) == 1:
        return modes_needed.pop()
    # Mixed: caller must handle. For now, prefer image_studio if present.
    if "image_studio" in modes_needed:
        return "image_studio"
    return None


class PipelineError(Exception):
    pass


def load_pipeline(name: str) -> dict:
    for c in [PIPELINE_DIR / f"{name}.yaml", PIPELINE_DIR / f"{name}.yml"]:
        if c.is_file():
            with c.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
    raise PipelineError(f"pipeline '{name}' not found in {PIPELINE_DIR}")


# ── LLM step helper ───────────────────────────────────────────────────────

def call_llm_step(stage: dict, *, brief: str, state: dict) -> dict:
    """Call llama-swap OpenAI-compat endpoint. Parses JSON if requested."""
    cfg = stage["llm_step"]
    model = cfg.get("model", "qwopus-27b")
    # Per-stage endpoint override. Default LLAMASWAP_URL (:8080) only serves
    # llama-swap-managed models; the persistent dual-3090 unit
    # (qwen3.6-27b-autoround) is reachable only via the :8084 Gate, which is
    # what config.yaml's llm.base_url points to. Honor a manifest base_url so
    # a stage can target the Gate without a global runner change.
    base_url = (cfg.get("base_url") or LLAMASWAP_URL).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    system = cfg.get("system_prompt", "")
    temperature = cfg.get("temperature", 0.7)
    max_tokens = cfg.get("max_tokens", 2000)
    parse_json = cfg.get("parse_json", False)

    user = brief
    # Hydrate user message with prior stage state for context
    if state.get("stage_outputs"):
        ctx_parts = []
        for s_name, s_out in state["stage_outputs"].items():
            if not s_out:
                continue
            ctx_parts.append(f"{s_name}: {json.dumps(s_out, default=str)[:500]}")
        if ctx_parts:
            user = f"{brief}\n\nPrior context:\n" + "\n".join(ctx_parts)

    log.info("  llm_step: model=%s temp=%s (prompt %d chars)", model, temperature, len(user))

    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=300,
        )
        if r.status_code != 200:
            raise PipelineError(f"llm_step HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        msg = body["choices"][0]["message"]
        content = msg.get("content", "") or ""
        # Thinking-mode fallback: if content is empty but reasoning_content has
        # the JSON answer (model used all tokens on CoT and ran out), scan
        # reasoning for the last JSON block.
        if not content.strip() and msg.get("reasoning_content"):
            reasoning = msg["reasoning_content"]
            # Look for a complete JSON object in reasoning
            import re
            matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", reasoning, re.DOTALL)
            # Try from longest to shortest; shallow first
            for m in sorted(matches, key=len, reverse=True):
                try:
                    json.loads(m)
                    log.info("  llm_step: extracted JSON from reasoning_content (%d chars)", len(m))
                    content = m
                    break
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as exc:
        raise PipelineError(f"llm_step request failed: {exc}")

    log.info("  llm_step: got %d chars (content)", len(content))

    if parse_json:
        # Try direct parse, else scan for JSON object
        txt = content.strip()
        # Strip common wrappers
        if txt.startswith("```json"):
            txt = txt[len("```json"):].strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
        if txt.startswith("```"):
            txt = txt[3:].strip()
        # Find first '{' and matching '}'
        first = txt.find("{")
        last = txt.rfind("}")
        if first >= 0 and last > first:
            txt = txt[first:last + 1]
        try:
            return json.loads(txt)
        except json.JSONDecodeError as exc:
            log.error("llm_step JSON parse failed: %s", exc)
            log.error("raw content (500 char): %s", content[:500])
            raise PipelineError(f"llm_step output not valid JSON: {exc}")
    return {"raw": content}


# ── Tool invocation helpers ────────────────────────────────────────────────

def invoke_shim(tool_name: str, inputs: dict) -> Optional[dict]:
    """Invoke a single shim by name; return {artifacts, data, duration_s} or None."""
    tool = registry._tools.get(tool_name)
    if tool is None:
        log.warning("  tool %s not registered", tool_name)
        return None
    try:
        result = tool.execute(inputs)
        if getattr(result, "success", False):
            return {
                "tool": tool_name,
                "data": result.data,
                "artifacts": list(result.artifacts or []),
                "duration_s": result.duration_seconds,
                "model": result.model,
            }
        err = getattr(result, "error", "unknown")
        log.warning("  %s → failed: %s", tool_name, str(err)[:200])
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("  %s → raised %s: %s", tool_name, type(exc).__name__, exc)
        return None


def try_first_working(tool_names: list[str], inputs: dict) -> Optional[dict]:
    """Try tools in order, return first successful result."""
    for name in tool_names:
        log.info("    trying %s", name)
        result = invoke_shim(name, inputs)
        if result:
            log.info("    ✓ %s (%.1fs, %d artifacts)", name, result["duration_s"],
                     len(result["artifacts"]))
            return result
    return None


# ── Stage executor ─────────────────────────────────────────────────────────

def run_stage(stage: dict, *, brief: str, character: str, project: str,
              state: dict) -> Optional[dict]:
    """Execute one stage. Returns stage output dict or None."""
    name = stage["name"]
    required = stage.get("required", True)
    log.info("=== stage: %s ===", name)

    # 1. LLM step
    if stage.get("llm_step"):
        try:
            parsed = call_llm_step(stage, brief=brief, state=state)
            output_key = stage["llm_step"].get("output_key", "llm_output")
            return {output_key: parsed}
        except PipelineError as exc:
            if required:
                raise
            log.warning("  (optional) llm_step failed: %s", exc)
            return None

    # Determine if stage needs a GPU-mode lease; hold for entire stage run
    stage_mode = _mode_for_stage(stage)

    # 2. Iteration over prior stage output
    iterate_over = stage.get("iterate_over")
    if iterate_over:
        # iterate_over like "narration_script.panels"
        parts = iterate_over.split(".")
        data: Any = state["stage_outputs"]
        for p in parts:
            # Walk through stage_outputs looking for the key
            if p in data:
                data = data[p]
            else:
                # Walk inside each stage's output dict
                found = False
                for so in data.values() if isinstance(data, dict) else []:
                    if isinstance(so, dict) and p in so:
                        data = so[p]
                        found = True
                        break
                if not found:
                    if required:
                        raise PipelineError(f"iterate_over path '{iterate_over}' not found; have {list((state['stage_outputs'] or {}).keys())}")
                    return None

        if not isinstance(data, list):
            if required:
                raise PipelineError(f"iterate_over '{iterate_over}' resolved to {type(data).__name__}, need list")
            return None

        iterate_key = stage.get("iterate_key", "prompt")
        tool_names = stage.get("tools_available") or []
        tool_inputs = stage.get("tool_inputs") or {}

        artifacts: list[str] = []
        per_item: list[dict] = []

        # Wrap iteration in a single mode lease if tools need one
        if stage_mode:
            log.info("  stage mode lease: %s (wrapping %d iterations)", stage_mode, len(data))
            lease_ctx = stargate_mode_lease(stage_mode, duration_minutes=60)
        else:
            from contextlib import nullcontext
            lease_ctx = nullcontext()

        with lease_ctx:
            for i, item in enumerate(data):
                item_input = item.get(iterate_key) if isinstance(item, dict) else str(item)
                inputs = {
                    "prompt": item_input, "text": item_input, "query": item_input,
                    "character": character, "project": project,
                    **tool_inputs,
                }
                log.info("  iter %d/%d: %s", i + 1, len(data), (item_input or "")[:60])
                result = try_first_working(tool_names, inputs)
                if result:
                    artifacts.extend(result["artifacts"])
                    per_item.append(result)
                else:
                    if required:
                        raise PipelineError(f"stage {name} iteration {i} failed")

        return {
            "artifacts": artifacts,
            "per_item": per_item,
            "produces": stage.get("produces") or [],
        }

    # 3. Single-shot tool-chain
    tools_available = stage.get("tools_available") or []
    if tools_available:
        inputs = {
            "prompt": brief, "text": brief, "query": brief,
            "url": brief if brief.startswith(("http://", "https://")) else "",
            "character": character, "project": project,
            **(stage.get("tool_inputs") or {}),
        }
        # Wrap single-shot in mode lease too if needed
        if stage_mode:
            with stargate_mode_lease(stage_mode, duration_minutes=30):
                result = try_first_working(tools_available, inputs)
        else:
            result = try_first_working(tools_available, inputs)
        if result is None and required:
            raise PipelineError(f"stage {name}: all {len(tools_available)} tools failed")
        return result

    # 4. Empty stage (compose handled elsewhere, publish is a marker)
    if stage.get("composition"):
        return {"composition": stage["composition"]}

    log.info("  (no tools / no llm — marker stage)")
    return {}


# ── Remotion compose ───────────────────────────────────────────────────────

def compose_short(pipeline: dict, state: dict, output_mp4: Path) -> bool:
    """For stargate_short: stitch 6 images + 6 audios via Remotion StargateShort."""
    stargate_cfg = pipeline.get("stargate") or {}
    composition = stargate_cfg.get("composition", "StargateShort")

    # Gather panel images + narration audios from iteration stages
    panel_images: list[str] = []
    narration_audios: list[str] = []

    pgen = state["stage_outputs"].get("panel_gen") or {}
    for item in pgen.get("per_item") or []:
        panel_images.extend(item.get("artifacts") or [])

    ngen = state["stage_outputs"].get("narration") or {}
    for item in ngen.get("per_item") or []:
        narration_audios.extend(item.get("artifacts") or [])

    if not panel_images:
        log.error("compose: no panel_images collected from panel_gen")
        return False

    # Panel metadata from script LLM output — key differs per pipeline
    script_out = state["stage_outputs"].get("script") or {}
    nscript = (script_out.get("narration_script")
               or script_out.get("panel_script")
               or {})
    panels_meta = nscript.get("panels") or []

    # Remotion's headless Chromium blocks file:/// URLs ("Not allowed to load local resource").
    # Stage assets into a run-specific public dir, reference them by relative path, and pass
    # --public-dir so staticFile() resolves from our staged dir.
    import shutil
    public_dir = output_mp4.parent / f".{output_mp4.stem}_assets"
    public_dir.mkdir(parents=True, exist_ok=True)

    staged_panels = []
    for i, img in enumerate(panel_images):
        src = Path(img)
        if not src.is_file():
            log.warning("compose: panel image missing: %s", img)
            continue
        dst = public_dir / f"panel_{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        staged_panels.append(dst.name)

    staged_audios = []
    for i, aud in enumerate(narration_audios):
        if not aud:
            staged_audios.append(None)
            continue
        src = Path(aud)
        if not src.is_file():
            staged_audios.append(None)
            continue
        dst = public_dir / f"narration_{i:02d}{src.suffix}"
        shutil.copy2(src, dst)
        staged_audios.append(dst.name)

    # Compose props — per-composition shape
    if composition == "Comic":
        props_data = {
            "panels": [
                {
                    "image_path": staged_panels[i],
                    "scene_description": (panels_meta[i].get("image_prompt", "")[:80]
                                          if i < len(panels_meta) else ""),
                    "dialogue": (panels_meta[i].get("dialogue") or []
                                 if i < len(panels_meta) else []),
                }
                for i in range(len(staged_panels))
            ],
            "title": pipeline.get("name", "comic"),
            "grid": "auto",
            "panel_duration_seconds": stargate_cfg.get("panel_duration_seconds", 4),
        }
    else:
        # StargateShort default
        props_data = {
            "panels": [
                {
                    "image_path": staged_panels[i] if i < len(staged_panels) else "",
                    "narration_text": (panels_meta[i].get("narration_text", "")
                                        if i < len(panels_meta) else ""),
                    "narration_audio_path": staged_audios[i] if i < len(staged_audios) else None,
                    "duration_seconds": stargate_cfg.get("panel_duration_seconds", 10),
                }
                for i in range(len(staged_panels))
            ],
            "title": pipeline.get("name", "stargate_short"),
            "fps": stargate_cfg.get("fps", 30),
        }

    remotion_dir = ROOT / "remotion-composer"
    props_file = output_mp4.parent / f"{output_mp4.stem}.props.json"
    props_file.write_text(json.dumps(props_data), encoding="utf-8")

    cmd = [
        "npx", "remotion", "render",
        str(remotion_dir / "src" / "index.tsx"),
        composition,
        str(output_mp4),
        f"--props={props_file}",
        f"--public-dir={public_dir}",
        "--log=info",
        "--concurrency=4",
    ]
    log.info("  remotion: %s", " ".join(cmd[:5]))
    # Prefer Node 22 via fnm
    env = {**__import__("os").environ}
    try:
        r = subprocess.run(cmd, cwd=str(remotion_dir), capture_output=True,
                           text=True, timeout=1800, env=env)
        if r.returncode != 0:
            log.error("Remotion stdout tail: %s", r.stdout[-300:])
            log.error("Remotion stderr tail: %s", r.stderr[-500:])
            return False
        log.info("  compose: %.1f MB", output_mp4.stat().st_size / 1e6 if output_mp4.is_file() else 0)
        return output_mp4.is_file()
    except subprocess.TimeoutExpired:
        log.error("remotion render timed out after 30 min")
        return False


# ── Output registration ───────────────────────────────────────────────────

def register_output(file_path: str, metadata: dict) -> None:
    try:
        r = requests.post(
            f"{AGENT_API_URL}/output/record",
            json={"file_path": file_path, "category": "productions", "metadata": metadata},
            timeout=10,
        )
        if r.status_code == 200:
            log.info("  ✓ registered in output_db (%s)", r.json().get("output_id"))
        else:
            log.warning("  /output/record: HTTP %s — %s", r.status_code, r.text[:200])
    except requests.RequestException as exc:
        log.warning("  /output/record unreachable (%s)", exc)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline")
    parser.add_argument("--brief", required=True)
    parser.add_argument("--character", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--output-dir", default="/home/edson/stargate/output/productions/")
    parser.add_argument("--num-panels", type=int, default=0, help="Override pipeline default")
    args = parser.parse_args()

    t_start = time.time()
    run_id = uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pipeline %s (run %s) brief=%r character=%s project=%s",
             args.pipeline, run_id, args.brief[:80], args.character, args.project)

    try:
        pipeline = load_pipeline(args.pipeline)
    except PipelineError as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1

    if args.num_panels > 0:
        pipeline.setdefault("stargate", {})["num_panels"] = args.num_panels

    state: dict[str, Any] = {"stage_outputs": {}}

    try:
        for stage in pipeline.get("stages") or []:
            try:
                out = run_stage(stage, brief=args.brief, character=args.character,
                                project=args.project, state=state)
                state["stage_outputs"][stage["name"]] = out or {}
            except PipelineError as exc:
                log.error("stage %s FATAL: %s", stage["name"], exc)
                print(json.dumps({
                    "success": False,
                    "pipeline": args.pipeline, "run_id": run_id,
                    "error": str(exc), "failed_stage": stage["name"],
                    "stage_outputs": {k: (v is not None) for k, v in state["stage_outputs"].items()},
                }))
                return 2

        # Compose
        final_mp4 = output_dir / f"{args.pipeline}_{run_id}.mp4"
        compose_ok = False
        if any(s.get("produces") == ["final_mp4"] or "compose" in s["name"] for s in pipeline["stages"]):
            compose_ok = compose_short(pipeline, state, final_mp4)

        # Sidecar
        metadata = {
            "pipeline": args.pipeline,
            "brief": args.brief,
            "character": args.character,
            "project": args.project,
            "stages_completed": list(state["stage_outputs"].keys()),
            "duration_s": time.time() - t_start,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "final_mp4_exists": compose_ok,
        }
        sidecar = output_dir / f"{args.pipeline}_{run_id}.meta.json"
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Register if compose succeeded
        if compose_ok and final_mp4.is_file():
            register_output(str(final_mp4), metadata)

        result = {
            "success": compose_ok,
            "pipeline": args.pipeline, "run_id": run_id,
            "output_path": str(final_mp4) if compose_ok else None,
            "sidecar_path": str(sidecar),
            "duration_s": round(time.time() - t_start, 1),
        }
        print(json.dumps(result))
        return 0 if compose_ok else 3

    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected: %s", exc)
        print(json.dumps({"success": False, "pipeline": args.pipeline, "run_id": run_id,
                          "error": f"{type(exc).__name__}: {exc}"}))
        return 4


if __name__ == "__main__":
    sys.exit(main())
