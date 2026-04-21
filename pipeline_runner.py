"""Stargate-local pipeline runner — minimal CLI for OpenMontage pipelines.

Fork-owned module (not in upstream). Executes pipeline_defs/<name>.yaml
stages in sequence against the tool_registry, with brief / character /
project passed through to each shim.

Usage:
    python -m pipeline_runner <pipeline_name> \\
        --brief "60-second explainer about sunscreen" \\
        [--character attenborough] \\
        [--project sunscreen-research] \\
        [--output-dir /home/edson/stargate/output/productions/] \\
        [--num-panels 3]  # for comic_stargate

Prints per-stage progress to stdout, writes final MP4/PNG to output dir,
posts to Agent API POST /output/record on success.

See docs/specs/openmontage.md §"Invocation patterns" for more.
"""

from __future__ import annotations

import argparse
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

# Structured log for operator visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline_runner")

# Ensure tools/ registry is discovered
sys.path.insert(0, str(ROOT))
from tools.tool_registry import registry  # noqa: E402
registry.discover()


class PipelineError(Exception):
    pass


def load_pipeline(name: str) -> dict:
    candidates = [PIPELINE_DIR / f"{name}.yaml", PIPELINE_DIR / f"{name}.yml"]
    for c in candidates:
        if c.is_file():
            with c.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
    raise PipelineError(f"pipeline '{name}' not found in {PIPELINE_DIR}")


def get_tools_for_capability(capability: str, allowlist: list[str] | None = None) -> list:
    """Return registered tools matching capability, filtered by allowlist."""
    tools = []
    for name, cls in registry._tools.items():
        if getattr(cls, "capability", "") != capability:
            continue
        if allowlist is not None and name not in allowlist:
            continue
        tools.append(cls)
    return tools


def try_tools_in_order(tools: list, inputs: dict, stage_name: str) -> Optional[dict]:
    """Try each tool in order; return first successful result dict.

    Each tool's execute returns a ToolResult; we treat .success=True as OK.
    """
    for tool_cls in tools:
        tool = tool_cls() if callable(tool_cls) else tool_cls
        log.info("  [%s] trying %s", stage_name, tool.name)
        try:
            result = tool.execute(inputs)
            if getattr(result, "success", False):
                log.info("    ✓ %s succeeded (%.1fs)", tool.name, getattr(result, "duration_seconds", 0))
                return {
                    "tool": tool.name,
                    "data": result.data,
                    "artifacts": result.artifacts,
                    "duration_s": result.duration_seconds,
                    "model": result.model,
                }
            err = getattr(result, "error", "unknown")
            log.warning("    ✗ %s failed: %s", tool.name, err[:200])
        except Exception as exc:  # noqa: BLE001
            log.warning("    ✗ %s raised %s: %s", tool.name, type(exc).__name__, exc)
    return None


def run_stage(stage: dict, *, brief: str, character: str, project: str,
              config: dict, state: dict) -> dict:
    """Execute one pipeline stage. Returns {produces_key: value}."""
    stage_name = stage["name"]
    produces = stage.get("produces") or []
    tools_available = stage.get("tools_available") or []
    required = stage.get("required", True)

    log.info("=== stage: %s ===", stage_name)

    if not tools_available:
        log.info("  (no tools declared; stage is a placeholder)")
        return {}

    # Map tool names to capabilities via registry
    allowlist_providers = config.get("providers", {}).get("allowlist", {})
    capabilities_by_tool = {}
    for tn in tools_available:
        tool_cls = registry._tools.get(tn)
        if tool_cls:
            capabilities_by_tool[tn] = getattr(tool_cls, "capability", "")

    # Group tools by capability
    by_cap: dict[str, list] = {}
    for tn, cap in capabilities_by_tool.items():
        by_cap.setdefault(cap, []).append(registry._tools[tn])

    # For each capability group, try tools in declared order
    stage_outputs: dict[str, Any] = {}
    for cap, tool_list in by_cap.items():
        inputs = {
            "query": brief,       # research/web_search
            "prompt": brief,      # image_gen / video_gen
            "text": brief,        # tts (fallback when no script)
            "url": brief if brief.startswith("http") else "",  # url_extract
            "character": character,
            "project": project,
        }
        # Hand off any state from prior stages (e.g. panel_script to panel_gen)
        inputs.update(state.get("stage_inputs", {}).get(stage_name, {}))

        result = try_tools_in_order(tool_list, inputs, stage_name)
        if result:
            stage_outputs[cap] = result

    # Map capability results to produces keys
    if produces:
        primary = produces[0]
        # Take whichever capability produced artifacts first
        for cap, result in stage_outputs.items():
            if result.get("artifacts"):
                stage_outputs[primary] = result
                break
            stage_outputs.setdefault(primary, result)

    if required and not stage_outputs and tools_available:
        raise PipelineError(f"stage '{stage_name}' produced nothing with tools {tools_available}")

    return stage_outputs


def compose_with_remotion(composition: str, props: dict, output_mp4: Path) -> None:
    """Invoke Remotion renderMedia via npx."""
    remotion_dir = ROOT / "remotion-composer"
    props_file = output_mp4.parent / f"{output_mp4.stem}.props.json"
    props_file.write_text(json.dumps(props), encoding="utf-8")

    cmd = [
        "npx", "remotion", "render",
        str(remotion_dir / "src" / "index.tsx"),
        composition,
        str(output_mp4),
        f"--props={props_file}",
        "--log=warn",
    ]
    log.info("  remotion: %s", " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=str(remotion_dir), capture_output=True,
                           text=True, timeout=1800)
        if r.returncode != 0:
            log.error("Remotion error: %s", r.stderr[-500:])
            raise PipelineError(f"remotion render failed: {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        raise PipelineError("remotion render timeout (30 min)")


def register_output(file_path: str, category: str, metadata: dict) -> None:
    """POST to Agent API /output/record. Best-effort — logs failure, doesn't raise."""
    try:
        r = requests.post(
            f"{AGENT_API_URL}/output/record",
            json={"file_path": file_path, "category": category, "metadata": metadata},
            timeout=10,
        )
        if r.status_code == 200:
            body = r.json()
            log.info("  output registered: id=%s", body.get("output_id"))
        else:
            log.warning("  /output/record returned %s: %s", r.status_code, r.text[:200])
    except requests.RequestException as exc:
        log.warning("  /output/record unreachable (%s); skipping registration", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stargate OpenMontage pipeline runner")
    parser.add_argument("pipeline", help="Pipeline name (without .yaml)")
    parser.add_argument("--brief", required=True, help="Brief / user request text")
    parser.add_argument("--character", default="", help="Stargate character id")
    parser.add_argument("--project", default="", help="Stargate project slug")
    parser.add_argument("--output-dir", default="/home/edson/stargate/output/productions/")
    parser.add_argument("--num-panels", type=int, default=3, help="Comic panels count")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    t_start = time.time()
    run_id = uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pipeline %s (run %s) — brief=%r", args.pipeline, run_id, args.brief[:100])

    try:
        pipeline = load_pipeline(args.pipeline)
        config = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).is_file() else {}
    except Exception as exc:
        log.error("failed to load: %s", exc)
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1

    state: dict[str, Any] = {"stage_inputs": {}, "stage_outputs": {}}
    try:
        for stage in pipeline.get("stages") or []:
            outputs = run_stage(
                stage, brief=args.brief, character=args.character, project=args.project,
                config=config, state=state,
            )
            state["stage_outputs"][stage["name"]] = outputs
            # Propagate key artifacts to downstream stages
            if "panel_images" in outputs or any("artifacts" in (o or {}) for o in outputs.values()):
                pass  # further state threading could be added here

        # Compose final artifact if pipeline has a `compose` stage
        compose_stage = next((s for s in (pipeline.get("stages") or []) if s["name"] == "compose"), None)
        final_path: Optional[Path] = None
        if compose_stage and pipeline.get("stargate", {}).get("composition"):
            composition = pipeline["stargate"]["composition"]
            panel_images = []
            panel_gen = state["stage_outputs"].get("panel_gen") or {}
            # Gather panel artifacts in order
            for cap_result in panel_gen.values():
                if isinstance(cap_result, dict) and cap_result.get("artifacts"):
                    panel_images.extend(cap_result["artifacts"])

            if not panel_images:
                # Fallback: 3 blank panels so we still emit something
                panel_images = ["placeholder.png"] * args.num_panels

            props = {
                "panels": [
                    {"image_path": p, "scene_description": f"panel {i+1}",
                     "dialogue": []}
                    for i, p in enumerate(panel_images[: args.num_panels])
                ],
                "title": args.pipeline,
                "panel_duration_seconds": 3,
                "grid": "auto",
            }
            final_path = output_dir / f"{args.pipeline}_{run_id}.mp4"
            try:
                compose_with_remotion(composition, props, final_path)
            except PipelineError as exc:
                log.warning("compose stage failed: %s — emitting sidecar only", exc)
                final_path = None

        # Sidecar metadata
        metadata = {
            "pipeline": args.pipeline,
            "brief": args.brief,
            "character": args.character,
            "project": args.project,
            "stages": list(state["stage_outputs"].keys()),
            "duration_s": time.time() - t_start,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        sidecar_path = output_dir / f"{args.pipeline}_{run_id}.meta.json"
        sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Register via Agent API if we produced a final MP4
        if final_path and final_path.is_file() and \
                config.get("output", {}).get("auto_register", True):
            register_output(str(final_path), "productions", metadata)

        result = {
            "success": True,
            "pipeline": args.pipeline,
            "run_id": run_id,
            "output_path": str(final_path) if final_path else None,
            "sidecar_path": str(sidecar_path),
            "duration_s": time.time() - t_start,
        }
        # Last line is parsable JSON for OpenClaw subprocess
        print(json.dumps(result))
        return 0

    except PipelineError as exc:
        log.error("pipeline failed: %s", exc)
        print(json.dumps({"success": False, "error": str(exc),
                          "pipeline": args.pipeline, "run_id": run_id}))
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("unexpected: %s", exc)
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}",
                          "pipeline": args.pipeline, "run_id": run_id}))
        return 3


if __name__ == "__main__":
    sys.exit(main())
