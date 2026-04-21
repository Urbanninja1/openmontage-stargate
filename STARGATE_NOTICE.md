# Stargate Fork — Lane 6 Provider Shims

This is `Urbanninja1/openmontage-stargate`, a fork of [`calesthio/OpenMontage`](https://github.com/calesthio/OpenMontage) (AGPL-3.0, retained).

## Fork-owned files (never synced from upstream)

These files are Stargate-local provider shims + helpers. Sync script `bin/openmontage-sync.sh` (in the Stargate repo) refuses to merge upstream changes that touch any `*_stargate*` file — none should exist upstream since `_stargate` is our reserved suffix.

### Shims (10)
- `tools/audio/fish_stargate.py` — Fish S2-PRO SGLang-Omni (capability=tts)
- `tools/audio/kokoro_stargate.py` — Kokoro CPU (capability=tts) — potentially upstreamable as answer to issue #37
- `tools/audio/indextts_stargate.py` — IndexTTS-2 disentanglement (capability=tts)
- `tools/audio/ensemble_stargate.py` — VibeVoice-Large 4-speaker (capability=multi_speaker_tts) — no upstream equivalent
- `tools/video/comfy_wan_stargate.py` — ComfyUI Wan 2.2 (capability=video_gen)
- `tools/video/comfy_hunyuan_stargate.py` — ComfyUI HunyuanVideo 1.5 (capability=video_gen)
- `tools/graphics/comfy_image_stargate.py` — ComfyUI FLUX 2 + Klein (capability=image_gen)
- `tools/analysis/searxng_stargate.py` — SearXNG (capability=web_search)
- `tools/analysis/crawl4ai_stargate.py` — Crawl4AI (capability=url_extract)
- `tools/analysis/rag_platform_stargate.py` — Projects RAG (capability=research) — no upstream equivalent

### Shared helpers (3)
- `tools/_stargate_character.py` — 14-character YAML loader + brief-text detection
- `tools/_stargate_comfy.py` — ComfyUI dispatch (resolve_workflow, render_placeholders, inject_lora_stack, submit_prompt, poll_until_done)
- `tools/_stargate_mode_lease.py` — pipeline-scoped mode lease via RAG Platform

### Stargate-local pipeline
- `pipeline_defs/comic_stargate.yaml` — illustrated-story / multi-panel comic (no upstream equivalent)

### Config
- `config.yaml` — disables cloud providers, routes to Stargate local services
- `STARGATE_NOTICE.md` — this file

## AGPL-3.0 notice

The original calesthio/OpenMontage is AGPL-3.0. This fork preserves the LICENSE unchanged. We publish the fork source publicly per personal-use-hygiene (even though solo-user Tailscale-only deployment doesn't strictly trigger §13 distribution obligations).

Stargate's own source tree does NOT import any OpenMontage Python — OpenMontage is always invoked as a separate subprocess. This is architectural cleanliness, not a license boundary.

## Upstream watch

- **PR #29 (native ComfyUI provider)** — if merged, our `comfy_*_stargate.py` shims pivot to thin URL/quality-tier overrides.
- **Issue #37 (Kokoro)** — we plan to upstream `kokoro_stargate.py` as the answer.
- **Issue #33 (ComfyUI umbrella)** — resolved by PR #29.

## Sync SOP

```bash
cd /home/edson/stargate  # Stargate repo
bash bin/openmontage-sync.sh --dry-run     # preview
bash bin/openmontage-sync.sh               # merge
cd external/openmontage-src                # fork
git push origin stargate/main              # update public fork
```
