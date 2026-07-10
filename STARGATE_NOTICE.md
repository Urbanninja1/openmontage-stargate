# Stargate Fork — Lane 6 Provider Shims

This is `Urbanninja1/openmontage-stargate`, a fork of [`calesthio/OpenMontage`](https://github.com/calesthio/OpenMontage) (AGPL-3.0, retained).

## Fork-owned files (never synced from upstream)

These files are Stargate-local provider shims + helpers. Sync script `bin/openmontage-sync.sh` (in the Stargate repo) refuses to merge upstream changes that touch any `*_stargate*` file — none should exist upstream since `_stargate` is our reserved suffix.

### Shims (13)

**Audio (5):**
- `tools/audio/fish_stargate.py` — Fish S2-PRO SGLang-Omni (capability=tts)
- `tools/audio/kokoro_stargate.py` — Kokoro CPU (capability=tts) — potentially upstreamable as answer to issue #37
- `tools/audio/indextts_stargate.py` — IndexTTS-2 disentanglement (capability=tts)
- `tools/audio/ensemble_stargate.py` — VibeVoice-Large 4-speaker (capability=multi_speaker_tts) — no upstream equivalent
- `tools/audio/music_stargate.py` — ACE-Step local music generation (capability=music_generation) — no local upstream equivalent (upstream uses ElevenLabs / Suno APIs)

**Video (2):**
- `tools/video/comfy_wan_stargate.py` — ComfyUI Wan 2.2 Lightning T2V/I2V + SVI + Fun Control (capability=video_generation)
- `tools/video/comfy_hunyuan_stargate.py` — ComfyUI HunyuanVideo 1.5 for face-heavy (capability=video_generation)

**Graphics (1):**
- `tools/graphics/comfy_image_stargate.py` — ComfyUI FLUX 2 Dev + Klein 9B/4B (capability=image_generation)

**Analysis (3):**
- `tools/analysis/searxng_stargate.py` — SearXNG (capability=web_search); also registers a `WebSearchStargate` subclass under the bare name `web_search` to satisfy pipeline YAMLs that list the phantom upstream tool
- `tools/analysis/crawl4ai_stargate.py` — Crawl4AI (capability=url_extract)
- `tools/analysis/parakeet_stargate.py` — Parakeet v3 STT (capability=analysis) — auto-resamples to 16 kHz mono; fallback to upstream `transcriber` (faster-whisper)

**Avatar (2):**
- `tools/avatar/lip_sync_stargate.py` — wraps LatentSync workflow (capability=avatar) — drop-in for upstream `lip_sync` (wav2lip)
- `tools/avatar/talking_head_stargate.py` — wraps FLOAT + InfiniteTalk (capability=avatar) — drop-in for upstream `talking_head` (sadtalker); FLOAT requires ComfyUI-FLOAT patched for transformers 5.5 compat (Lane 6 fix shipped)

### Shared helpers (3)
- `tools/_stargate_character.py` — 14-character YAML loader + brief-text detection
- `tools/_stargate_comfy.py` — ComfyUI dispatch (resolve_workflow, render_placeholders, inject_lora_stack, submit_prompt, poll_until_done, download_artifacts, validate_output_nodes). GPU ownership remains entirely with Stargate's mode manager and the ComfyUI unit.
- `tools/_stargate_mode_lease.py` — compatibility context manager over the authoritative Agent API mode transition. It always restores `off` on exit.

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
