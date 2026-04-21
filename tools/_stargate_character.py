"""Stargate character-system integration for OpenMontage shims.

Fork-owned helper module — never synced from upstream.

Reads character YAMLs at ~/stargate/data/characters/<id>.yaml directly
(file-based, no Python import of Stargate core — license/architectural
clean-cut). Provides:
  - load_character(id) -> CharacterBinding | None
  - detect_character_in_brief(brief) -> str | None    (fuzzy match)

Used by:
  - fish_stargate, kokoro_stargate, indextts_stargate → voice reference
    path + emotion ref path + sampling override
  - ensemble_stargate → multi-speaker scene assembly
  - comfy_image_stargate → character LoRA path routing
  - script-stage (via LLM system prompt injection) → persona card +
    character canon RAG collection

See docs/specs/openmontage.md §"Stargate shim template" for usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None  # graceful degradation; shims detect + skip character binding


CHARACTERS_ROOT = Path(
    os.environ.get("STARGATE_CHARACTERS_DIR", "/home/edson/stargate/data/characters")
)


@dataclass
class VoiceBinding:
    """Single-engine voice binding for a character."""
    engine: str                           # "fish-s2" | "kokoro" | "indextts-2" | "voxcpm2"
    reference_path: Optional[str] = None
    emotion_ref_path: Optional[str] = None
    voice_id: Optional[str] = None        # Kokoro presets etc.
    extra: dict[str, Any] = None


@dataclass
class CharacterBinding:
    """Resolved character binding — superset of voice + persona + retrieval."""
    id: str
    display_name: str
    primary_voice_engine: str
    voice_bindings: dict[str, VoiceBinding]    # {engine: VoiceBinding}
    persona_card_path: Optional[str] = None
    lorebook_path: Optional[str] = None
    qdrant_collection: Optional[str] = None
    sampling: dict[str, Any] = None             # temp, min_p, dry, xtc
    model: Optional[str] = None
    tags: list[str] = None


@lru_cache(maxsize=32)
def load_character(char_id: str) -> Optional[CharacterBinding]:
    """Load character YAML + ancillary files into a CharacterBinding.

    Returns None if char_id not found or yaml unavailable.
    Cached (LRU 32) to avoid re-reading per shim call.
    """
    if yaml is None:
        return None

    yaml_path = CHARACTERS_ROOT / f"{char_id}.yaml"
    if not yaml_path.is_file():
        return None

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if raw.get("disabled"):
        return None

    # Voice bindings — per-engine
    voice_bindings: dict[str, VoiceBinding] = {}
    for engine, binding in (raw.get("voice_bindings") or {}).items():
        if isinstance(binding, dict):
            voice_bindings[engine] = VoiceBinding(
                engine=engine,
                reference_path=binding.get("reference_path"),
                emotion_ref_path=binding.get("emotion_ref_path"),
                voice_id=binding.get("voice_id"),
                extra={k: v for k, v in binding.items() if k not in {
                    "reference_path", "emotion_ref_path", "voice_id"
                }},
            )

    retrieval = raw.get("retrieval") or {}

    # Ancillary file paths (relative to characters dir)
    char_dir = CHARACTERS_ROOT / char_id
    persona_card = char_dir / "card.json"
    lorebook = char_dir / "lorebook.json"

    return CharacterBinding(
        id=char_id,
        display_name=raw.get("display_name", char_id),
        primary_voice_engine=raw.get("primary_voice_engine") or (
            (raw.get("primary_voice") or {}).get("engine", "kokoro")
        ),
        voice_bindings=voice_bindings,
        persona_card_path=str(persona_card) if persona_card.is_file() else None,
        lorebook_path=str(lorebook) if lorebook.is_file() else None,
        qdrant_collection=retrieval.get("qdrant_collection"),
        sampling=raw.get("sampling"),
        model=raw.get("model"),
        tags=raw.get("tags") or [],
    )


def detect_character_in_brief(brief: str) -> Optional[str]:
    """Fuzzy-match character name in brief text. Returns char_id or None.

    Strategy: scan all non-disabled characters; match display_name (case-
    insensitive substring) or explicit tags (length > 4 to avoid false
    positives). Returns first match.
    """
    if not brief:
        return None

    brief_lower = brief.lower()

    if not CHARACTERS_ROOT.is_dir():
        return None

    for yaml_path in sorted(CHARACTERS_ROOT.glob("*.yaml")):
        char_id = yaml_path.stem
        binding = load_character(char_id)
        if binding is None:
            continue
        if binding.display_name.lower() in brief_lower:
            return char_id
        for tag in binding.tags or []:
            if len(tag) > 4 and tag.lower() in brief_lower:
                return char_id

    return None


def voice_binding_for(char_id: str, engine_hint: str) -> Optional[VoiceBinding]:
    """Convenience: get the VoiceBinding for a specific engine preference.

    Falls back to primary_voice_engine if engine_hint not bound.
    """
    char = load_character(char_id)
    if char is None:
        return None
    if engine_hint in char.voice_bindings:
        return char.voice_bindings[engine_hint]
    return char.voice_bindings.get(char.primary_voice_engine)
