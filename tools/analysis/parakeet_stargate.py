"""Stargate Parakeet STT provider — CPU ONNX transcription.

Fork-owned module — never synced from upstream.

Routes OpenMontage `analysis` capability transcription calls to Stargate's
Parakeet v3 STT service (NVIDIA `nemo-parakeet-tdt-0.6b-v3`, CPU ONNX)
on http://127.0.0.1:8090. Produces segment-level JSON compatible with
`subtitle_gen.py` (OpenMontage's native SRT/VTT generator).

Why Parakeet over Whisper for Stargate:
  - Runs CPU-only (no GPU contention with image/video mode)
  - 25-language auto-detect
  - ~0.05 real-time factor on Threadripper Pro (20x real-time)
  - Always-on systemd unit (`stargate-parakeet.service` on :8090)

Fallback: upstream `transcriber` (faster-whisper) if Parakeet :8090 down.

Output shape: segments[] with {text, start, end} — does NOT provide
word-level timestamps. For word-level timing (SRT karaoke), use upstream
transcriber with model_size=large-v3.

Lane 6 §Q5 — closes subtitle gap for shimmed OpenMontage pipelines.
See docs/solutions/integration-issues/2026-04-21-lane6-voice-references-missing.md
for unrelated voice reference gap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


PARAKEET_URL = "http://127.0.0.1:8090"
OUTPUT_DIR_DEFAULT = Path("/home/edson/stargate/output/productions/_subtitles")


class ParakeetStargate(BaseTool):
    name = "parakeet_stargate"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "analysis"
    provider = "stargate"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = []  # HTTP-only; Parakeet service runs via systemd
    install_instructions = (
        "Already provisioned on Stargate — systemd unit "
        "`stargate-parakeet.service` on :8090. No agent-side setup."
    )

    capabilities = [
        "transcribe",
        "language_detect",
        "multilingual_25",
        "offline_generation",
    ]
    supports = {
        "word_timestamps": False,     # Parakeet v3 emits segment-level only
        "diarization": False,         # use upstream transcriber + whisperx for diar
        "streaming": False,
        "multilingual": True,
    }
    best_for = [
        "fast CPU-only batch transcription (20x real-time)",
        "always-on subtitle generation without GPU mode switch",
        "25-language auto-detect when language unknown",
    ]
    not_good_for = [
        "word-by-word karaoke highlight (use transcriber + large-v3)",
        "speaker diarization (use pyannote via analysis/transcriber + whisperx)",
    ]

    quality_score = 0.88
    historical_success_rate = 0.97
    latency_p50_seconds = 5.0

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=1024, vram_mb=0, network_required=False)
    retry_policy = RetryPolicy(max_retries=2, backoff_seconds=2.0, retryable_errors=["timeout", "503"])

    input_schema = {
        "type": "object",
        "required": ["input_path"],
        "properties": {
            "input_path": {"type": "string", "description": "Path to audio or video file on host."},
            "language": {"type": "string", "default": "auto", "description": "ISO code or 'auto' for detect."},
            "model": {"type": "string", "default": "parakeet"},
            "output_dir": {"type": "string", "description": "Directory for transcript JSON."},
            "output_path": {"type": "string", "description": "Direct transcript JSON path."},
            "response_format": {"type": "string", "enum": ["json", "verbose_json"], "default": "verbose_json"},
            "operation": {"type": "string", "enum": ["generate", "rank"], "default": "generate"},
            "preferred_provider": {"type": "string"},
            "allowed_providers": {"type": "array", "items": {"type": "string"}},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "segments": {"type": "array", "description": "{text,start,end} triples."},
            "language": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "transcript_path": {"type": "string"},
            "model": {"type": "string"},
        },
    }

    fallback = "transcriber"
    fallback_tools = ["transcriber"]
    agent_skills = ["speech-to-text"]

    def get_status(self) -> ToolStatus:
        try:
            r = requests.get(f"{PARAKEET_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return ToolStatus.AVAILABLE
        except requests.RequestException:
            pass
        return ToolStatus.UNAVAILABLE

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 10.0  # ~20x real-time on Threadripper

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()

        if inputs.get("operation") == "rank":
            return ToolResult(
                success=True,
                data={"ranked": True, "quality_score": self.quality_score, "cost_usd": 0.0},
            )

        input_path = Path(inputs["input_path"])
        if not input_path.is_file():
            return ToolResult(success=False, error=f"input not found: {input_path}")

        language = inputs.get("language", "auto")
        response_format = inputs.get("response_format", "verbose_json")

        # Output path — write transcript JSON so subtitle_gen can consume segments.
        output_path = inputs.get("output_path")
        if not output_path:
            out_dir = Path(inputs.get("output_dir") or OUTPUT_DIR_DEFAULT)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(out_dir / f"{input_path.stem}.transcript.json")

        # Parakeet v3 requires 16kHz mono; auto-resample if needed.
        # Downstream Stargate TTS outputs are commonly 24kHz or 48kHz.
        upload_path = self._ensure_16k_mono(input_path)

        try:
            with open(upload_path, "rb") as fh:
                files = {"file": (input_path.name, fh, "audio/wav")}
                data = {
                    "model": inputs.get("model", "parakeet"),
                    "language": language,
                    "response_format": response_format,
                }
                r = requests.post(
                    f"{PARAKEET_URL}/v1/audio/transcriptions",
                    files=files,
                    data=data,
                    timeout=600,  # CPU transcription of ~10 min audio ≈ 30s
                )
            if r.status_code != 200:
                return ToolResult(
                    success=False,
                    error=f"Parakeet HTTP {r.status_code}: {r.text[:200]}",
                    duration_seconds=time.time() - t_start,
                )
            body = r.json()
        except requests.RequestException as exc:
            return ToolResult(
                success=False,
                error=f"Parakeet HTTP error: {exc}",
                duration_seconds=time.time() - t_start,
            )

        # verbose_json → {text, segments, language, duration}
        # json          → {text}
        segments = body.get("segments") or []
        if not segments and body.get("text"):
            # Non-verbose format — fabricate a single segment spanning the file.
            # subtitle_gen will chunk it by max_chars.
            duration = self._audio_duration(str(input_path))
            segments = [{"text": body["text"], "start": 0.0, "end": duration}]

        # Normalize segment shape — some implementations use "start_time"/"end_time".
        norm_segments = []
        for s in segments:
            text = s.get("text") or ""
            start = s.get("start") or s.get("start_time") or s.get("t0") or 0.0
            end = s.get("end") or s.get("end_time") or s.get("t1") or 0.0
            norm_segments.append({"text": str(text).strip(), "start": float(start), "end": float(end)})

        Path(output_path).write_text(
            json.dumps(
                {
                    "segments": norm_segments,
                    "language": body.get("language") or language,
                    "duration_seconds": body.get("duration") or self._audio_duration(str(input_path)),
                    "model": "nemo-parakeet-tdt-0.6b-v3",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return ToolResult(
            success=True,
            data={
                "segments": norm_segments,
                "language": body.get("language") or language,
                "duration_seconds": body.get("duration") or self._audio_duration(str(input_path)),
                "transcript_path": output_path,
                "model": "nemo-parakeet-tdt-0.6b-v3",
            },
            artifacts=[output_path],
            model="parakeet-v3",
            duration_seconds=time.time() - t_start,
        )

    @staticmethod
    def _audio_duration(path: str) -> float:
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=5,
            )
            return float(r.stdout.strip() or "0")
        except (subprocess.SubprocessError, ValueError):
            return 0.0

    @staticmethod
    def _ensure_16k_mono(input_path: Path) -> Path:
        """Parakeet v3 expects 16 kHz mono PCM. Resample if needed.

        Returns the original path if already 16 kHz mono, else a new path
        under /tmp/ with the resampled audio.
        """
        import subprocess
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "stream=sample_rate,channels",
                 "-of", "csv=p=0", str(input_path)],
                capture_output=True, text=True, timeout=5,
            )
            first = probe.stdout.strip().splitlines()[0] if probe.stdout.strip() else ""
            sr, ch = first.split(",") if "," in first else ("", "")
            if sr == "16000" and ch == "1":
                return input_path
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        # Resample to 16kHz mono
        import tempfile, uuid
        tmp_out = Path(tempfile.gettempdir()) / f"parakeet_{uuid.uuid4().hex[:8]}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(input_path), "-ar", "16000", "-ac", "1", str(tmp_out)],
            check=True, timeout=300,
        )
        return tmp_out
