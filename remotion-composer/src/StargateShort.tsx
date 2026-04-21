// Stargate-local — 60-second animated-short composition.
// Slideshow with ken-burns motion, per-panel narration audio + caption strip.
// Fork-owned; no upstream equivalent. Used by pipeline_defs/stargate_short.yaml.

import React from "react";
import {
  AbsoluteFill, Audio, Img, interpolate, Sequence, staticFile,
  useCurrentFrame, useVideoConfig,
} from "remotion";

export interface StargateShortPanel {
  image_path: string;                  // absolute path to FLUX-generated PNG
  narration_text: string;              // caption shown over panel
  narration_audio_path?: string | null; // absolute path to Kokoro/Fish WAV
  duration_seconds: number;            // how long this panel is on screen
}

export interface StargateShortProps {
  [key: string]: unknown;
  panels: StargateShortPanel[];
  title?: string;
  fps?: number;
}

const DEFAULT_PANEL_DURATION = 10;

export const StargateShort: React.FC<StargateShortProps> = (props) => {
  const { fps } = useVideoConfig();
  const panels = props.panels || [];

  let cumulativeFrames = 0;

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {panels.map((panel, i) => {
        const duration = (panel.duration_seconds ?? DEFAULT_PANEL_DURATION) * fps;
        const from = cumulativeFrames;
        cumulativeFrames += duration;
        return (
          <Sequence key={i} from={from} durationInFrames={duration}>
            <PanelSlide panel={panel} panelIndex={i} />
          </Sequence>
        );
      })}
      {props.title && <TitleBar title={props.title} />}
    </AbsoluteFill>
  );
};

const PanelSlide: React.FC<{ panel: StargateShortPanel; panelIndex: number }> = ({
  panel, panelIndex,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const totalFrames = (panel.duration_seconds ?? DEFAULT_PANEL_DURATION) * fps;

  // Ken-burns: gentle zoom + pan, alternating direction
  const zoomDir = panelIndex % 2 === 0 ? 1 : -1;
  const scale = interpolate(frame, [0, totalFrames], [1.0, 1.12 * (zoomDir > 0 ? 1 : 1)], {
    extrapolateRight: "clamp",
  });
  const translateX = interpolate(frame, [0, totalFrames], [0, 60 * zoomDir], {
    extrapolateRight: "clamp",
  });

  // Fade in/out across panel boundary
  const opacity = interpolate(
    frame,
    [0, Math.min(12, totalFrames / 6), totalFrames - Math.min(12, totalFrames / 6), totalFrames],
    [0, 1, 1, 0],
    { extrapolateRight: "clamp" },
  );

  // Resolve image path: pipeline_runner stages relative names into --public-dir;
  // fallback to absolute/http for backwards compat.
  let src = panel.image_path;
  if (src) {
    if (src.startsWith("http://") || src.startsWith("https://")) {
      // leave as-is
    } else if (src.startsWith("/")) {
      src = "file://" + src;
    } else {
      try {
        src = staticFile(src);
      } catch {
        // Fall through — raw relative path
      }
    }
  }

  return (
    <AbsoluteFill style={{ opacity }}>
      {src ? (
        <Img
          src={src}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: `scale(${scale}) translateX(${translateX}px)`,
          }}
        />
      ) : (
        <div style={{ background: "#222", width: "100%", height: "100%" }} />
      )}

      {/* Caption strip */}
      {panel.narration_text && (
        <div style={{
          position: "absolute", bottom: 80, left: "8%", right: "8%",
          padding: "24px 32px",
          background: "rgba(0, 0, 0, 0.65)",
          borderLeft: "4px solid #F59E0B",
          color: "#fff",
          fontFamily: "Inter, sans-serif",
          fontSize: 32,
          lineHeight: 1.35,
          letterSpacing: "-0.01em",
          fontWeight: 500,
        }}>
          {panel.narration_text}
        </div>
      )}

      {/* Audio — resolve relative names via staticFile */}
      {panel.narration_audio_path && (() => {
        let audioSrc = panel.narration_audio_path;
        if (audioSrc.startsWith("/")) {
          audioSrc = "file://" + audioSrc;
        } else if (!audioSrc.startsWith("http://") && !audioSrc.startsWith("https://")) {
          try {
            audioSrc = staticFile(audioSrc);
          } catch {
            // pass through
          }
        }
        return <Audio src={audioSrc} volume={1.0} />;
      })()}
    </AbsoluteFill>
  );
};

const TitleBar: React.FC<{ title: string }> = ({ title }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 30, 90, 120], [1, 1, 1, 0], {
    extrapolateRight: "clamp",
  });
  return (
    <div style={{
      position: "absolute", top: 32, left: 32,
      padding: "12px 20px",
      background: "rgba(255, 255, 255, 0.08)",
      color: "#fff",
      fontFamily: "'IBM Plex Mono', monospace",
      fontSize: 18,
      letterSpacing: "0.1em",
      opacity,
      textTransform: "uppercase",
      border: "1px solid rgba(255, 255, 255, 0.18)",
    }}>
      {title}
    </div>
  );
};
