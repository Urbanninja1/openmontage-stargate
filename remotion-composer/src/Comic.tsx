// Stargate-local Comic composition — multi-panel comic with optional dialogue bubbles.
// Used by pipeline_defs/comic_stargate.yaml. NO upstream equivalent.
//
// Props include `panels: ComicPanel[]` with image paths + optional dialogue,
// and the composition renders a 3×1, 2×2, or N×N grid with fade-in animation
// per panel and dialogue bubbles overlaid.
//
// Fork-owned; never synced from upstream.

import React from "react";
import {
  AbsoluteFill,
  Img,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

export interface ComicDialogue {
  speaker: string; // character display name for speech-bubble label
  text: string;
  position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
}

export interface ComicPanel {
  image_path: string;       // absolute path OR staticFile-relative
  scene_description: string;
  character?: string;
  dialogue?: ComicDialogue[];
}

export interface ComicProps {
  // Index signature — required by Remotion's Composition generic constraint
  [key: string]: unknown;
  panels: ComicPanel[];
  title?: string;
  subtitle?: string;
  grid?: "1x3" | "3x1" | "2x2" | "auto";
  panel_duration_seconds?: number;    // each panel holds for this long (MP4 mode)
  background_color?: string;
  bubble_color?: string;
  bubble_text_color?: string;
  font_family?: string;
}

const DEFAULTS: Required<Pick<ComicProps,
  "grid" | "panel_duration_seconds" | "background_color" | "bubble_color" |
  "bubble_text_color" | "font_family">> = {
  grid: "auto",
  panel_duration_seconds: 3,
  background_color: "#111",
  bubble_color: "#fff",
  bubble_text_color: "#111",
  font_family: "'Comic Sans MS', 'Marker Felt', cursive",
};

function pickGrid(panels: number, grid: ComicProps["grid"]) {
  if (grid && grid !== "auto") return grid;
  if (panels === 1) return "1x3" as const;
  if (panels === 2) return "3x1" as const;
  if (panels === 3) return "3x1" as const;
  if (panels === 4) return "2x2" as const;
  return "2x2" as const;
}

export const Comic: React.FC<ComicProps> = (props) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const merged = { ...DEFAULTS, ...props };
  const grid = pickGrid(merged.panels.length, merged.grid);

  return (
    <AbsoluteFill style={{ background: merged.background_color }}>
      {merged.title ? (
        <div
          style={{
            position: "absolute",
            top: 40,
            left: 0,
            right: 0,
            textAlign: "center",
            color: "#fff",
            fontFamily: merged.font_family,
            fontSize: 64,
            fontWeight: 700,
            letterSpacing: "-0.02em",
          }}
        >
          {merged.title}
        </div>
      ) : null}

      {merged.subtitle ? (
        <div
          style={{
            position: "absolute",
            top: 120,
            left: 0,
            right: 0,
            textAlign: "center",
            color: "#bbb",
            fontFamily: merged.font_family,
            fontSize: 28,
          }}
        >
          {merged.subtitle}
        </div>
      ) : null}

      <div style={gridContainerStyle(grid, merged.title ? 180 : 60)}>
        {merged.panels.map((panel, i) => (
          <Sequence
            key={i}
            from={Math.round(i * fps * merged.panel_duration_seconds)}
            durationInFrames={Math.round(fps * merged.panel_duration_seconds)}
          >
            <div style={panelContainerStyle(grid, i)}>
              <PanelImage panel={panel} />
              {(panel.dialogue || []).map((d, j) => (
                <DialogueBubble
                  key={j}
                  dialogue={d}
                  fontFamily={merged.font_family}
                  bubbleColor={merged.bubble_color}
                  textColor={merged.bubble_text_color}
                />
              ))}
            </div>
          </Sequence>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const PanelImage: React.FC<{ panel: ComicPanel }> = ({ panel }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateRight: "clamp" });

  // Support file:// and staticFile-style paths
  let src = panel.image_path;
  if (!src.startsWith("http") && !src.startsWith("file://") && !src.startsWith("/")) {
    try {
      src = staticFile(src);
    } catch {
      // fall through with raw path
    }
  } else if (src.startsWith("/")) {
    src = "file://" + src;
  }

  return (
    <Img
      src={src}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        opacity,
        border: "6px solid #000",
      }}
    />
  );
};

const DialogueBubble: React.FC<{
  dialogue: ComicDialogue;
  fontFamily: string;
  bubbleColor: string;
  textColor: string;
}> = ({ dialogue, fontFamily, bubbleColor, textColor }) => {
  const positionStyle: React.CSSProperties = positionFor(dialogue.position ?? "bottom-left");

  return (
    <div style={{ ...positionStyle, maxWidth: "60%" }}>
      <div
        style={{
          background: bubbleColor,
          color: textColor,
          padding: "16px 20px",
          borderRadius: 20,
          border: "3px solid #000",
          fontFamily,
          fontSize: 22,
          lineHeight: 1.3,
          boxShadow: "4px 4px 0 rgba(0,0,0,0.85)",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 4, fontSize: 14, opacity: 0.65 }}>
          {dialogue.speaker}
        </div>
        <div>{dialogue.text}</div>
      </div>
    </div>
  );
};

function positionFor(pos: ComicDialogue["position"]): React.CSSProperties {
  const base = { position: "absolute" as const, zIndex: 2 };
  switch (pos) {
    case "top-left":
      return { ...base, top: 16, left: 16 };
    case "top-right":
      return { ...base, top: 16, right: 16 };
    case "bottom-right":
      return { ...base, bottom: 16, right: 16 };
    case "bottom-left":
    default:
      return { ...base, bottom: 16, left: 16 };
  }
}

function gridContainerStyle(grid: "1x3" | "3x1" | "2x2", topPad: number): React.CSSProperties {
  let gridTemplate = "";
  if (grid === "1x3") gridTemplate = "1fr 1fr 1fr / 1fr";
  if (grid === "3x1") gridTemplate = "1fr / 1fr 1fr 1fr";
  if (grid === "2x2") gridTemplate = "1fr 1fr / 1fr 1fr";
  return {
    position: "absolute",
    top: topPad,
    left: 40,
    right: 40,
    bottom: 40,
    display: "grid",
    grid: gridTemplate,
    gap: 20,
  };
}

function panelContainerStyle(grid: "1x3" | "3x1" | "2x2", i: number): React.CSSProperties {
  return {
    position: "relative",
    background: "#222",
    width: "100%",
    height: "100%",
    overflow: "hidden",
  };
}
