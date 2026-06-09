import React from "react";
import {
  AbsoluteFill, Sequence, Img, Audio, staticFile,
  useCurrentFrame, useVideoConfig, interpolate, spring, Easing,
} from "remotion";
import { SEGMENTS } from "./segments.js";

export const WIDTH = 1920;
export const HEIGHT = 1080;
export const FPS = 30;
export const TOTAL_FRAMES = SEGMENTS.reduce((a, s) => a + s.durationInFrames, 0);

const FONT =
  '"Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const BG_FROM = "#0b1220";
const BG_TO = "#0f1b2e";

const Background = () => (
  <AbsoluteFill style={{ background: `radial-gradient(1200px 700px at 70% 0%, #14253f 0%, ${BG_TO} 45%, ${BG_FROM} 100%)` }}>
    <AbsoluteFill style={{ background: "radial-gradient(900px 500px at 12% 95%, rgba(52,211,153,0.10), transparent 60%)" }} />
  </AbsoluteFill>
);

const fadeInOut = (frame, dur, edge = 12) =>
  interpolate(frame, [0, edge, dur - edge, dur], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

const BrowserWindow = ({ children, accent }) => (
  <div style={{
    width: 1380, borderRadius: 16, overflow: "hidden",
    boxShadow: "0 40px 90px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)",
    background: "#0d1526",
  }}>
    <div style={{ height: 46, display: "flex", alignItems: "center", gap: 9, padding: "0 18px",
      background: "linear-gradient(#1b2740,#141f33)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
      <span style={{ width: 13, height: 13, borderRadius: 99, background: "#ff5f57" }} />
      <span style={{ width: 13, height: 13, borderRadius: 99, background: "#febc2e" }} />
      <span style={{ width: 13, height: 13, borderRadius: 99, background: "#28c840" }} />
      <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
        <div style={{ background: "rgba(255,255,255,0.07)", color: "#9fb3c8", fontFamily: FONT,
          fontSize: 18, padding: "6px 18px", borderRadius: 8, display: "flex", gap: 8 }}>
          <span style={{ color: accent }}>🔒</span> parselyfi.streamlit.app
        </div>
      </div>
    </div>
    <div style={{ height: 800, overflow: "hidden", position: "relative" }}>{children}</div>
  </div>
);

const KenBurns = ({ src, progress }) => {
  const scale = 1.0 + 0.07 * progress;
  const ty = interpolate(progress, [0, 1], [0, -26]);
  return (
    <Img src={staticFile(src)} style={{
      position: "absolute", top: 0, left: 0, width: 1380, height: "auto",
      transform: `scale(${scale}) translateY(${ty}px)`, transformOrigin: "top center",
    }} />
  );
};

const LowerThird = ({ step, accent, title, subtitle, frame }) => {
  const slide = interpolate(frame, [4, 22], [40, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const op = interpolate(frame, [4, 22], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", padding: "0 0 60px 0" }}>
      <div style={{ background: "linear-gradient(transparent, rgba(7,12,22,0.92) 55%)", height: 320,
        position: "absolute", bottom: 0, left: 0, right: 0 }} />
      <div style={{ position: "relative", transform: `translateY(${slide}px)`, opacity: op,
        display: "flex", alignItems: "center", gap: 28, padding: "0 90px" }}>
        {step ? (
          <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 26, color: accent,
            border: `2px solid ${accent}`, borderRadius: 10, padding: "8px 14px", letterSpacing: 1 }}>
            {step} <span style={{ color: "#5b7290" }}>/ 03</span>
          </div>
        ) : null}
        <div style={{ width: 5, height: 92, background: accent, borderRadius: 9 }} />
        <div>
          <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 58, color: "#eaf2ff", lineHeight: 1.05 }}>{title}</div>
          <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 30, color: "#9fb3c8", marginTop: 8 }}>{subtitle}</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const FeatureSlide = ({ seg }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const progress = interpolate(frame, [0, durationInFrames], [0, 1], { extrapolateRight: "clamp" });
  const imgs = seg.images || [];
  const showSecond = imgs.length > 1
    ? interpolate(frame, [durationInFrames * 0.5, durationInFrames * 0.62], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : 0;
  return (
    <AbsoluteFill style={{ opacity: fadeInOut(frame, durationInFrames) }}>
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "flex-start", paddingTop: 44 }}>
        <BrowserWindow accent={seg.accent}>
          {imgs[0] && <div style={{ position: "absolute", inset: 0, opacity: 1 - showSecond }}><KenBurns src={imgs[0]} progress={progress} /></div>}
          {imgs[1] && <div style={{ position: "absolute", inset: 0, opacity: showSecond }}><KenBurns src={imgs[1]} progress={progress} /></div>}
        </BrowserWindow>
      </AbsoluteFill>
      <LowerThird step={seg.step} accent={seg.accent} title={seg.title} subtitle={seg.subtitle} frame={frame} />
    </AbsoluteFill>
  );
};

const Logo = ({ size = 120 }) => (
  <span style={{ fontSize: size }}>🌱</span>
);

const IntroCard = ({ seg }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const pop = spring({ frame, fps, config: { damping: 200 } });
  const op = fadeInOut(frame, durationInFrames, 14);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", opacity: op }}>
      <div style={{ transform: `scale(${interpolate(pop, [0, 1], [0.86, 1])})`, textAlign: "center" }}>
        <Logo size={130} />
        <div style={{ fontFamily: FONT, fontWeight: 900, fontSize: 132, color: "#eaf2ff", letterSpacing: -2, marginTop: 4 }}>
          Parsely<span style={{ color: "#34d399" }}>Fi</span>
        </div>
        <div style={{ height: 6, width: interpolate(pop, [0, 1], [0, 360]), background: "#34d399",
          borderRadius: 9, margin: "18px auto 0" }} />
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 38, color: "#9fb3c8", marginTop: 28 }}>{seg.subtitle}</div>
        <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 24, color: "#5b7290", marginTop: 26, display: "flex", gap: 26, justifyContent: "center" }}>
          <span style={{ color: "#8E75B2" }}>Gemini 3.5 Flash</span>
          <span style={{ color: "#0A7CFF" }}>LinkUp sources</span>
          <span style={{ color: "#34d399" }}>ElevenLabs</span>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const OutroCard = ({ seg }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const pop = spring({ frame, fps, config: { damping: 200 } });
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", opacity: fadeInOut(frame, durationInFrames, 14) }}>
      <div style={{ textAlign: "center", transform: `scale(${interpolate(pop, [0, 1], [0.9, 1])})` }}>
        <Logo size={96} />
        <div style={{ fontFamily: FONT, fontWeight: 900, fontSize: 92, color: "#eaf2ff", marginTop: 6 }}>
          Parsely<span style={{ color: "#34d399" }}>Fi</span>
        </div>
        <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 34, color: "#34d399", marginTop: 18 }}>{seg.subtitle}</div>
        <div style={{ fontFamily: FONT, fontWeight: 500, fontSize: 26, color: "#9fb3c8", marginTop: 14 }}>github.com/HomenShum/Parselyfi</div>
      </div>
    </AbsoluteFill>
  );
};

const ProgressBar = () => {
  const frame = useCurrentFrame();
  const w = interpolate(frame, [0, TOTAL_FRAMES], [0, WIDTH], { extrapolateRight: "clamp" });
  return <div style={{ position: "absolute", bottom: 0, left: 0, height: 6, width: w, background: "#34d399" }} />;
};

// Standalone single-feature loop for the per-feature README previews.
export const FeatureClip = ({ seg }) => (
  <AbsoluteFill style={{ background: BG_FROM }}>
    <Background />
    <FeatureSlide seg={seg} />
  </AbsoluteFill>
);

export const Demo = () => {
  let from = 0;
  return (
    <AbsoluteFill style={{ background: BG_FROM }}>
      <Background />
      {SEGMENTS.map((seg, i) => {
        const at = from;
        from += seg.durationInFrames;
        return (
          <Sequence key={i} from={at} durationInFrames={seg.durationInFrames} name={seg.title}>
            {seg.narration ? <Audio src={staticFile(seg.narration)} /> : null}
            {seg.kind === "intro" && <IntroCard seg={seg} />}
            {seg.kind === "feature" && <FeatureSlide seg={seg} />}
            {seg.kind === "outro" && <OutroCard seg={seg} />}
          </Sequence>
        );
      })}
      <ProgressBar />
    </AbsoluteFill>
  );
};
