import React from "react";
import { AbsoluteFill, Img, staticFile, useCurrentFrame, interpolate, Easing } from "remotion";

export const WT_FPS = 30;
export const WT_W = 1920;
export const WT_H = 1080;

const FONT = '"Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const IMG_W = 1360;
const CAP_VW = 1280;                 // capture viewport CSS width
const IMG_H = Math.round(IMG_W * 800 / CAP_VW);   // preserve 1280x800 aspect
const SX = IMG_W / 1280, SY = IMG_H / 800;        // cursor coord -> displayed-image px

export const wtDuration = (wt) => wt.steps.reduce((a, s) => a + (s.hold || 60), 0);

const Background = () => (
  <AbsoluteFill style={{ background: "radial-gradient(1300px 760px at 68% -5%, #14253f 0%, #0f1b2e 46%, #0b1220 100%)" }}>
    <AbsoluteFill style={{ background: "radial-gradient(900px 520px at 10% 100%, rgba(52,211,153,0.10), transparent 60%)" }} />
  </AbsoluteFill>
);

// macOS-style arrow pointer.
const Pointer = ({ x, y, opacity }) => (
  <svg width="34" height="34" viewBox="0 0 24 24" style={{ position: "absolute", left: x, top: y, opacity, transform: "translate(-2px,-2px)", filter: "drop-shadow(0 3px 5px rgba(0,0,0,0.5))", zIndex: 30 }}>
    <path d="M5 3 L5 20 L9.5 15.5 L12.5 22 L15 21 L12 14.5 L18.5 14.5 Z" fill="#fff" stroke="#0b1220" strokeWidth="1.4" strokeLinejoin="round" />
  </svg>
);

const Ripple = ({ x, y, lf, accent }) => {
  const t = interpolate(lf, [20, 46], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const size = interpolate(t, [0, 1], [6, 84]);
  const op = interpolate(t, [0, 0.15, 1], [0, 0.85, 0]);
  return (
    <div style={{ position: "absolute", left: x - size / 2, top: y - size / 2, width: size, height: size, borderRadius: "50%", border: `3px solid ${accent}`, opacity: op, zIndex: 29 }} />
  );
};

const Chrome = ({ accent }) => (
  <div style={{ height: 44, display: "flex", alignItems: "center", gap: 9, padding: "0 18px", background: "linear-gradient(#1b2740,#141f33)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
    <span style={{ width: 12, height: 12, borderRadius: 99, background: "#ff5f57" }} />
    <span style={{ width: 12, height: 12, borderRadius: 99, background: "#febc2e" }} />
    <span style={{ width: 12, height: 12, borderRadius: 99, background: "#28c840" }} />
    <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
      <div style={{ background: "rgba(255,255,255,0.07)", color: "#9fb3c8", fontFamily: FONT, fontSize: 17, padding: "5px 16px", borderRadius: 8 }}>
        <span style={{ color: accent }}>🔒</span> parselyfi.streamlit.app
      </div>
    </div>
  </div>
);

export const Walkthrough = ({ wt }) => {
  const frame = useCurrentFrame();
  const steps = wt.steps || [];
  if (!steps.length) return <AbsoluteFill style={{ background: "#0b1220" }} />;

  const starts = [];
  let acc = 0;
  for (const s of steps) { starts.push(acc); acc += s.hold || 60; }
  let i = steps.findIndex((s, k) => frame >= starts[k] && frame < starts[k] + (steps[k].hold || 60));
  if (i < 0) i = steps.length - 1;
  const lf = frame - starts[i];
  const cur = steps[i];
  const prev = steps[i - 1];

  const sp = (p) => (p ? { x: p.x * SX, y: p.y * SY } : null);
  const curPos = sp(cur.cursor);
  const prevPos = sp(prev && prev.cursor);

  // Cursor glide from the previous target to this step's target.
  let cursor = null, cursorOp = 0;
  if (curPos) {
    const from = prevPos || curPos;
    const t = interpolate(lf, [0, 18], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
    cursor = { x: from.x + (curPos.x - from.x) * t, y: from.y + (curPos.y - from.y) * t };
    cursorOp = interpolate(lf, [0, 8], [prevPos ? 1 : 0, 1], { extrapolateRight: "clamp" });
  }

  // Image crossfade (prev under, current fading in).
  const fadeIn = interpolate(lf, [0, 11], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  // Caption slide-in.
  const capY = interpolate(lf, [3, 20], [26, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const capOp = interpolate(lf, [3, 20], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const winLeft = (WT_W - IMG_W) / 2;
  const winTop = 70;

  return (
    <AbsoluteFill style={{ background: "#0b1220" }}>
      <Background />

      {/* Feature title header */}
      <div style={{ position: "absolute", top: 22, left: winLeft, display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 30 }}>🌱</span>
        <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 28, color: "#eaf2ff" }}>{wt.title}</div>
        <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 16, color: wt.accent, border: `2px solid ${wt.accent}`, borderRadius: 8, padding: "3px 10px" }}>
          Step {i + 1} / {steps.length}
        </div>
      </div>

      {/* Browser window with the captured UI state + animated pointer */}
      <div style={{ position: "absolute", left: winLeft, top: winTop, width: IMG_W, borderRadius: 14, overflow: "hidden", boxShadow: "0 36px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)", background: "#0d1526" }}>
        <Chrome accent={wt.accent} />
        <div style={{ position: "relative", width: IMG_W, height: IMG_H, overflow: "hidden", background: "#fff" }}>
          {prev && <Img src={staticFile(prev.img)} style={{ position: "absolute", top: 0, left: 0, width: IMG_W }} />}
          <Img src={staticFile(cur.img)} style={{ position: "absolute", top: 0, left: 0, width: IMG_W, opacity: fadeIn }} />
          {cursor && <Ripple x={cursor.x} y={cursor.y} lf={cur.click ? lf : -999} accent={wt.accent} />}
          {cursor && <Pointer x={cursor.x} y={cursor.y} opacity={cursorOp} />}
        </div>
      </div>

      {/* Caption lower-third */}
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 34 }}>
        <div style={{ transform: `translateY(${capY}px)`, opacity: capOp, display: "flex", alignItems: "center", gap: 18, background: "rgba(7,12,22,0.72)", border: `1px solid rgba(255,255,255,0.08)`, borderRadius: 14, padding: "14px 26px", backdropFilter: "blur(4px)" }}>
          <div style={{ width: 5, height: 40, background: wt.accent, borderRadius: 8 }} />
          <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 34, color: "#eaf2ff" }}>{cur.caption}</div>
        </div>
        {/* progress dots */}
        <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
          {steps.map((_, k) => (
            <div key={k} style={{ width: k === i ? 30 : 10, height: 10, borderRadius: 99, background: k === i ? wt.accent : "rgba(255,255,255,0.22)", transition: "all .2s" }} />
          ))}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
