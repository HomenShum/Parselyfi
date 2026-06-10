import React from "react";
import { AbsoluteFill, Img, staticFile, useCurrentFrame, interpolate, Easing } from "remotion";

export const WT_FPS = 30;
export const WT_W = 1920;
export const WT_H = 1080;

const FONT = '"Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif';
const IMG_W = 1360;
const CAP_VW = 1280;                               // capture viewport CSS width
const IMG_H = Math.round(IMG_W * 800 / CAP_VW);    // preserve 1280x800 aspect
const SX = IMG_W / 1280, SY = IMG_H / 800;         // cursor coord -> displayed-image px

// Per-step "camera": zoom toward the click on action steps; pull back, gently
// zoomed + centered, on the result/loading states (the result is scrolled to
// the viewport centre at capture time). Pan/glide between steps (Arcade-style).
const ACTION_SCALE = 1.36, RESULT_SCALE = 1.14, OPEN_SCALE = 1.04;
const camTarget = (step) =>
  step.cursor
    ? { s: ACTION_SCALE, fx: step.cursor.x * SX, fy: step.cursor.y * SY }
    : { s: RESULT_SCALE, fx: IMG_W / 2, fy: IMG_H / 2 };

export const wtDuration = (wt) => wt.steps.reduce((a, s) => a + (s.hold || 60), 0);

// A BURST step holds a captured frame SEQUENCE (loading/streaming motion). Pick the
// frame for this local time — play once over ~72% of the hold, then rest on the last.
const burstFrame = (step, lf) => {
  if (step && step.burst && step.imgs && step.imgs.length) {
    const N = step.imgs.length;
    const playFrames = Math.max(N, Math.floor((step.hold || 60) * 0.72));
    const idx = Math.min(N - 1, Math.max(0, Math.floor(lf / (playFrames / N))));
    return step.imgs[idx];
  }
  return (step && (step.img || (step.imgs && step.imgs[step.imgs.length - 1]))) || null;
};

const Background = () => (
  <AbsoluteFill style={{ background: "radial-gradient(1300px 760px at 68% -5%, #14253f 0%, #0f1b2e 46%, #0b1220 100%)" }}>
    <AbsoluteFill style={{ background: "radial-gradient(900px 520px at 10% 100%, rgba(52,211,153,0.10), transparent 60%)" }} />
  </AbsoluteFill>
);

const Pointer = ({ x, y, opacity }) => (
  <svg width="46" height="46" viewBox="0 0 24 24" style={{ position: "absolute", left: x, top: y, opacity, transform: "translate(-3px,-2px)", filter: "drop-shadow(0 4px 6px rgba(0,0,0,0.55))", zIndex: 30 }}>
    <path d="M5 3 L5 20 L9.5 15.5 L12.5 22 L15 21 L12 14.5 L18.5 14.5 Z" fill="#fff" stroke="#0b1220" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);

const Ripple = ({ x, y, lf, accent }) => {
  const t = interpolate(lf, [20, 48], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const size = interpolate(t, [0, 1], [8, 104]);
  const op = interpolate(t, [0, 0.18, 1], [0, 0.9, 0]);
  return <div style={{ position: "absolute", left: x - size / 2, top: y - size / 2, width: size, height: size, borderRadius: "50%", border: `4px solid ${accent}`, opacity: op, zIndex: 29 }} />;
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
  const total = acc;
  let i = steps.findIndex((s, k) => frame >= starts[k] && frame < starts[k] + (steps[k].hold || 60));
  if (i < 0) i = steps.length - 1;
  const lf = frame - starts[i];
  const cur = steps[i];
  const prev = steps[i - 1];

  // ---- Camera: ease from previous target to this step's target (pre-move delay
  // then a gentle glide), so the eye registers context before the camera moves.
  const tgt = camTarget(cur);
  const prevTgt = i > 0 ? camTarget(prev) : { s: OPEN_SCALE, fx: IMG_W / 2, fy: IMG_H / 2 };
  const ct = interpolate(lf, [6, 26], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const s = prevTgt.s + (tgt.s - prevTgt.s) * ct;
  const fx = prevTgt.fx + (tgt.fx - prevTgt.fx) * ct;
  const fy = prevTgt.fy + (tgt.fy - prevTgt.fy) * ct;
  let tx = IMG_W / 2 - fx * s, ty = IMG_H / 2 - fy * s;
  tx = Math.min(0, Math.max(IMG_W - IMG_W * s, tx));   // keep the scaled image covering the frame
  ty = Math.min(0, Math.max(IMG_H - IMG_H * s, ty));

  // ---- Pointer glide (in image-space; the camera scales it along with the UI).
  let cursor = null, cursorOp = 0;
  if (cur.cursor) {
    const c = { x: cur.cursor.x * SX, y: cur.cursor.y * SY };
    const from = prev && prev.cursor ? { x: prev.cursor.x * SX, y: prev.cursor.y * SY } : c;
    const t = interpolate(lf, [0, 18], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
    cursor = { x: from.x + (c.x - from.x) * t, y: from.y + (c.y - from.y) * t };
    cursorOp = interpolate(lf, [0, 8], [prev && prev.cursor ? 1 : 0, 1], { extrapolateRight: "clamp" });
  }

  const curImg = burstFrame(cur, lf);
  const prevImg = prev ? burstFrame(prev, 1e9) : null;
  const fadeIn = interpolate(lf, [0, 11], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const capY = interpolate(lf, [4, 22], [26, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const capOp = interpolate(lf, [4, 22], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const progress = (starts[i] + Math.min(lf, cur.hold || 60)) / total;

  const winLeft = (WT_W - IMG_W) / 2;
  const winTop = 70;

  return (
    <AbsoluteFill style={{ background: "#0b1220" }}>
      <Background />

      <div style={{ position: "absolute", top: 22, left: winLeft, display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 30 }}>🌱</span>
        <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 28, color: "#eaf2ff" }}>{wt.title}</div>
        <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 16, color: wt.accent, border: `2px solid ${wt.accent}`, borderRadius: 8, padding: "3px 10px" }}>
          Step {i + 1} / {steps.length}
        </div>
      </div>

      {/* Browser window — overflow clips the zoomed camera */}
      <div style={{ position: "absolute", left: winLeft, top: winTop, width: IMG_W, borderRadius: 14, overflow: "hidden", boxShadow: "0 36px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.06)", background: "#0d1526" }}>
        <Chrome accent={wt.accent} />
        <div style={{ position: "relative", width: IMG_W, height: IMG_H, overflow: "hidden", background: "#fff" }}>
          {/* Camera: zoom + pan toward the active region */}
          <div style={{ position: "absolute", top: 0, left: 0, width: IMG_W, height: IMG_H, transformOrigin: "0 0", transform: `translate(${tx}px, ${ty}px) scale(${s})` }}>
            {prevImg && <Img src={staticFile(prevImg)} style={{ position: "absolute", top: 0, left: 0, width: IMG_W }} />}
            <Img src={staticFile(curImg)} style={{ position: "absolute", top: 0, left: 0, width: IMG_W, opacity: fadeIn }} />
            {cursor && <Ripple x={cursor.x} y={cursor.y} lf={cur.click ? lf : -999} accent={wt.accent} />}
            {cursor && <Pointer x={cursor.x} y={cursor.y} opacity={cursorOp} />}
          </div>
        </div>
      </div>

      {/* Caption lower-third */}
      <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 46 }}>
        <div style={{ transform: `translateY(${capY}px)`, opacity: capOp, display: "flex", alignItems: "center", gap: 18, background: "rgba(7,12,22,0.78)", border: "1px solid rgba(255,255,255,0.10)", borderRadius: 14, padding: "14px 28px", backdropFilter: "blur(4px)" }}>
          <div style={{ width: 5, height: 40, background: wt.accent, borderRadius: 8 }} />
          <div style={{ fontFamily: FONT, fontWeight: 700, fontSize: 34, color: "#eaf2ff" }}>{cur.caption}</div>
        </div>
      </AbsoluteFill>

      {/* Progress bar (bottom) */}
      <div style={{ position: "absolute", bottom: 0, left: 0, height: 6, width: WT_W * progress, background: wt.accent }} />
    </AbsoluteFill>
  );
};
